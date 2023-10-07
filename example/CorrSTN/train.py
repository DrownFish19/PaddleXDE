import os
from time import time

import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from args import args
from corrstn import CorrSTN
from dataset import TrafficFlowDataset
from metrics import regressor_scores
from paddle.io import DataLoader
from tqdm import tqdm
from utils import (
    CosineAnnealingWithWarmupDecay,
    EarlyStopping,
    Logger,
    multichannel_norm_adj,
)


class Trainer:
    def __init__(self, training_args):

        self.training_args = training_args

        self.folder_dir = f"MAE_${training_args.model_name}_elayer${training_args.encoder_num_layers}_"
        f"dlayer${training_args.decoder_num_layers}_head${training_args.head}_dm${training_args.d_model}_"
        f"einput${training_args.encoder_input_size}_dinput${training_args.decoder_input_size}_"
        f"doutput${training_args.decoder_output_size}_nlabel%d_ntruth%d_drop${training_args.dropout}_"
        f"lr{training_args.learning_rate}_wd${training_args.weight_decay}_bs${training_args.batch_size}_"
        f"topk${training_args.top_k}_att${training_args.attention}_trepoch${training_args.train_epochs}_"
        f"finepoch${training_args.finetune_epochs}"

        self.save_path = os.path.join(
            "experiments", training_args.dataset_name, self.folder_dir
        )
        self.logger = Logger("CorrSTN", os.path.join(self.save_path, "log.txt"))

        self.logger.info(f"save folder: ${self.folder_dir}")
        self.logger.info(f"save path  : ${self.save_path}")
        self.logger.info(f"log  file  : ${self.logger.log_file}")

        if (self.start_epoch == 0) and (not os.path.exists(self.save_path)):
            # 从头开始训练，就要重新构建文件夹
            os.makedirs(self.save_path, exist_ok=True)
            self.logger.info(f"create params directory ${self.save_path}")
        elif (self.start_epoch == 0) and (os.path.exists(self.save_path)):
            os.makedirs(self.save_path, exist_ok=True)
            self.logger.warning(f"Please delete the old directory ${self.save_path}")
        elif (self.start_epoch > 0) and (os.path.exists(self.save_path)):
            # 从中间开始训练，就要保证原来的目录存在
            self.logger.info(f"train from params directory ${self.save_path}")

        self.early_stopping = EarlyStopping(patience=training_args.patience, delta=0.0)

        self._init_train()
        self._build_data()
        self._build_model()
        self._build_optim()

    def _init_train(self):
        pass

    def _build_model(self):
        nn.initializer.set_global_initializer(
            nn.initializer.XavierUniform(), nn.initializer.XavierUniform()
        )

        self.net = CorrSTN(self.training_args, norm_adj_matrix=self.norm_adj_matrix)

        if self.training_args.continue_training:
            params_filename = os.path.join(
                self.save_path, f"epoch_${self.start_epoch}.params"
            )
            self.net.load_state_dict(paddle.load(params_filename))
            self.logger.info(f"load weight from: ${params_filename}")

        self.logger.info(self.net)

        total_param = 0
        self.logger.info("Net's state_dict:")
        for param_tensor in self.net.state_dict():
            self.logger.info(
                f"${param_tensor} \t ${self.net.state_dict()[param_tensor].size()}"
            )
            total_param += np.prod(self.net.state_dict()[param_tensor].size())
        self.logger.info(f"Net's total params: ${total_param}.")

        self.criterion1 = nn.L1Loss()  # 定义损失函数
        self.criterion2 = nn.MSELoss()  # 定义损失函数

    def _build_data(self):
        self.train_dataset = TrafficFlowDataset(self.training_args, "train")
        self.val_dataset = TrafficFlowDataset(self.training_args, "val")
        self.test_dataset = TrafficFlowDataset(self.training_args, "test")

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_args.batch_size,
            shuffle=True,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.training_args.batch_size,
            shuffle=True,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.training_args.batch_size,
            shuffle=True,
        )

        adj_mx = np.load(os.path.join(self.training_args.adj_matrix))[-1:, :, :]
        self.norm_adj_matrix = paddle.from_numpy(multichannel_norm_adj(adj_mx))

    def _build_optim(self):
        self.lr_scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=self.training_args.learning_rate,
            min_lr=self.training_args.learning_rate * 0.1,
            warmup_step=200,
            decay_step=2000,
        )

        # 定义优化器，传入所有网络参数
        self.optimizer = optim.AdamW(
            parameters=self.net.parameters(),
            learning_rate=self.lr_scheduler,
            weight_decay=self.training_args.weight_decay,
        )

        self.logger.info("Optimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            self.logger.info(f"${var_name} \t ${self.optimizer.state_dict()[var_name]}")

    def train(self):
        self.logger.info("start train...")

        s_time = time()
        best_val_loss = np.inf
        best_epoch = 0
        global_step = 0
        epoch = self.start_epoch
        epoch_limit_step = 300

        while epoch < self.train_epochs + self.finetune_epochs:
            # finetune => load best trainging model
            if epoch == self.train_epochs:
                self._init_finetune()

            self.net.train()  # ensure dropout layers are in train mode
            tr_s_time = time()
            epoch_step = 0

            for batch_index, batch_data in enumerate(self.train_dataloader):
                src, tgt = batch_data
                _, training_loss = self.train_one_step(src, tgt)
                epoch_step += 1
                global_step += 1

            self.logger.info(f"epoch: ${epoch}, train time cost:${time() - tr_s_time}")
            self.logger.info(f"epoch: ${epoch}, total time cost:${time() - s_time}")

            # save parameters
            params_filename = os.path.join(self.save_path, f"epoch_${epoch}.params")
            paddle.save(self.net.state_dict(), params_filename)
            self.logger.info(f"save parameters to file: ${params_filename}")

            # apply model on the validation data set
            val_loss = self.compute_val_loss()
            test_loss = self.compute_test_loss()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                self.logger.info(f"best_epoch: ${best_epoch}")

            self.logger.info(f"val_loss: ${val_loss}")
            self.logger.info(f"test_loss: ${test_loss}")

            self.early_stopping(val_loss=val_loss)
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping")
                break
            else:
                epoch += 1

        self.logger.info(f"best epoch: ${best_epoch}")
        self.logger.info("apply the best val model on the test dataset ...")

        params_filename = os.path.join(self.save_path, f"epoch_${best_epoch}.params")
        self.logger.info(f"load weight from: ${params_filename}")
        self.net.load_state_dict(paddle.load(params_filename))
        self.compute_test_loss()

    def _init_finetune(self):
        self.logger.info("Start FineTune Training")
        params_filename = os.path.join(self.save_path, "epoch_${best_epoch}.params")
        self.net.load_state_dict(paddle.load(params_filename))
        self.logger.info(f"load weight from: ${params_filename}")

        self.early_stopping.reset()

        self.optimizer._learning_rate.max_lr = self.training_args.learning_rate * 0.1
        self.optimizer._learning_rate.min_lr = self.training_args.learning_rate * 0.01

    def train_one_step(self, src, tgt):
        B, T1, N, F = src.shape
        B, T2, N, F = tgt.shape
        encoder_input = src
        decoder_input = tgt
        encoder_idx = paddle.arange(0, T1)
        decoder_idx = paddle.arange(T1, T1 + T2)

        decoder_output = self.net(
            src=encoder_input,
            src_idx=encoder_idx,
            tgt=decoder_input,
            tgt_idx=decoder_idx,
        )
        loss = self.criterion1(decoder_output, tgt)

        if self.net.training:
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()
            return decoder_output, loss
        else:
            return decoder_output

    def compute_val_loss(self):
        self.net.eval()

        with paddle.no_grad():
            tmp = []  # 记录了所有batch的loss
            start_time = time()
            for batch_index, batch_data in enumerate(self.val_dataloader):
                src, src_date, tgt, tgt_date, tgt_state = batch_data
                src = src.type(paddle.FloatTensor).to(self.DEVICE)
                src_date = src_date.type(paddle.FloatTensor).to(self.DEVICE)
                tgt = tgt.type(paddle.FloatTensor).to(self.DEVICE)
                tgt_date = tgt_date.type(paddle.FloatTensor).to(self.DEVICE)

                predict_output, loss = self.run_one_batch(
                    src, src_date, tgt, tgt_date, None, finetune=True
                )
                tmp.append(loss)
                self.sw.add_scalar("validation loss step", loss, self.val_step)
                self.val_step += 1

            validation_loss = sum(tmp) / len(tmp)
            print("validation cost time: %.4fs" % (time() - start_time), flush=True)
            print("validation_loss     : %.4f" % validation_loss, flush=True)
        return validation_loss

    def compute_test_loss(self, in_training=False):
        self.net.eval()

        with paddle.no_grad():
            predictions = []
            grounds = []
            tgt_states = []
            start_time = time()
            for batch_index, batch_data in enumerate(self.test_dataloader):
                src, src_date, tgt, tgt_date, tgt_state = batch_data
                src = src.type(paddle.FloatTensor).to(self.DEVICE)
                src_date = src_date.type(paddle.FloatTensor).to(self.DEVICE)
                tgt = tgt.type(paddle.FloatTensor).to(self.DEVICE)
                tgt_date = tgt_date.type(paddle.FloatTensor).to(self.DEVICE)

                predict_output, _ = self.run_one_batch(
                    src, src_date, tgt, tgt_date, None, finetune=True
                )

                predictions.append(predict_output.detach().cpu().numpy())
                grounds.append(tgt.detach().cpu().numpy())
                tgt_states.append(tgt_state)  # numpy
            print("test time on whole data:%.2fs" % (time() - start_time), flush=True)

            preds = np.concatenate(predictions, axis=0)  # [B,N,T,1]
            trues = np.concatenate(grounds, axis=0)  # [B,N,T,F]
            states = np.concatenate(tgt_states, axis=0)  # [B,N,T,1]
            preds = self.test_data_set.inverse_transform(preds, axis=-1)  # [B,N,T,1]
            trues = self.test_data_set.inverse_transform(
                trues[:, :, :, -1:], axis=-1
            )  # [B,N,T,1]

            print("prediction:", preds.shape, flush=True)
            print("target    :", trues.shape, flush=True)

            B, N, T, _ = trues.shape
            for n_idx in range(5):
                for t_idx in range(T):
                    self.sw.add_scalar(
                        "Turbine%d/pred" % n_idx, preds[3, n_idx, t_idx, 0], t_idx
                    )
                    self.sw.add_scalar(
                        "Turbine%d/true" % n_idx, trues[3, n_idx, t_idx, 0], t_idx
                    )

            # 计算误差
            all_mae, all_rmse = [], []
            for idx in range(trues.shape[1]):
                states_i = states[:, idx, :, :].reshape(-1)
                preds_i = preds[:, idx, :, :].reshape(-1)[np.where(states_i <= 2)]
                trues_i = trues[:, idx, :, :].reshape(-1)[np.where(states_i <= 2)]
                _mae, _rmse = regressor_scores(
                    prediction=preds_i / 1000, gt=trues_i / 1000
                )
                all_mae.append(_mae)
                all_rmse.append(_rmse)
                if not in_training:
                    print("Turbine ID %3d MAE: %.2f" % (idx, _mae), flush=True)
                    print("Turbine ID %3d RMSE: %.2f" % (idx, _rmse), flush=True)
                self.sw.add_scalar("Turbine ID MAE", _mae, idx)
                self.sw.add_scalar("Turbine ID RMSE", _rmse, idx)

            mae_score = np.array(all_mae).sum()
            rmse_score = np.array(all_rmse).sum()
            score = (mae_score + rmse_score) / 2
            print("all MAE : %.2f" % mae_score, flush=True)
            print("all RMSE: %.2f" % rmse_score, flush=True)
            print("Score   : %.2f" % score, flush=True)

    def run_test(self, epoch):
        params_filename = os.path.join(self.save_path, "epoch_%s.params" % epoch)
        print("load weight from:", params_filename, flush=True)
        self.net.load_state_dict(paddle.load(params_filename))
        self.compute_test_loss()

    def forcast(self, epoch, test_dataset=None):

        if test_dataset is None:
            test_dataset = self.test_data_set

        params_filename = os.path.join(self.save_path, "epoch_%s.params" % epoch)
        print("load weight from:", params_filename, flush=True)
        self.net.load_state_dict(
            paddle.load(params_filename, map_location=paddle.device("cpu"))
        )

        self.net.eval()

        source_len = max(
            self.num_of_hours * self.points_per_hour,
            (self.num_of_days * 24 + self.day_recent_hours) * self.points_per_hour,
        )
        target_len = self.total_preds * self.points_per_hour  # 48 * 6
        with paddle.no_grad():
            src = (
                paddle.from_numpy(test_dataset.data_x[:, -source_len:, :])
                .type(paddle.FloatTensor)
                .unsqueeze(0)
                .to(self.DEVICE)
            )
            src_date = (
                paddle.from_numpy(test_dataset.date_str[:, -source_len:, :])
                .type(paddle.FloatTensor)
                .unsqueeze(0)
                .to(self.DEVICE)
            )
            tgt = (
                paddle.from_numpy(test_dataset.data_x[:, -target_len:, :])
                .type(paddle.FloatTensor)
                .unsqueeze(0)
                .to(self.DEVICE)
            )
            tgt_date = (
                paddle.from_numpy(test_dataset.date_str[:, -target_len:, :])
                .type(paddle.FloatTensor)
                .unsqueeze(0)
                .to(self.DEVICE)
            )

            predict_output, _ = self.run_one_batch(
                src, src_date, tgt, tgt_date, None, finetune=True
            )
            predict_output = test_dataset.inverse_transform(
                predict_output.squeeze(0), axis=-1
            )
            return predict_output


if __name__ == "__main__":
    # scheme = RunModel()

    # scheme.train()
    pass

    # scheme.forcast(51)

    # scheme.run_test(51)
