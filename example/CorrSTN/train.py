import os
from time import time

import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from args import args
from corrstn import CorrSTN
from dataset import TrafficFlowDataset
from paddle.io import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import (
    CosineAnnealingWithWarmupDecay,
    EarlyStopping,
    Logger,
    masked_mape_np,
    norm_adj_matrix,
)

# paddle.seed(2357)

class Trainer:
    def __init__(self, training_args):

        self.training_args = training_args

        self.folder_dir = (
            f"MAE_{training_args.model_name}_elayer{training_args.encoder_num_layers}_"
            + f"dlayer{training_args.decoder_num_layers}_head{training_args.head}_dm{training_args.d_model}_"
            + f"einput{training_args.encoder_input_size}_dinput{training_args.decoder_input_size}_"
            + f"doutput{training_args.decoder_output_size}_drop{training_args.dropout}_"
            + f"lr{training_args.learning_rate}_wd{training_args.weight_decay}_bs{training_args.batch_size}_"
            + f"topk{training_args.top_k}_att{training_args.attention}_trepoch{training_args.train_epochs}_"
            + f"finepoch{training_args.finetune_epochs}"
        )

        self.save_path = os.path.join(
            "experiments", training_args.dataset_name, self.folder_dir
        )
        os.makedirs(self.save_path, exist_ok=True)
        self.logger = Logger("CorrSTN", os.path.join(self.save_path, "log.txt"))

        if training_args.start_epoch == 0:
            self.logger.info(f"create params directory {self.save_path}")
        elif training_args.start_epoch > 0:
            self.logger.info(f"train from params directory {self.save_path}")

        self.logger.info(f"save folder: {self.folder_dir}")
        self.logger.info(f"save path  : {self.save_path}")
        self.logger.info(f"log  file  : {self.logger.log_file}")

        self.early_stopping = EarlyStopping(patience=training_args.patience, delta=0.0)

        self._init_train()
        self._build_data()
        self._build_model()
        self._build_optim()

    def _init_train(self):
        pass

    def _build_data(self):
        self.train_dataset = TrafficFlowDataset(self.training_args, "train")
        self.val_dataset = TrafficFlowDataset(self.training_args, "val")
        self.test_dataset = TrafficFlowDataset(self.training_args, "test")

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.eval_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.training_args.batch_size,
            shuffle=False,
            drop_last=True,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.training_args.batch_size,
            shuffle=False,
            drop_last=True,
        )

    def _build_model(self):
        adj_mx = np.load(os.path.join(self.training_args.adj_path))[0, :, :]
        self.norm_adj_matrix = paddle.to_tensor(
            norm_adj_matrix(adj_mx), dtype=paddle.get_default_dtype()
        )

        nn.initializer.set_global_initializer(
            nn.initializer.XavierUniform(), nn.initializer.XavierUniform()
        )

        self.net = CorrSTN(self.training_args, adj_matrix=self.norm_adj_matrix)

        if self.training_args.continue_training:
            # params_filename = os.path.join(
            #     self.save_path, f"epoch_{self.start_epoch}.params"
            # )
            params_filename = os.path.join(
                self.save_path, f"epoch_best.params"
            )
            self.net.load_state_dict(paddle.load(params_filename))
            self.logger.info(f"load weight from: {params_filename}")

        self.logger.info(self.net)

        total_param = 0
        self.logger.info("Net's state_dict:")
        for param_tensor in self.net.state_dict():
            self.logger.info(
                f"{param_tensor} \t {self.net.state_dict()[param_tensor].shape}"
            )
            total_param += np.prod(self.net.state_dict()[param_tensor].shape)
        self.logger.info(f"Net's total params: {total_param}.")

        self.criterion1 = nn.L1Loss()  # 定义损失函数
        self.criterion2 = nn.MSELoss()  # 定义损失函数

    def _build_optim(self):
        self.lr_scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=self.training_args.learning_rate,
            min_lr=self.training_args.learning_rate * 0.1,
            warmup_step=10,
            decay_step=30,
        )

        # 定义优化器，传入所有网络参数
        self.optimizer = optim.Adam(
            parameters=self.net.parameters(),
            learning_rate=self.lr_scheduler,
            weight_decay=self.training_args.weight_decay,
        )

        self.logger.info("Optimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            self.logger.info(f"{var_name} \t {self.optimizer.state_dict()[var_name]}")

    def train(self):
        self.logger.info("start train...")

        s_time = time()
        best_eval_loss = np.inf
        best_epoch = 0
        global_step = 0
        epoch = self.training_args.start_epoch

        while (
            epoch < self.training_args.train_epochs + self.training_args.finetune_epochs
        ):
            # finetune => load best trainging model
            if epoch == self.training_args.train_epochs:
                self._init_finetune()

            self.net.train()  # ensure dropout layers are in train mode
            tr_s_time = time()
            epoch_step = 0
            self.lr_scheduler.step()
            for batch_index, batch_data in enumerate(self.train_dataloader):
                src, tgt = batch_data
                _, training_loss = self.train_one_step(src, tgt)
                # self.logger.info(f"training_loss: {training_loss.numpy()}")
                epoch_step += 1
                global_step += 1
            self.logger.info(f"learning_rate: {self.optimizer.get_lr()}")
            self.logger.info(f"epoch: {epoch}, train time cost:{time() - tr_s_time}")
            self.logger.info(f"epoch: {epoch}, total time cost:{time() - s_time}")

            # apply model on the validation data set
            eval_loss = self.compute_eval_loss()
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                self.logger.info(f"best_epoch: {best_epoch}")
                self.logger.info(f"eval_loss: {eval_loss.numpy()}")
                self.compute_test_loss()
                # save parameters
                # params_filename = os.path.join(self.save_path, f"epoch_{epoch}.params")
                # params_filename = os.path.join(self.save_path, f"epoch_best.params")
                # paddle.save(self.net.state_dict(), params_filename)
                # self.logger.info(f"save parameters to file: {params_filename}")

            self.early_stopping(val_loss=eval_loss)
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping")
                break
            else:
                epoch += 1

        self.logger.info(f"best epoch: {best_epoch}")
        self.logger.info("apply the best val model on the test dataset ...")

        # params_filename = os.path.join(self.save_path, f"epoch_{best_epoch}.params")
        # params_filename = os.path.join(self.save_path, f"epoch_best.params")
        # self.logger.info(f"load weight from: {params_filename}")
        # self.net.set_state_dict(paddle.load(params_filename))
        self.compute_test_loss()

    def _init_finetune(self):
        self.logger.info("Start FineTune Training")
        # params_filename = os.path.join(self.save_path, f"epoch_{best_epoch}.params")
        params_filename = os.path.join(self.save_path, f"epoch_best.params")
        # self.net.set_state_dict(paddle.load(params_filename))
        self.logger.info(f"load weight from: {params_filename}")

        self.early_stopping.reset()

        self.optimizer._learning_rate.max_lr = self.training_args.learning_rate * 0.1
        self.optimizer._learning_rate.min_lr = self.training_args.learning_rate * 0.01

    def train_one_step(self, src, tgt):
        # fix his_index for his_len = 2016
        # fix_week = paddle.arange(start=0, end=self.training_args.tgt_len)
        fix_day = paddle.arange(
            start=self.training_args.his_len - 288,
            end=self.training_args.his_len - 288 + 12,
        )
        fix_hour = paddle.arange(
            start=self.training_args.his_len - 12, end=self.training_args.his_len
        )
        # tgt_idx = paddle.arange(
        #     start=self.training_args.his_len,
        #     end=self.training_args.his_len + self.training_args.tgt_len,
        # )
        encoder_idx = [fix_day, fix_hour]
        encoder_input = paddle.index_select(src, paddle.concat(encoder_idx), axis=2)
        encoder_idx = paddle.concat(encoder_idx).expand(
            [self.training_args.batch_size, self.training_args.num_nodes, -1]
        )
        decoder_output = self.net(src=encoder_input, src_idx=encoder_idx, tgt=tgt)
        # decoder_output = paddle.where(tgt == -1, tgt, decoder_output)
        loss = self.criterion2(decoder_output, tgt)

        if self.net.training:
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()
        return decoder_output, loss

    def compute_eval_loss(self):
        self.net.eval()

        with paddle.no_grad():
            all_eval_loss = []  # 记录了所有batch的loss
            start_time = time()
            for batch_index, batch_data in enumerate(self.eval_dataloader):
                src, tgt = batch_data
                predict_output, eval_loss = self.train_one_step(src, tgt)
                all_eval_loss.append(eval_loss)

            eval_loss = sum(all_eval_loss) / len(all_eval_loss)
            self.logger.info(f"eval cost time: {time() - start_time}s")
            self.logger.info(f"eval_loss: {eval_loss.numpy()}")
        return eval_loss
    
    def test_val_one_step(self, src, tgt):
        fix_day = paddle.arange(
            start=self.training_args.his_len - 288,
            end=self.training_args.his_len - 288 + 12,
        )
        fix_hour = paddle.arange(
            start=self.training_args.his_len - 12, end=self.training_args.his_len
        )
        encoder_idx = [fix_day, fix_hour]
        encoder_input = paddle.index_select(src, paddle.concat(encoder_idx), axis=2)
        encoder_idx = paddle.concat(encoder_idx).expand(
            [self.training_args.batch_size, self.training_args.num_nodes, -1]
        )
        
        decoder_start_inputs = tgt[:, :, :1, :]
        decoder_input_list = [decoder_start_inputs]
        
        encoder_output = self.net.encode(encoder_input, encoder_idx)

        for step in range(self.training_args.tgt_len):
            decoder_inputs = paddle.concat(decoder_input_list, axis=2)
            predict_output = self.net.decode(decoder_inputs, encoder_output)
            decoder_input_list = [decoder_start_inputs, predict_output]
            
        return predict_output, None
    
    def compute_test_loss(self, in_training=False):
        self.net.eval()

        with paddle.no_grad():
            preds = []
            tgts = []
            start_time = time()
            for batch_index, batch_data in enumerate(self.test_dataloader):
                src, tgt = batch_data
                predict_output, _ = self.test_val_one_step(src, tgt)

                preds.append(predict_output.detach().numpy())
                tgts.append(tgt.detach().numpy())
            self.logger.info(f"test time on whole data: {time() - start_time}s")

            preds = np.concatenate(preds, axis=0)  # [B,N,T,1]
            trues = np.concatenate(tgts, axis=0)  # [B,N,T,F]
            preds = self.test_dataset.inverse_transform(preds, axis=-1)  # [B,N,T,1]
            trues = self.test_dataset.inverse_transform(trues, axis=-1)  # [B,N,T,1]

            self.logger.info(f"preds: {str(preds.shape)}")
            self.logger.info(f"tgts: {trues.shape}")

            # 计算误差
            excel_list = []
            prediction_length = trues.shape[2]

            for i in range(prediction_length):
                assert preds.shape[0] == trues.shape[0]
                mae = mean_absolute_error(preds[:, :, i, 0], trues[:, :, i, 0])
                rmse = mean_squared_error(preds[:, :, i, 0], trues[:, :, i, 0]) ** 0.5
                mape = masked_mape_np(preds[:, :, i, 0], trues[:, :, i, 0], 0)
                self.logger.info(f"{i} MAE: {mae}")
                self.logger.info(f"{i} RMSE: {rmse}")
                self.logger.info(f"{i} MAPE: {mape}")
                excel_list.extend([mae, rmse, mape])

            # print overall results
            mae = mean_absolute_error(preds.reshape(-1, 1), trues.reshape(-1, 1))
            rmse = mean_squared_error(preds.reshape(-1, 1), trues.reshape(-1, 1)) ** 0.5
            mape = masked_mape_np(preds.reshape(-1, 1), trues.reshape(-1, 1), 0)
            self.logger.info(f"all MAE: {mae}")
            self.logger.info(f"all RMSE: {rmse}")
            self.logger.info(f"all MAPE: {mape}")
            excel_list.extend([mae, rmse, mape])
            self.logger.info(excel_list)

    def run_test(self, epoch):
        params_filename = os.path.join(self.save_path, "epoch_%s.params" % epoch)
        print("load weight from:", params_filename, flush=True)
        self.net.load_state_dict(paddle.load(params_filename))
        self.compute_test_loss()


if __name__ == "__main__":
    trainer = Trainer(training_args=args)
    trainer.train()
