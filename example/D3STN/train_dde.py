import contextlib
import os
from time import time

import args
import numpy as np
import paddle
import paddle.distributed as dist
import paddle.io as io
import paddle.nn as nn
import paddle.optimizer as optim
from d3stn import D3STN, DecoderIndex
from dataset import TrafficFlowDataset
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
)
from paddle.nn.initializer import Constant, XavierUniform
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import (
    CosineAnnealingWithWarmupDecay,
    EarlyStopping,
    Logger,
    get_adjacency_matrix_2direction,
    masked_mape_np,
    norm_adj_matrix,
)
from visualdl import LogWriter

from paddlexde.functional import ddeint
from paddlexde.solver.fixed_solver import RK4, Euler, Midpoint
from paddlexde.version import commit
from paddlexde.xde.base_dde import HistoryIndex


class Trainer:
    def __init__(self, training_args):
        dist.init_parallel_env()

        self.training_args = training_args

        self.folder_dir = (
            f"MAE_{training_args.model_name}_elayer{training_args.encoder_num_layers}_"
            + f"dlayer{training_args.decoder_num_layers}_head{training_args.head}_dm{training_args.d_model}_"
            + f"einput{training_args.encoder_input_size}_dinput{training_args.decoder_input_size}_"
            + f"doutput{training_args.decoder_output_size}_elen{training_args.his_len}_"
            + f"dlen{training_args.tgt_len}_drop{training_args.dropout}_"
            + f"lr{training_args.learning_rate}_wd{training_args.weight_decay}_bs{training_args.batch_size}_"
            + f"topk{training_args.top_k}_att{training_args.attention}_trepoch{training_args.train_epochs}_"
            + f"finepoch{training_args.finetune_epochs}_dde"
        )

        self.save_path = os.path.join(
            "experiments", training_args.dataset_name, self.folder_dir
        )

        if dist.get_rank() == 0:
            os.makedirs(self.save_path, exist_ok=True)
            self.writer = LogWriter(logdir=os.path.join(self.save_path, "visualdl"))
        if dist.get_world_size() > 1:
            dist.barrier()
        self.logger = Logger("D3STN", os.path.join(self.save_path, "log.txt"))
        self.writer = LogWriter(logdir=os.path.join(self.save_path, "visualdl"))

        if training_args.start_epoch == 0:
            self.logger.info(f"create params directory {self.save_path}")
        elif training_args.start_epoch > 0:
            self.logger.info(f"train from params directory {self.save_path}")

        self.logger.info(f"git commit: {commit}")
        self.logger.info(f"save folder: {self.folder_dir}")
        self.logger.info(f"save path  : {self.save_path}")
        self.logger.info(f"log  file  : {self.logger.log_file}")

        args_message = "\n".join(
            [f"{k:<20}: {v}" for k, v in vars(training_args).items()]
        )
        self.logger.info(f"training_args  : \n{args_message}")
        self.finetune = False
        self.early_stopping = EarlyStopping(patience=training_args.patience, delta=0.0)

        self._build_data()
        self._build_model()
        self._build_optim()

    def _build_data(self):
        self.train_dataset = TrafficFlowDataset(self.training_args, "train")
        self.val_dataset = TrafficFlowDataset(self.training_args, "val")
        self.test_dataset = TrafficFlowDataset(self.training_args, "test")

        if self.training_args.distribute and dist.get_world_size() > 1:
            assert self.training_args.batch_size % dist.get_world_size() == 0
            self.training_args.batch_size = int(
                self.training_args.batch_size / dist.get_world_size()
            )

        train_sampler = io.DistributedBatchSampler(
            self.train_dataset,
            batch_size=self.training_args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        eval_sampler = io.DistributedBatchSampler(
            self.val_dataset,
            batch_size=self.training_args.batch_size,
            drop_last=True,
        )
        test_sampler = io.DistributedBatchSampler(
            self.test_dataset,
            batch_size=self.training_args.batch_size,
            drop_last=True,
        )
        self.train_dataloader = io.DataLoader(
            self.train_dataset, batch_sampler=train_sampler
        )
        self.eval_dataloader = io.DataLoader(
            self.val_dataset, batch_sampler=eval_sampler
        )
        self.test_dataloader = io.DataLoader(
            self.test_dataset, batch_sampler=test_sampler
        )

        # 保持输入序列长度为12
        self.fix_week = paddle.arange(
            start=self.training_args.his_len - 2016,
            end=self.training_args.his_len - 2016 + 12,
        )
        self.fix_day = paddle.arange(
            start=self.training_args.his_len - 288,
            end=self.training_args.his_len - 288 + 12,
        )
        self.fix_hour = paddle.arange(
            start=self.training_args.his_len - 12,
            end=self.training_args.his_len,
        )
        self.fix_pred = paddle.arange(
            start=self.training_args.his_len,
            end=self.training_args.his_len + 12,
        )
        self.fix_pred = paddle.ones(shape=[self.training_args.tgt_len]) * (
            self.training_args.his_len - 1
        )

        encoder_idx = []
        decoder_idx = [self.fix_pred]

        # for week
        if self.training_args.his_len >= 2016:
            encoder_idx.append(self.fix_week)

        # for day
        elif self.training_args.his_len >= 288:
            encoder_idx.append(self.fix_day)

        # for hour
        elif self.training_args.his_len >= 12:
            encoder_idx.append(self.fix_hour)

        # concat all
        encoder_idx = paddle.concat(encoder_idx)
        decoder_idx = paddle.concat(decoder_idx)

        self.init_encoder_idx = encoder_idx
        self.init_decoder_idx = decoder_idx

        self.encoder_idx = paddle.create_parameter(
            shape=encoder_idx.shape, dtype="float32"
        )
        self.decoder_idx = paddle.create_parameter(
            shape=decoder_idx.shape, dtype="float32"
        )
        self.encoder_idx.set_value(paddle.cast(encoder_idx, "float32"))
        self.decoder_idx.set_value(paddle.cast(decoder_idx, "float32"))

        self.logger.info(f"encoder_idx: {self.encoder_idx}")
        self.logger.info(f"decoder_idx: {self.decoder_idx}")

    def _build_model(self):
        default_dtype = paddle.get_default_dtype()
        adj_matrix, _ = get_adjacency_matrix_2direction(
            self.training_args.adj_path, self.training_args.num_nodes
        )
        adj_matrix = paddle.to_tensor(norm_adj_matrix(adj_matrix), default_dtype)

        sc_matrix = np.load(self.training_args.sc_path)[0, :, :]
        sc_matrix = paddle.to_tensor(norm_adj_matrix(sc_matrix), default_dtype)

        nn.initializer.set_global_initializer(XavierUniform(), Constant(value=0.0))

        self.net = D3STN(
            self.training_args,
            adj_matrix=adj_matrix,
            sc_matrix=sc_matrix,
        )

        if self.training_args.continue_training:
            self.load()

        if self.training_args.distribute and dist.get_world_size() > 1:
            self.net = paddle.DataParallel(self.net)

        self.logger.debug(self.net)

        total_param = 0
        self.logger.debug("Net's state_dict:")
        for param_tensor in self.net.state_dict():
            self.logger.debug(
                f"{param_tensor} \t {self.net.state_dict()[param_tensor].shape}"
            )
            total_param += np.prod(self.net.state_dict()[param_tensor].shape)
        self.logger.debug(f"Net's total params: {total_param}.")

        self.criterion = nn.L1Loss()  # 定义损失函数

    def _build_optim(self):
        self.lr_scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=1,
            min_lr=0.1,
            warmup_step=self.training_args.warmup_step,
            decay_step=self.training_args.decay_step,
        )

        parameters = [
            {
                "params": self.net.parameters(),
                "learning_rate": self.training_args.learning_rate,
            },
            {
                "params": [self.decoder_idx],
                "learning_rate": self.training_args.learning_rate * 0.1,
            },
            {
                "params": [self.encoder_idx],
                "learning_rate": self.training_args.learning_rate * 0.1,
            },
        ]

        # 定义优化器，传入所有网络参数
        self.optimizer = optim.Adam(
            parameters=parameters,
            learning_rate=self.lr_scheduler,
            weight_decay=float(self.training_args.weight_decay),
            multi_precision=True,
        )

        self.logger.info("Optimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            self.logger.info(f"{var_name} \t {self.optimizer.state_dict()[var_name]}")

        if self.training_args.solver == "euler":
            self.dde_solver = Euler
        elif self.training_args.solver == "midpoint":
            self.dde_solver = Midpoint
        elif self.training_args.solver == "rk4":
            self.dde_solver = RK4

        self.logger.info(f"dde_solver: {self.dde_solver}")

    def save(self, epoch=None):
        if dist.get_rank() != 0:
            return

        if epoch is not None:
            params_filename = os.path.join(self.save_path, f"epoch_{epoch}.params")
            encoder_idx_filename = os.path.join(self.save_path, f"epoch_{epoch}.enidx")
            decoder_idx_filename = os.path.join(self.save_path, f"epoch_{epoch}.deidx")
        else:
            params_filename = os.path.join(self.save_path, "epoch_best.params")
            encoder_idx_filename = os.path.join(self.save_path, "epoch_best.enidx")
            decoder_idx_filename = os.path.join(self.save_path, "epoch_best.deidx")
        paddle.save(self.net.state_dict(), params_filename)
        paddle.save(self.encoder_idx, encoder_idx_filename)
        paddle.save(self.decoder_idx, decoder_idx_filename)
        self.logger.info(f"save parameters to file: {params_filename}")

    def load(self, epoch=None):
        if epoch is not None:
            params_filename = os.path.join(self.save_path, f"epoch_{epoch}.params")
            encoder_idx_filename = os.path.join(self.save_path, f"epoch_{epoch}.enidx")
            decoder_idx_filename = os.path.join(self.save_path, f"epoch_{epoch}.deidx")
        else:
            params_filename = os.path.join(self.save_path, "epoch_best.params")
            encoder_idx_filename = os.path.join(self.save_path, "epoch_best.enidx")
            decoder_idx_filename = os.path.join(self.save_path, "epoch_best.deidx")

        self.net.set_state_dict(paddle.load(params_filename))
        self.encoder_idx.set_value(paddle.load(encoder_idx_filename))
        self.decoder_idx.set_value(paddle.load(decoder_idx_filename))
        self.logger.info(f"load weight from: {params_filename}")

    def train(self):
        self.logger.info("start train...")

        s_time = time()
        best_eval_loss = np.inf
        best_epoch = 0
        global_step = 0
        epoch = self.training_args.start_epoch

        self.train_func = self.train_one_step
        while (
            epoch < self.training_args.train_epochs + self.training_args.finetune_epochs
        ):
            # finetune => load best training model
            if epoch == self.training_args.train_epochs:
                self.compute_test_loss(epoch)
                self._init_finetune()
                self.train_func = self.finetune_one_step

            self.net.train()  # ensure dropout layers are in train mode
            tr_s_time = time()
            epoch_step = 0
            self.lr_scheduler.step()
            for batch_index, batch_data in enumerate(self.train_dataloader):
                src, tgt = batch_data
                src = paddle.cast(src, paddle.get_default_dtype())
                tgt = paddle.cast(tgt, paddle.get_default_dtype())
                _, training_loss = self.train_func(src, tgt)
                self.writer.add_scalar("train/loss", training_loss, global_step)
                self.writer.add_scalar("train/lr", self.optimizer.get_lr(), global_step)
                epoch_step += 1
                global_step += 1
            self.logger.info(f"learning_rate: {self.optimizer.get_lr()}")
            self.logger.info(f"epoch: {epoch}, train time cost:{time() - tr_s_time}")
            self.logger.info(f"epoch: {epoch}, total time cost:{time() - s_time}")

            # apply model on the validation data set
            eval_loss = self.compute_eval_loss(epoch)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                self.logger.info(f"best_epoch: {best_epoch}")
                self.logger.info(f"eval_loss: {eval_loss}")
                self.compute_test_loss(epoch)
                # save parameters
                self.save(epoch=epoch)
                self.save()

            self.early_stopping(val_loss=eval_loss)
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping")
                if epoch < self.training_args.train_epochs:
                    epoch = self.training_args.train_epochs
                else:
                    break
            else:
                epoch += 1

        self.logger.info(f"best epoch: {best_epoch}")
        self.logger.info("apply the best val model on the test dataset ...")

        self.load()
        self.compute_test_loss(epoch)

    def _init_finetune(self):
        self.logger.info("Start FineTune Training")
        self.load()

        self.early_stopping.reset()

        parameters = [
            {
                "params": self.net.parameters(),
                "learning_rate": self.training_args.learning_rate,
            },
            {
                "params": [self.decoder_idx],
                "learning_rate": 0.0,
            },
            {
                "params": [self.encoder_idx],
                "learning_rate": 0.0,
            },
        ]

        # 定义优化器，传入所有网络参数
        self.optimizer = optim.Adam(
            parameters=parameters,
            learning_rate=1.0,
            weight_decay=float(self.training_args.weight_decay),
            multi_precision=True,
        )
        self.finetune = True

    def train_one_step(self, src, tgt):
        """_summary_

        Args:
            src (_type_): [B,N,T,D]
            tgt (_type_): [B,N,T,D]

        Returns:
            _type_: _description_
        """
        self.net.train()

        y0 = DecoderIndex.apply(
            lags=self.decoder_idx,
            his=src,
            his_span=paddle.arange(self.training_args.his_len),
        )

        preds, encoder_input = ddeint(
            func=self.net,
            y0=y0,
            t_span=paddle.arange(1 + 1),
            lags=self.encoder_idx,
            his=src,
            his_span=paddle.arange(self.training_args.his_len),
            solver=self.dde_solver,
            fixed_solver_interp="",
        )
        pred_len = y0.shape[-2]
        preds = preds[:, :, -pred_len:, :1]

        delay_log_softmax = paddle.nn.functional.log_softmax(
            encoder_input[..., :1], axis=-2
        )
        tgt_softmax = paddle.nn.functional.softmax(tgt[..., :1], axis=-2)
        kl_loss = paddle.nn.functional.kl_div(delay_log_softmax, tgt_softmax)
        loss = self.criterion(preds, tgt[..., :1]) + 0.01 * kl_loss
        loss.backward()
        if self.training_args.distribute and dist.get_world_size() > 1:
            fused_allreduce_gradients([self.encoder_idx, self.decoder_idx], None)
        self.optimizer.step()
        self.optimizer.clear_grad()

        return preds, loss

    def finetune_one_step(self, src, tgt):
        """_summary_

        Args:
            src (_type_): [B,N,T,D]
            tgt (_type_): [B,N,T,D]

        Returns:
            _type_: _description_
        """
        self.net.train()

        y0 = DecoderIndex.apply(
            lags=self.decoder_idx,
            his=src,
            his_span=paddle.arange(self.training_args.his_len),
        )
        encoder_input = HistoryIndex.apply(
            lags=self.encoder_idx,
            his=src,
            his_span=paddle.arange(self.training_args.his_len),
        )

        if self.training_args.distribute and dist.get_world_size() > 1:
            encoder_func = self.net._layers.encode
            decoder_func = self.net._layers.decode
        else:
            encoder_func = self.net.encode
            decoder_func = self.net.decode

        def dist_no_sync():
            if self.training_args.distribute and dist.get_world_size() > 1:
                return self.net.no_sync()
            else:
                return contextlib.nullcontext()

        with dist_no_sync():
            encoder_output = encoder_func(encoder_input)
            preds, _ = ddeint(
                func=decoder_func,
                y0=y0,
                t_span=paddle.arange(1 + 1),
                lags=None,
                his=encoder_output,
                his_span=None,
                solver=self.dde_solver,
                his_processed=True,
                fixed_solver_interp="",
            )
            pred_len = y0.shape[-2]
            preds = preds[:, :, -pred_len:, :1]

            loss = self.criterion(preds, tgt[..., :1])
            loss.backward()

        if self.training_args.distribute and dist.get_world_size() > 1:
            fused_allreduce_gradients(
                list(self.net.parameters()) + [self.encoder_idx, self.decoder_idx], None
            )
        self.optimizer.step()
        self.optimizer.clear_grad()

        return preds, loss

    def eval_one_step(self, src, tgt):
        self.net.eval()
        y0 = DecoderIndex.apply(
            lags=self.decoder_idx,
            his=src,
            his_span=paddle.arange(self.training_args.his_len),
        )
        encoder_input = HistoryIndex.apply(
            lags=self.encoder_idx,
            his=src,
            his_span=paddle.arange(self.training_args.his_len),
        )

        if self.training_args.distribute and dist.get_world_size() > 1:
            encoder_func = self.net._layers.encode
            decoder_func = self.net._layers.decode
        else:
            encoder_func = self.net.encode
            decoder_func = self.net.decode

        encoder_output = encoder_func(encoder_input)

        preds, _ = ddeint(
            func=decoder_func,
            y0=y0,
            t_span=paddle.arange(1 + 1),
            lags=None,
            his=encoder_output,
            his_span=None,
            solver=self.dde_solver,
            his_processed=True,
            fixed_solver_interp="",
        )
        pred_len = y0.shape[-2]
        preds = preds[:, :, -pred_len:, :1]

        loss = self.criterion(preds, tgt[..., :1])

        return preds, loss

    def test_one_step(self, src, tgt):
        self.net.eval()
        y0 = DecoderIndex.apply(
            lags=self.decoder_idx,
            his=src,
            his_span=paddle.arange(self.training_args.his_len),
        )
        encoder_input = HistoryIndex.apply(
            lags=self.encoder_idx,
            his=src,
            his_span=paddle.arange(self.training_args.his_len),
        )

        if self.training_args.distribute and dist.get_world_size() > 1:
            encoder_func = self.net._layers.encode
            decoder_func = self.net._layers.decode
        else:
            encoder_func = self.net.encode
            decoder_func = self.net.decode

        encoder_output = encoder_func(encoder_input)

        preds, _ = ddeint(
            func=decoder_func,
            y0=y0,
            t_span=paddle.arange(1 + 1),
            lags=None,
            his=encoder_output,
            his_span=None,
            solver=self.dde_solver,
            his_processed=True,
            fixed_solver_interp="",
        )
        pred_len = y0.shape[-2]
        preds = preds[:, :, -pred_len:, :1]

        loss = self.criterion(preds, tgt[..., :1])

        init_encoder_input = HistoryIndex.apply(
            lags=self.init_encoder_idx,
            his=src,
            his_span=paddle.arange(self.training_args.his_len),
        )

        init_delay_log_softmax = paddle.nn.functional.log_softmax(
            init_encoder_input[..., :1], axis=-2
        )
        delay_log_softmax = paddle.nn.functional.log_softmax(
            encoder_input[..., :1], axis=-2
        )
        tgt_softmax = paddle.nn.functional.softmax(tgt[..., :1], axis=-2)
        init_kl_loss = paddle.nn.functional.kl_div(init_delay_log_softmax, tgt_softmax)
        kl_loss = paddle.nn.functional.kl_div(delay_log_softmax, tgt_softmax)
        return preds, loss, kl_loss, init_kl_loss

    def compute_eval_loss(self, epoch=-1):
        with paddle.no_grad():
            all_eval_loss = paddle.zeros([1], dtype=paddle.get_default_dtype())
            start_time = time()
            for batch_index, batch_data in enumerate(self.eval_dataloader):
                src, tgt = batch_data
                src = paddle.cast(src, paddle.get_default_dtype())
                tgt = paddle.cast(tgt, paddle.get_default_dtype())
                predict_output, eval_loss = self.eval_one_step(src, tgt)

                all_eval_loss += eval_loss

            eval_loss = (all_eval_loss / len(self.eval_dataloader)).cpu().numpy()

            all_eval_loss = []
            if dist.get_world_size() > 1:
                dist.all_gather_object(all_eval_loss, eval_loss)
                eval_loss = np.mean(
                    [all_eval_loss[i] for i in range(dist.get_world_size())]
                )
                self.logger.info(f"eval cost time: {time() - start_time}s")
                self.logger.info(f"eval_loss: {eval_loss}")
                paddle.device.cuda.empty_cache()
        return eval_loss

    def compute_test_loss(self, epoch=-1):
        with paddle.no_grad():
            preds = []
            tgts = []
            kl_loss_all = 0.0
            init_kl_loss_all = 0.0
            start_time = time()
            for batch_index, batch_data in enumerate(self.test_dataloader):
                src, tgt = batch_data
                src = paddle.cast(src, paddle.get_default_dtype())
                tgt = paddle.cast(tgt, paddle.get_default_dtype())
                predict_output, test_loss, kl_loss, init_kl_loss = self.test_one_step(
                    src, tgt
                )

                preds.append(predict_output)
                tgts.append(tgt[..., :1])
                kl_loss_all += kl_loss.numpy()
                init_kl_loss_all += init_kl_loss.numpy()
            self.logger.info(f"test time on whole data: {time() - start_time} s")

            preds = paddle.concat(preds, axis=0)  # [B,N,T,1]
            trues = paddle.concat(tgts, axis=0)  # [B,N,T,F]
            kl_loss = kl_loss_all / len(self.eval_dataloader)  # [M]
            init_kl_loss = init_kl_loss_all / len(self.eval_dataloader)  # [M]

            # [B,N,T,1]
            preds = self.test_dataset.inverse_transform(preds, axis=-1).numpy()
            # [B,N,T,1]
            trues = self.test_dataset.inverse_transform(trues, axis=-1).numpy()

            if dist.get_world_size() > 1:
                all_preds = []
                all_trues = []
                all_kl_loss = []
                all_init_kl_loss = []
                dist.all_gather_object(all_preds, preds)
                dist.all_gather_object(all_trues, trues)
                dist.all_gather_object(all_kl_loss, kl_loss)
                dist.all_gather_object(all_init_kl_loss, init_kl_loss)
                if dist.get_rank() == 0:
                    preds = np.concatenate(
                        [all_preds[i] for i in range(dist.get_world_size())], axis=0
                    )
                    trues = np.concatenate(
                        [all_trues[i] for i in range(dist.get_world_size())], axis=0
                    )
                    kl_loss = np.mean(
                        [all_kl_loss[i] for i in range(dist.get_world_size())], axis=0
                    )
                    init_kl_loss = np.mean(
                        [all_init_kl_loss[i] for i in range(dist.get_world_size())],
                        axis=0,
                    )
                    paddle.device.cuda.empty_cache()
                else:
                    paddle.device.cuda.empty_cache()
                    return

            self.logger.info(f"preds: {preds.shape}")
            self.logger.info(f"tgts: {trues.shape}")
            self.logger.info(f"kl_loss: {kl_loss}")
            self.logger.info(f"init_kl_loss: {init_kl_loss}")

            from utils import smis

            smis_score = smis(
                trues.reshape(trues.shape[0], -1),
                preds.reshape(preds.shape[0], -1),
                m=288,
                level=0.95,
            )
            self.logger.info(f"smis: {smis_score}")

            # 计算误差
            excel_list = []
            prediction_length = trues.shape[2]

            for i in range(prediction_length):
                assert preds.shape[0] == trues.shape[0]
                mae = mean_absolute_error(trues[:, :, i, 0], preds[:, :, i, 0])
                rmse = mean_squared_error(trues[:, :, i, 0], preds[:, :, i, 0]) ** 0.5
                mape = masked_mape_np(trues[:, :, i, 0], preds[:, :, i, 0], 0)
                self.logger.info(f"{i} MAE: {mae}")
                self.logger.info(f"{i} RMSE: {rmse}")
                self.logger.info(f"{i} MAPE: {mape}")
                excel_list.extend([mae, rmse, mape])

            # print overall results
            mae = mean_absolute_error(trues.reshape(-1, 1), preds.reshape(-1, 1))
            rmse = mean_squared_error(trues.reshape(-1, 1), preds.reshape(-1, 1)) ** 0.5
            mape = masked_mape_np(trues.reshape(-1, 1), preds.reshape(-1, 1), 0)
            self.logger.info(f"all MAE: {mae}")
            self.logger.info(f"all RMSE: {rmse}")
            self.logger.info(f"all MAPE: {mape}")
            excel_list.extend([mae, rmse, mape])
            self.logger.info(excel_list)

    def run_test(self):
        self.load()
        self.compute_test_loss()


if __name__ == "__main__":
    trainer = Trainer(training_args=args.args)
    trainer.train()
    trainer.run_test()
