import copy
import logging
import math
import os
from math import sqrt

import numpy as np
import paddle
import paddle.nn as nn
import pandas as pd
from paddle.optimizer.lr import LRScheduler


class CosineAnnealingWithWarmupDecay(LRScheduler):
    def __init__(
        self, max_lr, min_lr, warmup_step, decay_step, last_epoch=-1, verbose=False
    ):
        self.decay_step = decay_step
        self.warmup_step = warmup_step
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(CosineAnnealingWithWarmupDecay, self).__init__(
            max_lr, last_epoch, verbose
        )

    def get_lr(self):
        if self.warmup_step > 0 and self.last_epoch <= self.warmup_step:
            return float(self.max_lr) * (self.last_epoch) / self.warmup_step

        if self.last_epoch > self.decay_step:
            return self.min_lr

        num_step_ = self.last_epoch - self.warmup_step
        decay_step_ = self.decay_step - self.warmup_step
        decay_ratio = float(num_step_) / float(decay_step_)
        coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


class LinearAnnealingWithWarmupDecay(LRScheduler):
    def __init__(
        self, max_lr, min_lr, warmup_step, decay_step, last_epoch=-1, verbose=False
    ):

        self.decay_step = decay_step
        self.warmup_step = warmup_step
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(LinearAnnealingWithWarmupDecay, self).__init__(
            max_lr, last_epoch, verbose
        )

    def get_lr(self):
        if self.warmup_step > 0 and self.last_epoch <= self.warmup_step:
            return float(self.max_lr) * (self.last_epoch) / self.warmup_step

        if self.last_epoch > self.decay_step:
            return self.min_lr

        num_step_ = self.last_epoch - self.warmup_step
        decay_step_ = self.decay_step - self.warmup_step
        decay_ratio = float(num_step_) / float(decay_step_)
        coeff = 1.0 - decay_ratio
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


class Logger:
    def __init__(self, name, log_file):
        self.name = name
        self.log_file = log_file
        self.logger = self._get_logger()

    def _get_logger(self):
        # 创建一个logger
        logger = logging.getLogger(self.name)
        # 设置logger的日志级别
        logger.setLevel(logging.DEBUG)
        # 创建一个handler，用于写入日志文件
        file_handler = logging.FileHandler(self.log_file)
        # 定义handler的输出格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        # 给logger添加handler
        logger.addHandler(file_handler)
        return logger

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def debug(self, message):
        self.logger.debug(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


def clones(module, N):
    """
    Produce N identical layers.
    :param module: nn.Layer
    :param N: int
    :return: paddle.nn.LayerList
    """
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


def sym_norm_adj(W):
    """
    compute Symmetric normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    """
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)  # 为邻居矩阵加上自连接
    D = np.diag(np.sum(W, axis=1))
    sym_norm_adj_matrix = np.dot(np.sqrt(D), W)
    sym_norm_adj_matrix = np.dot(sym_norm_adj_matrix, np.sqrt(D))

    return sym_norm_adj_matrix


def norm_adj_matrix(W):
    """
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    """
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)  # 为邻接矩阵加上自连接
    D = np.diag(1.0 / np.sum(W, axis=1))
    normed = np.dot(D, W)

    return normed


def multichannel_norm_adj(A):
    norm_adj_list = []
    for c in range(A.shape[0]):
        norm_adj_list.append(normed(A[c]))

    output = np.stack(norm_adj_list)
    return output


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    """
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    """
    if "npy" in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        distaneA = np.zeros(
            (int(num_of_vertices), int(num_of_vertices)), dtype=np.float32
        )

        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:
            with open(id_filename, "r") as f:
                id_dict = {
                    int(i): idx for idx, i in enumerate(f.read().strip().split("\n"))
                }  # 把节点id（idx）映射成从0开始的索引
            with open(distance_df_filename, "r") as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA
        else:  # distance file中的id直接从0开始
            with open(distance_df_filename, "r") as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def get_adjacency_matrix_2direction(
    distance_df_filename, num_of_vertices, id_filename=None
):
    """
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    """
    if "npy" in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        distaneA = np.zeros(
            (int(num_of_vertices), int(num_of_vertices)), dtype=np.float32
        )

        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:
            with open(id_filename, "r") as f:
                id_dict = {
                    int(i): idx for idx, i in enumerate(f.read().strip().split("\n"))
                }  # 把节点id（idx）映射成从0开始的索引
            with open(distance_df_filename, "r") as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    distaneA[id_dict[j], id_dict[i]] = distance
            return A, distaneA
        else:  # distance file中的id直接从0开始
            with open(distance_df_filename, "r") as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance
            return A, distaneA


def get_matrix(base_path, filename, turbine_nums):
    npy_file_path = os.path.join(base_path, filename + ".npy")
    if os.path.exists(npy_file_path):
        return np.load(npy_file_path)
    else:
        df_raw = pd.read_csv(os.path.join(base_path, filename))
        res = np.zeros((turbine_nums, turbine_nums))
        for i in range(turbine_nums):
            for j in range(turbine_nums):
                res[i, j] = sqrt(
                    (df_raw.values[i, 1] - df_raw.values[j, 1]) ** 2
                    + (df_raw.values[i, 2] - df_raw.values[j, 2]) ** 2
                )
        np.save(npy_file_path, res)
    return res


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}",
                flush=True,
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
