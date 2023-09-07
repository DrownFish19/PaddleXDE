from typing import Union

import paddle
from paddle import nn

from .base_xde import BaseXDE


class BaseDDE(BaseXDE):
    """Base class for all ODEs."""

    def __init__(
        self,
        func: Union[nn.Layer, callable],
        y0: Union[tuple, paddle.Tensor],
        t_span: Union[list, paddle.Tensor],
        lags: Union[list, paddle.Tensor],
        history: paddle.Tensor,
    ):
        super(BaseDDE, self).__init__(name="ODE", var_nums=1, y0=y0, t_span=t_span)

        self.func = func
        self.lags = lags
        self.history = history

    def handle(self, h, ts):
        pass

    def move(self, t0, dt, y0):
        dy = self.call_func(t0, y0)
        return dy

    def fuse(self, dy, dt, y0):
        # 测试是够还存在振动
        y = dy * dt + y0
        _lambda = 0.001
        return (dy - _lambda * y) * dt + y0

        # return dy * dt + y0

    def call_func(self, t, y0):
        y0 = self.unflatten(y0, length=1)
        self.init_lags()
        input_history = paddle.index_select(self.history, self.lags)
        dy = self.func(t, y0, input_history)
        dy = self.flatten(dy)
        return dy

    def init_lags(self):
        # TODO 不同时刻初始化不同lags
        pass


class HistoryIndex(nn.autograd.Function):
    def forward(ctx):
        # 传入lags, history,
        # 计算序列位置对应位置的梯度，并保存至backward
        pass

    def backward(ctx):
        # 计算history相应的梯度，并提取forward中保存的梯度，用于计算lag的梯度
        # 在计算的过程中，无需更新history，仅更新lags即可

        pass
