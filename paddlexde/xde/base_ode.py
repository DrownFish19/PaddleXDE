from typing import Union

import paddle
from paddle import nn

from .base_xde import BaseXDE


class BaseODE(BaseXDE):
    """Base class for all ODEs."""

    def __init__(
        self,
        func: Union[nn.Layer, callable],
        y0: Union[tuple, paddle.Tensor],
        t_span: Union[list, paddle.Tensor],
    ):
        super(BaseODE, self).__init__(name="ODE", var_nums=1, y0=y0, t_span=t_span)
        self.func = func

    def handle(self, h, ts):
        pass

    def move(self, t0, dt, y0):
        dy = self.call_func(t0, y0)
        return dy

    def fuse(self, dy, dt, y0):
        # 测试是否存在振动
        y = dy * dt + y0
        _lambda = 0.001
        return (dy - _lambda * y) * dt + y0

        # return dy * dt + y0

    def call_func(self, t, y0):
        y0 = self.unflatten(y0, length=1)
        dy = self.func(t, y0)
        dy = self.flatten(dy)
        return dy
