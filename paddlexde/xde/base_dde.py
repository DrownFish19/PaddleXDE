from typing import Union

import paddle
from paddle import nn

from ..utils.misc import flat_to_shape
from .base_xde import BaseXDE


class BaseDDE(BaseXDE):
    """Base class for all ODEs."""

    def __init__(
        self,
        func: Union[nn.Layer, callable],
        y0: Union[tuple, paddle.Tensor],
        t_span: Union[list, paddle.Tensor],
        lags: Union[list, paddle.Tensor],
    ):
        super(BaseDDE, self).__init__(name="ODE", var_nums=1, y0=y0, t_span=t_span)

        self.func = func
        self.lags = lags

    def handle(self, h, ts):
        pass

    def move(self, t0, dt, y0):
        if self.is_tuple:
            dy = self.func(t0, flat_to_shape(y0, (), self.shapes, self.num_elements))
            dy = paddle.concat([dy_.reshape([-1]) for dy_ in dy])
        else:
            dy = self.func(t0, y0)
        return paddle.stack([dy])

    def fuse(self, dy, dt, y0):
        # 测试是够还存在振动
        y = dy[0] * dt + y0
        _lambda = 0.001
        return (dy[0] - _lambda * y) * dt + y0

        # return dy[0] * dt + y0

    def get_dy(self, dy):
        return dy[0]
