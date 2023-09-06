from typing import Union

import paddle
from paddle import nn

from .base_xde import BaseXDE


class BaseCDE(BaseXDE):
    """Base class for all CDEs."""

    def __init__(
        self,
        f: Union[nn.Layer, callable],
        y0: Union[tuple, paddle.Tensor],
        t_span: Union[list, paddle.Tensor],
    ):
        super(BaseCDE, self).__init__(name="CDE", var_nums=1)
        self.X = None
        self.control_gradient = None
        self.f = f
        self.t_span = t_span

    def handle(self, h, ts):
        self.X = h

    def move(self, t0, dt, y0):
        # todo 通过handle计算的数据求导数，计算control_gradient, 然后计算之后的数据
        dy = self.f(t0, y0)
        return [dy]

    def fuse(self, dy, dt, y0):
        return y0 + dy * dt
