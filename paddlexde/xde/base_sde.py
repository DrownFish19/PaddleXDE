from typing import Union

import paddle
from paddle import nn

from ..utils.brownian import BrownianInterval
from ..utils.misc import flat_to_shape
from .base_xde import BaseXDE


class BaseSDE(BaseXDE):
    """Base class for all ODEs."""

    def __init__(
        self,
        f: Union[nn.Layer, callable],
        g: Union[nn.Layer, callable],
        y0: Union[tuple, paddle.Tensor],
        t: Union[list, paddle.Tensor],
        reverse=False,
    ):
        super(BaseSDE, self).__init__(name="SDE", var_nums=2, y0=y0, t=t)
        self.f = f
        self.g = g

        if self.is_tuple:
            self.batch_size = self.shapes[1][0]
            self.state_size = self.shapes[1][1]
        else:
            self.batch_size = y0.shape[0]
            self.state_size = y0.shape[1]

        self.bm = BrownianInterval(
            t0=t[0], t1=t[-1], size=(self.batch_size, self.state_size)
        )
        if reverse:
            t.clip(0)

    def handle(self, h, ts):
        pass

    def move(self, t0, dt, y0):
        t1 = t0 + dt
        I_k = self.bm(t0, t1)

        if self.is_tuple:
            dy = self.f(t0, flat_to_shape(y0, (), self.shapes, self.num_elements))
            dy = paddle.concat([dy_.reshape([-1]) for dy_ in dy])

            dg = self.g(t0, flat_to_shape(y0, (), self.shapes, self.num_elements))
            dg = paddle.concat([dy_.reshape([-1]) for dy_ in dg])

        else:
            dy = self.f(t0, y0)
            dg = self.g(t0, y0) * I_k
        return paddle.stack([dy, dg])

    def fuse(self, dy, dt, y0):
        return y0 + dy[0] * dt + dy[1]

    def get_dy(self, dy):
        return dy[0]
