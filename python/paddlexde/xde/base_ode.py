import paddle
import paddle.nn as nn

from .base_xde import BaseXDE


class BaseODE(BaseXDE):
    """Base class for all ODEs.

    """

    def __init__(self, func: nn.Layer):
        super(BaseODE, self).__init__(name="ODE", var_nums=1)
        self.func = func

    def handle(self, h, ts):
        pass

    def move(self, t0, dt, y0):
        dy = self.func(t0, y0)
        return paddle.stack([dy])

    def fuse(self, dy, dt, y0):
        return dy[0] * dt + y0

    def get_dy(self, dy):
        return dy[0]
