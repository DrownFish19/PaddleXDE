from .base_xde import BaseXDE
from ..types import Layer


class BaseDDE(BaseXDE):
    """Base class for all ODEs.

    """

    def __init__(self, f: Layer, dts):
        super(BaseDDE, self).__init__(name="DDE", var_nums=1)
        self.f = f
        self.dts = dts

    def handle(self, h, ts):
        pass

    def move(self, t0, dt, y0):
        dy = self.f(t0, y0, self.dts)
        return [dy]

    def fuse(self, dy, dt, y0):
        return y0 + dy[0] * dt

    def get_dy(self, dy):
        return dy[0]
