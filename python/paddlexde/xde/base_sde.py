from .base_xde import BaseXDE
from ..types import Layer


class BaseSDE(BaseXDE):
    """Base class for all ODEs.

    """

    def __init__(self, f: Layer, g: Layer):
        super(BaseSDE, self).__init__(name="SDE", var_nums=2)
        self.f = f
        self.g = g

    def handle(self, h, ts):
        pass

    def move(self, t0, dt, y0):
        t1 = t0 + dt
        dy = self.f(t0, y0)
        dg = self.g(t0, y0)
        return [dy, dg]

    def fuse(self, dy, dt, y0):
        return y0 + dy[0] * dt + dy[1]

    def get_dy(self, dy):
        return dy[0]
