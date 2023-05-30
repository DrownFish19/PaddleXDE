from .base_xde import BaseXDE
from ..types import LayerOrFunction, TupleOrTensor
from ..utils.brownian import BrownianInterval


class BaseSDE(BaseXDE):
    """Base class for all ODEs.

    """

    def __init__(self, f: LayerOrFunction, g: LayerOrFunction, bm: BrownianInterval, y0: TupleOrTensor, t):
        super(BaseSDE, self).__init__(name="SDE", var_nums=2, y0=y0, t=t)
        self.f = f
        self.g = g
        self.bm = bm

    def handle(self, h, ts):
        pass

    def move(self, t0, dt, y0):
        t1 = t0 + dt
        I_k = self.bm(t0, t1)

        dy = self.f(t0, y0)
        dg = self.g(t0, y0) * I_k
        return [dy, dg]

    def fuse(self, dy, dt, y0):
        return y0 + dy[0] * dt + dy[1]

    def get_dy(self, dy):
        return dy[0]
