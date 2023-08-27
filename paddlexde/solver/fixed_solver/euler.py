from ..base_fixed_solver import FixedSolver


class Euler(FixedSolver):
    order = 1

    def step(self, t0, t1, y0):
        dt = t1 - t0
        dy = self.move(t0, dt, y0)
        y1 = self.fuse(dy, dt, y0)
        return y1, self.get_dy(dy)
