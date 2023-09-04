from ..base_fixed_solver import FixedSolver


class RK4(FixedSolver):
    order = 4

    def step(self, t0, t1, y0):
        f0 = self.move(t0, t1 - t0, y0)
        # return self.rk4_step_func(t0, t1, y0, f0=f0)
        return self.rk4_alt_step_func(t0, t1, y0, f0=f0), f0
