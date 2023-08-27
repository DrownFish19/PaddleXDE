from ..base_fixed_solver import FixedSolver


class Midpoint(FixedSolver):
    order = 2

    def step(self, t0, t1, y0):
        dt = t1 - t0
        half_dt = 0.5 * dt

        dy_half = self.move(t0, half_dt, y0)
        y_half = self.fuse(dy_half, half_dt, y0)

        t_half = t0 + half_dt
        dy = self.move(t_half, dt, y_half)
        y1 = self.fuse(dy, dt, y0)

        return y1, self.get_dy(dy)
