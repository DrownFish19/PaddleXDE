import abc

import paddle

from ..types import Scalar, Tensor
from ..xde.base_xde import BaseXDE

_one_third = 1 / 3
_two_thirds = 2 / 3
_one_sixth = 1 / 6


class FixedSolver(metaclass=abc.ABCMeta):
    order: int

    def __init__(
        self,
        xde: BaseXDE,
        y0: Tensor,
        step_size: Scalar = None,
        grid_constructor: object = None,
        interp: str = "linear",
        perturb: bool = False,
        **kwargs,
    ):
        """API for solvers with possibly adaptive time stepping.

        :param xde: BaseXDE, including BaseODE, BaseSDE, BaseCDE and so on.
        :param y0:
        :param step_size:
        :param grid_constructor:
        :param interp:
        :param perturb:
        :param kwargs:
        """
        self.xde = xde
        self.y0 = y0
        self.dtype = y0.dtype
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb

        self.atol = kwargs["atol"]
        self.rtol = kwargs["rtol"]
        self.norm = kwargs["norm"]

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError(
                    "step_size and grid_constructor are mutually exclusive arguments."
                )

        self.move = self.xde.move
        self.fuse = self.xde.fuse
        self.get_dy = self.xde.get_dy

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = paddle.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = paddle.arange(0, niters, dtype=t.dtype) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer

        return _grid_constructor

    @abc.abstractmethod
    def step(self, t0, t1, y0):
        """Propose a step with step size from time t to time next_t, with
         current state y.

        :param t0:
        :param t1:
        :param y0:
        :return:
        """
        pass

    def integrate(self, t):
        time_grid = self.grid_constructor(self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = paddle.empty([len(t), *self.y0.shape], dtype=self.y0.dtype)
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            y1, dy0 = self.step(t0, t1, y0)

            while j < len(t) and t1 >= t[j]:
                if self.interp == "linear":
                    solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                elif self.interp == "cubic":
                    _, dy1 = self.step(t1, t1, y1)
                    solution[j] = self._cubic_hermite_interp(
                        t0, y0, dy0, t1, y1, dy1, t[j]
                    )
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                j += 1
            y0 = y1

        return solution

    @staticmethod
    def _cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = t1 - t0
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    @staticmethod
    def _linear_interp(t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)

    def rk4_step_func(self, t0, t1, y0, f0=None):
        dt = t1 - t0
        half_dt = dt * 0.5
        t_half = t0 + half_dt

        k1 = f0
        if k1 is None:
            k1 = self.move(t0, dt, y0)

        k2 = self.move(t_half, half_dt, self.fuse(k1, half_dt, y0))
        k3 = self.move(t_half, half_dt, self.fuse(k2, half_dt, y0))
        k4 = self.move(t1, half_dt, self.fuse(k3, dt, y0))

        return (
            self.fuse(k1, dt, y0)
            + 2 * self.fuse(k2, dt, y0)
            + 2 * self.fuse(k3, dt, y0)
            + self.fuse(k4, dt, y0)
        ) * _one_sixth

    def rk4_alt_step_func(self, t0, t1, y0, f0=None):
        """Smaller error with slightly more compute."""
        dt = t1 - t0

        dt_one_third = dt * _one_third
        dt_two_thirds = dt * _two_thirds

        t_one_third = t0 + dt_one_third
        t_two_thirds = t0 + dt_two_thirds

        k1 = f0
        if k1 is None:
            k1 = self.move(t0, dt, y0)

        k2 = self.move(t_one_third, dt_one_third, self.fuse(k1, dt_one_third, y0))
        k3 = self.move(
            t_two_thirds,
            dt_one_third,
            self.fuse([n2 - n1 * _one_third for (n1, n2) in zip(k1, k2)], dt, y0),
        )
        k4 = self.move(
            t1,
            t_one_third,
            self.fuse([n1 - n2 + n3 for (n1, n2, n3) in zip(k1, k2, k3)], dt, y0),
        )

        return (
            self.fuse(k1, dt, y0)
            + 3 * self.fuse(k2, dt, y0)
            + 3 * self.fuse(k3, dt, y0)
            + self.fuse(k4, dt, y0)
        ) * 0.125
