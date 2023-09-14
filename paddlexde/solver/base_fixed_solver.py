import abc
from typing import Union

import paddle

from ..interpolation.functional import cubic_hermite_interp, linear_interp
from ..xde.base_xde import BaseXDE

_one_third = 1 / 3
_two_thirds = 2 / 3
_one_sixth = 1 / 6


class FixedSolver(metaclass=abc.ABCMeta):
    order: int

    def __init__(
        self,
        xde: BaseXDE,
        y0: paddle.Tensor,
        step_size: Union[float, paddle.Tensor] = None,
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
        raise NotImplementedError

    def integrate(self, t_span: paddle.Tensor):
        """_summary_

        Args:
            t_span (paddle.Tensor): [batch_size, pred_len]

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        batch_size, pred_len = t_span.shape
        # time_grid = self.grid_constructor(self.y0, t_span)
        time_grid = t_span
        assert paddle.equal_all(time_grid[..., 0], t_span[..., 0])
        assert paddle.equal_all(time_grid[..., -1], t_span[..., -1])

        # sol solution [pred_len, batch_size, dims]
        sol = paddle.empty(shape=[pred_len] + self.y0.shape, dtype=self.y0.dtype)
        sol[0] = self.y0

        y0 = self.y0

        for i in range(1, pred_len):
            t0, t1 = time_grid[..., i - 1 : i], time_grid[..., i : i + 1]
            y1, dy0 = self.step(t0, t1, y0)

            # while j < pred_len and paddle.greater_equal(t1, t_span[..., j]):
            if self.interp == "linear":
                sol[i] = linear_interp(t0, t1, y0, y1, t_span[..., i : i + 1])
            elif self.interp == "cubic":
                y2, dy1 = self.step(t1, t1, y1)
                sol[i] = cubic_hermite_interp(
                    t0, y0, dy0, t1, y1, dy1, t_span[..., i : i + 1]
                )
            else:
                raise ValueError(f"Unknown interpolation method {self.interp}")

            y0 = y1

        return sol

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
            self.fuse(k1 - k2 * _one_third, dt, y0),
        )
        k4 = self.move(
            t1,
            t_one_third,
            self.fuse(k1 - k2 + k3, dt, y0),
        )

        return (
            self.fuse(k1, dt, y0)
            + 3 * self.fuse(k2, dt, y0)
            + 3 * self.fuse(k3, dt, y0)
            + self.fuse(k4, dt, y0)
        ) * 0.125
