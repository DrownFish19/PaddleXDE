import bisect
import collections
from typing import Union

import paddle

from ..utils.ode_utils import (
    PaddleAssign,
    compute_error_ratio,
    interp_evaluate,
    interp_fit,
    optimal_step_size,
    sort_tvals,
)
from ..xde import BaseXDE
from .base_adaptive_solver import AdaptiveSolver

_ButcherTableau = collections.namedtuple(
    "_ButcherTableau", "alpha, beta, c_sol, c_error"
)

_RungeKuttaState = collections.namedtuple(
    "_RungeKuttaState", "y1, f1, t0, t1, dt, interp_coeff"
)


class AdaptiveRKSolver(AdaptiveSolver):
    order: int
    tableau: _ButcherTableau
    mid: paddle.Tensor

    def __init__(
        self,
        xde: BaseXDE,
        y0: paddle.Tensor,
        rtol: Union[float, paddle.Tensor],
        atol: Union[float, paddle.Tensor],
        min_step: Union[float, paddle.Tensor] = 0,
        max_step: Union[float, paddle.Tensor] = float("inf"),
        first_step=None,
        step_t=None,
        jump_t=None,
        safety=0.9,
        ifactor=10.0,
        dfactor=0.2,
        max_num_steps=2**31 - 1,
        dtype=paddle.float32,
        **kwargs
    ):
        super(AdaptiveRKSolver, self).__init__(xde=xde, dtype=dtype, y0=y0, **kwargs)

        # We use mixed precision. y has its original dtype (probably float32), whilst all 'time'-like objects use
        # `dtype` (defaulting to float64).
        # dtype = paddle.promote_types(dtype, y0.abs().dtype)

        self.rtol = paddle.to_tensor(rtol, dtype=dtype)
        self.atol = paddle.to_tensor(atol, dtype=dtype)
        self.min_step = paddle.to_tensor(min_step, dtype=dtype)
        self.max_step = paddle.to_tensor(max_step, dtype=dtype)
        self.first_step = (
            None if first_step is None else paddle.to_tensor(first_step, dtype=dtype)
        )
        self.safety = paddle.to_tensor(safety, dtype=dtype)
        self.ifactor = paddle.to_tensor(ifactor, dtype=dtype)
        self.dfactor = paddle.to_tensor(dfactor, dtype=dtype)
        self.max_num_steps = paddle.to_tensor(max_num_steps, dtype=paddle.int32)
        self.dtype = dtype

        self.step_t = None if step_t is None else paddle.to_tensor(step_t, dtype=dtype)
        self.jump_t = None if jump_t is None else paddle.to_tensor(jump_t, dtype=dtype)

        # Copy from class to instance to set device
        self.tableau = _ButcherTableau(
            alpha=self.tableau.alpha.astype(dtype=y0.dtype),
            beta=[b.astype(dtype=y0.dtype) for b in self.tableau.beta],
            c_sol=self.tableau.c_sol.astype(dtype=y0.dtype),
            c_error=self.tableau.c_error.astype(dtype=y0.dtype),
        )
        self.mid = self.mid.astype(dtype=y0.dtype)

    def _before_integrate(self, t_span):
        t0 = t_span[0]
        f0 = self.move(t_span[0], t_span[1] - t_span[0], self.y0)
        if self.first_step is None:
            first_step = self.select_initial_step(
                t_span[0], self.y0, self.order - 1, self.rtol, self.atol
            )
        else:
            first_step = self.first_step
        self.rk_state = _RungeKuttaState(
            self.y0, f0, t_span[0], t_span[0], first_step, [self.y0] * 5
        )

        # Handle step_t and jump_t arguments.
        if self.step_t is None:
            step_t = paddle.to_tensor([], dtype=self.dtype)
        else:
            step_t = sort_tvals(self.step_t, t0)
        if self.jump_t is None:
            jump_t = paddle.to_tensor([], dtype=self.dtype)
        else:
            jump_t = sort_tvals(self.jump_t, t0)
        # counts = paddle.concat([step_t, jump_t]).unique(return_counts=True)[1]
        # if (counts > 1).any():
        #     raise ValueError("`step_t` and `jump_t` must not have any repeated elements between them.")

        self.step_t = step_t
        self.jump_t = jump_t
        self.next_step_index = min(
            bisect.bisect(self.step_t.tolist(), t_span[0]), len(self.step_t) - 1
        )
        self.next_jump_index = min(
            bisect.bisect(self.jump_t.tolist(), t_span[0]), len(self.jump_t) - 1
        )

    def step(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t > self.rk_state.t1:
            assert (
                n_steps < self.max_num_steps
            ), "max_num_steps exceeded ({}>={})".format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_step(self.rk_state)
            n_steps += 1
        return interp_evaluate(
            self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t
        )

    def _runge_kutta_step(self, y0, f0, t0, dt, t1, tableau):
        """Take an arbitrary Runge-Kutta step and estimate error.
        Args:
            func: Function to evaluate like `func(t, y)` to compute the time derivative of `y`.
            y0: Tensor initial value for the state.
            f0: Tensor initial value for the derivative, computed from `func(t0, y0)`.
            t0: float64 scalar Tensor giving the initial time.
            dt: float64 scalar Tensor giving the size of the desired time step.
            t1: float64 scalar Tensor giving the end time; equal to t0 + dt. This is used (rather than t0 + dt) to ensure
                floating point accuracy when needed.
            tableau: _ButcherTableau describing how to take the Runge-Kutta step.
        Returns:
            Tuple `(y1, f1, y1_error, k)` giving the estimated function value after
            the Runge-Kutta step at `t1 = t0 + dt`, the derivative of the state at `t1`,
            estimated error at `t1`, and a list of Runge-Kutta coefficients `k` used for
            calculating these terms.
        """

        t_dtype = y0.abs().dtype

        t0 = t0.astype(t_dtype)
        dt = dt.astype(t_dtype)
        t1 = t1.astype(t_dtype)

        # We use an unchecked assign to put data into k without incrementing its _version counter, so that the backward
        # doesn't throw an (overzealous) error about in-place correctness. We know that it's actually correct.
        k = paddle.empty([*f0.shape, len(tableau.alpha) + 1], dtype=y0.dtype)
        k.stop_gradient = False
        k = PaddleAssign.apply(target=k, value=f0, index=(..., 0))
        for i, (alpha_i, beta_i) in enumerate(zip(tableau.alpha, tableau.beta)):
            if alpha_i == 1.0:
                # Always step to perturbing just before the end time, in case of discontinuities.
                ti = t1
                # perturb = Perturb.PREV
            else:
                ti = t0 + alpha_i * dt
                # perturb = Perturb.NONE
            yi = y0 + paddle.sum(k[..., : i + 1] * (beta_i * dt), axis=-1).reshape(
                y0.shape
            )
            f = self.move(ti, dt, yi)
            k = PaddleAssign.apply(target=k, value=f, index=(..., i + 1))

        if not (
            tableau.c_sol[-1] == 0 and (tableau.c_sol[:-1] == tableau.beta[-1]).all()
        ):
            # This property (true for Dormand-Prince) lets us save a few FLOPs.
            yi = y0 + paddle.sum(k * (dt * tableau.c_sol), axis=-1).reshape(y0.shape)

        y1 = yi
        f1 = k[..., -1]
        y1_error = paddle.sum(k * (dt * tableau.c_error), axis=-1)
        return y1, f1, y1_error, k

    def _adaptive_step(self, rk_state):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, interp_coeff = rk_state
        # self.func.callback_step(t0, y0, dt)
        t1 = t0 + dt
        # dtypes: self.y0.dtype (probably float32); self.dtype (probably float64)
        # used for state and timelike objects respectively.
        # Then:
        # y0.dtype == self.y0.dtype
        # f0.dtype == self.y0.dtype
        # t0.dtype == self.dtype
        # dt.dtype == self.dtype
        # for coeff in interp_coeff: coeff.dtype == self.y0.dtype

        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, "underflow in dt {}".format(dt.item())
        assert paddle.isfinite(y0).all(), "non-finite values in state `y`: {}".format(
            y0
        )

        ########################################################
        #     Make step, respecting prescribed grid points     #
        ########################################################

        on_step_t = False
        if len(self.step_t):
            next_step_t = self.step_t[self.next_step_index]
            on_step_t = t0 < next_step_t < t0 + dt
            if on_step_t:
                t1 = next_step_t
                dt = t1 - t0

        on_jump_t = False
        if len(self.jump_t):
            next_jump_t = self.jump_t[self.next_jump_index]
            on_jump_t = t0 < next_jump_t < t0 + dt
            if on_jump_t:
                on_step_t = False
                t1 = next_jump_t
                dt = t1 - t0

        # Must be arranged as doing all the step_t handling, then all the jump_t handling, in case we
        # trigger both. (i.e. interleaving them would be wrong.)

        y1, f1, y1_error, k = self._runge_kutta_step(
            y0, f0, t0, dt, t1, tableau=self.tableau
        )
        # dtypes:
        # y1.dtype == self.y0.dtype
        # f1.dtype == self.y0.dtype
        # y1_error.dtype == self.dtype
        # k.dtype == self.y0.dtype

        ########################################################
        #                     Error Ratio                      #
        ########################################################
        error_ratio = compute_error_ratio(
            y1_error, self.rtol, self.atol, y0, y1, self.norm
        )
        accept_step = error_ratio <= 1

        # Handle min max stepping
        if dt > self.max_step:
            accept_step = False
        if dt <= self.min_step:
            accept_step = True

        # dtypes:
        # error_ratio.dtype == self.dtype

        ########################################################
        #                   Update RK State                    #
        ########################################################
        if accept_step:
            # self.func.callback_accept_step(t0, y0, dt)
            t_next = t1
            y_next = y1
            interp_coeff = self._interp_fit(y0, y_next, k, dt)
            if on_step_t:
                if self.next_step_index != len(self.step_t) - 1:
                    self.next_step_index += 1
            if on_jump_t:
                if self.next_jump_index != len(self.jump_t) - 1:
                    self.next_jump_index += 1
                # We've just passed a discontinuity in f; we should update f to match the side of the discontinuity
                # we're now on.
                # f1 = self.func(t_next, y_next, perturb=Perturb.NEXT)
                f1 = self.func(t_next, y_next)
            f_next = f1
        else:
            # self.func.callback_reject_step(t0, y0, dt)
            t_next = t0
            y_next = y0
            f_next = f0
        dt_next = optimal_step_size(
            dt, error_ratio, self.safety, self.ifactor, self.dfactor, self.order
        )
        dt_next = dt_next.clip(self.min_step, self.max_step)
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)
        return rk_state

    def _interp_fit(self, y0, y1, k, dt):
        """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
        dt = dt.astype(y0.dtype)
        y_mid = y0 + paddle.sum(k * (dt * self.mid), axis=-1).reshape(y0.shape)
        f0 = k[..., 0]
        f1 = k[..., -1]
        return interp_fit(y0, y1, y_mid, f0, f1, dt)
