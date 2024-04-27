from typing import Union

import paddle

from ..utils.ode_utils import _rms_norm
from ..xde import BaseDDE
from .ddeint_utils import DelayIndex


def ddeint(
    drift_f: callable,
    delay_f: callable,
    y0: Union[tuple, paddle.Tensor],
    y_t_span,
    delay_t_span,
    delay,
    delay_t,
    solver,
    options: object = {"norm": _rms_norm},
    fixed_solver_interp="linear",
    **kwargs,
):
    """Integrate a system of delay differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.
    """
    delay = DelayIndex.apply(delay_t_span, delay, delay_t)
    xde = BaseDDE(
        drift_f=drift_f,
        delay_f=delay_f,
        y0=y0,
        delay=delay,
        y_t_span=y_t_span,
        **kwargs,
    )
    s = solver(xde, xde.input_y0, interp=fixed_solver_interp, **options)
    solution = s.integrate(y_t_span)

    return solution
