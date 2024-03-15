from typing import Union

import paddle

from ..utils.ode_utils import _rms_norm
from ..xde import BaseDDE


def ddeint(
    func: callable,
    y0: Union[tuple, paddle.Tensor],
    t_span,
    lags,
    his,
    his_span,
    solver,
    rtol=1e-7,
    atol=1e-9,
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

    xde = BaseDDE(
        func,
        y0=y0,
        t_span=t_span,
        lags=lags,
        his=his,
        his_span=his_span,
        **kwargs,
    )

    s = solver(
        xde=xde, y0=xde.y0, rtol=rtol, atol=atol, interp=fixed_solver_interp, **options
    )
    solution = s.integrate(t_span)

    return solution
