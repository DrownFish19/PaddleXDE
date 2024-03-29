from typing import Union

import paddle

from ..utils.ode_utils import _rms_norm
from ..xde import BaseODE


def odeint(
    func: callable,
    y0: Union[tuple, paddle.Tensor],
    t_span,
    solver,
    *,
    rtol=1e-7,
    atol=1e-9,
    options: object = {"norm": _rms_norm}
):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.
    """

    xde = BaseODE(func, y0=y0, t_span=t_span)

    s = solver(xde=xde, y0=xde.y0, rtol=rtol, atol=atol, **options)
    solution = s.integrate(t_span)

    solution = xde.format(solution)

    return solution
