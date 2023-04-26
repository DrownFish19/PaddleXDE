from ..utils.ode_utils import _rms_norm
from ..xde import BaseODE


def odeint(xde: BaseODE, y0, t, solver, *, rtol=1e-7, atol=1e-9, options: object = {'norm': _rms_norm}):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.
    """
    s = solver(xde=xde, y0=y0, rtol=rtol, atol=atol, **options)
    solution = s.integrate(t)

    return solution
