from ..types import TupleOrTensor
from ..utils.ode_utils import _rms_norm
from ..xde import BaseSDE


def sdeint(drift: callable, diffusion: callable, y0: TupleOrTensor, t, solver, *, rtol=1e-7, atol=1e-9, reverse=False,
           options: object = {'norm': _rms_norm}):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.
    """

    xde = BaseSDE(f=drift, g=diffusion, y0=y0, t=t, reverse=reverse)

    s = solver(xde=xde, y0=xde.y0, rtol=rtol, atol=atol, **options)
    solution = s.integrate(t)

    solution = xde.format(solution)

    return solution
