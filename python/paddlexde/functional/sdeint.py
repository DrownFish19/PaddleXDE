from ..types import TupleOrTensor
from ..utils.brownian import BrownianInterval
from ..utils.ode_utils import _rms_norm
from ..xde import BaseSDE


def sdeint(drift: callable, diffusion: callable, y0: TupleOrTensor, t, solver, *, rtol=1e-7, atol=1e-9, options: object = {'norm': _rms_norm}):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.
    """

    bm = BrownianInterval(t0=0.0, t1=1.0, size=(1, 1))

    xde = BaseSDE(f=drift, g=diffusion, bm=bm, y0=y0, t=t)

    s = solver(xde=xde, y0=xde.y0, rtol=rtol, atol=atol, **options)
    solution = s.integrate(t)

    solution = xde.format(solution)

    return solution
