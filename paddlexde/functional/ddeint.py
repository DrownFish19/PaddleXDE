from typing import Union

import paddle

from ..utils.ode_utils import _rms_norm
from ..xde import BaseDDE


def ddeint(
    func: callable,
    y0: Union[tuple, paddle.Tensor],
    t_span,
    lags,
    history,
    solver,
    *,
    rtol=1e-7,
    atol=1e-9,
    options: object = {"norm": _rms_norm}
):
    """Integrate a system of delay differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.
    """

    xde = BaseDDE(func, y0=y0, t_span=t_span, lags=lags, history=history)
    # TODO 此处仅需要传入his，lags如果设定为固定值则需要传入，如果设定为动态计算，每次计算batch时，会初始优化出lags
    # 实现自适应的延迟计算
    # 如果最后存在需要，可以返回每个batch的lags数据，进行可视化查看

    s = solver(xde=xde, y0=xde.y0, rtol=rtol, atol=atol, **options)
    solution = s.integrate(t_span)

    solution = xde.format(solution)

    return solution
