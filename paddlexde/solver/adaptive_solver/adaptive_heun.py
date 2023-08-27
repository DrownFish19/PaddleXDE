import paddle

from ..base_adaptive_solver_rk import AdaptiveRKSolver, _ButcherTableau

_ADAPTIVE_HEUN_TABLEAU = _ButcherTableau(
    alpha=paddle.to_tensor([1.0], dtype=paddle.float64),
    beta=[
        paddle.to_tensor([1.0], dtype=paddle.float64),
    ],
    c_sol=paddle.to_tensor([0.5, 0.5], dtype=paddle.float64),
    c_error=paddle.to_tensor(
        [
            0.5,
            -0.5,
        ],
        dtype=paddle.float64,
    ),
)

_AH_C_MID = paddle.to_tensor([0.5, 0.0], dtype=paddle.float64)


class AdaptiveHeun(AdaptiveRKSolver):
    order = 2
    tableau = _ADAPTIVE_HEUN_TABLEAU
    mid = _AH_C_MID
