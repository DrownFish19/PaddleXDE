import paddle

from ..base_adaptive_solver_rk import AdaptiveRKSolver, _ButcherTableau

_FEHLBERG2_TABLEAU = _ButcherTableau(
    alpha=paddle.to_tensor([1 / 2, 1.0], dtype=paddle.float64),
    beta=[
        paddle.to_tensor([1 / 2], dtype=paddle.float64),
        paddle.to_tensor([1 / 256, 255 / 256], dtype=paddle.float64),
    ],
    c_sol=paddle.to_tensor([1 / 512, 255 / 256, 1 / 512], dtype=paddle.float64),
    c_error=paddle.to_tensor([-1 / 512, 0, 1 / 512], dtype=paddle.float64),
)

_FE_C_MID = paddle.to_tensor([0.0, 0.5, 0.0], dtype=paddle.float64)


class Fehlberg2(AdaptiveRKSolver):
    order = 2
    tableau = _FEHLBERG2_TABLEAU
    mid = _FE_C_MID
