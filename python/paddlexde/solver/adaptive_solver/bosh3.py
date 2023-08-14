import paddle

from ..base_adaptive_solver_rk import _ButcherTableau, AdaptiveRKSolver

_BOGACKI_SHAMPINE_TABLEAU = _ButcherTableau(
    alpha=paddle.to_tensor([1 / 2, 3 / 4, 1.], dtype=paddle.float64),
    beta=[paddle.to_tensor([1 / 2], dtype=paddle.float64),
          paddle.to_tensor([0., 3 / 4], dtype=paddle.float64),
          paddle.to_tensor([2 / 9, 1 / 3, 4 / 9], dtype=paddle.float64)
          ],
    c_sol=paddle.to_tensor([2 / 9, 1 / 3, 4 / 9, 0.], dtype=paddle.float64),
    c_error=paddle.to_tensor([2 / 9 - 7 / 24, 1 / 3 - 1 / 4, 4 / 9 - 1 / 3, -1 / 8], dtype=paddle.float64),
)

_BS_C_MID = paddle.to_tensor([0., 0.5, 0., 0.], dtype=paddle.float64)


class Bosh3(AdaptiveRKSolver):
    order = 3
    tableau = _BOGACKI_SHAMPINE_TABLEAU
    mid = _BS_C_MID
