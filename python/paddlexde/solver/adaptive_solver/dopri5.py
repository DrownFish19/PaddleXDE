import paddle

from ..base_adaptive_solver_rk import AdaptiveRKSolver, _ButcherTableau

_DORMAND_PRINCE_SHAMPINE_TABLEAU = _ButcherTableau(
    alpha=paddle.to_tensor(
        [1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0], dtype=paddle.float64
    ),
    beta=[
        paddle.to_tensor([1 / 5], dtype=paddle.float64),
        paddle.to_tensor([3 / 40, 9 / 40], dtype=paddle.float64),
        paddle.to_tensor([44 / 45, -56 / 15, 32 / 9], dtype=paddle.float64),
        paddle.to_tensor(
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
            dtype=paddle.float64,
        ),
        paddle.to_tensor(
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
            dtype=paddle.float64,
        ),
        paddle.to_tensor(
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
            dtype=paddle.float64,
        ),
    ],
    c_sol=paddle.to_tensor(
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        dtype=paddle.float64,
    ),
    c_error=paddle.to_tensor(
        [
            35 / 384 - 1951 / 21600,
            0,
            500 / 1113 - 22642 / 50085,
            125 / 192 - 451 / 720,
            -2187 / 6784 - -12231 / 42400,
            11 / 84 - 649 / 6300,
            -1.0 / 60.0,
        ],
        dtype=paddle.float64,
    ),
)

DPS_C_MID = paddle.to_tensor(
    [
        6025192743 / 30085553152 / 2,
        0,
        51252292925 / 65400821598 / 2,
        -2691868925 / 45128329728 / 2,
        187940372067 / 1594534317056 / 2,
        -1776094331 / 19743644256 / 2,
        11237099 / 235043384 / 2,
    ],
    dtype=paddle.float64,
)


class Dopri5(AdaptiveRKSolver):
    order = 5
    tableau = _DORMAND_PRINCE_SHAMPINE_TABLEAU
    mid = DPS_C_MID
