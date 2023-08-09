import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pytest
from paddlexde.functional import sdeint, sdeint_adjoint
from paddlexde.solver.adaptive_solver import Fehlberg2

from tests.solver_tests.problems import construct_problem

batch_size, state_size, t_size = 3, 1, 100
ts = paddle.linspace(0, 1, t_size)
y0 = paddle.full(shape=[batch_size, state_size], fill_value=0.1)


class f(nn.Layer):
    def __init__(self):
        super().__init__()
        self.theta = paddle.create_parameter(shape=[1], dtype=paddle.float32)  # Scalar

    def forward(self, t, y):
        return paddle.sin(t) + self.theta * y


class g(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, t, y):
        return 0.3 * F.sigmoid(paddle.cos(t) * paddle.exp(-y))


@pytest.mark.api_fixed_solver_base
def test_fixed_solver_base_rk4_constant_sdeint():
    # drift = f()
    diffusion = g()

    f, y0, t_points, sol = construct_problem(
        dtype=paddle.float32, ode="constant", reverse=False
    )  # todo:修改sde problem创建函数
    y = sdeint(
        drift=f, diffusion=diffusion, y0=y0.unsqueeze(-1), t=t_points, solver=Fehlberg2
    )
    assert paddle.allclose(sol.unsqueeze(-1), y, rtol=4e-3)


@pytest.mark.api_fixed_solver_base
def test_fixed_solver_base_rk4_constant_sdeint_adjoint():
    # drift = f()
    diffusion = g()

    f, y0, t_points, sol = construct_problem(
        dtype=paddle.float32, ode="constant", reverse=False
    )  # todo:修改sde problem创建函数
    y = sdeint_adjoint(
        drift=f, diffusion=diffusion, y0=y0.unsqueeze(-1), t=t_points, solver=Fehlberg2
    )

    paddle.autograd.grad(outputs=y, inputs=y0.unsqueeze(-1), allow_unused=True)
    assert paddle.allclose(sol.unsqueeze(-1), y, rtol=4e-3)
