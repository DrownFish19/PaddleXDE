import paddle
import pytest

from paddlexde.functional import odeint
from paddlexde.solver.fixed_solver import Euler, Midpoint, RK4, AdamsBashforthMoulton
from paddlexde.xde import BaseODE
from problems import construct_problem

EPS = {paddle.float32: 1e-4, paddle.float64: 1e-12, paddle.complex64: 1e-4}


@pytest.mark.api_fixed_solver_base
def test_fixed_solver_base_euler_constant():
    f, y0, t_points, sol = construct_problem(dtype=paddle.float32, ode="constant", reverse=False)
    f = BaseODE(func=f)
    y = odeint(f, y0, t_points, solver=Euler)
    assert paddle.allclose(sol, y, rtol=4e-3)


@pytest.mark.api_fixed_solver_base
def test_fixed_solver_base_midpoint_constant():
    f, y0, t_points, sol = construct_problem(dtype=paddle.float32, ode="constant", reverse=False)
    f = BaseODE(func=f)
    y = odeint(f, y0, t_points, solver=Midpoint)
    assert paddle.allclose(sol, y, rtol=4e-3)


@pytest.mark.api_fixed_solver_base
def test_fixed_solver_base_rk4_constant():
    f, y0, t_points, sol = construct_problem(dtype=paddle.float32, ode="constant", reverse=False)
    f = BaseODE(func=f)
    y = odeint(f, y0, t_points, solver=RK4)
    assert paddle.allclose(sol, y, rtol=4e-3)


@pytest.mark.api_fixed_solver_base
def test_fixed_solver_base_adam_constant():
    f, y0, t_points, sol = construct_problem(dtype=paddle.float32, ode="constant", reverse=False)
    f = BaseODE(func=f)
    y = odeint(f, y0, t_points, solver=AdamsBashforthMoulton)
    assert paddle.allclose(sol, y, rtol=4e-3)
