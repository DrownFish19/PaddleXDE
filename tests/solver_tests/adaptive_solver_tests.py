import paddle
import pytest
from paddlexde.functional.odeint import odeint
from paddlexde.solver.adaptive_solver import (
    AdaptiveHeun,
    Bosh3,
    Dopri5,
    Dopri8,
    Fehlberg2,
)

from tests.solver_tests.problems import construct_problem


@pytest.mark.api_adaptive_solver_base
def test_adaptive_solver_base_bosh3_sine():
    f, y0, t_points, sol = construct_problem(
        dtype=paddle.float32, ode="sine", reverse=False
    )
    y = odeint(f, y0, t_points, solver=Bosh3)
    assert paddle.allclose(sol, y, rtol=4e-3)


@pytest.mark.api_adaptive_solver_base
def test_adaptive_solver_base_bosh3_linear():
    f, y0, t_points, sol = construct_problem(
        dtype=paddle.float32, ode="linear", reverse=False
    )
    y = odeint(f, y0, t_points, solver=Bosh3)
    assert paddle.allclose(sol, y, rtol=4e-3)


@pytest.mark.api_adaptive_solver_base
def test_adaptive_solver_base_dopri5_sine():
    f, y0, t_points, sol = construct_problem(
        dtype=paddle.float32, ode="sine", reverse=False
    )
    y = odeint(f, y0, t_points, solver=Dopri5)
    assert paddle.allclose(sol, y, rtol=4e-3)


@pytest.mark.api_adaptive_solver_base
def test_adaptive_solver_base_dopri5_linear():
    f, y0, t_points, sol = construct_problem(
        dtype=paddle.float32, ode="linear", reverse=False
    )
    y = odeint(f, y0, t_points, solver=Dopri5)
    assert paddle.allclose(sol, y, rtol=4e-3)


@pytest.mark.api_adaptive_solver_base
def test_adaptive_solver_base_dopri8_sine():
    f, y0, t_points, sol = construct_problem(
        dtype=paddle.float32, ode="sine", reverse=False
    )
    y = odeint(f, y0, t_points, solver=Dopri8)
    assert paddle.allclose(sol, y, rtol=4e-3)


@pytest.mark.api_adaptive_solver_base
def test_adaptive_solver_base_dopri8_linear():
    f, y0, t_points, sol = construct_problem(
        dtype=paddle.float32, ode="linear", reverse=False
    )
    y = odeint(f, y0, t_points, solver=Dopri8)
    assert paddle.allclose(sol, y, rtol=4e-3)


@pytest.mark.api_adaptive_solver_base
def test_adaptive_solver_base_fehlberg2_sine():
    f, y0, t_points, sol = construct_problem(
        dtype=paddle.float32, ode="sine", reverse=False
    )
    y = odeint(f, y0, t_points, solver=Fehlberg2)
    assert paddle.allclose(sol, y, rtol=4e-3)


@pytest.mark.api_adaptive_solver_base
def test_adaptive_solver_base_fehlberg2_linear():
    f, y0, t_points, sol = construct_problem(
        dtype=paddle.float32, ode="linear", reverse=False
    )
    y = odeint(f, y0, t_points, solver=Fehlberg2)
    assert paddle.allclose(sol, y, rtol=4e-3)


@pytest.mark.api_adaptive_solver_base
def test_adaptive_solver_base_adaptiveheun_sine():
    f, y0, t_points, sol = construct_problem(
        dtype=paddle.float32, ode="sine", reverse=False
    )
    y = odeint(f, y0, t_points, solver=AdaptiveHeun)
    assert paddle.allclose(sol, y, rtol=4e-3)


@pytest.mark.api_adaptive_solver_base
def test_adaptive_solver_base_adaptiveheun_linear():
    f, y0, t_points, sol = construct_problem(
        dtype=paddle.float32, ode="linear", reverse=False
    )
    y = odeint(f, y0, t_points, solver=AdaptiveHeun)
    assert paddle.allclose(sol, y, rtol=1e-2)
