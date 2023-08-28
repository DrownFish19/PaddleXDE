import unittest

import paddle

from paddlexde.functional import odeint, odeint_adjoint
from paddlexde.solver.adaptive_solver import (
    AdaptiveHeun,
    Bosh3,
    Dopri5,
    Dopri8,
    Fehlberg2,
)
from tests.testing_utils import construct_problem


class TestAdaptiveSolversSineForODE(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        f, y0, t, sol = construct_problem(
            dtype=paddle.float32,
            ode="sine",
            reverse=False,
        )
        self.f = f
        self.y0 = y0
        self.t = t
        self.sol = sol

        self.xdeints = [odeint, odeint_adjoint]
        self.solvers = [Bosh3, Dopri5, Dopri8, Fehlberg2, AdaptiveHeun]

    def test_bosh3(self):
        y = odeint(self.f, self.y0, self.t, solver=Bosh3)
        assert paddle.allclose(self.sol, y, rtol=4e-3)

    def test_dopri5(self):
        y = odeint(self.f, self.y0, self.t, solver=Dopri5)
        assert paddle.allclose(self.sol, y, rtol=4e-3)

    def test_dopri8(self):
        y = odeint(self.f, self.y0, self.t, solver=Dopri8)
        assert paddle.allclose(self.sol, y, rtol=4e-3)

    def test_fehlberg2(self):
        y = odeint(self.f, self.y0, self.t, solver=Fehlberg2)
        assert paddle.allclose(self.sol, y, rtol=4e-3)

    def test_adaptive_heun(self):
        y = odeint(self.f, self.y0, self.t, solver=AdaptiveHeun)
        assert paddle.allclose(self.sol, y, rtol=4e-3)


class TestAdaptiveSolversLinearForODE(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        f, y0, t, sol = construct_problem(
            dtype=paddle.float32,
            ode="linear",
            reverse=False,
        )
        self.f = f
        self.y0 = y0
        self.t = t
        self.sol = sol

        self.xdeints = [odeint, odeint_adjoint]
        self.solvers = [Bosh3, Dopri5, Dopri8, Fehlberg2, AdaptiveHeun]

    def test_bosh3(self):
        y = odeint(self.f, self.y0, self.t, solver=Bosh3)
        assert paddle.allclose(self.sol, y, rtol=4e-3)

    def test_dopri5(self):
        y = odeint(self.f, self.y0, self.t, solver=Dopri5)
        assert paddle.allclose(self.sol, y, rtol=4e-3)

    def test_dopri8(self):
        y = odeint(self.f, self.y0, self.t, solver=Dopri8)
        assert paddle.allclose(self.sol, y, rtol=4e-3)

    def test_fehlberg2(self):
        y = odeint(self.f, self.y0, self.t, solver=Fehlberg2)
        assert paddle.allclose(self.sol, y, rtol=4e-3)

    def test_adaptive_heun(self):
        y = odeint(self.f, self.y0, self.t, solver=AdaptiveHeun)
        assert paddle.allclose(self.sol, y, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
