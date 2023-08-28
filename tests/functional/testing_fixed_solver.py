import unittest

import paddle

from paddlexde.functional import odeint, odeint_adjoint
from paddlexde.solver.fixed_solver import RK4, AdamsBashforthMoulton, Euler, Midpoint
from tests.testing_utils import construct_problem


class TestFixedSolversSineForODE(unittest.TestCase):
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
        self.solvers = [Euler, Midpoint, RK4, AdamsBashforthMoulton]

    def test_bosh3(self):
        for xdeint in self.xdeints:
            y = xdeint(self.f, self.y0, self.t, solver=Euler)
            assert paddle.allclose(self.sol, y, rtol=4e-3)

    def test_dopri5(self):
        for xdeint in self.xdeints:
            y = xdeint(self.f, self.y0, self.t, solver=Midpoint)
            assert paddle.allclose(self.sol, y, rtol=4e-3)

    def test_dopri8(self):
        for xdeint in self.xdeints:
            y = xdeint(self.f, self.y0, self.t, solver=RK4)
            assert paddle.allclose(self.sol, y, rtol=4e-3)

    def test_fehlberg2(self):
        for xdeint in self.xdeints:
            y = xdeint(self.f, self.y0, self.t, solver=AdamsBashforthMoulton)
            assert paddle.allclose(self.sol, y, rtol=4e-3)


class TestFixedSolversLinearForODE(unittest.TestCase):
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
        self.solvers = [Euler, Midpoint, RK4, AdamsBashforthMoulton]

    def test_bosh3(self):
        for xdeint in self.xdeints:
            y = xdeint(self.f, self.y0, self.t, solver=Euler)
            assert paddle.allclose(self.sol, y, rtol=4e-3)

    def test_dopri5(self):
        for xdeint in self.xdeints:
            y = xdeint(self.f, self.y0, self.t, solver=Midpoint)
            assert paddle.allclose(self.sol, y, rtol=4e-3)

    def test_dopri8(self):
        for xdeint in self.xdeints:
            y = xdeint(self.f, self.y0, self.t, solver=RK4)
            assert paddle.allclose(self.sol, y, rtol=4e-3)

    def test_fehlberg2(self):
        for xdeint in self.xdeints:
            y = xdeint(self.f, self.y0, self.t, solver=AdamsBashforthMoulton)
            assert paddle.allclose(self.sol, y, rtol=4e-3)


if __name__ == "__main__":
    unittest.main()
