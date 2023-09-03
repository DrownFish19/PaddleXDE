import unittest

import paddle

from paddlexde.functional import odeint, odeint_adjoint
from paddlexde.solver.fixed_solver import RK4, AdamsBashforthMoulton, Euler, Midpoint
from tests.testing_utils import construct_problem


class TestFixedSolversConstantForODE(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        f, y0, t, sol = construct_problem(
            dtype=paddle.float32,
            ode="constant",
            reverse=False,
        )
        self.f = f
        self.y0 = y0
        self.t = t
        self.sol = sol

        self.xdeints = [odeint, odeint_adjoint]
        self.solvers = [Euler, Midpoint, RK4, AdamsBashforthMoulton]

    def test_euler(self):
        for xdeint in self.xdeints:
            y = xdeint(self.f, self.y0, self.t, solver=Euler)
            assert paddle.allclose(self.sol, y, rtol=1e-2)

    def test_midpoint(self):
        for xdeint in self.xdeints:
            y = xdeint(self.f, self.y0, self.t, solver=Midpoint)
            assert paddle.allclose(self.sol, y, rtol=1e-2)

    def test_rk4(self):
        for xdeint in self.xdeints:
            y = xdeint(self.f, self.y0, self.t, solver=RK4)
            assert paddle.allclose(self.sol, y, rtol=1e-2)

    def test_adam_bashforth_moulton(self):
        for xdeint in self.xdeints:
            y = xdeint(self.f, self.y0, self.t, solver=AdamsBashforthMoulton)
            assert paddle.allclose(self.sol, y, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
