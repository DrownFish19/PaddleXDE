import math

import numpy as np
import paddle
import scipy.linalg


class ConstantODE(paddle.nn.Layer):
    def __init__(self):
        super(ConstantODE, self).__init__()
        self.a = paddle.create_parameter(
            shape=[1],
            dtype=paddle.float32,
            default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor(0.2)),
        )
        self.b = paddle.create_parameter(
            shape=[1],
            dtype=paddle.float32,
            default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor(3.0)),
        )

    def forward(self, t, y):
        return self.a + (y - (self.a * t + self.b)) ** 5

    def y_exact(self, t):
        return (self.a * t + self.b).unsqueeze(-1)


class SineODE(paddle.nn.Layer):
    def forward(self, t, y):
        return 2 * y / t + t**4 * paddle.sin(2 * t) - t**2 + 4 * t**3

    def y_exact(self, t):
        return (
            -0.5 * t**4 * paddle.cos(2 * t)
            + 0.5 * t**3 * paddle.sin(2 * t)
            + 0.25 * t**2 * paddle.cos(2 * t)
            - t**3
            + 2 * t**4
            + (math.pi - 0.25) * t**2
        ).unsqueeze(-1)


class LinearODE(paddle.nn.Layer):
    def __init__(self, dim=10):
        super(LinearODE, self).__init__()
        self.dim = dim
        U = paddle.randn([dim, dim]) * 0.1
        A = 2 * U - (U + U.transpose([1, 0]))
        self.A = paddle.create_parameter(
            shape=A.shape,
            dtype=paddle.float32,
            default_initializer=paddle.nn.initializer.Assign(A),
        )
        self.initial_val = np.ones((dim, 1))
        self.nfe = 0

    def forward(self, t, y):
        self.nfe += 1
        return paddle.mm(self.A, y.reshape([self.dim, 1])).reshape([-1])

    def y_exact(self, t):
        t_numpy = t.detach().cpu().numpy()
        A_np = self.A.detach().cpu().numpy()
        ans = []
        for t_i in t_numpy:
            ans.append(np.matmul(scipy.linalg.expm(A_np * t_i), self.initial_val))
        return paddle.stack([paddle.to_tensor(ans_) for ans_ in ans]).reshape(
            [len(t_numpy), self.dim]
        )


PROBLEMS = {"constant": ConstantODE, "linear": LinearODE, "sine": SineODE}
DTYPES = (paddle.float32, paddle.float64, paddle.complex64)
DEVICES = ["cpu"]
FIXED_METHODS = ("euler", "midpoint", "rk4", "explicit_adams", "implicit_adams")
ADAMS_METHODS = ("explicit_adams", "implicit_adams")
ADAPTIVE_METHODS = ("adaptive_heun", "fehlberg2", "bosh3", "dopri5", "dopri8")
SCIPY_METHODS = ("scipy_solver",)
METHODS = FIXED_METHODS + ADAPTIVE_METHODS + SCIPY_METHODS


def construct_problem(npts=10, ode="constant", reverse=False, dtype=paddle.float32):
    f = PROBLEMS[ode]()

    t_points = paddle.linspace(1, 8, npts, dtype=paddle.float32)
    sol = f.y_exact(t_points).astype(dtype)

    def _flip(x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = paddle.arange(x.size(dim) - 1, -1, -1, dtype=paddle.int64)
        return x[tuple(indices)]

    if reverse:
        t_points = _flip(t_points, 0).clone().detach()
        sol = _flip(sol, 0).clone().detach()

    return f, paddle.assign(sol[0]), t_points, sol


if __name__ == "__main__":
    f = SineODE()
    t_points = paddle.linspace(1, 8, 100)
    sol = f.y_exact(t_points)

    import matplotlib.pyplot as plt

    plt.plot(t_points.cpu().numpy(), sol.cpu().numpy())
    plt.show()
