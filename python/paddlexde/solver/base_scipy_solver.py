import abc

import paddle
from scipy.integrate import solve_ivp


class ScipyWrapperODESolver(metaclass=abc.ABCMeta):

    def __init__(self, func, y0, rtol, atol, min_step=0, max_step=float('inf'), solver="LSODA", **unused_kwargs):
        unused_kwargs.pop('norm', None)
        unused_kwargs.pop('grid_points', None)
        unused_kwargs.pop('eps', None)
        del unused_kwargs

        self.dtype = y0.dtype
        self.device = y0.device
        self.shape = y0.shape
        self.y0 = y0.detach().cpu().numpy().reshape(-1)
        self.rtol = rtol
        self.atol = atol
        self.min_step = min_step
        self.max_step = max_step
        self.solver = solver
        self.func = convert_func_to_numpy(func, self.shape, self.device, self.dtype)

    def integrate(self, t):
        if t.numel() == 1:
            return paddle.to_tensor(self.y0)[None].astype(self.dtype)
        t = t.detach().cpu().numpy()
        sol = solve_ivp(
            self.func,
            t_span=[t.min(), t.max()],
            y0=self.y0,
            t_eval=t,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
            min_step=self.min_step,
            max_step=self.max_step
        )
        sol = paddle.to_tensor(sol.y).T.astype(self.dtype)
        sol = sol.reshape(-1, *self.shape)
        return sol

    @classmethod
    def valid_callbacks(cls):
        return set()


def convert_func_to_numpy(func, shape, device, dtype):
    def np_func(t, y):
        t = paddle.to_tensor(t).astype(dtype)
        y = paddle.reshape(paddle.to_tensor(y).astype(dtype), shape)
        with paddle.no_grad():
            f = func(t, y)
        return f.detach().cpu().numpy().reshape(-1)

    return np_func
