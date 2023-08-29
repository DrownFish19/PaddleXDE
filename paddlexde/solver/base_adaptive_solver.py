import abc

import paddle


class AdaptiveSolver(metaclass=abc.ABCMeta):
    def __init__(self, xde, dtype: type, y0: paddle.Tensor, norm, **unused_kwargs):
        self.dtype = dtype
        self.y0 = y0
        self.norm = norm

        self.xde = xde
        self.move = self.xde.move
        self.fuse = self.xde.fuse
        self.get_dy = self.xde.get_dy

    @abc.abstractmethod
    def _before_integrate(self, t):
        pass

    @abc.abstractmethod
    def step(self, next_t):
        raise NotImplementedError

    def integrate(self, t):
        solution = paddle.empty([len(t), *self.y0.shape], dtype=self.y0.dtype)
        solution[0] = self.y0
        t = t.astype(self.dtype)
        self._before_integrate(t)
        for i in range(1, len(t)):
            solution[i] = self.step(t[i])
        return solution

    def select_initial_step(self, t0, y0, order, rtol, atol, f0=None):
        """Empirically select a good initial step.

        The algorithm is described in [1]_.

        References
        ----------
        .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
               Equations I: Nonstiff Problems", Sec. II.4, 2nd edition.
        """

        dtype = y0.dtype
        t_dtype = t0.dtype

        if f0 is None:
            f0 = self.move(t0, 0, y0)

        scale = atol + paddle.abs(y0) * rtol

        d0 = self.norm(y0 / scale).abs()
        d1 = self.norm(self.get_dy(f0) / scale).abs()

        if d0 < 1e-5 or d1 < 1e-5:
            h0 = paddle.to_tensor(1e-6, dtype=dtype)
        else:
            h0 = 0.01 * d0 / d1
        h0 = h0.abs()

        y1 = self.fuse(f0, h0, y0)
        f1 = self.move(t0 + h0, 0, y1)

        d2 = paddle.abs(self.norm((self.get_dy(f1) - self.get_dy(f0)) / scale) / h0)

        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = paddle.max(paddle.to_tensor(1e-6, dtype=dtype), h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1.0 / float(order + 1))
        h1 = h1.abs()

        return paddle.fmin(100.0 * h0, h1).astype(t_dtype)
