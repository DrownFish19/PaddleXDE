import paddle
from .interpnd import NDInterpolationBase


class LinearInterpolation(NDInterpolationBase):
    """Calculates the linear interpolation to the batch of controls given. Also calculates its derivative."""

    def __init__(self, coeffs, t=None, **kwargs):
        """
        Arguments:
            coeffs: As returned by linear_interpolation_coeffs.
            t: As passed to linear_interpolation_coeffs. (If it was passed. If you are using neural CDEs then you **do
                not need to use this argument**. See the Further Documentation in README.md.)
        """
        super(LinearInterpolation, self).__init__(**kwargs)

        if t is None:
            t = paddle.linspace(
                0, coeffs.size(-2) - 1, coeffs.size(-2), dtype=coeffs.dtype
            )

        derivs = (coeffs[..., 1:, :] - coeffs[..., :-1, :]) / (
            t[1:] - t[:-1]
        ).unsqueeze(-1)

        self.register_buffer("_t", t)
        self.register_buffer("_coeffs", coeffs)
        self.register_buffer("_derivs", derivs)

    @property
    def grid_points(self):
        return self._t

    @property
    def interval(self):
        return paddle.stack([self._t[0], self._t[-1]])

    def _interpret_t(self, t):
        t = paddle.to_tensor(t, dtype=self._derivs.dtype)
        maxlen = self._derivs.size(-2) - 1
        # clamp because t may go outside of [t[0], t[-1]]; this is fine
        index = paddle.bucketize(t.detach(), self._t.detach()).sub(1).clamp(0, maxlen)
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index]
        return fractional_part, index

    def evaluate(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        prev_coeff = self._coeffs[..., index, :]
        next_coeff = self._coeffs[..., index + 1, :]
        prev_t = self._t[index]
        next_t = self._t[index + 1]
        diff_t = next_t - prev_t
        return prev_coeff + fractional_part * (
            next_coeff - prev_coeff
        ) / diff_t.unsqueeze(-1)

    def derivative(self, t):
        fractional_part, index = self._interpret_t(t)
        deriv = self._derivs[..., index, :]
        return deriv
