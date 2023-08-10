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
                0, coeffs.shape[-2] - 1, coeffs.shape[-2], dtype=coeffs.dtype
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

    def interpolate(self, t):
        t = paddle.to_tensor(t, dtype=self._derivs.dtype)
        maxlen = self._derivs.shape[-2] - 1
        # clamp because t may go outside of [t[0], t[-1]]; this is fine
        index = (paddle.bucketize(t.detach(), self._t.detach()) - 1).clip(0, maxlen)
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index]
        return fractional_part, index

    def evaluate(self, t):
        fractional_part, index = self.interpolate(t)
        fractional_part = fractional_part.unsqueeze(-1)
        prev_coeff = self._coeffs[..., int(index), :]
        next_coeff = self._coeffs[..., int(index) + 1, :]
        prev_t = self._t[index]
        next_t = self._t[index + 1]
        diff_t = next_t - prev_t
        return prev_coeff + fractional_part * (
            next_coeff - prev_coeff
        ) / diff_t.unsqueeze(-1)

    def derivative(self, t):
        fractional_part, index = self.interpolate(t)
        deriv = self._derivs[..., index, :]
        return deriv


class CubicSpline(NDInterpolationBase):
    """Calculates the cubic spline approximation to the batch of controls given. Also calculates its derivative.

    Example:
        # (2, 1) are batch dimensions. 7 is the time dimension (of the same length as t). 3 is the channel dimension.
        x = torch.rand(2, 1, 7, 3)
        coeffs = natural_cubic_coeffs(x)
        # ...at this point you can save coeffs, put it through PyTorch's Datasets and DataLoaders, etc...
        spline = CubicSpline(coeffs)
        point = torch.tensor(0.4)
        # will be a tensor of shape (2, 1, 3), corresponding to batch and channel dimensions
        out = spline.derivative(point)
    """

    def __init__(self, coeffs, t=None, **kwargs):
        """
        Arguments:
            coeffs: As returned by `torchcde.natural_cubic_coeffs`.
            t: As passed to linear_interpolation_coeffs. (If it was passed. If you are using neural CDEs then you **do
                not need to use this argument**. See the Further Documentation in README.md.)
        """
        super(CubicSpline, self).__init__(**kwargs)

        if t is None:
            t = paddle.linspace(
                0,
                coeffs.shape[-2],
                coeffs.shape[-2] + 1,
                dtype=coeffs.dtype,
            )

        channels = coeffs.shape[-1] // 4
        if channels * 4 != coeffs.shape[-1]:  # check that it's a multiple of 4
            raise ValueError("Passed invalid coeffs.")
        a, b, two_c, three_d = (
            coeffs[..., :channels],
            coeffs[..., channels : 2 * channels],
            coeffs[..., 2 * channels : 3 * channels],
            coeffs[..., 3 * channels :],
        )

        self.register_buffer("_t", t)
        self.register_buffer("_a", a)
        self.register_buffer("_b", b)
        # as we're typically computing derivatives, we store the multiples of these coefficients that are more useful
        self.register_buffer("_two_c", two_c)
        self.register_buffer("_three_d", three_d)

    @property
    def grid_points(self):
        return self._t

    @property
    def interval(self):
        return paddle.stack([self._t[0], self._t[-1]])

    def _interpret_t(self, t):
        t = paddle.to_tensor(t, dtype=self._b.dtype)
        maxlen = self._b.shape[-2] - 1
        # clamp because t may go outside of [t[0], t[-1]]; this is fine
        index = paddle.bucketize(t.detach(), self._t.detach()).sub(1).clamp(0, maxlen)
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index]
        return fractional_part, index

    def evaluate(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        inner = (
            0.5 * self._two_c[..., index, :]
            + self._three_d[..., index, :] * fractional_part / 3
        )
        inner = self._b[..., index, :] + inner * fractional_part
        return self._a[..., index, :] + inner * fractional_part

    def derivative(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        inner = (
            self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part
        )
        deriv = self._b[..., index, :] + inner * fractional_part
        return deriv
