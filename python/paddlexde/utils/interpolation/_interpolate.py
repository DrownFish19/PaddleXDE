import paddle

from .interpnd import InterpolationBase


class LinearInterpolation(InterpolationBase):
    """Calculates the linear interpolation to the batch of controls given. Also calculates its derivative."""

    def __init__(self, series, t=None, **kwargs):
        """
        Arguments:
        -----------
            series: As returned by linear_interpolation_coeffs.
            t: As passed to linear_interpolation_coeffs. (If it was passed. If you are using neural CDEs then you **do
                not need to use this argument**. See the Further Documentation in README.md.)

        Example:
        -----------
        ```
            # (2, 1) are batch dimensions. 7 is the time dimension (of the same length as t). 3 is the channel dimension.
            x = paddle.rand(2, 1, 7, 3)
            coeffs = natural_cubic_coeffs(x)
            # ...at this point you can save coeffs, put it through PyTorch's Datasets and DataLoaders, etc...
            spline = CubicSpline(coeffs)
            point = paddle.tensor(0.4)
            # will be a tensor of shape (2, 1, 3), corresponding to batch and channel dimensions
            out = spline.derivative(point)
        ```
        """
        super(LinearInterpolation, self).__init__(**kwargs)

        if t is None:
            t = paddle.linspace(
                0, series.shape[-2] - 1, series.shape[-2], dtype=series.dtype
            )

        derivs = (series[..., 1:, :] - series[..., :-1, :]) / (
            t[1:] - t[:-1]
        ).unsqueeze(-1)

        self.register_buffer("_t", t)
        self.register_buffer("_coeffs", series)
        self.register_buffer("_derivs", derivs)

    def interpolate(self, t):
        t = paddle.to_tensor(t, dtype=self._derivs.dtype)
        maxlen = self._derivs.shape[-2] - 1
        # clamp because t may go outside of [t[0], t[-1]]; this is fine
        index = (paddle.bucketize(t.detach(), self._t.detach()) - 1).clip(0, maxlen)
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index : index + 1]
        return fractional_part, index

    def evaluate(self, t):
        fractional_part, index = self.interpolate(t)
        fractional_part = fractional_part.unsqueeze(-1)
        prev_coeff = self._coeffs[..., index : index + 1, :].squeeze(-2)
        next_coeff = self._coeffs[..., index + 1 : index + 2, :].squeeze(-2)
        prev_t = self._t[index : index + 1]
        next_t = self._t[index + 1 : index + 2]
        diff_t = next_t - prev_t
        return prev_coeff + fractional_part * (
            next_coeff - prev_coeff
        ) / diff_t.unsqueeze(-1)

    def derivative(self, t):
        fractional_part, index = self.interpolate(t)
        deriv = self._derivs[..., index : index + 1, :]
        return deriv


class CubicHermiteSpline(InterpolationBase):
    def __init__(self, series, t=None, **kwargs):
        """_summary_

        Arguments:
        -----------
            series (tensor): B T D
            t (_type_, optional): _description_. Defaults to None.

        Example:
        -----------
        ```
            # (2, 1) are batch dimensions. 7 is the time dimension (of the same length as t). 3 is the channel dimension.
            x = paddle.rand(2, 1, 7, 3)
            coeffs = natural_cubic_coeffs(x)
            # ...at this point you can save coeffs, put it through PyTorch's Datasets and DataLoaders, etc...
            spline = CubicSpline(coeffs)
            point = paddle.tensor(0.4)
            # will be a tensor of shape (2, 1, 3), corresponding to batch and channel dimensions
            out
        """
        super().__init__(series, t, **kwargs)

        # build cubic hemite spline matrix H
        indices = [[0, 0, 0, 0, 1, 1, 1, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3, 2, 0]]
        values = [2.0, -2.0, 1.0, 1.0, -3.0, 3.0, -2.0, -1.0, 1.0, 1]
        dense_shape = [4, 4]
        h = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)

        self._h = h

    def ts(self, t, der=False, z=1.0):
        t = z * t
        # 判断是否是微分过程
        if not der:
            return paddle.to_tensor([t**3, t**2, t, 1]).unsqueeze(-1)  # [1, 4]
        else:
            paddle.to_tensor([3 * t**2, 2 * t, 1, 0]).unsqueeze(-1)  # [1, 4]

    def ps(self, index):
        p = paddle.concat(
            [
                self._series[..., index : index + 1, :],
                self._series[..., index + 1, index + 2, :],
                self._derivs[..., index, index + 1, :],
                self._derivs[..., index + 1 : index + 2, :],
            ],
            axis=-2,
        )
        return p
