import paddle

from .interpnd import NDInterpolationBase


class LinearInterpolation(NDInterpolationBase):
    """Calculates the linear interpolation to the batch of controls given. Also calculates its derivative."""

    def __init__(self, series, t=None, **kwargs):
        """
        Arguments:
            series: As returned by linear_interpolation_coeffs.
            t: As passed to linear_interpolation_coeffs. (If it was passed. If you are using neural CDEs then you **do
                not need to use this argument**. See the Further Documentation in README.md.)
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
        deriv = self._derivs[..., index, :]
        return deriv

class CubicHermiteSpline(NDInterpolationBase):
    
    def __init__(self, series, t=None, **kwargs):
        """_summary_

        Args:
            series (tensor): B T D
            t (_type_, optional): _description_. Defaults to None.
        """
        super(CubicHermiteSpline).__init__()

        if t is None:
            t = paddle.linspace(
                0,
                series.shape[-2],
                series.shape[-2] + 1,
                dtype=series.dtype,
            )

        # build cubic hemite spline matrix H
        indices = [[0, 0, 0, 0, 1, 1, 1, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3, 2, 0]]
        values = [2.0, -2.0, 1.0, 1.0, -3.0, 3.0, -2.0, -1.0, 1.0, 1]
        dense_shape = [4, 4]
        h = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)

        # build cubic hermite spline P
        derivs = (series[..., 1:, :] - series[..., :-1, :]) / (t[1:] - t[:-1])  # [B, T-1, D]
        derivs = paddle.concat([derivs, derivs[..., -1:, :]], axis=-2)  # [B T D] # 最后一个点的梯度采用上一个点的梯度
        
        self.register_buffer("_t", t)
        self.register_buffer("_h", h)
        self.register_buffer("_series", series)
        self.register_buffer("_derivs", derivs)


    def Ts(self,t, der=False):
        # 判断是否是微分过程
        if not der:
            ts = [t**3,t**2,t,1]
        else :
            ts = [3*t**2,2*t,1,0]
        
        return paddle.to_tensor(ts)
    
    
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
        deriv = self._derivs[..., index, :]
        return deriv
    
    
    
chp = CubicHermiteSpline()

print(chp.h.to_dense())
        
        