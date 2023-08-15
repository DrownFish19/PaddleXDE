import abc

import paddle
from paddle import nn


class InterpolationBase(nn.Layer, metaclass=abc.ABCMeta):
    def __init__(self, series, t=None, **kwargs):
        super().__init__()

        if t is None:
            t = paddle.linspace(
                0,
                series.shape[-2],
                series.shape[-2] + 1,
                dtype=series.dtype,
            )
        series = paddle.cast(series, dtype="float32")
        t = paddle.cast(t, dtype="float32")
        # build cubic hermite spline P
        derivs = (series[..., 1:, :] - series[..., :-1, :]) / (
            t[1:] - t[:-1]
        ).unsqueeze(
            -1
        )  # [B, T-1, D]
        derivs = paddle.concat(
            [derivs, derivs[..., -1:, :]], axis=-2
        )  # [B T D] # 最后一个点的梯度采用上一个点的梯度

        self._t = t
        self._series = series
        self._derivs = derivs

    @property
    def grid_points(self):
        """The time points."""
        return self._t

    @property
    def interval(self):
        """The time interval between time points."""
        return paddle.stack([self._t[0], self._t[-1]])

    def interpolate(self, t):
        """Calculates the index of the given time point t in the list of time points.

        Args:
            t (_type_): time point t

        Raises:
            NotImplementedError:

        Retuns:
            The index of the given time point t in the list of time points.
        """
        t = paddle.to_tensor(t, dtype=self._derivs.dtype)
        maxlen = self._derivs.shape[-2] - 1
        # clamp because t may go outside of [t[0], t[-1]]; this is fine
        index = (paddle.bucketize(t.detach(), self._t.detach()) - 1).clip(0, maxlen)
        # will never access the last element of self._t; this is correct behaviour
        diff_t = (
            self._t[index + 1 : index + 2] - self._t[index : index + 1]
        ).reciprocal()  # 计算t和下界的差

        p = self.ps(index)

        return diff_t, p

    def evaluate(self, t):
        """Calculates the value at the time point t.

        Args:
            t (_type_): The time point t

        Raises:
            NotImplementedError:

        Retruns:
            The value at the time point t.
        """

        diff_t, p = self.interpolate(t)  # diff_t
        ts_tensor = self.ts(t, der=False, z=diff_t)
        result = ts_tensor @ self._h.to_dense() @ p
        return result

    def derivative(self, t):
        """Calculates the derivative of the function at the point t.

        Args:
            t (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            The derivative of the function at the point t.
        """
        diff_t, p = self.interpolate(t)
        return self.ts(t, der=True, z=diff_t) @ self._h.to_dense() @ p

    @abc.abstractmethod
    def ts(self, t, der=False, z=1.0):
        """_summary_

        Args:
            t (_type_): _description_
            der (bool, optional): _description_. Defaults to False.
            z (float, optional): _description_. Defaults to 1.0.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ps(self, index):
        """return P tensor

        Args:
            index (int): index of time point

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
