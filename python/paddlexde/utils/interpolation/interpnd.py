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

        diffs_t = t[1:] - t[:-1]
        diffs_t1 = paddle.concat([diffs_t, diffs_t[-1:]])
        diffs_t2 = paddle.concat([diffs_t[:1], diffs_t1[:-1]])
        diffs_t3 = paddle.concat([diffs_t[:1], diffs_t2[:-1]])
        diffs_t4 = paddle.concat([diffs_t[:1], diffs_t3[:-1]])

        series1 = series
        series2 = paddle.concat([series1[..., 1:, :], series[..., -1:, :]], axis=-2)
        series3 = paddle.concat([series2[..., 1:, :], series[..., -1:, :]], axis=-2)
        series4 = paddle.concat([series3[..., 1:, :], series[..., -1:, :]], axis=-2)

        norm_series1 = series1 / diffs_t1.unsqueeze(-1)
        norm_series2 = series2 / diffs_t2.unsqueeze(-1)
        norm_series3 = series3 / diffs_t3.unsqueeze(-1)
        norm_series4 = series4 / diffs_t4.unsqueeze(-1)

        diffs_series = series[..., 1:, :] - series[..., :-1, :]
        diffs_series = paddle.concat([diffs_series, diffs_series[..., -1:, :]], axis=-2)

        # 梯度 # [B T D] # 最后一个点的梯度采用上一个点的梯度
        derivs = diffs_series / diffs_t1.unsqueeze(-1)  # [B, T-1, D]
        derivs = paddle.concat([derivs, derivs[..., -1:, :]], axis=-2)

        self._t = t
        self._diff_t = diffs_t1
        self._norm_series1 = norm_series1
        self._norm_series2 = norm_series2
        self._norm_series3 = norm_series3
        self._norm_series4 = norm_series4
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

    def interpolate(self, t, der=False):
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

        norm_t = (t - self._t[index : index + 1]) / self._diff_t[index : index + 1]

        ps_tensor = self.ps(index)
        ts_tensor = self.ts(norm_t, der=der)

        return ts_tensor, ps_tensor, index

    def evaluate(self, t):
        """Calculates the value at the time point t.

        Args:
            t (_type_): The time point t

        Raises:
            NotImplementedError:

        Retruns:
            The value at the time point t.
        """

        ts_tensor, ps_tensor, index = self.interpolate(t, der=False)
        result = ts_tensor @ self._h.to_dense() @ ps_tensor
        result *= self._diff_t[index : index + 1]
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
        ts_tensor, ps_tensor, index = self.interpolate(t, der=True)
        result = ts_tensor @ self._h.to_dense() @ ps_tensor
        result *= self._diff_t[index : index + 1]
        return result

    @abc.abstractmethod
    def ts(self, t, der=False):
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
