import abc

import paddle
from paddle import nn


class InterpolationBase(nn.Layer, metaclass=abc.ABCMeta):
    def __init__(self, series, t=None, **kwargs):
        """_summary_

        Args:
            series (_type_): [B, T, D]
            t (_type_, optional): [B, T]. Defaults to None.

            the B dim can be ignored
        """
        super().__init__()

        if t is None:
            t = paddle.linspace(
                0,
                series.shape[-2],
                series.shape[-2] + 1,
                dtype=series.dtype,
            ).expand(series.shape[:-1])
        series = paddle.cast(series, dtype="float32")
        t = paddle.cast(t, dtype="float32")

        series_arr, scale_t = self._make_series(series=series, t=t)
        derivs = self._make_derivative(series, t)

        self._t = t
        self._scale_t = scale_t
        self._series_arr = series_arr
        self._series = series
        self._derivs = derivs

        if len(series.shape) == 2:
            self._seq_len, self._dims = series.shape
            self._batch_size = None
        elif len(series.shape) == 3:
            self._batch_size, self._seq_len, self._dims = series.shape
        else:
            raise ValueError

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
            t (_type_): time point t [B, T]
            B can be ignored => [T]

        Raises:
            NotImplementedError:

        Retuns:
            The index of the given time point t in the list of time points.
        """
        t = paddle.to_tensor(t, dtype=self._series.dtype)
        if len(t.shape) == 1:
            t = t.expand([1, t.shape[-1]])
        maxlen = self._series.shape[-2] - 1

        # [B, T], [B, T], [B, T]
        t_shape, _t_shape, _scale_t_shape = t.shape, self._t.shape, self._scale_t.shape

        index = []
        norm_t = []

        t = paddle.reshape(t, shape=[-1, t_shape[-1]])
        _t = paddle.reshape(self._t, shape=[-1, _t_shape[-1]])
        _scale_t = paddle.reshape(self._scale_t, shape=[-1, _scale_t_shape[-1]])

        for bs in range(t_shape[0]):
            t_i, _t_i, _scale_t_i = t[bs], _t[bs], _scale_t[bs]
            # clamp because t may go outside of [t[0], t[-1]]; this is fine
            # will never access the last element of self._t; this is correct behaviour
            index_i = (paddle.bucketize(t_i, _t_i) - 1).clip(0, maxlen)  # [T]
            norm_t_i = (t_i - _t_i[index_i]) / _scale_t_i[index_i]  # [T]
            index.append(index_i)
            norm_t.append(norm_t_i)

        index = paddle.stack(index, axis=0).reshape(t_shape)  # [B, T]
        norm_t = paddle.stack(norm_t, axis=0).reshape(t_shape)  # [B, T]

        # [B, T, 1, M], M is 2 for linear, 4 for cubic
        ts_tensor = self.ts(norm_t, der=der)
        # [B, T, M, D], M is 2 for linear, 4 for cubic
        ps_tensor = self.ps(index)

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
        # [B, T, 1, M], [B, T, M, D], [B, T]
        ts_tensor, ps_tensor, index = self.interpolate(t, der=False)
        # [B, T, 1, D] => [B, T, D]
        result = (ts_tensor @ self._h.to_dense() @ ps_tensor).squeeze(-2)
        axis_b = paddle.arange(self._batch_size)[:, None]
        axis_index = index[:, :]
        scale = self._scale_t[axis_b, axis_index].unsqueeze(-1)  # [B, T, 1]
        result *= scale
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
        # [B, T, 1, M], [B, T, M, D], [B, T]
        ts_tensor, ps_tensor, index = self.interpolate(t, der=True)
        # [B, T, 1, D] => [B, T, D]
        result = (ts_tensor @ self._h.to_dense() @ ps_tensor).squeeze(-2)
        # result *= self._scale_t[index : index + 1]
        return result

    @abc.abstractmethod
    def _make_series(self, series, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _make_derivative(self, series, t):
        raise NotImplementedError

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
            index (int): index of time point, [T]

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
