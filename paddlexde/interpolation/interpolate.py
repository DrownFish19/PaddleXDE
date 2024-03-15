import paddle

from .interpolate_base import InterpolationBase


class LinearInterpolation(InterpolationBase):
    """Calculates the linear interpolation to the batch of controls given. Also calculates its derivative."""

    def __init__(self, series, t=None):
        """_summary_

        Arguments:
        -----------
            series (tensor): B T D
            t (_type_, optional): _description_. Defaults to None. [T]

        Example:
        -----------
        ```
        series = paddle.stack(
            [paddle.cast(paddle.arange(0, 2, 0.001), dtype="float32"), paddle.zeros([2000])],
            axis=-1,
        ).unsqueeze(0)
        series = paddle.sin(series)
        t = paddle.arange(0, 2000, 1)

        chp = LinearInterpolation(series, t)
        print(chp.evaluate(99))
        print(chp.derivative(99))
        ```
        """
        super().__init__(series, t)

        # build cubic hemite spline matrix H
        indices = [[0, 0, 1], [0, 1, 0]]
        values = [-1.0, 1.0, 1.0]
        dense_shape = [2, 2]
        h = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)

        self._h = h

    def _make_series(self, series, t):
        """_summary_

        Args:
            series (_type_): [B, T, D]
            t (_type_): [T]

            Note: B can be ignored

        Returns:
            _type_: _description_
        """
        scale = t[1:] - t[:-1]
        scale1 = paddle.concat([scale, scale[-1:]])
        scale2 = paddle.concat([scale[:1], scale1[:-1]])

        series1 = series
        series2 = paddle.concat([series1[..., 1:, :], series[..., -1:, :]], axis=-2)

        # [B, T, 2, D]
        series_r = paddle.stack(
            [series1 / scale1.unsqueeze(-1), series2 / scale2.unsqueeze(-1)], axis=-2
        )

        return series_r, scale1

    def _make_derivative(self, series, t):
        return None  # 线性插值不需要梯度信息

    def ts(self, t, der=False):
        if not der:  # 判断是否是微分过程
            t_list = [t, paddle.ones_like(t)]
        else:
            t_list = [paddle.ones_like(t), paddle.zeros_like(t)]

        t_tensor = paddle.stack(t_list, axis=-1).unsqueeze(-2)
        return t_tensor  # [B, T, 1, M]

    def ps(self, index):
        """_summary_

        Args:
            index (_type_): [T]

        Returns:
            _type_: _description_
        """
        p_tensor = paddle.stack(
            [
                paddle.index_select(self._series_arr[..., 0, :], index, axis=-2),
                paddle.index_select(self._series_arr[..., 1, :], index, axis=-2),
            ],
            axis=-2,
        )
        # [B, T, M, D]
        return p_tensor


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
        series = paddle.stack(
            [paddle.cast(paddle.arange(0, 2, 0.001), dtype="float32"), paddle.zeros([2000])],
            axis=-1,
        ).unsqueeze(0)
        series = paddle.sin(series)
        t = paddle.arange(0, 2000, 1)

        chp = CubicHermiteSpline(series, t)
        print(chp.evaluate(99))
        print(chp.derivative(22.2))
        ```
        """
        super().__init__(series, t, **kwargs)

        # build cubic hemite spline matrix H
        indices = [[0, 0, 0, 0, 1, 1, 1, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3, 2, 0]]
        values = [2.0, -2.0, 1.0, 1.0, -3.0, 3.0, -2.0, -1.0, 1.0, 1]
        dense_shape = [4, 4]
        h = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)

        self._h = h

    def _make_series(self, series, t):
        """_summary_

        Args:
            series (_type_): [B, X, T, D]
            t (_type_): [T]

            Note: B and X can be ignored

        Returns:
            _type_: _description_
        """
        scale = t[1:] - t[:-1]
        scale1 = paddle.concat([scale, scale[-1:]])
        scale2 = paddle.concat([scale[:1], scale1[:-1]])

        series1 = series
        series2 = paddle.concat([series1[..., 1:, :], series[..., -1:, :]], axis=-2)

        # [B, T, S, D]
        series_r = paddle.stack(
            [series1 / scale1.unsqueeze(-1), series2 / scale2.unsqueeze(-1)], axis=-2
        )

        return series_r, scale1

    def _make_derivative(self, series, t):
        """_summary_

        Args:
            series (_type_): [B, T, D]
            t (_type_): [T]

            Note: B can be ignored

        Returns:
            _type_: _description_
        """
        diffs_t = t[1:] - t[:-1]
        diffs_t1 = paddle.concat([diffs_t, diffs_t[-1:]])

        diffs_series = series[..., 1:, :] - series[..., :-1, :]
        diffs_series = paddle.concat([diffs_series, diffs_series[..., -1:, :]], axis=-2)

        # 梯度 # [B T D] # 最后一个点的梯度采用上一个点的梯度
        derivs = diffs_series / diffs_t1.unsqueeze(-1)  # [B, T-1, D]
        derivs = paddle.concat([derivs, derivs[..., -1:, :]], axis=-2)

        return derivs

    def ts(self, t, der=False):
        if not der:  # 判断是否是微分过程
            t_list = [t**3, t**2, t, paddle.ones_like(t)]
        else:
            t_list = [3 * t**2, 2 * t, paddle.ones_like(t), paddle.zeros_like(t)]

        t_tensor = paddle.stack(t_list, axis=-1).unsqueeze(-2)
        return t_tensor  # [B, T, 1, M]

    def ps(self, index):
        p_tensor = paddle.stack(
            [
                paddle.index_select(self._series_arr[..., 0, :], index, axis=-2),
                paddle.index_select(self._series_arr[..., 1, :], index, axis=-2),
                paddle.index_select(self._derivs, index, axis=-2),
                paddle.index_select(self._derivs, index + 1, axis=-2),
            ],
            axis=-2,
        )
        # [B, T, M, D]
        return p_tensor


class BezierSpline(InterpolationBase):
    def __init__(self, series, t=None, **kwargs):
        """_summary_

        Arguments:
        -----------
            series (tensor): B T D
            t (_type_, optional): _description_. Defaults to None.

        Example:
        -----------
        ```
        series = paddle.stack(
            [paddle.cast(paddle.arange(0, 2, 0.001), dtype="float32"), paddle.zeros([2000])],
            axis=-1,
        ).unsqueeze(0)
        series = paddle.sin(series)
        t = paddle.arange(0, 2000, 1)

        chp = BezierSpline(series, t)
        print(chp.evaluate(99))
        print(chp.derivative(22.2))
        ```
        """
        super().__init__(series, t, **kwargs)

        # build cubic hemite spline matrix H
        indices = [[0, 0, 0, 0, 1, 1, 1, 2, 2, 3], [0, 1, 2, 3, 0, 1, 2, 0, 1, 0]]
        values = [-1.0, 3.0, -3.0, 1.0, 3.0, -6.0, 3.0, -3.0, 3.0, 1.0]
        dense_shape = [4, 4]
        h = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)

        self._h = h

    def _make_series(self, series, t):
        """_summary_

        Args:
            series (_type_): [B, T, D]
            t (_type_): [B, T]

        Returns:
            _type_: _description_
        """
        scale = t[3:] - t[:-3]
        scale1 = paddle.concat([scale, scale[-1:], scale[-1:], scale[-1:]])
        scale2 = paddle.concat([scale[:1], scale1[:-1]])
        scale3 = paddle.concat([scale[:1], scale2[:-1]])
        scale4 = paddle.concat([scale[:1], scale3[:-1]])

        series1 = series
        series2 = paddle.concat([series1[..., 1:, :], series[..., -1:, :]], axis=-2)
        series3 = paddle.concat([series2[..., 1:, :], series[..., -1:, :]], axis=-2)
        series4 = paddle.concat([series3[..., 1:, :], series[..., -1:, :]], axis=-2)

        # [B, T, 4, D]
        series_r = paddle.stack(
            [
                series1 / scale1.unsqueeze(-1),
                series2 / scale2.unsqueeze(-1),
                series3 / scale3.unsqueeze(-1),
                series4 / scale4.unsqueeze(-1),
            ],
            axis=-2,
        )

        return series_r, scale1

    def _make_derivative(self, series, t):
        return None  # 不需要计算梯度

    def ts(self, t, der=False):
        if not der:  # 判断是否是微分过程
            t_list = [t**3, t**2, t, paddle.ones_like(t)]
        else:
            t_list = [3 * t**2, 2 * t, paddle.ones_like(t), paddle.zeros_like(t)]

        t_tensor = paddle.stack(t_list, axis=-1).unsqueeze(-2)
        return t_tensor  # [B, T, 1, M]

    def ps(self, index):
        p_tensor = paddle.stack(
            [
                paddle.index_select(self._series_arr[..., 0, :], index, axis=-2),
                paddle.index_select(self._series_arr[..., 1, :], index, axis=-2),
                paddle.index_select(self._series_arr[..., 2, :], index, axis=-2),
                paddle.index_select(self._series_arr[..., 3, :], index, axis=-2),
            ],
            axis=-2,
        )
        # [B, T, M, D]
        return p_tensor
