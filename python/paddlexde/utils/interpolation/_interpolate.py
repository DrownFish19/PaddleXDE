import paddle

from .interpnd import InterpolationBase


class LinearInterpolation(InterpolationBase):
    """Calculates the linear interpolation to the batch of controls given. Also calculates its derivative."""

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

        chp = LinearInterpolation(series, t)
        print(chp.evaluate(99))
        print(chp.derivative(99))
        ```
        """
        super().__init__(series, t, **kwargs)

        # build cubic hemite spline matrix H
        indices = [[0, 0, 1], [0, 1, 0]]
        values = [-1.0, 1.0, 1.0]
        dense_shape = [2, 2]
        h = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)

        self._h = h

    def ts(self, t, der=False):
        if not der:  # 判断是否是微分过程
            t_list = [t, paddle.ones_like(t)]
        else:
            t_list = [paddle.ones_like(t), paddle.zeros_like(t)]

        t_tensor = paddle.concat(t_list).unsqueeze(0)
        return t_tensor  # [1, 4]

    def ps(self, index):
        p = paddle.concat(
            [
                self._norm_series1[..., index : index + 1, :],
                self._norm_series2[..., index : index + 1, :],
            ],
            axis=-2,
        )
        return p


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

    def ts(self, t, der=False):
        if not der:  # 判断是否是微分过程
            t_list = [t**3, t**2, t, paddle.ones_like(t)]
        else:
            t_list = [3 * t**2, 2 * t, paddle.ones_like(t), paddle.zeros_like(t)]

        t_tensor = paddle.concat(t_list).unsqueeze(0)
        return t_tensor  # [1, 4]

    def ps(self, index):
        p_tensor = paddle.concat(
            [
                self._norm_series1[..., index : index + 1, :],
                self._norm_series2[..., index : index + 1, :],
                self._derivs[..., index : index + 1, :],
                self._derivs[..., index + 1 : index + 2, :],
            ],
            axis=-2,
        )
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

    def ts(self, t, der=False):
        if not der:  # 判断是否是微分过程
            t_list = [t**3, t**2, t, paddle.ones_like(t)]
        else:
            t_list = [3 * t**2, 2 * t, paddle.ones_like(t), paddle.zeros_like(t)]

        t_tensor = paddle.concat(t_list).unsqueeze(0)
        return t_tensor  # [1, 4]

    def ps(self, index):
        p_tensor = paddle.concat(
            [
                self._norm_series1[..., index : index + 1, :],
                self._norm_series2[..., index : index + 1, :],
                self._norm_series3[..., index : index + 1, :],
                self._norm_series4[..., index : index + 1, :],
            ],
            axis=-2,
        )
        return p_tensor
