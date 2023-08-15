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
        indices = [[0, 0, 1], [0, 1, 0]]
        values = [-1.0, 1.0, 1.0]
        dense_shape = [2, 2]
        h = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)

        self._h = h

    def ts(self, t, der=False, z=1.0):
        t = z * t
        # 判断是否是微分过程
        if not der:
            return paddle.concat([t, paddle.to_tensor([1.0])]).unsqueeze(0)  # [1, 4]
        else:
            return paddle.concat(
                [paddle.to_tensor([1.0]), paddle.to_tensor([0.0])]
            ).unsqueeze(
                0
            )  # [1, 4]

    def ps(self, index):
        p = paddle.concat(
            [
                self._series[..., index : index + 1, :],
                self._series[..., index + 1 : index + 2, :],
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
            return paddle.concat(
                [t**3, t**2, t, paddle.to_tensor([1.0])]
            ).unsqueeze(
                0
            )  # [1, 4]
        else:
            return paddle.concat(
                [3 * t**2, 2 * t, paddle.to_tensor([1.0]), paddle.to_tensor([0.0])]
            ).unsqueeze(
                0
            )  # [1, 4]

    def ps(self, index):
        p = paddle.concat(
            [
                self._series[..., index : index + 1, :],
                self._series[..., index + 1 : index + 2, :],
                self._derivs[..., index : index + 1, :],
                self._derivs[..., index + 1 : index + 2, :],
            ],
            axis=-2,
        )
        return p


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
        indices = [[0, 0, 0, 0, 1, 1, 1, 2, 2, 3], [0, 1, 2, 3, 0, 1, 2, 0, 1, 0]]
        values = [-1.0, 3.0, -3.0, 1.0, 3.0, -6.0, 3.0, -3.0, 3.0, 1.0]
        dense_shape = [4, 4]
        h = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)

        self._h = h

    def ts(self, t, der=False, z=1.0):
        t = z * t
        # 判断是否是微分过程
        if not der:
            return paddle.concat(
                [t**3, t**2, t, paddle.to_tensor([1.0])]
            ).unsqueeze(
                0
            )  # [1, 4]
        else:
            return paddle.concat(
                [3 * t**2, 2 * t, paddle.to_tensor([1.0]), paddle.to_tensor([0.0])]
            ).unsqueeze(
                0
            )  # [1, 4]

    def ps(self, index):
        p = paddle.concat(
            [
                self._series[..., index : index + 1, :],
                self._series[..., index + 1 : index + 2, :],
                self._series[..., index + 2 : index + 3, :],
                self._series[..., index + 3 : index + 4, :],
            ],
            axis=-2,
        )
        return p
