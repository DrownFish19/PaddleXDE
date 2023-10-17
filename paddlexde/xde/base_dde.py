from typing import Union

import paddle
from paddle import autograd, nn

from ..interpolation.interpolate import (
    BezierSpline,
    CubicHermiteSpline,
    LinearInterpolation,
)
from .base_xde import BaseXDE


class BaseDDE(BaseXDE):
    """
    Base class for all DDEs.
    """

    def __init__(
        self,
        func: Union[nn.Layer, callable],
        y0: Union[tuple, paddle.Tensor],
        t_span: Union[list, paddle.Tensor],
        lags: Union[list, paddle.Tensor],
        his: paddle.Tensor,
        his_span: paddle.Tensor,
    ):
        # TODO 此处传入的数据值需要进行改变
        # 如果lags不存在梯度值，则不需要进行初始化和更新，采取固定lags的形式
        # 如果lags存在梯度，证明lags可以进行更新
        # 如果lags为None，则选择动态初始化lags
        super(BaseDDE, self).__init__(name="DDE", var_nums=1, y0=y0, t_span=t_span)

        self.func = func
        self.lags = lags
        self.y_lags = HistoryIndex.apply(
            func=func, t0=t_span[0], y0=y0, lags=lags, his=his, his_span=his_span
        )
        self.his = his
        self.his_span = his_span
        self.init_y0(y0)

    def init_y0(self, input):
        self.y0 = input

    def handle(self, h, ts):
        pass

    def move(self, t0, dt, y0):
        # self.init_lags()
        # input_history = paddle.index_select(self.history, self.lags)

        # y_lags [B, T, D]  T是选择后的序列长度

        dy = self.call_func(t0, y0, self.lags, self.y_lags)
        return dy

    def fuse(self, dy, dt, y0):
        y = dy * dt + y0
        _lambda = 0.001
        return (dy - _lambda * y) * dt + y0

        # return dy * dt + y0

    def call_func(self, t, y0, lags, y_lags):
        # y0 = self.unflatten(y0, length=1)
        dy = self.func(t, y0, lags, y_lags)
        # dy = self.flatten(dy)
        return dy

    def init_lags(self):
        # TODO 不同时刻初始化不同lags
        pass

    def flatten(self, input):
        return input

    def unflatten(self, input, length):
        return input


class HistoryIndex(autograd.PyLayer):
    def forward(ctx, func, t0, y0, lags, his, his_span, interp_method="linear"):
        """
        计算给定输入序列的未来值，并返回计算结果。
        传入lags, history,
        计算序列位置对应位置的梯度, 并保存至backward

        Args:
            ctx (): 动态图计算上下文对象。
            xde (): 未来值的输入序列, BaseXDE类型。
            lags (paddle.Tensor): 用多少个过去的值来计算未来的这个值（未来值的滞后量）。
            history (paddle.Tensor): 用于计算未来值的过去输入序列。
            interp_method (str, optional): 插值方法，取值为 "linear"（线性插值）,"cubic"（三次样条插值）或 "bez"（贝塞尔插值）。默认为 "linear"。

        Returns:
            paddle.Tensor: 计算结果，形状为 [batch_size, len_t, dims]。

        Raises:
            NotImplementedError: 如果interp_method不是上述三种情况之一, 将抛出NotImplementedError异常。
        """
        with paddle.no_grad():
            if interp_method == "linear":
                interp = LinearInterpolation(his, his_span)
            elif interp_method == "cubic":
                interp = CubicHermiteSpline(his, his_span)
            elif interp_method == "bez":
                interp = BezierSpline(his, his_span)
            else:
                raise NotImplementedError

            y_lags = interp.evaluate(lags)
            derivative_lags = interp.derivative(lags)

            ctx.t0 = t0
            ctx.y0 = y0
            ctx.lags = lags
            ctx.func = func

            ctx.save_for_backward(y_lags, derivative_lags)

        return y_lags

    def backward(ctx, grad_y):
        # 计算history相应的梯度，并提取forward中保存的梯度，用于计算lag的梯度
        # 在计算的过程中，无需更新history，仅更新lags即可

        t0 = ctx.t0
        y0 = ctx.y0
        lags = ctx.lags
        func = ctx.func

        y_lags, derivative_lags = ctx.saved_tensor()

        _y_lags = y_lags.detach()
        _y_lags = paddle.assign(y_lags)
        _y_lags.stop_gradient = False

        _lags = lags.detach()
        _lags = paddle.assign(lags)

        with paddle.set_grad_enabled(True):
            output = func(t0, y0, _lags, _y_lags)
        paddle.autograd.backward([output], [grad_y], True)

        return None, None, _y_lags.grad * derivative_lags * 1000, None, None
        # return None, grad_y_lags * derivative_lags, None, None, None
