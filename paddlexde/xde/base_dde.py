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
    """Base class for all ODEs."""

    def __init__(
        self,
        func: Union[nn.Layer, callable],
        y0: Union[tuple, paddle.Tensor],
        t_span: Union[list, paddle.Tensor],
        lags: Union[list, paddle.Tensor],
        history: paddle.Tensor,
    ):
        super(BaseDDE, self).__init__(name="ODE", var_nums=1, y0=y0, t_span=t_span)

        self.func = func
        self.lags = lags
        self.history = history

    def handle(self, h, ts):
        pass

    def move(self, t0, dt, y0):
        # self.init_lags()
        # input_history = paddle.index_select(self.history, self.lags)
        dy = self.call_func(t0, y0)
        return dy

    def fuse(self, dy, dt, y0):
        # 测试是够还存在振动
        y = dy * dt + y0
        _lambda = 0.001
        return (dy - _lambda * y) * dt + y0

        # return dy * dt + y0

    def call_func(self, t, y0, lags, y_lags):
        y0 = self.unflatten(y0, length=1)
        dy = self.func(t, y0, lags, y_lags)
        dy = self.flatten(dy)
        return dy

    def init_lags(self):
        # TODO 不同时刻初始化不同lags
        pass


class HistoryIndex(autograd.PyLayer):
    def forward(
        ctx, xde: BaseXDE, t, y0, lags, history, h_span, interp_method="linear"
    ):
        """
        计算给定输入序列的未来值，并返回计算结果。
        传入lags, history,
        计算序列位置对应位置的梯度, 并保存至backward

        Args:
            ctx (): 动态图计算上下文对象。
            xde (): 未来值的输入序列, BaseXDE类型。
            lags (paddle.Tensor): 用多少个过去的值来计算未来的这个值（未来值的滞后量）。
            history (paddle.Tensor): 用于计算未来值的过去输入序列。
            h_span (int): 未来多少步的预测。
            interp_method (str, optional): 插值方法，取值为 "linear"（线性插值）,"cubic"（三次样条插值）或 "bez"（贝塞尔插值）。默认为 "linear"。

        Returns:
            paddle.Tensor: 计算结果，形状为 [batch_size, len_t, dims]。

        Raises:
            NotImplementedError: 如果interp_method不是上述三种情况之一, 将抛出NotImplementedError异常。
        """
        if interp_method == "linear":
            interp = LinearInterpolation(history, h_span)
        elif interp_method == "cubic":
            interp = CubicHermiteSpline(history, h_span)
        elif interp_method == "bez":
            interp = BezierSpline(history, h_span)
        else:
            raise NotImplementedError

        batch_size, len_t, dims = history.shape  # [B, T, D]
        axis_b = paddle.arange(batch_size)[:, None, None]
        axis_index = lags[:, :, None]
        axis_d = paddle.arange(dims)[None, None, :]
        his = history[axis_b, axis_index, axis_d]

        ctx.t = t
        ctx.y0 = y0
        ctx.lags = lags
        ctx.derivative_lags = interp.derivative(lags)
        ctx.xde = xde
        ctx.his = his

        return his

    def backward(ctx, grad_y):
        # 计算history相应的梯度，并提取forward中保存的梯度，用于计算lag的梯度
        # 在计算的过程中，无需更新history，仅更新lags即可

        t = ctx.t
        y0 = ctx.y0
        lags = ctx.lags
        derivative_lags = ctx.derivative_lags
        xde = ctx.xde
        his = ctx.his

        eval = xde.call_func(t, y0, lags, his)

        grad_his = paddle.grad(
            outputs=[eval],
            inputs=[his],
            grad_outputs=-grad_y,
            allow_unused=True,
            retain_graph=True,
        )
        return grad_his * derivative_lags
