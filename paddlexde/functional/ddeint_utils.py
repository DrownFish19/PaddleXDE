import paddle

from ..interpolation.interpolate import (
    BezierSpline,
    CubicHermiteSpline,
    LinearInterpolation,
)


class DelayIndex(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, delay_t_span, delay, delay_t=None, interp_method="cubic"):
        """计算给定输入序列的未来值，并返回计算结果。

        传入lags, history,
        计算序列位置对应位置的梯度, 并保存至backward

        Args:
            ctx (_type_): 动态图计算上下文对象。
            history_index (_type_): 用多少个过去的值来计算未来的这个值（未来值的滞后量）。
            history (_type_): 用于计算未来值的过去输入序列。
            history_timestamp (_type_): 过去输入序列的时间点，支持非均匀分布。
            interp_method (str, optional): 插值方法，取值为 "linear"（线性插值）,"cubic"（三次样条插值）或 "bez"（贝塞尔插值）。默认为 "cubic"。.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        # his [batch_size, sequence_len, hidden_dim]
        his_len = delay.shape[-2]
        if delay_t is None:
            delay_t = paddle.arange(his_len)

        with paddle.no_grad():
            if interp_method == "linear":
                interp = LinearInterpolation(delay, delay_t)
            elif interp_method == "cubic":
                interp = CubicHermiteSpline(delay, delay_t)
            elif interp_method == "bez":
                interp = BezierSpline(delay, delay_t)
            else:
                raise NotImplementedError

            indexed_history = interp.evaluate(delay_t_span)

            derivative_indexed_history = interp.derivative(delay_t_span)
            ctx.save_for_backward(derivative_indexed_history)

        return indexed_history

    @staticmethod
    def backward(ctx, grad_y):
        # 计算history相应的梯度，并提取forward中保存的梯度，用于计算lag的梯度
        # 在计算的过程中，无需更新history，仅更新lags即可
        (derivative_indexed_history,) = ctx.saved_tensor()
        grad = grad_y * derivative_indexed_history
        grad = paddle.mean(paddle.sum(grad, axis=[1, 3]), axis=[0])
        return grad, None, None
        # return None, grad_y_lags * derivative_lags, None, None, None
