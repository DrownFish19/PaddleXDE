from typing import Union

import paddle
from paddle import nn

from .base_xde import BaseXDE


class BaseDDE(BaseXDE):
    """
    Base class for all DDEs.
    """

    def __init__(
        self,
        drift_f: Union[nn.Layer, callable],
        delay_f: Union[nn.Layer, callable],
        y0: Union[tuple, paddle.Tensor],
        delay: Union[tuple, paddle.Tensor],
        y_t_span: Union[list, paddle.Tensor],
        **kwargs,
    ):
        """_summary_

        Args:
            func (Union[nn.Layer, callable]): the differential equation func
                                                which input include delay, y0 and
                                                kwargs
            y0 (Union[tuple, paddle.Tensor]): input data
            t_span (Union[list, paddle.Tensor]): output timestamp span
            lags (Union[list, paddle.Tensor]): input delay index,
                                                which is used to make the delay data
            his (paddle.Tensor): [B,N,T,D] input all delay
            his_span (paddle.Tensor): all delay timestamp
        """
        super(BaseDDE, self).__init__(name="DDE", var_nums=1)

        self.drift_f = drift_f
        self.delay_f = delay_f
        self.input_delay = delay
        self.input_y0 = y0
        self.y_t_span = y_t_span
        self.kwargs = kwargs

        self.delays_hidden_state = self.delay_f(self.input_delay, **kwargs)

    def move(self, t0, dt, y0):
        # self.init_lags()
        # input_history = paddle.index_select(self.history, self.lags)

        # y_lags [B, T, D]  T是选择后的序列长度
        dy = self.drift_f(self.delays_hidden_state, y0, t0, t0 + dt, **self.kwargs)
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
