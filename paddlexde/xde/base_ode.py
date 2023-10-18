from typing import Union

import paddle
from paddle import nn

from .base_xde import BaseXDE


class BaseODE(BaseXDE):
    """
    Base class for all ODEs.
    """

    def __init__(
        self,
        func: Union[nn.Layer, callable],
        y0: Union[tuple, paddle.Tensor],
        t_span: Union[list, paddle.Tensor],
    ):
        """_summary_

        Args:
            func (Union[nn.Layer, callable]): _description_
            y0 (Union[tuple, paddle.Tensor]): paddle.Tensor shape is (B, T, D), T=1, tuple shape is TODO
            t_span (Union[list, paddle.Tensor]): shape is (T), T=pred_len
        """
        super(BaseODE, self).__init__(name="ODE", var_nums=1, y0=y0, t_span=t_span)
        """
        super() will create
        self.t_span,
        self.batch_size,
        self.pred_len,
        self.shapes, TODO
        self.numels, TODO
        self.y0 (after reshape) TODO
        """

        self.func = func
        self.init_y0(y0)

    def init_y0(self, y0):
        self.y0 = y0

    def handle(self, h, ts):
        pass

    def move(self, t0, dt, y0):
        dy = self.call_func(t0, y0)
        return dy

    def fuse(self, dy, dt, y0):
        # 增加负反馈，减少振动，加快收敛
        # y = dy * dt + y0
        # _lambda = 0.001
        # return (dy - _lambda * y) * dt + y0

        # 原始写法
        return dy * dt + y0

    def call_func(self, t, y0):
        dy = self.func(t, y0)
        return dy
