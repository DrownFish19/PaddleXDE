from typing import Union

import paddle
from paddle import nn

from ..utils.input import ModelInputOutput as mio
from .base_xde import BaseXDE


class BaseODE(BaseXDE):
    """Base class for all ODEs."""

    def __init__(
        self,
        func: Union[nn.Layer, callable],
    ):
        super(BaseODE, self).__init__(name="ODE")
        self.func = func

    def handle(self, h, ts):
        pass

    def move(
        self,
        t0: Union[float, paddle.Tensor],
        dt: Union[float, paddle.Tensor],
        y0: dict,
    ) -> dict:
        """_summary_

        Args:
            t0 (Union[float, paddle.Tensor]): _description_
            dt (Union[float, paddle.Tensor]): _description_
            y0 (dict): _description_

        Returns:
            dict: _description_
        """

        return self.func(t0, y0)

    def fuse(
        self,
        dy: dict,
        dt: Union[float, paddle.Tensor],
        y0: dict,
    ) -> dict:
        """_summary_

        Args:
            dy (dict): _description_
            dt (Union[float, paddle.Tensor]): _description_
            y0 (dict): _description_

        Returns:
            dict: _description_
        """
        # # 测试是否存在振动
        # y = mio.get_dy(dy) * dt + mio.get_y0(y0)
        # _lambda = 0.001
        # return (mio.get_dy(dy) - _lambda * y) * dt + mio.get_y0(y0)

        _y0 = mio.get_y0(y0)
        _dy = mio.get_dy(dy)

        y1 = _dy * dt + _y0

        # _d_grad_t = mio.get_d_grad_t(dy)
        # _d_grad_paras = mio.get_d_grad_paras(dy)

        mio.create_input(y0=y1)

        return
