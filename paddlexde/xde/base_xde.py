from abc import ABC, abstractmethod

import paddle
import paddle.nn as nn

from paddlexde.types import TupleOrTensor
from paddlexde.utils.misc import flat_to_shape


class BaseXDE(ABC, nn.Layer):
    """Base class for all ODEs.

    Inheriting from this class ensures `noise_type` and `sde_type` are valid attributes, which the solver depends on.
    """

    def __init__(self, name, var_nums, y0: TupleOrTensor, t):
        super(BaseXDE, self).__init__()
        self.name = name
        self.var_nums = var_nums  # 返回值数量

        if isinstance(y0, tuple):
            self.is_tuple = True
            self.shapes = [y0_.shape for y0_ in y0]
            self.num_elements = [paddle.numel(y0_) for y0_ in y0]
            self.y0 = paddle.concat([y0_.reshape([-1]) for y0_ in y0])

        else:
            self.is_tuple = False
            self.shapes = [y0.shape]
            self.num_elements = [paddle.numel(y0)]
            self.y0 = y0

        self.t = t
        self.length = len(t)

    @abstractmethod
    def handle(self, h, ts):
        """
        数据预处理
        :param h:
        :param ts:
        :return:
        """
        pass

    @abstractmethod
    def move(self, t0, dt, y0):
        """
        计算单次dy
        :param t0:
        :param dt:
        :param y0:
        :return:
        """
        pass

    @abstractmethod
    def fuse(self, dy, dt, y0):
        """
        根据dt融合dy至y0
        :param dy:
        :param dt:
        :param y0:
        :return:
        """
        pass

    @abstractmethod
    def get_dy(self, dy):
        """
        获取当前的dy
        :param dy:
        :return:
        """
        pass

    def format(self, sol):
        if self.is_tuple:
            return flat_to_shape(sol, (len(self.t),), self.shapes, self.num_elements)
        else:
            return sol

    def method(self):
        print(f"current method is {self.name}.")
        return self.name
