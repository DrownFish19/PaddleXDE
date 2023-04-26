from abc import abstractmethod, ABC

import paddle.nn as nn


class BaseXDE(ABC, nn.Layer):
    """Base class for all ODEs.

    Inheriting from this class ensures `noise_type` and `sde_type` are valid attributes, which the solver depends on.
    """

    def __init__(self, name="XDE", var_nums=1):
        super(BaseXDE, self).__init__()
        self.name = name
        self.var_nums = var_nums  # 返回值数量

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

    def method(self):
        print(f"current method is {self.name}.")
        return self.name
