from abc import ABC, abstractmethod

import paddle.nn as nn

from paddlexde.utils.misc import flat_to_shape


class BaseXDE(ABC, nn.Layer):
    """Base class for all ODEs.

    Inheriting from this class ensures `noise_type` and `sde_type` are valid attributes, which the solver depends on.
    """

    def __init__(
        self,
        name,
    ):
        super(BaseXDE, self).__init__()
        self.name = name

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

    def format(self, sol):
        # todo @DrownFish19
        if self.is_tuple:
            return flat_to_shape(sol, (len(self.t),), self.shapes, self.num_elements)
        else:
            return sol

    def method(self):
        print(f"current method is {self.name}.")
        return self.name
