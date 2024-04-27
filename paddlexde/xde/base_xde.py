from abc import ABC, abstractmethod

import paddle.nn as nn


class BaseXDE(ABC, nn.Layer):
    """
    Base class for all ODEs.
    """

    def __init__(
        self,
        name,
        var_nums,
    ):
        super(BaseXDE, self).__init__()
        self.name = name
        self.var_nums = var_nums  # 返回值数量

    def method(self):
        print(f"current method is {self.name}.")
        return self.name

    @abstractmethod
    def init_y0(self, input):
        # if isinstance(input, tuple) or isinstance(input, list):
        #     self.shapes = [_tensor.shape for _tensor in input]
        #     self.num_elements = [
        #         paddle.numel(_tensor) / self.batch_size for _tensor in input
        #     ]
        #     self.y0 = paddle.concat(
        #         [_tensor.reshape([self.batch_size, -1]) for _tensor in input], axis=-1
        #     )  # [batch_size, -1]
        # elif isinstance(input, paddle.Tensor):
        #     self.shapes = [input.shape]
        #     self.num_elements = [paddle.numel(input) / self.batch_size]
        #     self.y0 = input.reshape([self.batch_size, -1])
        # else:
        #     raise NotImplementedError
        raise NotImplementedError

    @abstractmethod
    def handle(self, h, ts):
        """
        数据预处理
        :param h:
        :param ts:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def move(self, t0, dt, y0):
        """
        计算单次dy
        :param t0:
        :param dt:
        :param y0:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def fuse(self, dy, dt, y0):
        """
        根据dt融合dy至y0
        :param dy:
        :param dt:
        :param y0:
        :return:
        """
        raise NotImplementedError

    def unflatten(self, input, length):
        raise NotImplementedError

    def flatten(self, input):
        raise NotImplementedError

    @abstractmethod
    def call_func(self, **kwargs):
        raise NotImplementedError
