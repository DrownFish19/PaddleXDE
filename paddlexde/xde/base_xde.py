from abc import ABC, abstractmethod
from typing import Union

import paddle
import paddle.nn as nn


class BaseXDE(ABC, nn.Layer):
    """
    Base class for all ODEs.
    """

    def __init__(
        self,
        name,
        var_nums,
        y0: Union[tuple, paddle.Tensor],
        t_span: Union[list, paddle.Tensor],
    ):
        super(BaseXDE, self).__init__()
        self.name = name
        self.var_nums = var_nums  # 返回值数量

        self.t_span = t_span
        self.batch_size, self.pred_len = t_span.shape

        self.init_y0(y0)  # shapes, numels, y0

    def init_y0(self, input):
        if isinstance(input, tuple) or isinstance(input, list):
            self.shapes = [_tensor.shape for _tensor in input]
            self.num_elements = [
                paddle.numel(_tensor) / self.batch_size for _tensor in input
            ]
            self.y0 = paddle.concat(
                [_tensor.reshape([self.batch_size, -1]) for _tensor in input], axis=-1
            )  # [batch_size, -1]
        elif isinstance(input, paddle.Tensor):
            self.shapes = [input.shape]
            self.num_elements = [paddle.numel(input) / self.batch_size]
            self.y0 = input.reshape([self.batch_size, -1])
        else:
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

    def format(self, sol):
        sol = self.unflatten(sol, self.pred_len)
        sol = paddle.transpose(sol, perm=[1, 0, 2, 3]).squeeze(-2)
        return sol

    def method(self):
        print(f"current method is {self.name}.")
        return self.name

    def unflatten(self, input, length):
        batch_size = input.shape[0]  # 默认第一维是batch_size

        length_shape = [] if length == 1 else [length]
        if len(self.shapes) == 1:
            return input.reshape(length_shape + self.shapes[0])

        output = []
        total = 0

        # TODO:此处需要验证
        for shape, num_ele in zip(self.shapes, self.num_elements):
            next_total = total + num_ele
            output.append(
                input[..., total:next_total].reshape(
                    [batch_size] + length_shape + shape
                )
            )
            total = next_total
        return tuple(output)

    def flatten(self, input):
        if isinstance(input, tuple) or isinstance(input, list):
            output = paddle.concat(
                [_tensor.reshape([self.batch_size, -1]) for _tensor in input]
            )
        elif isinstance(input, paddle.Tensor):
            output = input.reshape([self.batch_size, -1])
        else:
            raise NotImplementedError

        return output

    @abstractmethod
    def call_func(self, **kwargs):
        raise NotImplementedError
