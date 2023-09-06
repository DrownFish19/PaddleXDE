from abc import ABC, abstractmethod
from typing import Union

import paddle
import paddle.nn as nn


class BaseXDE(ABC, nn.Layer):
    """Base class for all ODEs.

    Inheriting from this class ensures `noise_type` and `sde_type` are valid attributes, which the solver depends on.
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
        # y0输入后全部进行flatten操作，进行展平
        # 后续计算过程中，如果需要调用func操作，可以通过xde进行输入，在xde中进行unflatten操作，然后将数据传给func
        # 当全部数据计算结束后，进行unflatten操作，需要注意sol已经增加多个，需要根据数据形状进行展开
        self.init_y0(y0)  # shapes, numels, y0
        self.t_span = t_span
        self.length = len(t_span)

    def init_y0(self, tensors):
        if isinstance(tensors, tuple) or isinstance(tensors, list):
            self.shapes = [_tensor.shape for _tensor in tensors]
            self.num_elements = [paddle.numel(_tensor) for _tensor in tensors]
            self.y0 = paddle.concat([_tensor.reshape([-1]) for _tensor in tensors])
        elif isinstance(tensors, paddle.Tensor):
            self.shapes = [tensors.shape]
            self.num_elements = [paddle.numel(tensors)]
            self.y0 = tensors.reshape([-1])
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
        return self.unflatten(sol, self.length)

    def method(self):
        print(f"current method is {self.name}.")
        return self.name

    def unflatten(self, tensor, length):
        if length == 1:
            return tensor.reshape(self.shapes[0])

        tensors_r = []
        total = 0

        for shape, num_ele in zip(self.shapes, self.num_elements):
            next_total = total + num_ele
            # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
            if len(shape) == 0:
                tensors_r.append(tensor[..., total:next_total].reshape((*(length,), 0)))
            else:
                tensors_r.append(
                    tensor[..., total:next_total].reshape((*(length,), *shape))
                )
            total = next_total
        return tuple(tensors_r)

    def flatten(self, tensors):
        if isinstance(tensors, tuple) or isinstance(tensors, list):
            tensors_r = paddle.concat([_tensor.reshape([-1]) for _tensor in tensors])
        elif isinstance(tensors, paddle.Tensor):
            tensors_r = tensors.reshape([-1])
        else:
            raise NotImplementedError

        return tensors_r

    @abstractmethod
    def call_func(self, **kwargs):
        raise NotImplementedError
