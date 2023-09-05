from abc import ABC, abstractmethod
from typing import Union

import paddle
import paddle.nn as nn

from paddlexde.utils.misc import flat_to_shape


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

        self.t = t_span
        self.length = len(t_span)

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
    
    def unflatten(self, tensor, length, shapes, numels):
        tensor_list = []
        total = 0

        for shape, num_ele in zip(shapes, numels):
            next_total = total + num_ele
            # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
            if len(shape) == 0:
                tensor_list.append(tensor[..., total:next_total].reshape((*length, 0)))
            else:
                tensor_list.append(tensor[..., total:next_total].reshape((*length, *shape)))
            total = next_total
        return tuple(tensor_list)
    
    
    def flatten(self):
        self.shapes = [y0_.shape for y0_ in y0]
        self.num_elements = [paddle.numel(y0_) for y0_ in y0]
        self.y0 = paddle.concat([y0_.reshape([-1]) for y0_ in y0])
        
    def call_dunc(self, **kwargs):
        pass

