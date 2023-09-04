from typing import Union

import paddle


def get_variable_name(var):
    return list(dict(abc=var).keys())[0]


class ModelInputOutput:

    _type = "type"
    _input = "type_input"
    _output = "type_output"

    # input
    # ode variables (required)
    y0 = "y0"

    # dde varivales (dde required)
    lags = "lags"

    # grad variables for adjoint methods
    grad_y = "grad_y"
    grad_t = "grad_t"
    grad_paras = "grad_paras"

    # dde grad variables
    grad_lags = "grad_lags"

    # output
    # ode variables (required)
    dy = "dy"

    # dde varivales (dde required)
    d_lags = "d_lags"

    # grad variables for adjoint methods
    d_grad_y = "d_grad_y"
    d_grad_t = "d_grad_t"
    d_grad_paras = "d_grad_paras"

    # dde grad variables
    d_grad_lags = "d_grad_lags"

    _inputs = [y0, lags, grad_y, grad_t, grad_paras, grad_lags]
    _outputs = [dy, d_lags, d_grad_y, d_grad_t, d_grad_paras, d_grad_lags]

    @classmethod
    def create_input(cls, y0, **kwargs):
        res = dict()

        res[cls._type] = cls._input
        # add ode variables
        res[cls.y0] = y0

        # dde varivales (dde required)
        res[cls.lags] = kwargs.get(cls.lags)

        # add grad variables
        res[cls.grad_y] = kwargs.get(cls.grad_y)
        res[cls.grad_t] = kwargs.get(cls.grad_t)
        res[cls.grad_paras] = kwargs.get(cls.grad_paras)

        # dde grad variables
        res[cls.grad_lags] = kwargs.get(cls.grad_lags)

        return res

    @classmethod
    def create_output(cls, dy, **kwargs):
        res = dict()

        res[cls._type] = cls._output
        # ode variables (required)
        res[cls.dy] = dy

        # dde varivales (dde required)
        res[cls.d_lags] = kwargs.get(cls.d_lags)

        # grad variables for adjoint methods
        res[cls.d_grad_y] = kwargs.get(cls.d_grad_y)
        res[cls.d_grad_t] = kwargs.get(cls.d_grad_t)
        res[cls.d_grad_paras] = kwargs.get(cls.d_grad_paras)

        # dde grad variables
        res[cls.d_grad_lags] = kwargs.get(cls.d_grad_lags)

        return res

    # for input
    @classmethod
    def get_y0(cls, x: dict):
        return x.get(cls.y0)

    @classmethod
    def get_lags(cls, x: dict):
        return x.get(cls.lags)

    @classmethod
    def get_grad_y(cls, x: dict):
        return x.get(cls.grad_y)

    @classmethod
    def get_grad_t(cls, x: dict):
        return x.get(cls.grad_t)

    @classmethod
    def get_grad_paras(cls, x: dict):
        return x.get(cls.grad_paras)

    @classmethod
    def get_grad_lags(cls, x: dict):
        return x.get(cls.grad_lags)

    # for output
    @classmethod
    def get_dy(cls, x: dict):
        return x.get(cls.dy)

    @classmethod
    def get_d_lags(cls, x: dict):
        return x.get(cls.d_lags)

    @classmethod
    def get_d_grad_y(cls, x: dict):
        return x.get(cls.d_grad_y)

    @classmethod
    def get_d_grad_t(cls, x: dict):
        return x.get(cls.d_grad_t)

    @classmethod
    def get_d_grad_paras(cls, x: dict):
        return x.get(cls.d_grad_paras)

    @classmethod
    def get_d_grad_lags(cls, x: dict):
        return x.get(cls.d_grad_lags)

    @classmethod
    def add(cls, x: dict, y: Union[dict, paddle.Tensor, float]):
        res = dict({cls._type: x[cls._type]})
        if isinstance(y, dict):
            x_list = cls._inputs if x[cls._type] == cls._input else cls._outputs
            y_list = cls._inputs if y[cls._type] == cls._input else cls._outputs

            for k_x, k_y in zip(x_list, y_list):
                res[k_x] = (
                    x[k_x] + y[k_y]
                    if x[k_x] is not None and y[k_y] is not None
                    else None
                )
            return res

        if isinstance(y, float) or isinstance(y, paddle.Tensor):
            member_list = cls._inputs if x[cls._type] == cls._input else cls._outputs
            for k_x in member_list:
                res[k_x] = x[k_x] + y if x[k_x] is not None else None
            return res

    @classmethod
    def sub(cls, x: dict, y: Union[dict, paddle.Tensor, float]):
        res = dict({cls._type: x[cls._type]})
        if isinstance(y, dict):
            x_list = cls._inputs if x[cls._type] == cls._input else cls._outputs
            y_list = cls._inputs if y[cls._type] == cls._input else cls._outputs

            for k_x, k_y in zip(x_list, y_list):
                res[k_x] = (
                    x[k_x] - y[k_y]
                    if x[k_x] is not None and y[k_y] is not None
                    else None
                )
            return res

        if isinstance(y, float) or isinstance(y, paddle.Tensor):
            member_list = cls._inputs if x[cls._type] == cls._input else cls._outputs
            for k_x in member_list:
                res[k_x] = x[k_x] - y if x[k_x] is not None else None
            return res

    @classmethod
    def mul(cls, x: dict, y: Union[dict, paddle.Tensor, float]):
        res = dict({cls._type: x[cls._type]})
        if isinstance(y, dict):
            x_list = cls._inputs if x[cls._type] == cls._input else cls._outputs
            y_list = cls._inputs if y[cls._type] == cls._input else cls._outputs

            for k_x, k_y in zip(x_list, y_list):
                res[k_x] = (
                    x[k_x] * y[k_y]
                    if x[k_x] is not None and y[k_y] is not None
                    else None
                )
            return res

        if isinstance(y, float) or isinstance(y, paddle.Tensor):
            member_list = cls._inputs if x[cls._type] == cls._input else cls._outputs
            for k_x in member_list:
                res[k_x] = x[k_x] * y if x[k_x] is not None else None
            return res

    @classmethod
    def div(cls, x: dict, y: Union[dict, paddle.Tensor, float]):
        res = dict({cls._type: x[cls._type]})
        if isinstance(y, dict):
            x_list = cls._inputs if x[cls._type] == cls._input else cls._outputs
            y_list = cls._inputs if y[cls._type] == cls._input else cls._outputs

            for k_x, k_y in zip(x_list, y_list):
                res[k_x] = (
                    x[k_x] / y[k_y]
                    if x[k_x] is not None and y[k_y] is not None
                    else None
                )
            return res

        if isinstance(y, float) or isinstance(y, paddle.Tensor):
            member_list = cls._inputs if x[cls._type] == cls._input else cls._outputs
            for k_x in member_list:
                res[k_x] = x[k_x] / y if x[k_x] is not None else None
            return res

    @classmethod
    def stack_op(cls, x: list):
        res = dict()
        for k in cls._inputs:
            res[k] = []

        for item in x:
            for k in cls._inputs:
                if item[k] is not None:
                    res[k].append(item[k])

        for k in cls._inputs:
            res[k] = paddle.stack(res[k]) if len(res[k]) > 0 else None

        return cls.create_input(**res)
