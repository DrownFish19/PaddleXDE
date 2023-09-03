def get_variable_name(var):
    return list(dict(abc=var).keys())[0]


class ModelInputOutput:

    # input
    # ode variables (required)
    y0 = "y0"
    t = "t"

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
    dt = "dt"

    # dde varivales (dde required)
    d_lags = "d_lags"

    # grad variables for adjoint methods
    d_grad_y = "d_grad_y"
    d_grad_t = "d_grad_t"
    d_grad_paras = "d_grad_paras"

    # dde grad variables
    d_grad_lags = "d_grad_lags"

    @classmethod
    def create_input(cls, y0, **kwargs):
        res = dict()

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
    def get_y0(cls, dic: dict):
        return dic.get(cls.y0)

    @classmethod
    def get_lags(cls, dic: dict):
        return dic.get(cls.lags)

    @classmethod
    def get_grad_y(cls, dic: dict):
        return dic.get(cls.grad_y)

    @classmethod
    def get_grad_t(cls, dic: dict):
        return dic.get(cls.grad_t)

    @classmethod
    def get_grad_paras(cls, dic: dict):
        return dic.get(cls.grad_paras)

    @classmethod
    def get_grad_lags(cls, dic: dict):
        return dic.get(cls.grad_lags)

    # for output
    @classmethod
    def get_dy(cls, dic: dict):
        return dic.get(cls.dy)

    @classmethod
    def get_d_lags(cls, dic: dict):
        return dic.get(cls.d_lags)

    @classmethod
    def get_d_grad_y(cls, dic: dict):
        return dic.get(cls.d_grad_y)

    @classmethod
    def get_d_grad_t(cls, dic: dict):
        return dic.get(cls.d_grad_t)

    @classmethod
    def get_d_grad_paras(cls, dic: dict):
        return dic.get(cls.d_grad_paras)

    @classmethod
    def get_d_grad_lags(cls, dic: dict):
        return dic.get(cls.d_grad_lags)
