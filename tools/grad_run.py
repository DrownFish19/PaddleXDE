import paddle
from paddle.autograd import PyLayer


# Inherit from PyLayer
class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x, **kwargs):
        y = 2 * x
        return y

    @staticmethod
    # forward has only one output, so there is only one gradient in the input of backward.
    def backward(ctx, dy):
        grad = dy
        # forward has only one input, so only one gradient tensor is returned.
        return grad


data = paddle.randn([2, 3], dtype="float64")
data.stop_gradient = False


z = cus_tanh.apply(data, func1=paddle.tanh)
z = z * 2
z.mean().backward()

print(data.grad)
