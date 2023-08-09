# Gumbel softmax trick:

import paddle
import paddle.nn.functional as F


def inverse_gumbel_cdf(y, mu, beta):
    return mu - beta * paddle.log(-paddle.log(y))


def gumbel_softmax_sampling(h, mu=0, beta=1, tau=0.1):
    """
    h : (N x K) tensor. Assume we need to sample a NxK tensor, each row is an independent r.v.
    """
    shape_h = h.shape
    p = F.softmax(h, axis=1)
    y = paddle.rand(shape_h) + 1e-25  # ensure all y is positive.
    g = inverse_gumbel_cdf(y, mu, beta)
    x = paddle.log(p) + g  # samples follow Gumbel distribution.
    # using softmax to generate one_hot vector:
    x = x / tau
    x = F.softmax(x, axis=1)  # now, the x approximates a one_hot vector.
    return x


N = 10  # 假设 有N个独立的离散变量需要采样
K = 3  # 假设 每个离散变量有3个取值
h = paddle.randn((N, K))  # 假设 h是由一个神经网络输出的tensor。

mu = 0
beta = 1
tau = 0.1

samples = gumbel_softmax_sampling(h, mu, beta, tau)

print(samples)
