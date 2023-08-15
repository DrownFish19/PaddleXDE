import paddle
from _interpolate import BezierSpline, CubicHermiteSpline, LinearInterpolation

series = paddle.arange(0, 100, 1).unsqueeze(0).unsqueeze(-1)
t = paddle.arange(0, 100, 1)

chp = CubicHermiteSpline(series, t)

print(chp.evaluate(1.0))
print(chp.derivative(1.0))


chp = LinearInterpolation(series, t)
print(chp.evaluate(1.0))
print(chp.derivative(1.0))

chp = BezierSpline(series, t)
print(chp.evaluate(1.0))
print(chp.derivative(1.0))
print(chp._h.to_dense())


import numpy as np
from scipy.interpolate import CubicHermiteSpline

ch = CubicHermiteSpline(
    np.linspace(0, 100, num=100),
    np.linspace(0, 100, num=100),
    np.linspace(1, 1, num=100),
)
sample3 = ch(np.linspace(1.5, 1.6, 100))
print(sample3)
