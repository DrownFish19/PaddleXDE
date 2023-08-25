import paddle

from paddlexde import BezierSpline, CubicHermiteSpline, LinearInterpolation

# series = paddle.arange(0, 100, 2).unsqueeze(0).unsqueeze(-1)
# t = paddle.arange(0, 100, 2)

series = paddle.stack(
    [paddle.cast(paddle.arange(0, 1000, 0.5), dtype="float32"), paddle.zeros([2000])],
    axis=-1,
).unsqueeze(0)
# series = paddle.sin(series)
t = paddle.arange(0, 2000, 1)

chp = LinearInterpolation(series, t)
print(chp.evaluate(23).numpy())
print(chp.derivative(23.2).numpy())

chp = CubicHermiteSpline(series, t)
print(chp.evaluate(23).numpy())
print(chp.derivative(23.2).numpy())

chp = BezierSpline(series, t)
print(chp.evaluate(23).numpy())
print(chp.derivative(23.2).numpy())

import numpy as np
from scipy.interpolate import CubicHermiteSpline

ch = CubicHermiteSpline(
    np.linspace(0, 100, num=50),
    np.linspace(0, 100, num=50),
    np.linspace(1, 1, num=50),
)
sample3 = ch(np.linspace(1.0, 1.2, 20))
print(sample3)
