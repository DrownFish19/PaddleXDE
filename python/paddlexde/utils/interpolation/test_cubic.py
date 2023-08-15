import paddle
from _interpolate import CubicHermiteSpline

series = paddle.arange(0, 100, 1).unsqueeze(0).unsqueeze(-1)
t = paddle.arange(0, 100, 1)

chp = CubicHermiteSpline(series, t)

print(chp._h.to_dense())
