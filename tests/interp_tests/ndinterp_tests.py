import paddle
import pytest
from data_generate import get_data
from paddlexde.utils.interpolation import CubicSpline, LinearInterpolation

paddle.device.set_device("cpu")


@pytest.mark.interp_linear_tests
def test_linear_interp():
    pX, pY = get_data()
    X = LinearInterpolation(pX)
    X0 = X.evaluate(X.interval[0])
    print(X0)
    # assert paddle.allclose(sol, y, rtol=4e-3)


@pytest.mark.interp_cubic_tests
def test_cubic_interp():
    pX, pY = get_data()
    X = CubicSpline(pX)
    X0 = X.evaluate(X.interval[0])
    print(X0)
    # assert paddle.allclose(sol, y, rtol=4e-3)


test_linear_interp()
# test_cubic_interp()
