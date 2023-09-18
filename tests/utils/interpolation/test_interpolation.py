import unittest

import paddle

from paddlexde.interpolation import (
    BezierSpline,
    CubicHermiteSpline,
    LinearInterpolation,
)


# 固定梯度
class TestInterpolationForFixedDeriv(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.series = paddle.stack(
            [
                paddle.cast(paddle.arange(0, 1000, 0.5), dtype="float32"),
                paddle.zeros([2000]),
            ],
            axis=-1,
        ).unsqueeze(
            0
        )  # [B, T, D]
        self.t = paddle.arange(0, 2000, 1).unsqueeze(0)  # [B, T]

        self.t_eval = paddle.to_tensor([1, 21.12])  # [B, T]
        self.val_tgt = (
            paddle.to_tensor([self.t_eval * 0.5, 0]).unsqueeze(0).unsqueeze(0)
        )  # [B, T, D]
        # [B, T, D]
        self.tgt_deri = paddle.to_tensor([0.5, 0]).unsqueeze(0).unsqueeze(0)

    def test_LinearInterpolation(self):
        interp = LinearInterpolation(self.series, self.t)
        assert paddle.allclose(self.val_tgt, interp.evaluate(self.t_eval), rtol=1e-4)
        assert paddle.allclose(self.tgt_deri, interp.derivative(self.t_eval), rtol=1e-4)

    def test_CubicHermiteSpline(self):
        interp = CubicHermiteSpline(self.series, self.t)
        assert paddle.allclose(self.val_tgt, interp.evaluate(self.t_eval), rtol=1e-4)
        assert paddle.allclose(self.tgt_deri, interp.derivative(self.t_eval), rtol=1e-4)

    def test_BezierSpline(self):
        interp = BezierSpline(self.series, self.t)
        assert paddle.allclose(self.val_tgt, interp.evaluate(self.t_eval), rtol=1e-4)
        assert paddle.allclose(self.tgt_deri, interp.derivative(self.t_eval), rtol=1e-4)


class TestInterpolationForDynamicDeriv(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.series = paddle.stack(
            [
                paddle.cast(paddle.arange(0, 20, 0.01), dtype="float32"),
                paddle.zeros([2000]),
            ],
            axis=-1,
        ).unsqueeze(
            0
        )  # [B, T, D]
        self.series = paddle.sin(self.series)
        self.t = paddle.arange(0, 20, 0.01).unsqueeze(0)  # [B, T]

        self.t_eval = paddle.to_tensor([1, 16.5])  # [B, T]
        self.val_tgt = paddle.sin(
            paddle.to_tensor([16.5, 0]).unsqueeze(0).unsqueeze(0)
        )  # [B, T, D]
        self.tgt_deri = paddle.cos(
            paddle.to_tensor([16.5, 0]).unsqueeze(0).unsqueeze(0)
        )  # [B, T, D]
        self.tgt_deri[:, :, 1] = 0

    def test_LinearInterpolation(self):
        interp = LinearInterpolation(self.series, self.t)
        assert paddle.allclose(self.val_tgt, interp.evaluate(self.t_eval), rtol=5e-2)
        assert paddle.allclose(self.tgt_deri, interp.derivative(self.t_eval), rtol=1e-2)

    def test_CubicHermiteSpline(self):
        interp = CubicHermiteSpline(self.series, self.t)
        assert paddle.allclose(self.val_tgt, interp.evaluate(self.t_eval), rtol=5e-2)
        assert paddle.allclose(self.tgt_deri, interp.derivative(self.t_eval), rtol=1e-2)

    def test_BezierSpline(self):
        interp = BezierSpline(self.series, self.t)
        assert paddle.allclose(self.val_tgt, interp.evaluate(self.t_eval), rtol=5e-2)
        assert paddle.allclose(self.tgt_deri, interp.derivative(self.t_eval), rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
