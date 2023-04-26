import paddle


def _linf_norm(tensor):
    return tensor.abs().max()


def _rms_norm(tensor):
    return tensor.abs().pow(2).mean().sqrt()


def _zero_norm(tensor):
    return 0.


def _mixed_norm(tensor_tuple):
    if len(tensor_tuple) == 0:
        return 0.
    return max([_rms_norm(tensor) for tensor in tensor_tuple])


def sort_tvals(tvals, t0):
    # TODO: add warning if tvals come before t0?
    tvals = tvals[tvals >= t0]
    return paddle.sort(tvals)


def interp_fit(y0, y1, y_mid, f0, f1, dt):
    """Fit coefficients for 4th order polynomial interpolation.

    Args:
        y0: function value at the start of the interval.
        y1: function value at the end of the interval.
        y_mid: function value at the mid-point of the interval.
        f0: derivative value at the start of the interval.
        f1: derivative value at the end of the interval.
        dt: width of the interval.

    Returns:
        List of coefficients `[a, b, c, d, e]` for interpolating with the polynomial
        `p = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e` for values of `x`
        between 0 (start of interval) and 1 (end of interval).
    """
    a = 2 * dt * (f1 - f0) - 8 * (y1 + y0) + 16 * y_mid
    b = dt * (5 * f0 - 3 * f1) + 18 * y0 + 14 * y1 - 32 * y_mid
    c = dt * (f1 - 4 * f0) - 11 * y0 - 5 * y1 + 16 * y_mid
    d = dt * f0
    e = y0
    return [e, d, c, b, a]


def interp_evaluate(coefficients, t0, t1, t):
    """Evaluate polynomial interpolation at the given time point.

    Args:
        coefficients: list of Tensor coefficients as created by `interp_fit`.
        t0: scalar float64 Tensor giving the start of the interval.
        t1: scalar float64 Tensor giving the end of the interval.
        t: scalar float64 Tensor giving the desired interpolation point.

    Returns:
        Polynomial interpolation of the coefficients at time `t`.
    """

    assert (t0 <= t) & (t <= t1), 'invalid interpolation, fails `t0 <= t <= t1`: {}, {}, {}'.format(t0, t, t1)
    x = (t - t0) / (t1 - t0)
    x = x.astype(coefficients[0].dtype)

    total = coefficients[0] + x * coefficients[1]
    x_power = x
    for coefficient in coefficients[2:]:
        x_power = x_power * x
        total = total + x_power * coefficient

    return total


def compute_error_ratio(error_estimate, rtol, atol, y0, y1, norm):
    error_tol = atol + rtol * paddle.fmax(y0.abs(), y1.abs())
    return norm(error_estimate / error_tol).abs()


@paddle.no_grad()
def optimal_step_size(last_step, error_ratio, safety, ifactor, dfactor, order):
    """Calculate the optimal size for the next step."""
    if error_ratio == 0:
        return last_step * ifactor
    if error_ratio < 1:
        dfactor = paddle.to_tensor(1, dtype=last_step.dtype)
    error_ratio = error_ratio.astype(last_step.dtype)
    exponent = paddle.to_tensor(order, dtype=last_step.dtype).reciprocal()
    factor = paddle.fmin(ifactor, paddle.fmax(safety / error_ratio ** exponent, dfactor))
    return last_step * factor


class PaddleAssign(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, target, value, index):
        ctx.index = index
        target[index] = value  # sneak past the version checker
        return target

    @staticmethod
    def backward(ctx, grad_target):
        return grad_target, grad_target[ctx.index], None
