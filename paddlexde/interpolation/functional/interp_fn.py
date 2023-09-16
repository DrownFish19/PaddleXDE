import paddle


def linear_interp(t0, t1, y0, y1, t):
    if paddle.equal_all(t, t0):
        return y0
    if paddle.equal_all(t, t1):
        return y1
    slope = (t - t0) / (t1 - t0)
    return y0 + slope * (y1 - y0)


def cubic_hermite_interp(t0, y0, dy0, t1, y1, dy1, t):
    h = (t - t0) / (t1 - t0)
    h00 = (1 + 2 * h) * (1 - h) * (1 - h)
    h10 = h * (1 - h) * (1 - h)
    h01 = h * h * (3 - 2 * h)
    h11 = h * h * (h - 1)
    dt = t1 - t0
    return h00 * y0 + h10 * dt * dy0 + h01 * y1 + h11 * dt * dy1
