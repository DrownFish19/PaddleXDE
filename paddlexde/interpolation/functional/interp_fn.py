def linear_interp(t0, t1, y0, y1, t):
    if t == t0:
        return y0
    if t == t1:
        return y1
    slope = (t - t0) / (t1 - t0)
    return y0 + slope * (y1 - y0)


def cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t):
    h = (t - t0) / (t1 - t0)
    h00 = (1 + 2 * h) * (1 - h) * (1 - h)
    h10 = h * (1 - h) * (1 - h)
    h01 = h * h * (3 - 2 * h)
    h11 = h * h * (h - 1)
    dt = t1 - t0
    return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1
