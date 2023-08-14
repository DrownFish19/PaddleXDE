import math

import matplotlib.pyplot as plt
import paddle


def get_data(num_timepoints=100):
    ######################
    # Now we need some data.
    # Here we have a simple example which generates some spirals, some going clockwise, some going anticlockwise.
    ######################
    t = paddle.linspace(0.0, 4 * math.pi, num_timepoints)

    start = paddle.rand([128]) * 2 * math.pi
    x_pos = paddle.cos(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos[:64] *= -1
    y_pos = paddle.sin(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos += 0.01 * paddle.randn(x_pos.shape)  # [128,100]
    y_pos += 0.01 * paddle.randn(y_pos.shape)  # [128,100]
    ######################
    # Easy to forget gotcha: time should be included as a channel; Neural CDEs need to be explicitly told the
    # rate at which time passes. Here, we have a regularly sampled dataset, so appending time is pretty simple.
    ######################

    X = paddle.stack(
        [paddle.concat([t.unsqueeze(0) for i in range(128)], axis=0), x_pos, y_pos],
        axis=-1,
    )  # [128,100,3]
    y = paddle.zeros(128)
    y[:64] = 1

    perm = paddle.randperm(128)
    X = X[perm]
    y = y[perm]

    ######################
    # X is a tensor of observations, of shape (batch=128, sequence=100, channels=3)
    # y is a tensor of labels, of shape (batch=128,), either 0 or 1 corresponding to anticlockwise or clockwise
    # respectively.
    ######################
    return X, y


pX, pY = get_data()
# 将 paddle tensor 转换为 numpy array
X_np = pX.numpy()
Y_np = pY.numpy()

# 绘制 X 数据
plt.figure(figsize=(10, 10))
plt.scatter(
    X_np[0, :, 1], X_np[0, :, 2], c=X_np[0, :, 0], cmap="coolwarm", edgecolors="black"
)
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Label")
plt.show()
