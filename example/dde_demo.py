import paddle
import paddle.nn as nn

from paddlexde.functional import ddeint
from paddlexde.solver.fixed_solver import Euler

B, T, D = 10, 128, 128
series = paddle.randn([B, T, D])  # [10, 128, 128]
t = paddle.linspace(0.0, 100.0, T)  # [128]

y_input = series[:, 100:101, :]  # [10, 1, 128]
t_intput = t[100:]  # [28]

y_lags = series[:, :100, :]  # [10, 100, 128]
t_lags = t[:100]  # [100]

y_tgts = series[:, 100:, :]  # [10, 28, 128]
t_tgts = t[100:]  # [28]


class DDEFunc(nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(D, D * 2)
        self.linear2 = nn.Linear(D * 2, D)

    def forward(self, t, y0, lags, his, **kwargs):
        h = self.linear1(y0)
        h_his = self.linear1(his)
        h_his = paddle.sum(h_his, axis=1, keepdim=True)
        re = self.linear2(h + h_his)
        return re


model = DDEFunc()
optimizer = paddle.optimizer.Adam(0.01, parameters=model.parameters())

sol = ddeint(
    func=model,
    y0=y_input,
    t_span=t_intput,
    lags=y_lags,
    history=y_lags,
    solver=Euler,
)

print(sol.shape)
loss = nn.functional.l1_loss(sol, y_tgts)

loss.backward()

optimizer.step()
optimizer.clear_grad()
