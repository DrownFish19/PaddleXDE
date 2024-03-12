import paddle
import paddle.nn as nn
from demo_utils import DemoUtils

from paddlexde.functional import odeint, odeint_adjoint
from paddlexde.solver.fixed_solver import RK4

paddle.seed(42)

demo_utils = DemoUtils()
if demo_utils.args.adjoint:
    xdeint = odeint_adjoint
else:
    xdeint = odeint


class ODEFunc(nn.Layer):
    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.sublayers():
            if isinstance(m, nn.Linear):
                m.weight.set_value(0.1 * paddle.randn(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))

    def forward(self, t, y):
        return self.net(y**3)


if __name__ == "__main__":
    pred_len = demo_utils.args.pred_len
    func = ODEFunc()
    optimizer = paddle.optimizer.RMSProp(
        parameters=func.parameters(), learning_rate=1e-3
    )

    stop = False
    global_step = 0
    while not stop:
        for batch_y0, batch_t, batch_y in demo_utils.dataloader:
            # batch_y0 : [B, D]
            # batch_t  : [B, T]
            # batch_y  : [B, T, D]
            t_span = paddle.linspace(0.0, 25.0, demo_utils.args.data_len)[:pred_len]
            pred_y = odeint(func, batch_y0, t_span, solver=RK4)
            loss = paddle.mean(paddle.abs(pred_y - batch_y))
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            global_step += 1
            if global_step >= demo_utils.args.max_steps:
                stop = True
                break

            if global_step % demo_utils.args.test_steps == 0:
                with paddle.no_grad():
                    y0 = demo_utils.data.true_y0.unsqueeze(0)  # [1, D]
                    t_span = demo_utils.data.t_span  # [T]
                    true_y = demo_utils.data.true_y.unsqueeze(0)  # [1, T, D]
                    pred_y = xdeint(func, y0, t_span, solver=RK4)
                    loss = paddle.mean(paddle.abs(pred_y - true_y))
                    print(
                        "Iter {:04d} | Total Loss {:.6f}".format(
                            global_step, loss.item()
                        )
                    )
                    demo_utils.visualize(pred_y, func, global_step)

    demo_utils.stop()
