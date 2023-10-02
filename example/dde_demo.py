import paddle
import paddle.nn as nn

from example.demo_utils import DemoUtils, SimpleDemoData
from paddlexde.functional import ddeint, ddeint_adjoint
from paddlexde.solver.fixed_solver import Euler

demo_utils = DemoUtils()
if demo_utils.args.adjoint:
    xdeint = ddeint_adjoint
else:
    xdeint = ddeint


class DDEDataset(SimpleDemoData):
    def __init__(self, his_len):
        super().__init__(demo_utils.args)

        self.his_len = his_len

    def __getitem__(self, idx):
        his = self.true_y[idx : idx + self.his_len, :]
        his_span = self.t_span[idx : idx + self.his_len]

        return super().__getitem__(idx + self.his_len) + (his, his_span)

    def __len__(self):
        return super().__len__() - self.his_len


class DDEFunc(nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, t, y0, lags, y_lags):
        """_summary_

        Args:
            t (_type_): [B, 1]
            y0 (_type_): [B, D]
            lags (_type_): [B, T] T为不一定连续的时刻序列, 此处样例中为20
            y_lags (_type_): [B, T, D] T为不一定连续的时刻序列, 此处序列与lags相对应

        Returns:
            _type_: [B, D], the shape should match y0 shape
        """
        h = self.linear1(y0).unsqueeze(-2)  # [10, 1, 256]
        h_his = self.linear1(y_lags)  # [10, 20, 256]
        h_his = paddle.sum(h_his, axis=1, keepdim=True)  # [10, 1, 256]
        re = self.linear2(h + h_his).squeeze(-2)  # [10, 128]
        return re


if __name__ == "__main__":
    his_len = demo_utils.args.his_len
    dde_dataset = DDEDataset(his_len=his_len)
    demo_utils.make_dataloader(dde_dataset)

    # [B, T], T is pred_len
    lags = paddle.randint(
        low=0, high=his_len, shape=[demo_utils.args.batch_size, 20]
    ).astype("float32")
    lags.stop_gradient = False

    func = DDEFunc()
    optimizer = paddle.optimizer.RMSProp(
        parameters=func.parameters() + [lags], learning_rate=1e-3
    )

    stop = False
    global_step = 0
    while not stop:
        for (
            batch_y0,
            batch_t_span,
            batch_y,
            batch_his,
            batch_his_span,
        ) in demo_utils.dataloader:
            # batch_y0       : [B, D]
            # batch_t_span   : [B, T], T is pres_len
            # batch_y        : [B, T, D], T is pred_len
            # batch_his      : [B, T, D], T is his_len
            # batch_his_span : [B, T], T is his_len
            pred_y = xdeint(
                func,
                batch_y0,
                batch_t_span,
                lags,
                batch_his,
                batch_his_span,
                solver=Euler,
            )
            loss = paddle.mean(paddle.abs(pred_y - batch_y))
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            lags = paddle.assign(lags.detach().clip(0, his_len))
            lags.stop_gradient = False

            global_step += 1
            # retain_graph=True
            if global_step >= demo_utils.args.max_steps:
                stop = True
                break

            if global_step % demo_utils.args.test_steps == 0:
                with paddle.no_grad():
                    data = demo_utils.data
                    y0 = data.true_y[his_len].unsqueeze(0)  # [1, D]
                    t_span = data.t_span[his_len + 1 :].unsqueeze(0)  # [1, T]
                    true_y = data.true_y[his_len + 1 :].unsqueeze(0)  # [1, T, D]
                    his = data.true_y[:his_len].unsqueeze(0)  # [1, T, D]
                    his_span = data.t_span[:his_len].unsqueeze(0)  # [1, T]
                    pred_y = xdeint(
                        func, y0, t_span, lags[0:1, ...], his, his_span, solver=Euler
                    )
                    loss = paddle.mean(paddle.abs(pred_y - true_y))
                    print(
                        "Iter {:04d} | Total Loss {:.6f}".format(
                            global_step, loss.item()
                        )
                    )
                    demo_utils.visualize(
                        pred_y,
                        func,
                        global_step,
                        t_span.squeeze(0),
                        true_y.squeeze(0),
                        vector_field=False,
                    )

    demo_utils.stop()
