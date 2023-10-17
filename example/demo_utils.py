import argparse
import time
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader, Dataset

from paddlexde.functional import odeint
from paddlexde.solver import RK4


class DemoUtils:
    def __init__(self) -> None:
        self.start_time = time.time()
        self.make_init()

    def make_init(self):
        paddle.seed(42)
        self.make_args()  # cmd input
        self.make_dataloader()  # dataloader
        self.make_image()

    def stop(self):
        plt.ioff()
        plt.show()

        print("training time:", time() - self.start_time, "s.")

    def make_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--solver", type=str, default="euler")
        parser.add_argument("--data_len", type=int, default=1000)
        parser.add_argument("--pred_len", type=int, default=32)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--max_steps", type=int, default=2000)
        parser.add_argument("--test_steps", type=int, default=20)
        parser.add_argument("--viz", type=bool, default=True)
        parser.add_argument("--adjoint", type=bool, default=False)

        # DDE
        parser.add_argument("--his_len", type=int, default=8)
        self.args = parser.parse_args()

    def make_dataloader(self, dataset=None):
        if dataset is None:
            self.data = SimpleDemoData(self.args)
        else:
            self.data = dataset
        self.dataloader = DataLoader(
            self.data,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

    def make_image(self):
        if self.args.viz:
            makedirs("png", exist_ok=True)
            self.fig = plt.figure(figsize=(12, 4), facecolor="white")
            self.ax_traj = self.fig.add_subplot(131, frameon=False)
            self.ax_phase = self.fig.add_subplot(132, frameon=False)
            self.ax_vecfield = self.fig.add_subplot(133, frameon=False)
            plt.ion()

    def visualize(
        self,
        pred_y,
        odefunc,
        steps,
        t_span=None,
        true_y=None,
        vector_field=True,
    ):
        if not self.args.viz:
            return

        t_span = (
            self.data.t_span.cpu().numpy() if t_span is None else t_span.cpu().numpy()
        )
        true_y = (
            self.data.true_y.cpu().numpy() if true_y is None else true_y.cpu().numpy()
        )
        pred_y = pred_y.cpu().numpy()[0]

        self.ax_traj.cla()
        self.ax_traj.set_title("Trajectories")
        self.ax_traj.set_xlabel("t")
        self.ax_traj.set_ylabel("x,y")
        self.ax_traj.plot(t_span, true_y[:, 0], t_span, true_y[:, 1], "g-")
        self.ax_traj.plot(t_span, pred_y[:, 0], "--", t_span, pred_y[:, 1], "b--")
        self.ax_traj.set_xlim(t_span.min(), t_span.max())
        self.ax_traj.set_ylim(-2, 2)
        # self.ax_traj.legend()

        self.ax_phase.cla()
        self.ax_phase.set_title("Phase Portrait")
        self.ax_phase.set_xlabel("x")
        self.ax_phase.set_ylabel("y")
        self.ax_phase.plot(true_y[:, 0], true_y[:, 1], "g-")
        self.ax_phase.plot(pred_y[:, 0], pred_y[:, 1], "b--")
        self.ax_phase.set_xlim(-2, 2)
        self.ax_phase.set_ylim(-2, 2)

        self.ax_vecfield.cla()
        self.ax_vecfield.set_title("Learned Vector Field")
        self.ax_vecfield.set_xlabel("x")
        self.ax_vecfield.set_ylabel("y")

        if vector_field:
            y, x = np.mgrid[-2:2:21j, -2:2:21j]
            input_y = paddle.to_tensor(
                np.stack([x, y], -1).reshape(21 * 21, 2), dtype=paddle.float32
            )

            dydt = odefunc(t=0, y=input_y).cpu().detach().numpy()
            mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
            dydt = dydt / mag
            dydt = dydt.reshape(21, 21, 2)

            self.ax_vecfield.streamplot(
                x, y, dydt[:, :, 0], dydt[:, :, 1], color="black"
            )
            self.ax_vecfield.set_xlim(-2, 2)
            self.ax_vecfield.set_ylim(-2, 2)

        self.fig.tight_layout()
        self.fig.show()
        self.fig.savefig("png/{:03d}".format(steps))
        plt.pause(0.001)


class Lambda(nn.Layer):
    def __init__(self, trans_matrix):
        super().__init__()
        self.trans_matrix = trans_matrix

    def forward(self, t, y):
        # y**3 => [B, D]
        # paddle.mm => [B, D] x [D, D] => [B, D]
        return paddle.mm(y**3, self.trans_matrix)


class SimpleDemoData(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.true_y0 = paddle.to_tensor([2.0, 0.0])  # [D]
        self.t_span = paddle.linspace(0.0, 25.0, config.data_len)  # [T], T is data_len
        self.trans_matrix = paddle.to_tensor([[-0.1, 2.0], [-2.0, -0.1]])  # [2, 2]

        with paddle.no_grad():
            # construst the train data
            # [T, D], T is data_len
            self.true_y = odeint(
                Lambda(self.trans_matrix),
                self.true_y0.unsqueeze(0),
                self.t_span,
                solver=RK4,
            ).squeeze(0)

    def __getitem__(self, idx):
        y0, t_span, tgt_y = self.true_y[idx, :], [], []
        for i in range(self.config.pred_len):
            t_span.append(self.t_span[idx + i])
            tgt_y.append(self.true_y[idx + i, :])
        try:
            t_span = paddle.concat(t_span)
        except:
            t_span = paddle.stack(t_span)
        tgt_y = paddle.stack(tgt_y)

        # [D], [T], [T, D], T is pred_len
        return y0, t_span, tgt_y

    def __len__(self):
        return self.config.data_len - self.config.pred_len


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
