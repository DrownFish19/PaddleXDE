import argparse
import os
import time

import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim

from paddlexde.solver.fixed_solver import Euler

parser = argparse.ArgumentParser("SDE demo")
parser.add_argument("--method", type=str, choices=["dopri5", "adams"], default="dopri5")
parser.add_argument("--data_size", type=int, default=1000)
parser.add_argument("--batch_time", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--niters", type=int, default=2000)
parser.add_argument("--test_freq", type=int, default=20)
parser.add_argument("--viz", type=bool, default=True)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--adjoint", type=bool, default=False)
args = parser.parse_args()

if args.adjoint:
    from paddlexde.functional import sdeint_adjoint as sdeint
else:
    from paddlexde.functional import sdeint

true_y0 = paddle.to_tensor([[2.0, 0.0]])
t = paddle.linspace(0.0, 25.0, args.data_size)
true_A = paddle.to_tensor([[-0.1, 2.0], [-2.0, -0.1]])


class Lambda_f(nn.Layer):
    def forward(self, t, y):
        return paddle.mm(2 * y, true_A)


class Lambda_g(nn.Layer):
    def forward(self, t, y):
        return y


with paddle.no_grad():
    true_y = sdeint(Lambda_f(), Lambda_g(), true_y0, t, solver=Euler)
    true_y = true_y.squeeze(1)


def get_batch():
    s = paddle.to_tensor(
        np.random.choice(
            np.arange(args.data_size - args.batch_time, dtype=np.int64),
            args.batch_size,
            replace=False,
        )
    )
    batch_y0 = true_y[s]  # (B, D)
    batch_t = t[: args.batch_time]  # (T)
    batch_y = paddle.stack(
        [true_y[s + i] for i in range(args.batch_time)], axis=0
    )  # (T, M, D)
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs("png")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), facecolor="white")
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.ion()


def visualize(true_y, pred_y, odefunc, itr):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title("Trajectories")
        ax_traj.set_xlabel("t")
        ax_traj.set_ylabel("x,y")
        ax_traj.plot(
            t.cpu().numpy(),
            true_y.cpu().numpy()[:, 0],
            t.cpu().numpy(),
            true_y.cpu().numpy()[:, 1],
            "g-",
        )
        ax_traj.plot(
            t.cpu().numpy(),
            pred_y.cpu().numpy()[:, 0, 0],
            "--",
            t.cpu().numpy(),
            pred_y.cpu().numpy()[:, 0, 1],
            "b--",
        )
        ax_traj.set_xlim(t.cpu().numpy().min(), t.cpu().numpy().max())
        ax_traj.set_ylim(-2, 2)
        # ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title("Phase Portrait")
        ax_phase.set_xlabel("x")
        ax_phase.set_ylabel("y")
        ax_phase.plot(true_y.cpu().numpy()[:, 0], true_y.cpu().numpy()[:, 1], "g-")
        ax_phase.plot(
            pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], "b--"
        )
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title("Learned Vector Field")
        ax_vecfield.set_xlabel("x")
        ax_vecfield.set_ylabel("y")

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = (
            odefunc(
                t=0,
                y=paddle.to_tensor(
                    np.stack([x, y], -1).reshape(21 * 21, 2), dtype=paddle.float32
                ),
            )
            .cpu()
            .detach()
            .numpy()
        )
        mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
        dydt = dydt / mag
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        fig.show()
        fig.savefig("png/{:03d}".format(itr))
        # plt.pause(0.001)


class SDEFunc(nn.Layer):
    def __init__(self):
        super(SDEFunc, self).__init__()

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


class SDEDiffusion(nn.Layer):
    def __init__(self):
        super(SDEDiffusion, self).__init__()

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
        return self.net(y**2)


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


if __name__ == "__main__":

    ii = 0

    func = SDEFunc()
    diffusion = SDEDiffusion()

    optimizer = optim.RMSProp(parameters=func.parameters(), learning_rate=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)

    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.clear_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = sdeint(func, diffusion, batch_y0, batch_t, solver=Euler)
        loss = paddle.mean(paddle.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with paddle.no_grad():
                pred_y = sdeint(func, diffusion, true_y0, t, solver=Euler)
                loss = paddle.mean(paddle.abs(pred_y - true_y))
                print("Iter {:04d} | Total Loss {:.6f}".format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()

    plt.ioff()
    plt.show()
