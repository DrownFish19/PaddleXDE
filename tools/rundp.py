# -*- coding: UTF-8 -*-
import numpy as np
import paddle
from paddle import nn

# 导入必要分布式训练的依赖包
from paddle.distributed import fleet, get_rank
from paddle.io import DataLoader, Dataset, DistributedBatchSampler

# 导入模型文件

base_lr = 0.1  # 学习率
momentum_rate = 0.9  # 冲量
l2_decay = 1e-4  # 权重衰减

epoch = 10  # 训练迭代次数
batch_num = 100  # 每次迭代的 batch 数
batch_size = 32  # 训练批次大小
class_dim = 102


# 设置数据读取器
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([3, 224, 224]).astype("float32")
        label = np.random.randint(0, class_dim - 1, (1,)).astype("int64")
        return image, label

    def __len__(self):
        return self.num_samples


# 设置优化器
def optimizer_setting(parameter_list=None):
    optimizer = paddle.optimizer.Momentum(
        learning_rate=base_lr,
        momentum=momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(l2_decay),
        parameters=parameter_list,
    )
    return optimizer


class Net(nn.Layer):
    def __init__(self):
        super().__init__()

        self.h = nn.Conv2D(3, 3, 1)

    def forward(self, *inputs, **kwargs):
        data01 = inputs[0]

        delay_data = kwargs["delay"]

        return self.h(data01) + self.h(delay_data)


# 设置训练函数
def train_model():
    # 初始化 Fleet 环境
    fleet.init(is_collective=True)

    # model = ResNet(BottleneckBlock, 50, num_classes=class_dim)

    model = Net()

    optimizer = optimizer_setting(parameter_list=model.parameters())
    # 通过 Fleet API 获取分布式 model，用于支持分布式训练
    model = fleet.distributed_model(model)
    optimizer = fleet.distributed_optimizer(optimizer)

    dataset = RandomDataset(batch_num * batch_size)
    # 设置分布式批采样器，用于数据并行训练
    sampler = DistributedBatchSampler(
        dataset, rank=get_rank(), batch_size=batch_size, shuffle=False, drop_last=True
    )
    train_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=1)

    for eop in range(epoch):
        model.train()

        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True

            out = model(img, delay_data=img)
            loss = paddle.nn.functional.mse_loss(input=out, label=img)
            avg_loss = paddle.mean(x=loss)

            avg_loss.backward()
            optimizer.step()
            model.clear_gradients()

            if batch_id % 5 == 0:
                print(
                    "[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f"
                    % (eop, batch_id, avg_loss, 0, 0)
                )


# 启动训练
if __name__ == "__main__":
    train_model()
