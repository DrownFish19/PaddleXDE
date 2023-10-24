import numpy as np
import paddle
from paddle.io import DataLoader, Dataset


class ScalerStd(object):
    """
    Desc: Normalization utilities with std mean
    """

    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        # type: (paddle.tensor) -> None
        """
        Desc:
            Fit the data
        Args:
            data:
        Returns:
            None
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Transform the data
        Args:
            data:
        Returns:
            The transformed data
        """
        mean = (
            paddle.tensor(self.mean).type_as(data).to(data.device)
            if paddle.is_tensor(data)
            else self.mean
        )
        std = (
            paddle.tensor(self.std).type_as(data).to(data.device)
            if paddle.is_tensor(data)
            else self.std
        )
        return (data - mean) / std

    def inverse_transform(self, data):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Restore to the original data
        Args:
            data: the transformed data
        Returns:
            The original data
        """

        mean = paddle.tensor(self.mean) if paddle.is_tensor(data) else self.mean
        std = paddle.tensor(self.std) if paddle.is_tensor(data) else self.std
        return (data * std) + mean


class ScalerMinMax(object):
    """
    Desc: Normalization utilities with min max
    """

    def __init__(self):
        self.min = 0.0
        self.max = 1.0

    def fit(self, data):
        # type: (paddle.tensor) -> None
        """
        Desc:
            Fit the data
        Args:
            data:
        Returns:
            None
        """
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def transform(self, data):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Transform the data
        Args:
            data:
        Returns:
            The transformed data
        """
        _min = paddle.to_tensor(self.min) if paddle.is_tensor(data) else self.min
        _max = paddle.to_tensor(self.max) if paddle.is_tensor(data) else self.max
        data = 1.0 * (data - _min) / (_max - _min)
        return 2.0 * data - 1.0

    def inverse_transform(self, data, axis=None):
        # type: (paddle.tensor, None) -> paddle.tensor
        """
        Desc:
            Restore to the original data
        Args:
            data: the transformed data
        Returns:
            The original data
        """

        _min = paddle.to_tensor(self.min) if paddle.is_tensor(data) else self.min
        _max = paddle.to_tensor(self.max) if paddle.is_tensor(data) else self.max
        data = (data + 1.0) / 2.0

        if axis is None:
            return 1.0 * data * (_max[axis] - _min[axis]) + _min[axis]
        else:
            return 1.0 * data * (_max[axis] - _min[axis]) + _min[axis]


class TrafficFlowDataset(Dataset):
    """
    Desc: Data preprocessing,
          Here, e.g.    15 days for training,
                        3 days for validation,
                        and 6 days for testing
    """

    def __init__(self, training_args, data_type="train"):
        super().__init__()
        self.training_args = training_args

        # [T, N, D]
        # D=3 for PEMS04 and PEMS08, D=1 for others
        self.origin_data = np.load(training_args.data_path)["data"].transpose([1, 0, 2])
        self.num_nodes, self.seq_len, self.dims = self.origin_data.shape

        self.train_ratio, self.val_ratio, self.test_ratio = map(
            int, training_args.split.split(":")
        )
        sum_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        self.train_ratio, self.val_ratio, self.test_ratio = (
            self.train_ratio / sum_ratio,
            self.val_ratio / sum_ratio,
            self.test_ratio / sum_ratio,
        )

        self.train_size = int(self.seq_len * self.train_ratio)
        self.val_size = int(self.seq_len * self.val_ratio)
        self.test_size = int(self.seq_len * self.test_ratio)
        self.data_type = data_type

        if training_args.scale:
            self.scaler = ScalerMinMax()
            train_data = self.origin_data[: self.train_size, :, :]
            self.scaler.fit(train_data.reshape(-1, train_data.shape[-1]))
            self.data = self.scaler.transform(self.origin_data).reshape(
                self.num_nodes, self.seq_len, self.dims
            )
        else:
            self.data = self.origin_data

        self.data = paddle.to_tensor(self.data, dtype=paddle.get_default_dtype())

    def __getitem__(self, index):

        if self.data_type == "train":
            index += 0
        elif self.data_type == "val":
            index += self.train_size - self.training_args.his_len
        else:
            index += self.train_size + self.val_size - self.training_args.his_len

        his_begin = index
        his_end = his_begin + self.training_args.his_len
        tgt_begin = his_end
        tgt_end = tgt_begin + self.training_args.tgt_len

        valid = True
        if "HZME" in self.training_args.dataset_name:
            if tgt_begin % 288 < 72 or tgt_end % 288 < 72:
                valid = False

        # [N, T, F]
        his = self.data[:, his_begin:his_end, :]
        tgt = self.data[:, tgt_begin:tgt_end, :]

        return his, tgt, valid

    def __len__(self):
        if self.data_type == "train":
            data_len = (
                self.train_size
                - self.training_args.his_len
                - self.training_args.tgt_len
            )
        elif self.data_type == "val":
            data_len = self.val_size - self.training_args.tgt_len
        else:
            data_len = self.test_size - self.training_args.tgt_len

        return data_len

    def inverse_transform(self, data, axis=None):
        if self.training_args.scale:
            return self.scaler.inverse_transform(data, axis)
        else:
            return data


if __name__ == "__main__":
    from args import args

    train_dataset = TrafficFlowDataset(args, "train")
    val_dataset = TrafficFlowDataset(args, "val")
    test_dataset = TrafficFlowDataset(args, "test")

    def collate_func(batch_data):
        src_list, tgt_list = [], []

        for item in batch_data:
            if item[2]:
                src_list.append(item[0])
                tgt_list.append(item[1])

        if len(src_list) == 0:
            src_list.append(item[0])
            tgt_list.append(item[1])

        return paddle.stack(src_list), paddle.stack(tgt_list)

    traing_dataloader = DataLoader(
        train_dataset, batch_size=10, shuffle=False, collate_fn=collate_func
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=10, shuffle=False, collate_fn=collate_func
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=10, shuffle=False, collate_fn=collate_func
    )

    print(test_dataloader[0])

    # for item in traing_dataloader:
    #     print(item[0].shape, item[1].shape)

    # for item in val_dataloader:
    #     print(item[0].shape, item[1].shape)

    # for item in test_dataloader:
    #     print(item[0].shape, item[1].shape)
