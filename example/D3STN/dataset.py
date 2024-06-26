import numpy as np
import paddle
from paddle.io import Dataset


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
        self.mean = np.mean(data)
        self.std = np.std(data)

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
        mean = paddle.to_tensor(self.mean) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std) if paddle.is_tensor(data) else self.std
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

        mean = paddle.to_tensor(self.mean) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std) if paddle.is_tensor(data) else self.std
        return (data * std) + mean


class ScalerMinMax(object):
    """
    Desc: Normalization utilities with min max
    """

    def __init__(self):
        self.min = 0.0
        self.max = 1.0

    def fit(self, data, axis=None):
        """
        Desc:
            Fit the data
        Args:
            data:
        Returns:
            None
        """
        self.min = (
            np.min(data) if axis is None else np.min(data, axis=axis, keepdims=True)
        )
        self.min = self.min * 0.0  # min must be zero

        self.max = (
            np.max(data) if axis is None else np.max(data, axis=axis, keepdims=True)
        )

    def transform(self, data):
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

    def inverse_transform(self, data):
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
        return 1.0 * data * (_max - _min) + _min


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
        origin_data = np.load(training_args.data_path)["data"].transpose([1, 0, 2])
        origin_data = origin_data[:, :, :1]
        self.num_nodes, self.seq_len, self.dims = origin_data.shape

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

        # Scaler
        # [N, T, D]
        if training_args.scale:
            self.scaler = ScalerMinMax()
            train_data = origin_data[:, : self.train_size, :]
            self.scaler.fit(train_data, axis=1)
            self.data = self.scaler.transform(origin_data)
        else:
            self.data = origin_data

        # Concat day of week and hour of day index
        index = np.arange(0, self.seq_len, step=1).reshape([1, -1, 1])
        index = index.repeat(self.num_nodes, axis=0)
        day_of_week_index = (index // 288) % 7
        hour_of_day_index = index % 288

        self.data = np.concatenate(
            [self.data, day_of_week_index, hour_of_day_index], axis=-1
        )

        if self.data_type == "train":
            data_len = self.train_size - self.training_args.tgt_len
        elif self.data_type == "val":
            data_len = self.val_size - self.training_args.tgt_len
        else:
            data_len = self.test_size - self.training_args.tgt_len

        self.his_pair = []
        self.tgt_pair = []
        for i in range(data_len):
            if self.data_type == "train":
                i += 0
            elif self.data_type == "val":
                i += self.train_size - self.training_args.his_len
            else:
                i += self.train_size + self.val_size - self.training_args.his_len

            his_begin = i
            his_end = his_begin + self.training_args.his_len
            tgt_begin = his_end
            tgt_end = tgt_begin + self.training_args.tgt_len

            if "HZME" in self.training_args.dataset_name and (
                tgt_begin % 288 < 72 or tgt_end % 288 < 72
            ):
                continue
            self.his_pair.append((his_begin, his_end))
            self.tgt_pair.append((tgt_begin, tgt_end))

    def __getitem__(self, index):
        his_begin, his_end = self.his_pair[index]
        tgt_begin, tgt_end = self.tgt_pair[index]

        his = self.data[:, his_begin:his_end, :]
        tgt = self.data[:, tgt_begin:tgt_end, :]
        # his_extend = his[:,-12:, :] + (his[:,-1:,:] - his[:,-12:-11,:])
        # his = np.concatenate([his, his_extend], axis=-2)

        return his, tgt

    def __len__(self):
        return len(self.his_pair)

    def inverse_transform(self, data):
        if self.training_args.scale:
            return self.scaler.inverse_transform(data)
        else:
            return data
