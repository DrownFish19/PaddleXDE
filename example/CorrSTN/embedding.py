import math
from typing import List

import numpy as np
import paddle
import paddle.nn as nn
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class SpatialPositionalEmbedding(nn.Layer):
    def __init__(self, args, gcn=None):
        super(SpatialPositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=args.dropout)
        self.embedding = paddle.nn.Embedding(args.num_nodes, args.d_model)
        self.gcn_smooth_layers = None
        if (gcn is not None) and (args.smooth_layer_num > 0):
            self.gcn_smooth_layers = nn.LayerList(
                [gcn for _ in range(args.smooth_layer_num)]
            )

    def forward(self, x):
        """
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        """
        batch, num_of_vertices, timestamps, _ = x.shape
        x_indexs = paddle.arange(num_of_vertices)  # (N)
        embed = self.embedding(x_indexs).unsqueeze(0)  # [N,D]->[1,N,D]
        if self.gcn_smooth_layers is not None:
            for _, l in enumerate(self.gcn_smooth_layers):
                embed = l(embed)  # [1,N,D] -> [1,N,D]
        x = x + embed.unsqueeze(-2)  # [B,N,T,D]+[1,N,1,D]
        return self.dropout(x)


class TemporalPositionalEmbedding(nn.Layer):
    def __init__(self, args, max_len):
        super(TemporalPositionalEmbedding, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(p=args.dropout)
        self.max_len = max_len
        self.d_model = args.d_model
        # computing the positional encodings once in log space
        pe = paddle.zeros([max_len, self.d_model])
        for pos in range(max_len):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.d_model)))
                pe[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / self.d_model))
                )

        pe = pe.expand([1, 1, max_len, self.d_model])  # [1,1,max_len,D]
        # register_buffer:
        # Adds a persistent buffer to the module.
        # This is typically used to register a buffer that should not to be considered a model parameter.
        self.register_buffer("pe", pe)

    def forward(self, x, lookup_index=None):
        """

        Args:
            x: (B, N, T, F)
            lookup_index: None or [B,N,T]

        Returns:(B, N, T, F)

        """

        if lookup_index is None:
            lookup_index = paddle.arange(x.shape[-2])
            # [B,N,T,D] + [1,1,T,D]
            x = x + paddle.index_select(self.pe, lookup_index, axis=-2)
        else:

            x = x + paddle.index_select(self.pe, lookup_index, axis=-2)

        return self.dropout(x.detach())


class TimeEmbedding:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class MinuteOfHour(TimeEmbedding):
    """Minute of hour encoded as value between [-1.0, 1.0]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.minute / 59.0 - 0.5) * 2.0


class HourOfDay(TimeEmbedding):
    """Hour of day encoded as value between [-1.0, 1.0]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.hour / 23.0 - 0.5) * 2.0


def time_features_from_frequency_str(freq_str: str) -> List[TimeEmbedding]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.Hour: [HourOfDay],
        offsets.Minute: [MinuteOfHour, HourOfDay],
    }
    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]


def time_embedding(dates, freq="10T"):
    dates = pd.to_datetime(list(dates))
    res = []
    for feat in time_features_from_frequency_str(freq):
        res.append(feat(dates))

    return np.vstack(res).transpose(1, 0)


if __name__ == "__main__":
    dataStrs = ["00:10", "12:30", "13:30", "22:20"]
    time_embedding_res = time_embedding(dates=dataStrs)
    print(time_embedding_res)
    print()
