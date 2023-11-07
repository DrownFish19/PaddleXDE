import math

import paddle
import paddle.nn as nn


class SpatialPositionalEmbedding(nn.Layer):
    def __init__(self, args, gcn=None):
        super(SpatialPositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=args.dropout)
        self.embedding = paddle.nn.Embedding(args.num_nodes, args.d_proj)
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
        return embed.unsqueeze(-2)  # [B,N,T,D]+[1,N,1,D]


class TemporalPositionalEmbedding(nn.Layer):
    def __init__(self, args, max_len):
        super(TemporalPositionalEmbedding, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(p=args.dropout)
        self.max_len = max_len
        self.d_model = args.d_proj
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
            embed = paddle.index_select(self.pe, lookup_index, axis=-2)
        else:
            if lookup_index.dtype != paddle.int64:
                lookup_index = paddle.cast(lookup_index, dtype="int64")
            embed = paddle.index_select(self.pe, lookup_index, axis=-2)

        return embed


class TemporalSectionEmbedding(nn.Layer):
    def __init__(self, args, section_nums, axis=1):
        """
        axis=1 indicate day of week
        axis=2 indicate hour of day
        """
        super(TemporalSectionEmbedding, self).__init__()
        self.axis = axis
        self.embedding = paddle.nn.Embedding(section_nums, args.d_sect)

    def forward(self, x):
        input = x[..., self.axis]
        input = paddle.clip(input,min=0,max=self.embedding._num_embeddings-1)
        input = paddle.cast(input, dtype=paddle.int32)
        return self.embedding(input)
