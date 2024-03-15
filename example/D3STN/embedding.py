import math

import paddle
import paddle.nn as nn


class SpatialPositionalEmbedding(nn.Layer):
    def __init__(self, args):
        super(SpatialPositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=args.dropout)
        self.embedding = paddle.nn.Embedding(args.num_nodes, int(args.d_model / 4))

    def forward(self, x):
        """
        :param x: (B, N, T, D)
        :return: (1, N, 1, D)
        """
        B, N, T, D = x.shape
        x_index = paddle.arange(N)  # (N)
        embed = self.embedding(x_index).unsqueeze([0, 2])  # [N,D]->[1,N,1,D]
        return embed


class TemporalPositionalEmbedding(nn.Layer):
    def __init__(self, args):
        super(TemporalPositionalEmbedding, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(p=args.dropout)
        self.max_len = max(args.his_len, args.tgt_len)
        self.d_model = int(args.d_model / 4)
        # computing the positional encodings once in log space
        pe = paddle.zeros([self.max_len, self.d_model])
        for pos in range(self.max_len):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.d_model)))
                pe[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / self.d_model))
                )

        pe = pe.unsqueeze([0, 1])  # [1,1,max_len,D]
        # register_buffer:
        # Adds a persistent buffer to the module.
        # This is typically used to register a buffer that should not to be considered a model parameter.
        self.register_buffer("pe", pe)

    def forward(self, x):
        """

        Args:
            x: (B, N, T, D)

        Returns:(1, 1, T, D)

        """
        lookup_index = paddle.arange(x.shape[-2])
        embed = paddle.index_select(self.pe, lookup_index, axis=-2)
        return embed


class TemporalSectionEmbedding(nn.Layer):
    def __init__(self, args):
        """
        axis=1 indicate day of week
        axis=2 indicate hour of day
        """
        super(TemporalSectionEmbedding, self).__init__()
        self.embedding_day = paddle.nn.Embedding(7, int(args.d_model / 8))
        self.embedding_minute = paddle.nn.Embedding(288, int(args.d_model / 8))

    def forward(self, d_idx, m_idx):
        d_idx = paddle.cast(d_idx, dtype=paddle.int64)
        m_idx = paddle.clip(m_idx, min=0, max=self.embedding_minute._num_embeddings - 1)
        m_idx = paddle.cast(m_idx, dtype=paddle.int64)
        embed = paddle.concat(
            [self.embedding_day(d_idx), self.embedding_minute(m_idx)],
            axis=-1,
        )
        return embed


class TrafficFlowEmbedding(nn.Layer):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dense = nn.Sequential(
            nn.Linear(self.args.input_size, self.args.d_model, bias_attr=True),
            nn.Silu(),
            nn.Linear(self.args.d_model, int(self.args.d_model / 4), bias_attr=True),
        )

        self.spatial_position_embedding = SpatialPositionalEmbedding(args=args)
        self.temporal_position_embedding = TemporalPositionalEmbedding(args=args)
        self.temporal_section_embedding = TemporalSectionEmbedding(args=args)
        self.layer_norm = nn.LayerNorm(self.args.d_model)

    def forward(self, x, d_idx, m_idx):
        x = self.dense(x)
        spatial_emb = self.spatial_position_embedding(x)
        temporal_emb = self.temporal_position_embedding(x)
        section_emb = self.temporal_section_embedding(d_idx, m_idx)

        spatial_emb = paddle.expand_as(spatial_emb, x)
        temporal_emb = paddle.expand_as(temporal_emb, x)
        section_emb = paddle.expand_as(section_emb, x)

        x = paddle.concat([x, spatial_emb, temporal_emb, section_emb], axis=-1)
        x = self.layer_norm(x)
        return x
