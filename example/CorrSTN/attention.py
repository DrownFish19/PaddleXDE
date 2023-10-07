import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from utils import clones


class VanillaAttention(nn.Layer):
    def __init__(self):
        super(VanillaAttention, self).__init__()

    def forward(self, query, key, value, mask=None, dropout=None):
        """
        :param query:  [B,N,H,T,D]
        :param key: [B,N,H,T,D]
        :param value: [B,N,H,T,D]
        :param mask: [B,1,1,T2,T2]
        :param dropout:
        :return: [B,N,H,T1,d], [B,N,H,T1,T2]
        """
        B, N, H, T, D = query.shape
        # [B,N,H,T1,T2]
        scores = paddle.matmul(query, key, transpose_y=True) / math.sqrt(D)

        if mask is not None:
            scores = scores + mask
        p_attn = F.softmax(scores, axis=-1)  # [B,N,H,T1,T2]
        if dropout is not None:
            p_attn = dropout(p_attn)

        return paddle.matmul(p_attn, value)  # [B,N,H,T1,T2] * [B,N,H,T1,D]


class CorrAttention(nn.Layer):
    def __init__(self, norm_sc_matrix, attention_top_k):
        super(CorrAttention, self).__init__()
        self.norm_sc_matrix = norm_sc_matrix
        self.k = attention_top_k
        self.DAT = DynamicAttentionLayer(dropout=0.0)

        sc_top = norm_sc_matrix.topk(self.k, sorted=False)
        sc_top_v = sc_top[0]  # [C, N, K]
        # [C, N, 1, K]
        sc_top_v = paddle.softmax(sc_top_v, dim=-1).unsqueeze(-1).transpose(-2, -1)
        self.register_buffer("sc_top_v", sc_top_v)
        self.sc_top_i = sc_top[1]  # [C, N, K]

    def forward(self, query, key, value, mask=None, dropout=None):
        B, N, H, T1, D = query.size()
        _, _, _, T2, _ = key.size()

        # [B,N,K,H,T,D] => [B,H,T,N,K,D]
        key_c = key[:, self.sc_top_i[0], :, :, :].permute(0, 3, 4, 1, 2, 5)
        key_o = (
            paddle.matmul(self.sc_top_v[0], key_c)
            .view(B, H, T2, N, D)
            .permute(0, 3, 1, 2, 4)
        )

        for c in range(1, self.norm_sc_matrix.shape[0]):
            # [B,N,K,H,T,D] => [B,H,T,N,K,D]
            key_c = key[:, self.sc_top_i[c], :, :, :].permute(0, 3, 4, 1, 2, 5)
            key_o += (
                paddle.matmul(self.sc_top_v[c], key_c)
                .view(B, H, T2, N, D)
                .permute(0, 3, 1, 2, 4)
            )

        key = key_o / self.norm_sc_matrix.shape[0]

        # [B,N,H,T1,T2]
        scores = paddle.matmul(query, key, transpose_y=True) / math.sqrt(D)

        if mask is not None:
            # [T1,T2]
            mask = paddle.ones(T1, T2, dtype=paddle.bool).to(query.device).triu(1)
            mask_cluster = mask[None, :].expand(B, N, H, T1, T2)  # # [B,H,N,T1,T2]
            scores = scores.masked_fill_(mask_cluster, -np.inf)
        p_attn = F.softmax(scores, dim=-1)  # [B,H,N,T1,T2]
        if dropout is not None:
            p_attn = dropout(p_attn)  # [B,H,N,T1,T2]

        return paddle.matmul(p_attn, value)  # [B,H,N,T1,D], [B,H,N,T1,T2]


class MultiHeadAttentionAwareTemporalContext(nn.Layer):
    def __init__(
        self,
        args,
        adj_matrix,
        query_conv_type="1DConv",
        key_conv_type="1DConv",
    ):
        super(MultiHeadAttentionAwareTemporalContext, self).__init__()

        assert args.d_model % args.head == 0
        self.head_dim = args.d_model // args.head
        self.heads = args.head

        # 2 linear layers: 1  for W^V, 1 for W^O
        self.linears = clones(nn.Linear(args.d_model, args.d_model), 2)

        # 构建aware_temporal_context
        self.padding_causal = args.kernel_size - 1
        self.padding_1DConv = (args.kernel_size - 1) // 2
        self.query_conv_type = query_conv_type
        self.key_conv_type = key_conv_type
        if query_conv_type == "1DConv":
            self.query_conv = nn.Conv2D(
                args.d_model,
                args.d_model,
                (1, args.kernel_size),
                padding=(0, self.padding_1DConv),
            )
        else:
            self.query_conv = nn.Conv2D(
                args.d_model,
                args.d_model,
                (1, args.kernel_size),
                padding=(0, self.padding_causal),
            )

        if key_conv_type == "1DConv":
            self.key_conv = nn.Conv2D(
                args.d_model,
                args.d_model,
                (1, args.kernel_size),
                padding=(0, self.padding_1DConv),
            )
        else:
            self.key_conv = nn.Conv2D(
                args.d_model,
                args.d_model,
                (1, args.kernel_size),
                padding=(0, self.padding_causal),
            )

        self.dropout = nn.Dropout(p=args.dropout)

        if args.attention == "Corr":
            self.attention = CorrAttention(
                norm_sc_matrix=adj_matrix,
                attention_top_k=args.top_k,
            )
            print("Attention Type:", args.attention)
        else:
            self.attention = VanillaAttention()
            print("Attention Type:", args.attention)

        self.attention_type = args.attention

    def subsequent_mask(self, size):
        """
        mask out subsequent positions.
        :param size: int
        :return: (1, size, size)
        """
        mask = paddle.full(
            [1, size, size],
            paddle.finfo(paddle.float32).min,
            dtype=paddle.float32,
        )
        mask = paddle.triu(mask, diagonal=1)
        return mask

    def aware_temporal(self, func, conv_type, data):
        B, N, T, D = data.shape
        data_conv = func(data.transpose([0, 3, 1, 2]))  # B, D, N, T
        data_conv = data_conv.transpose([0, 2, 3, 1])  # B, N, T, D

        if conv_type == "causal":
            data_conv = data_conv[:, :, : -self.padding_causal, :]

        # [B,N,T,D]
        return data_conv

    def forward(
        self,
        query,
        key,
        value,
        is_mask=False,
    ):
        """
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        Args:
            query: (batch, N, T, d_model)
            key: (batch, N, T, d_model)
            value: (batch, N, T, d_model)
            is_mask: bool
            query_multi_segment: bool
            key_multi_segment: bool

        Returns: (batch, N, T, d_model)

        """
        B, N, T, D = query.shape

        if is_mask:
            # (batch, 1, 1, T, T), same mask applied to all h heads.
            # (1, T', T')
            mask = self.subsequent_mask(T).unsqueeze([0, 1])
        else:
            mask = None

        query = self.aware_temporal(self.query_conv, self.query_conv_type, query)
        key = self.aware_temporal(self.key_conv, self.key_conv_type, key)
        value = self.linears[0](value)

        # convert [B,N,T,D] to [B,N,T,H,D] to [B,N,H,T,D]
        multi_head_shape = [B, N, -1, self.heads, self.head_dim]
        perm = [0, 1, 3, 2, 4]
        query = query.reshape(multi_head_shape).transpose(perm)
        key = key.reshape(multi_head_shape).transpose(perm)
        value = value.reshape(multi_head_shape).transpose(perm)

        # [B,N,H,T,d]
        if self.attention_type == "Prob":
            x = self.attention(query, key, value, is_mask=is_mask)
        elif self.attention_type == "Corr":
            x = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        else:
            x = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # [B,N,T,D]
        x = x.transpose(perm).reshape([B, N, -1, self.heads * self.head_dim])
        return self.linears[1](x)
