import math

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
    def __init__(self, norm_sc_matrix, attention_top_k: int):
        super(CorrAttention, self).__init__()
        self.norm_sc_matrix = norm_sc_matrix
        self.k = attention_top_k

        vals, indx = paddle.topk(norm_sc_matrix, self.k, axis=-1)
        self.vals = F.softmax(vals, axis=-1).unsqueeze(-2)  # [N, 1, K]
        self.indx = indx

    def forward(self, query, key, value, mask=None, dropout=None):
        B, N, H, T1, D = query.shape
        B, N, H, T2, D = key.shape

        # method 1
        # key = key.transpose([0, 2, 3, 1, 4])  # [B, H, T, N, D]
        # [B, H, T, N, D]
        # key_selected = paddle.stack(
        #     [
        #         paddle.index_select(key, index=self.indx[i], axis=-2) # [B, H, T, K, D]
        #         for i in range(N)
        #     ],
        #     axis=-3,
        # )
        # key = paddle.matmul(self.vals, key_selected).squeeze(-2)

        # method 2
        # key = key.transpose([0, 2, 3, 1, 4])  # [B, H, T, N, D]
        # key = paddle.concat(
        #     [
        #         paddle.matmul(
        #             self.vals[i], paddle.index_select(key, index=self.indx[i], axis=-2)
        #         )
        #         for i in range(N)
        #     ],
        #     axis=-2,
        # )

        # method 3
        # axis_b = paddle.arange(B)[:, None, None,None ,None, None]
        # axis_n = self.indx[None, :, :,None ,None, None]
        # axis_h = paddle.arange(H)[None, None, None,: ,None, None]
        # axis_t = paddle.arange(T2)[None, None, None,None ,:, None]
        # axis_d = paddle.arange(D)[None, None, None,None ,None, :]

        # # [B,N,K,H,T,D] => [B,H,T,N,K,D]
        # key_selected = (key[axis_b, axis_n, axis_h, axis_t, axis_d]
        #             .transpose([0, 3, 4, 1, 2, 5]))
        # key =paddle.matmul(self.vals, key_selected).squeeze(-2)

        # key = key.transpose([0, 3, 1, 2, 4])
        # key = key  # / math.sqrt(N)

        # [B,N,H,T1,T2]
        scores = paddle.matmul(query, key, transpose_y=True) / math.sqrt(D)

        if mask is not None:
            scores = scores + mask
        p_attn = F.softmax(scores, axis=-1)  # [B,H,N,T1,T2]
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
        self.training_args = args
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

        self.attention = VanillaAttention()

        self.attention_type = args.attention

        vals, indx = paddle.topk(adj_matrix, args.top_k, axis=-1)
        self.vals = F.softmax(vals, axis=-1).unsqueeze(-2)  # [N, 1, K]
        self.indx = indx

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

        if self.training_args.split_seq and T >= 12:
            data_conv = []
            for i in range(T // 12):
                # B, D, N, T
                res = func(data[:, :, i * 12 : (i + 1) * 12, :].transpose([0, 3, 1, 2]))
                res = res.transpose([0, 2, 3, 1])  # B, N, T, D

                if conv_type == "causal":
                    res = res[:, :, : -self.padding_causal, :]
                data_conv.append(res)
            data_conv = paddle.concat(data_conv, axis=-2)
        else:
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
        B, N, T2, D = key.shape
        if is_mask:
            # (batch, 1, 1, T, T), same mask applied to all h heads.
            # (1, T', T')
            mask = self.subsequent_mask(T).unsqueeze([0, 1])
        else:
            mask = None

        query = self.aware_temporal(self.query_conv, self.query_conv_type, query)
        key = self.aware_temporal(self.key_conv, self.key_conv_type, key)
        value = self.linears[0](value)

        if self.attention_type == "Corr":
            axis_b = paddle.arange(B)[:, None, None, None, None]
            axis_n = self.indx[None, :, :, None, None]
            axis_t = paddle.arange(T2)[None, None, None, :, None]
            axis_d = paddle.arange(D)[None, None, None, None, :]

            # [B,N,K,H,T,D] => [B,H,T,N,K,D]
            key_selected = key[axis_b, axis_n, axis_t, axis_d].transpose(
                [0, 3, 1, 2, 4]
            )
            key = paddle.matmul(self.vals, key_selected).squeeze(-2)
            key = key.transpose([0, 2, 1, 3])

        # convert [B,N,T,D] to [B,N,T,H,D] to [B,N,H,T,D]
        multi_head_shape = [B, N, -1, self.heads, self.head_dim]
        perm = [0, 1, 3, 2, 4]
        query = query.reshape(multi_head_shape).transpose(perm)
        key = key.reshape(multi_head_shape).transpose(perm)
        value = value.reshape(multi_head_shape).transpose(perm)

        # [B,N,H,T,d]
        x = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # [B,N,T,D]
        x = x.transpose(perm).reshape([B, N, -1, self.heads * self.head_dim])
        return self.linears[1](x)
