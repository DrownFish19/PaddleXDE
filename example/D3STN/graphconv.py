import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class GCN(nn.Layer):
    def __init__(self, training_args, d_model, norm_adj_matrix, norm_sc_matrix):
        super(GCN, self).__init__()
        self.args = training_args
        self.norm_adj_matrix = norm_adj_matrix
        self.norm_sc_matrix = norm_sc_matrix
        self.Theta = nn.Linear(
            d_model,
            d_model,
            bias_attr=False,
        )
        self.alpha = paddle.create_parameter(
            shape=[1],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(value=0.0),
        )
        self.beta = paddle.create_parameter(
            shape=[1],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(value=1.0),
        )

    def forward(self, x):
        """
        spatial graph convolution operation
        :param x: (batch_size, N, F_in)
        :return: (batch_size, N, F_out)
        """
        adj = paddle.add(
            self.alpha * self.norm_adj_matrix,
            self.beta * self.norm_sc_matrix,
        )

        if self.args.with_adj and not self.args.with_sc:
            adj = self.alpha * self.norm_adj_matrix
        elif not self.args.with_adj and self.args.with_sc:
            adj = self.beta * self.norm_sc_matrix
        elif self.args.with_adj and self.args.with_sc:
            adj = paddle.add(
                self.alpha * self.norm_adj_matrix, self.beta * self.norm_sc_matrix
            )
        else:
            adj = self.alpha * self.norm_adj_matrix

        x_gcn = paddle.matmul(adj, x)
        # [N,N][B,N,in]->[B,N,in]->[B,N,out]
        return F.silu(self.Theta(x_gcn))


class SpatialAttentionLayer(nn.Layer):
    """
    compute spatial attention scores
    """

    def __init__(self, dropout=0.0):
        super(SpatialAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        :param x: (B, N, T, D)
        :return: (B, T, N, N)
        """
        B, N, T, D = x.shape
        x = x.transpose([0, 2, 1, 3])  # [B,T,N,F_in]
        # [B,T,N,F_in][B,T,F_in,N]=[B*T,N,N]
        score = paddle.matmul(x, x, transpose_y=True) / math.sqrt(D)
        score = self.dropout(F.softmax(score, axis=-1))  # [B,T,N,N]
        return score


class SpatialAttentionGCN(nn.Layer):
    def __init__(self, args, adj_matrix, sc_matrix, is_scale=True):
        super(SpatialAttentionGCN, self).__init__()
        self.norm_adj = adj_matrix
        self.norm_sc = sc_matrix
        self.args = args
        self.linear = nn.Linear(args.d_model, args.d_model, bias_attr=False)
        self.is_scale = is_scale
        self.SAt = SpatialAttentionLayer(dropout=args.dropout)
        self.alpha = paddle.create_parameter(
            shape=[1],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(value=0.0),
        )
        self.beta = paddle.create_parameter(
            shape=[1],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(value=1.0),
        )

    def forward(self, x):
        """
        spatial graph convolution
        :param x: (B, N, T, F_in)
        :return: (B, N, T, F_out)
        """
        B, N, T, D = x.shape
        spatial_attention = self.SAt(x)  # [B, T, N, N]
        if self.is_scale:
            spatial_attention = spatial_attention / math.sqrt(self.args.d_model)
        x = x.transpose([0, 2, 1, 3])  # [B,T,N,D]

        if self.args.with_adj and not self.args.with_sc:
            adj = self.alpha * paddle.multiply(spatial_attention, self.norm_adj)
        elif not self.args.with_adj and self.args.with_sc:
            adj = self.beta * paddle.multiply(spatial_attention, self.norm_sc)
        elif self.args.with_adj and self.args.with_sc:
            adj = paddle.add(
                self.alpha * paddle.multiply(spatial_attention, self.norm_adj),
                self.beta * paddle.multiply(spatial_attention, self.norm_sc),
            )
        else:
            adj = self.alpha * paddle.multiply(spatial_attention, self.norm_adj)

        x_gcn = paddle.matmul(adj, x)
        # [B, N, T, D]
        return F.silu(self.linear(x_gcn).transpose([0, 2, 1, 3]))
