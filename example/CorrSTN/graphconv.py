import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class GCN(nn.Layer):
    def __init__(self, training_args, norm_adj_matrix):
        super(GCN, self).__init__()
        self.register_buffer("norm_adj_matrix", norm_adj_matrix)
        self.Theta = nn.Linear(
            training_args.d_model, training_args.d_model, bias_attr=False
        )

    def forward(self, x):
        """
        spatial graph convolution operation
        :param x: (batch_size, N, F_in)
        :return: (batch_size, N, F_out)
        """
        return F.relu(
            self.Theta(paddle.matmul(self.norm_adj_matrix, x))
        )  # [N,N][B,N,in]->[B,N,in]->[B,N,out]


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
    def __init__(self, args, adj_matrix, is_scale=False):
        super(SpatialAttentionGCN, self).__init__()
        self.register_buffer("norm_adj", adj_matrix)
        self.args = args
        self.linear = nn.Linear(args.d_model, args.d_model, bias_attr=False)
        self.is_scale = is_scale
        self.SAt = SpatialAttentionLayer(dropout=args.dropout)

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
        x = paddle.matmul(paddle.multiply(spatial_attention, self.norm_adj), x)

        # [B, N, T, D]
        return F.relu(self.linear(x).transpose([0, 2, 1, 3]))
