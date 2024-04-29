import paddle.nn as nn
from utils.utils import clones


class SublayerConnection(nn.Layer):
    """
    A residual connection followed by a layer norm
    """

    def __init__(self, size, dropout, residual_connection, use_layer_norm):
        super(SublayerConnection, self).__init__()
        self.residual_connection = residual_connection
        self.use_layer_norm = use_layer_norm
        self.dropout = nn.Dropout(dropout)
        if self.use_layer_norm:
            self.norm = nn.LayerNorm(size)

    def forward(self, x, sublayer):
        """
        :param x: (batch, N, T, d_model)
        :param sublayer: nn.Layer
        :return: (batch, N, T, d_model)
        """
        if self.residual_connection and self.use_layer_norm:
            return x + self.dropout(sublayer(self.norm(x)))
        if self.residual_connection and (not self.use_layer_norm):
            return x + self.dropout(sublayer(x))
        if (not self.residual_connection) and self.use_layer_norm:
            return self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Layer):
    def __init__(
        self,
        size,
        self_attn,
        gcn,
        dropout,
        residual_connection=True,
        use_layer_norm=True,
    ):
        super(EncoderLayer, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_layer_norm
        self.self_attn = self_attn
        self.feed_forward_gcn = gcn
        if residual_connection or use_layer_norm:
            self.sublayer = clones(
                SublayerConnection(size, dropout, residual_connection, use_layer_norm),
                2,
            )
        self.size = size

    def forward(self, x):
        """
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        """
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](
                x,
                lambda x: self.self_attn(x, x, x),
            )  # [B,N,T,D]
            return self.sublayer[1](x, self.feed_forward_gcn)  # [B,N,T,D]
        else:

            x = self.self_attn(x, x, x)  # [B,N,T,D]
            return self.feed_forward_gcn(x)  # [B,N,T,D]


class Encoder(nn.Layer):
    def __init__(self, layer, n):
        """
        :param layer:  EncoderLayer
        :param n:  int, number of EncoderLayers
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x):
        """
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DecoderLayer(nn.Layer):
    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        gcn,
        dropout,
        residual_connection=True,
        use_layer_norm=True,
    ):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward_gcn = gcn
        self.residual_connection = residual_connection
        self.use_layer_norm = use_layer_norm
        if residual_connection or use_layer_norm:
            self.sublayer = clones(
                SublayerConnection(size, dropout, residual_connection, use_layer_norm),
                3,
            )
        self.norm = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)
        self.norm3 = nn.LayerNorm(size)

    def forward(self, x, memory, past_key_values):
        """
        :param x: (batch_size, N, T', F_in)
        :param memory: (batch_size, N, T, F_in)
        :return: (batch_size, N, T', F_in)
        """
        m = memory

        # [B,N,T,D]
        x, past_key_values = self.self_attn(
            x, x, x, past_key_values, is_mask=True, use_cache=True
        )

        # [B,N,T,D]
        x = self.src_attn(x, m, m)
        x = self.feed_forward_gcn(x)
        return x, past_key_values


class Decoder(nn.Layer):
    def __init__(self, layer, n):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, past_key_values):
        """
        :param x: (batch, N, T', d_model)
        :param memory: (batch, N, T, d_model)
        :return:(batch, N, T', d_model)
        """
        for idx, layer in enumerate(self.layers):
            output = layer(x, memory, past_key_values[idx])
            x = output[0]
            past_key_values[idx] = output[1]
        return self.norm(x), past_key_values
