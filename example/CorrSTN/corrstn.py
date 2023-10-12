from copy import deepcopy

import paddle.nn as nn
from attention import MultiHeadAttentionAwareTemporalContext
from embedding import SpatialPositionalEmbedding, TemporalPositionalEmbedding
from endecoder import Decoder, DecoderLayer, Encoder, EncoderLayer
from graphconv import GCN, SpatialAttentionGCN


class CorrSTN(nn.Layer):
    def __init__(self, training_args, adj_matrix):
        super(CorrSTN, self).__init__()

        self.training_args = training_args

        self.encoder_dense = nn.Linear(
            training_args.encoder_input_size, training_args.d_model
        )
        self.decoder_dense = nn.Linear(
            training_args.decoder_input_size, training_args.d_model
        )

        self.encode_temporal_position = TemporalPositionalEmbedding(
            training_args, max_len=2048
        )
        self.decode_temporal_position = deepcopy(self.encode_temporal_position)

        self.encode_spatial_position = SpatialPositionalEmbedding(
            training_args, GCN(training_args, adj_matrix)
        )
        self.decode_spatial_position = deepcopy(self.encode_spatial_position)

        attn_ss = MultiHeadAttentionAwareTemporalContext(
            args=training_args,
            adj_matrix=adj_matrix,
            query_conv_type="1DConv",
            key_conv_type="1DConv",
        )
        attn_st = MultiHeadAttentionAwareTemporalContext(
            args=training_args,
            adj_matrix=adj_matrix,
            query_conv_type="causal",
            key_conv_type="1DConv",
        )
        attn_tt = MultiHeadAttentionAwareTemporalContext(
            args=training_args,
            adj_matrix=adj_matrix,
            query_conv_type="causal",
            key_conv_type="causal",
        )

        spatial_attention_gcn = SpatialAttentionGCN(
            args=training_args,
            adj_matrix=adj_matrix,
            is_scale=True,
        )

        encoderLayer = EncoderLayer(
            training_args.d_model,
            self_attn=deepcopy(attn_ss),
            gcn=deepcopy(spatial_attention_gcn),
            dropout=training_args.dropout,
            residual_connection=True,
            use_layer_norm=True,
        )
        decoderLayer = DecoderLayer(
            training_args.d_model,
            self_attn=deepcopy(attn_tt),
            src_attn=deepcopy(attn_st),
            gcn=deepcopy(spatial_attention_gcn),
            dropout=training_args.dropout,
            residual_connection=True,
            use_layer_norm=True,
        )

        self.encoder = Encoder(encoderLayer, training_args.encoder_num_layers)
        self.decoder = Decoder(decoderLayer, training_args.decoder_num_layers)

        self.generator = nn.Linear(
            training_args.d_model, training_args.decoder_output_size
        )

    def encode(self, src, lookup_index):
        src_dense = self.encoder_dense(src)
        src_tp_embedding = self.encode_temporal_position(src_dense, lookup_index)
        src_sp_embedding = self.encode_spatial_position(src_tp_embedding)
        encoder_output = self.encoder(src_sp_embedding)
        return encoder_output

    def decode(self, tgt, encoder_output):
        tgt_dense = self.decoder_dense(tgt)
        tgt_tp_embedding = self.decode_temporal_position(tgt_dense)
        tgt_sp_embedding = self.decode_spatial_position(tgt_tp_embedding)
        decoder_output = self.decoder(x=tgt_sp_embedding, memory=encoder_output)
        return self.generator(decoder_output)

    def forward(self, src, src_idx, tgt):
        src = src.reshape([-1, self.training_args.num_nodes] + src.shape[-2:])
        src_idx = src_idx.reshape(
            [-1, self.training_args.num_nodes] + src_idx.shape[-1:]
        )
        encoder_output = self.encode(src, src_idx)
        output = self.decode(tgt, encoder_output)
        return output


if __name__ == "__main__":
    import numpy as np
    import paddle
    import paddle.nn as nn
    from args import args
    from utils import norm_adj_matrix

    corr = np.load(args.adj_path)[0]
    adj_matrix = paddle.to_tensor(norm_adj_matrix(corr), paddle.get_default_dtype())

    nn.initializer.set_global_initializer(
        nn.initializer.XavierUniform(), nn.initializer.XavierUniform()
    )
    model = CorrSTN(args, adj_matrix)

    from dataset import TrafficFlowDataset
    from paddle.io import DataLoader

    train_dataset = TrafficFlowDataset(args, "train")
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False
    )
    his, tgt = next(iter(train_dataloader))
    his_index = paddle.randint(
        shape=[args.batch_size * args.num_nodes, 36], high=his.shape[-2], low=0
    )
    axis_bs = paddle.arange(args.batch_size * args.num_nodes)[:, None, None]
    axis_index = his_index[:, :, None]
    axis_dim = paddle.arange(1)[None, None, :]
    his = his.reshape([-1] + his.shape[-2:])
    his_input = his[axis_bs, axis_index, axis_dim]

    output = model(his_input, his_index, tgt)

    his_index = paddle.concat([paddle.arange(0, 12), paddle.arange(276, 288)]).expand(
        [args.batch_size * args.num_nodes, -1]
    )
    np.random.seed(42)
    encoder_input = paddle.to_tensor(np.ones([16 * 80, 24, 1]), dtype=paddle.float32)
    decoder_input = paddle.to_tensor(np.ones([16, 80, 12, 1]), dtype=paddle.float32)

    ckpt = "example/CorrSTN/ckpt/epoch_148.params"
    import torch

    torch_weight = torch.load(ckpt, map_location=torch.device("cpu"))
    for param_tensor in torch_weight:
        print(f"{param_tensor} \t {torch_weight[param_tensor].shape}")

    paddle_weight = model.state_dict()
    for param_tensor in paddle_weight:
        print(f"{param_tensor} \t {paddle_weight[param_tensor].shape}")

    from collections import OrderedDict

    new_weight_dict = OrderedDict()

    model(encoder_input, his_index, decoder_input)
