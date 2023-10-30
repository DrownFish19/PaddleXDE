from copy import deepcopy

import paddle.nn as nn
from attention import MultiHeadAttentionAwareTemporalContext
from embedding import SpatialPositionalEmbedding, TemporalPositionalEmbedding
from endecoder import Decoder, DecoderLayer, Encoder, EncoderLayer
from graphconv import GCN, SpatialAttentionGCN


class CorrSTN(nn.Layer):
    def __init__(self, training_args, adj_matrix, sc_matrix):
        super(CorrSTN, self).__init__()

        self.training_args = training_args

        self.encoder_dense = nn.Linear(
            training_args.encoder_input_size, training_args.d_model
        )
        self.decoder_dense = nn.Linear(
            training_args.decoder_input_size, training_args.d_model
        )

        self.encode_temporal_position = TemporalPositionalEmbedding(
            training_args, max_len=training_args.his_len
        )
        self.decode_temporal_position = TemporalPositionalEmbedding(
            training_args, max_len=training_args.tgt_len
        )

        self.encode_spatial_position = SpatialPositionalEmbedding(
            training_args, GCN(training_args, adj_matrix, sc_matrix)
        )
        self.decode_spatial_position = deepcopy(self.encode_spatial_position)

        attn_ss = MultiHeadAttentionAwareTemporalContext(
            args=training_args,
            adj_matrix=sc_matrix,
            query_conv_type="1DConv",
            key_conv_type="1DConv",
        )
        attn_st = MultiHeadAttentionAwareTemporalContext(
            args=training_args,
            adj_matrix=sc_matrix,
            query_conv_type="causal",
            key_conv_type="1DConv",
        )
        attn_tt = MultiHeadAttentionAwareTemporalContext(
            args=training_args,
            adj_matrix=sc_matrix,
            query_conv_type="causal",
            key_conv_type="causal",
        )

        spatial_attention_gcn = SpatialAttentionGCN(
            args=training_args,
            adj_matrix=adj_matrix,
            sc_matrix=sc_matrix,
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

    def encode(self, src_idx, src):
        src_dense = self.encoder_dense(src)
        src_tp_embedding = self.encode_temporal_position(src_dense, src_idx)
        src_sp_embedding = self.encode_spatial_position(src_tp_embedding)
        encoder_output = self.encoder(src_sp_embedding)
        return encoder_output

    def decode(self, encoder_output, tgt_idx=None, tgt=None):
        tgt_dense = self.decoder_dense(tgt)
        tgt_tp_embedding = self.decode_temporal_position(tgt_dense, tgt_idx)
        tgt_sp_embedding = self.decode_spatial_position(tgt_tp_embedding)
        decoder_output = self.decoder(x=tgt_sp_embedding, memory=encoder_output)
        return self.generator(decoder_output)

    def forward(self, src_idx, src, tgt_idx=None, tgt=None):
        encoder_output = self.encode(src_idx, src)
        output = self.decode(encoder_output, tgt_idx, tgt)
        return output


if __name__ == "__main__":
    import os

    # 将日志级别设置为6
    os.environ["GLOG_v"] = "6"
    import numpy as np
    import paddle
    import paddle.nn as nn
    from args import args
    from dataset import TrafficFlowDataset
    from paddle.io import DataLoader
    from paddle.nn.initializer import XavierUniform
    from utils import get_adjacency_matrix_2direction, norm_adj_matrix

    from paddlexde.functional import ddeint
    from paddlexde.solver import Euler

    default_dtype = paddle.get_default_dtype()
    adj_matrix, _ = get_adjacency_matrix_2direction(args.adj_path, 80)
    adj_matrix = paddle.to_tensor(norm_adj_matrix(adj_matrix), default_dtype)

    sc_matrix = np.load(args.sc_path)[0, :, :]
    sc_matrix = paddle.to_tensor(norm_adj_matrix(sc_matrix), default_dtype)

    nn.initializer.set_global_initializer(XavierUniform(), XavierUniform())
    model = CorrSTN(args, adj_matrix, sc_matrix)

    def collate_func(batch_data):
        src_list, tgt_list = [], []

        for item in batch_data:
            if item[2]:
                src_list.append(item[0])
                tgt_list.append(item[1])

        if len(src_list) == 0:
            src_list.append(item[0])
            tgt_list.append(item[1])

        return paddle.stack(src_list), paddle.stack(tgt_list)

    train_dataset = TrafficFlowDataset(args, "train")
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=collate_func
    )
    src, tgt = next(iter(train_dataloader))
    src_index = paddle.randint(shape=[12], high=src.shape[-2], low=0)
    src_input = paddle.index_select(src, src_index, axis=-2)
    decoder_input = paddle.concat([src[:, :, -1:, :], tgt[:, :, :-1, :]], axis=-2)

    # 1. call model foreward  (choose 1 or 2)
    preds = model(src_index, src_input, None, decoder_input)
    preds.backward()

    # 2. call ddeint (choose 1 or 2)
    y0 = decoder_input
    preds = ddeint(
        func=model,
        y0=y0,
        t_span=paddle.arange(args.tgt_len + 1),
        lags=src_index,
        his=src,
        his_span=paddle.arange(args.his_len),
        solver=Euler,
    )
    preds = preds[:, :, -args.tgt_len :, :]
    preds.backward()

    from paddleviz.paddleviz.viz import make_graph

    dot = make_graph(preds, dpi="600")
    dot.render("viz-result.gv", format="png", view=False)
