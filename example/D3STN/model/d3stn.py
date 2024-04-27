from copy import deepcopy

import paddle
import paddle.nn as nn
from model.attention import MultiHeadAttentionAwareTemporalContext
from model.embedding import TrafficFlowEmbedding
from model.endecoder import Decoder, DecoderLayer, Encoder, EncoderLayer
from model.graphconv import SpatialAttentionGCN

from paddlexde.functional.ddeint import ddeint
from paddlexde.solver import RK4, Euler, Midpoint


class D3STN(nn.Layer):
    def __init__(self, training_args, adj_matrix, sc_matrix):
        super(D3STN, self).__init__()

        self.training_args = training_args

        # 输出当前微分方程使用的优化器函数
        if self.training_args.solver == "euler":
            self.dde_solver = Euler
        elif self.training_args.solver == "midpoint":
            self.dde_solver = Midpoint
        elif self.training_args.solver == "rk4":
            self.dde_solver = RK4

        self.encoder_embedding = TrafficFlowEmbedding(args=training_args)
        self.decoder_embedding = TrafficFlowEmbedding(args=training_args)

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

        self.generator = nn.Linear(training_args.d_model, training_args.output_size)

    def encode(self, src, **kwargs):
        """_summary_

        Args:
            src (_type_): [B,N,T,F],经过DelayIndex选择的数据

        Returns:
            _type_: _description_
        """
        idx = kwargs["src_idx"]
        day_idx = kwargs["src_day_idx"]
        hour_idx = kwargs["src_hour_idx"]

        idx = paddle.cast(idx, dtype=paddle.int64)
        idx = paddle.clip(idx, min=0, max=self.training_args.his_len - 1)
        day_idx = paddle.index_select(day_idx, idx, axis=2)
        hour_idx = paddle.index_select(hour_idx, idx, axis=2)
        src_dense = self.encoder_embedding(src, idx, day_idx, hour_idx)

        encoder_output = self.encoder(src_dense)
        return encoder_output

    def decode(self, encoder_output, y0, t0, t1, **kwargs):
        idx = kwargs["tgt_idx"]
        day_idx = kwargs["tgt_day_idx"]
        hour_idx = kwargs["tgt_hour_idx"]

        lookup_index = paddle.to_tensor([t0], dtype="int32")
        idx = paddle.bucketize(lookup_index, idx)
        day_idx = paddle.index_select(day_idx, idx, axis=2)
        hour_idx = paddle.index_select(hour_idx, idx, axis=2)
        tgt_dense = self.decoder_embedding(y0, idx, day_idx, hour_idx)

        decoder_output = self.decoder(x=tgt_dense, memory=encoder_output)
        return self.generator(decoder_output)

    def forward(
        self,
        src: paddle.Tensor,
        src_idx: paddle.Tensor,
        src_day_idx: paddle.Tensor,
        src_hour_idx: paddle.Tensor,
        tgt: paddle.Tensor,
        tgt_idx: paddle.Tensor,
        tgt_day_idx: paddle.Tensor,
        tgt_hour_idx: paddle.Tensor,
        **kwargs,
    ):
        """_summary_

        Args:
            src (paddle.Tensor): 延迟输入数据 [B,N,T,F]
            src_idx (paddle.Tensor): 延迟输入数据的idx=[0,1,2,...]
            src_day_idx (paddle.Tensor): 延迟输入数据周idx=[0,0,0,...], min=0, max=6
            src_hour_idx (paddle.Tensor): 延迟输入数据小时idx=[0,0,0,...], min=0, max=23
            tgt (paddle.Tensor): 预测数据的idx=[0,1,2,...]
            tgt_idx (paddle.Tensor): 预测数据 [B,N,T,F]
            tgt_day_idx (paddle.Tensor): 预测数据周idx=[0,0,0,...], min=0, max=6
            tgt_hour_idx (paddle.Tensor): 预测数据小时idx=[0,0,0,...], min=0, max=23

        Returns:
            _type_: _description_
        """

        kwargs = {
            "src_idx": src_idx,
            "src_day_idx": src_day_idx,
            "src_hour_idx": src_hour_idx,
            "tgt_idx": tgt_idx,
            "tgt_day_idx": tgt_day_idx,
            "tgt_hour_idx": tgt_hour_idx,
        }

        y0 = src[:, :, -1:, :]
        delay_len = src.shape[2]
        output = ddeint(
            drift_f=self.decoder,
            delay_f=self.encoder,
            y0=y0,
            y_t_span=tgt_idx,
            delay_t_span=src_idx,
            delay=src,
            delay_t=paddle.arange(delay_len, dtype=paddle.float32),
            solver=self.dde_solver,
            fixed_solver_interp="",
            **kwargs,
        )

        return output
