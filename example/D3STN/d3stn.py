from copy import deepcopy

import paddle
from attention import MultiHeadAttentionAwareTemporalContext
from embedding import AdaptiveEmbedding, TemporalSectionEmbedding
from endecoder import Decoder, DecoderLayer, Encoder, EncoderLayer
from graphconv import SpatialAttentionGCN
from paddle import autograd, nn

from paddlexde.interpolation.interpolate import (
    BezierSpline,
    CubicHermiteSpline,
    LinearInterpolation,
)


class D3STN(nn.Layer):
    def __init__(self, training_args, adj_matrix, sc_matrix):
        super(D3STN, self).__init__()

        self.training_args = training_args

        self.encoder_dense = nn.Linear(
            training_args.encoder_input_size, training_args.d_proj
        )
        self.decoder_dense = nn.Linear(
            training_args.decoder_input_size, training_args.d_proj
        )
        self.temporal_section_week = TemporalSectionEmbedding(training_args, 7, axis=1)
        self.temporal_section_day = TemporalSectionEmbedding(training_args, 288, axis=2)
        if training_args.d_adaptive > 0:
            self.adaptive_embedding_encoder = AdaptiveEmbedding(training_args)
            self.adaptive_embedding_decoder = AdaptiveEmbedding(training_args)

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

    def encode(self, src):
        src_dense = self.encoder_dense(src[..., :1])
        embed_list = [src_dense]

        week_embedding = self.temporal_section_week(src)
        day_embedding = self.temporal_section_day(src)
        embed_list.append(week_embedding)
        embed_list.append(day_embedding)

        if self.training_args.d_adaptive > 0:
            adp_embedding = self.adaptive_embedding_encoder(src_dense)
            embed_list.append(adp_embedding)

        embed = paddle.concat(embed_list, axis=-1)

        encoder_output = self.encoder(embed)
        return encoder_output

    def decode(self, encoder_output, tgt):
        tgt_dense = self.decoder_dense(tgt[..., :1])
        embed_list = [tgt_dense]

        week_embedding = self.temporal_section_week(tgt)
        day_embedding = self.temporal_section_day(tgt)
        embed_list.append(week_embedding)
        embed_list.append(day_embedding)

        if self.training_args.d_adaptive > 0:
            adp_embedding = self.adaptive_embedding_encoder(tgt_dense)
            embed_list.append(adp_embedding)

        embed = paddle.concat(embed_list, axis=-1)

        decoder_output = self.decoder(x=embed, memory=encoder_output)
        return self.generator(decoder_output)

    def forward(self, src, tgt):
        encoder_output = self.encode(src)
        output = self.decode(encoder_output, tgt)
        return output


class DecoderIndex(autograd.PyLayer):
    @staticmethod
    def forward(ctx, lags, his, his_span, interp_method="cubic"):
        """
        计算给定输入序列的未来值，并返回计算结果。
        传入lags, history,
        计算序列位置对应位置的梯度, 并保存至backward

        Args:
            ctx (): 动态图计算上下文对象。
            xde (): 未来值的输入序列, BaseXDE类型。
            lags (paddle.Tensor): 用多少个过去的值来计算未来的这个值（未来值的滞后量）。
            history (paddle.Tensor): 用于计算未来值的过去输入序列。
            interp_method (str, optional): 插值方法，取值为 "linear"（线性插值）,"cubic"（三次样条插值）或 "bez"（贝塞尔插值）。默认为 "linear"。

        Returns:
            paddle.Tensor: 计算结果，形状为 [batch_size, len_t, dims]。

        Raises:
            NotImplementedError: 如果interp_method不是上述三种情况之一, 将抛出NotImplementedError异常。
        """
        with paddle.no_grad():
            if interp_method == "linear":
                interp = LinearInterpolation(his, his_span)
            elif interp_method == "cubic":
                interp = CubicHermiteSpline(his, his_span)
            elif interp_method == "bez":
                interp = BezierSpline(his, his_span)
            else:
                raise NotImplementedError

            y_lags = interp.evaluate(lags)

            derivative_lags = interp.derivative(lags)
            ctx.save_for_backward(derivative_lags)

        return y_lags

    @staticmethod
    def backward(ctx, grad_y):
        # 计算history相应的梯度，并提取forward中保存的梯度，用于计算lag的梯度
        # 在计算的过程中，无需更新history，仅更新lags即可
        (derivative_lags,) = ctx.saved_tensor()
        grad = grad_y * derivative_lags
        grad = paddle.sum(grad, axis=[0, 1, 3])
        return grad, None, None
        # return None, grad_y_lags * derivative_lags, None, None, None
