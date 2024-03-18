from copy import deepcopy

import paddle
import paddle.autograd as autograd
import paddle.nn as nn
from attention import MultiHeadAttentionAwareTemporalContext
from embedding import TrafficFlowEmbedding
from endecoder import Decoder, DecoderLayer, Encoder, EncoderLayer
from graphconv import SpatialAttentionGCN

from paddlexde.functional import ddeint
from paddlexde.interpolation.interpolate import (
    BezierSpline,
    CubicHermiteSpline,
    LinearInterpolation,
)
from paddlexde.solver.fixed_solver import RK4, Euler, Midpoint


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

        self.generator = nn.Linear(
            training_args.d_model, training_args.decoder_output_size
        )

    def encode(self, src, d_idx, m_idx):
        src_dense = self.encoder_embedding(src, d_idx, m_idx)
        encoder_output = self.encoder(src_dense)
        return encoder_output

    def decode(self, encoder_output, tgt, d_idx, m_idx):
        tgt_dense = self.decoder_embedding(tgt, d_idx, m_idx)
        decoder_output = self.decoder(x=tgt_dense, memory=encoder_output)
        return self.generator(decoder_output)

    def endecode(
        self,
        src,
        tgt,
        **kwargs,
    ):
        lags = paddle.cast(kwargs["lags"], dtype=paddle.int64)
        lags = paddle.clip(lags, min=0, max=self.training_args.his_len - 1)
        src_d_idx = paddle.index_select(kwargs["src_d_idx"], lags, axis=2)
        src_m_idx = paddle.index_select(kwargs["src_m_idx"], lags, axis=2)

        memory = self.encode(src, src_d_idx, src_m_idx)
        output = self.decode(memory, tgt, kwargs["tgt_d_idx"], kwargs["tgt_m_idx"])
        return output

    def forward(
        self,
        src: paddle.Tensor,
        src_idx: paddle.Tensor,
        tgt: paddle.Tensor,
        **kwargs,
    ):
        """_summary_

        Args:
            src (_type_): 作为微分方程中encoder输入
            tgt (_type_): 作为微分方程中decoder输入

        Returns:
            _type_: 预测结果
        """
        pred_span = paddle.arange(1 + 1)
        his_span = paddle.arange(self.training_args.his_len)
        output = ddeint(
            func=self.endecode,
            y0=tgt,
            t_span=pred_span,
            lags=src_idx,
            his=src,
            his_span=his_span,
            solver=self.dde_solver,
            fixed_solver_interp="",
            **kwargs,
        )

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
