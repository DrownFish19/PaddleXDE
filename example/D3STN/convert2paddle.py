from collections import OrderedDict

import numpy as np
import paddle
import torch
from args import args
from d3stn import D3STN
from utils import get_adjacency_matrix_2direction, norm_adj_matrix

encoder_layer_num = 4
decoder_layer_num = 4

mapping_dict = {
    "src_embed.0.weight": "encoder_dense.weight",
    "src_embed.0.bias": "encoder_dense.bias",
    "src_embed.1.pe": "encode_temporal_position.pe",
    "src_embed.2.embedding.weight": "encode_spatial_position.embedding.weight",
    "src_embed.2.gcn_smooth_layers.0.Theta.weight": "encode_spatial_position.gcn_smooth_layers.0.Theta.weight",
    "src_embed.2.gcn_smooth_layers.0.alpha": "encode_spatial_position.gcn_smooth_layers.0.alpha",  # TODO
    "src_embed.2.gcn_smooth_layers.0.beta": "encode_spatial_position.gcn_smooth_layers.0.beta",  # TODO
    "trg_embed.0.weight": "decoder_dense.weight",
    "trg_embed.0.bias": "decoder_dense.bias",
    "trg_embed.1.pe": "decode_temporal_position.pe",
    "trg_embed.2.embedding.weight": "decode_spatial_position.embedding.weight",
    "trg_embed.2.gcn_smooth_layers.0.Theta.weight": "decode_spatial_position.gcn_smooth_layers.0.Theta.weight",
    "trg_embed.2.gcn_smooth_layers.0.alpha": "decode_spatial_position.gcn_smooth_layers.0.alpha",  # TODO
    "trg_embed.2.gcn_smooth_layers.0.beta": "decode_spatial_position.gcn_smooth_layers.0.beta",  # TODO
    "prediction_generator.weight": "generator.weight",
    "prediction_generator.bias": "generator.bias",
    "encoder.norm.weight": "encoder.norm.weight",
    "encoder.norm.bias": "encoder.norm.bias",
    "decoder.norm.weight": "decoder.norm.weight",
    "decoder.norm.bias": "decoder.norm.bias",
}

en_decoder_mapping_dict = {
    ".src_attn.linears.0.weight": ".src_attn.linears.0.weight",
    ".src_attn.linears.0.bias": ".src_attn.linears.0.bias",
    ".src_attn.linears.1.weight": ".src_attn.linears.1.weight",
    ".src_attn.linears.1.bias": ".src_attn.linears.1.bias",
    ".src_attn.query_conv1Ds_aware_temporal_context.weight": ".src_attn.query_conv.weight",
    ".src_attn.query_conv1Ds_aware_temporal_context.bias": ".src_attn.query_conv.bias",
    ".src_attn.key_conv1Ds_aware_temporal_context.weight": ".src_attn.key_conv.weight",
    ".src_attn.key_conv1Ds_aware_temporal_context.bias": ".src_attn.key_conv.bias",
    ".self_attn.linears.0.weight": ".self_attn.linears.0.weight",
    ".self_attn.linears.0.bias": ".self_attn.linears.0.bias",
    ".self_attn.linears.1.weight": ".self_attn.linears.1.weight",
    ".self_attn.linears.1.bias": ".self_attn.linears.1.bias",
    ".self_attn.conv1Ds_aware_temporal_context.0.weight": ".self_attn.query_conv.weight",
    ".self_attn.conv1Ds_aware_temporal_context.0.bias": ".self_attn.query_conv.bias",
    ".self_attn.conv1Ds_aware_temporal_context.1.weight": ".self_attn.key_conv.weight",
    ".self_attn.conv1Ds_aware_temporal_context.1.bias": ".self_attn.key_conv.bias",
    ".feed_forward_gcn.gcn.alpha": ".feed_forward_gcn.alpha",
    ".feed_forward_gcn.gcn.beta": ".feed_forward_gcn.beta",
    ".feed_forward_gcn.gcn.Theta.weight": ".feed_forward_gcn.linear.weight",
    ".sublayer.0.norm.weight": ".sublayer.0.norm.weight",
    ".sublayer.0.norm.bias": ".sublayer.0.norm.bias",
    ".sublayer.1.norm.weight": ".sublayer.1.norm.weight",
    ".sublayer.1.norm.bias": ".sublayer.1.norm.bias",
    ".sublayer.2.norm.weight": ".sublayer.2.norm.weight",
    ".sublayer.2.norm.bias": ".sublayer.2.norm.bias",
}

for i in range(encoder_layer_num):
    for item in en_decoder_mapping_dict:
        mapping_dict[f"encoder.layers.{i}" + item] = (
            f"encoder.layers.{i}" + en_decoder_mapping_dict[item]
        )

for i in range(decoder_layer_num):
    for item in en_decoder_mapping_dict:
        mapping_dict[f"decoder.layers.{i}" + item] = (
            f"decoder.layers.{i}" + en_decoder_mapping_dict[item]
        )

for key in mapping_dict:
    print(key, mapping_dict[key])


default_dtype = paddle.get_default_dtype()
adj_matrix, _ = get_adjacency_matrix_2direction(args.adj_path, 80)
adj_matrix = paddle.to_tensor(norm_adj_matrix(adj_matrix), default_dtype)
sc_matrix = np.load(args.sc_path)[0, :, :]
sc_matrix = paddle.to_tensor(norm_adj_matrix(sc_matrix), default_dtype)

model = D3STN(args, adj_matrix, sc_matrix)

ckpt = "example/CorrSTN/ckpt/epoch_143.params"
torch_weight = torch.load(ckpt, map_location=torch.device("cpu"))
for param_tensor in torch_weight:
    print(f"{param_tensor} \t {torch_weight[param_tensor].shape}")

paddle_weight = model.state_dict()
for param_tensor in paddle_weight:
    print(f"{param_tensor} \t {paddle_weight[param_tensor].shape}")


new_weight_dict = OrderedDict()
for torch_key in torch_weight:
    paddle_key = mapping_dict[torch_key]
    torch_w = torch_weight[torch_key].numpy()

    if len(torch_w.shape) == 2 and "embedding" not in torch_key:
        new_weight_dict[paddle_key] = torch_w.T
    else:
        new_weight_dict[paddle_key] = torch_w

model.load_dict(new_weight_dict)

his_index = paddle.concat([paddle.arange(0, 12)])
encoder_input = paddle.to_tensor(np.ones([1, 80, 12, 1]), dtype=paddle.float32)
decoder_input = paddle.to_tensor(np.ones([1, 80, 12, 1]), dtype=paddle.float32)
print(model(encoder_input, his_index, decoder_input)[0, -1, :, :])

# for convert model with [hour]
# if output is close follows, the convert is ok.
# [0.29537824]
# [0.35994622]
# [0.40714896]
# [0.4344998 ]
# [0.37601328]
# [0.37747186]
# [0.37033072]
# [0.37977317]
# [0.40246618]
# [0.38128358]
# [0.33716473]
# [0.34575891]


# his_index = paddle.concat([paddle.arange(0, 24)])
# encoder_input = paddle.to_tensor(np.ones([1, 80, 24, 1]), dtype=paddle.float32)
# decoder_input = paddle.to_tensor(np.ones([1, 80, 12, 1]), dtype=paddle.float32)
# print(model(encoder_input, his_index, decoder_input)[0, -1, :, :])

# for convert model with [week, day, hour] input, the conv of corrstn in paddle is
# different from conv of corrstn in torch, which list as follows:
# paddle: conv(concat([week, day, hour]))
# torch: concat([conv(week), conv(day), conv(hour)])
