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
    "*.self_attn.conv1Ds_aware_temporal_context.0.weight": "*.self_attn.query_conv.weight",
    "*.self_attn.conv1Ds_aware_temporal_context.0.bias": "*.self_attn.query_conv.bias ",
    "*.self_attn.conv1Ds_aware_temporal_context.1.weight": "*.self_attn.key_conv.weight",
    "*.self_attn.conv1Ds_aware_temporal_context.1.bias": "*.self_attn.key_conv.bias",
    "*.feed_forward_gcn.gcn.alpha": "*.feed_forward_gcn.gcn.alpha",  # TODO
    "*.feed_forward_gcn.gcn.beta": "*.feed_forward_gcn.gcn.beta",  # TODO
    "*.feed_forward_gcn.gcn.Theta.weight": "*.feed_forward_gcn.linear.weight",
    "prediction_generator.weight": "generator.weight",
    "prediction_generator.bias": "generator.bias",
}
