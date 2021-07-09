{
  parseBool(s): std.asciiLower(s) == 'true',

  parsePaths(s):
    local paths = std.split(s, ',');
    if std.length(paths) == 1 then paths[0] else paths,

  devices(device_indices): {
    device_ids: [
      std.parseInt(device)
      for device in std.split(device_indices, ',')
    ],
    use_multi_devices: std.length(self.device_ids) > 1,
  },

  bert_config: {
    hidden_size: 768,
    num_hidden_layers: 12,
    num_attention_heads: 12,
    intermediate_size: 3072,
    hidden_act: 'gelu',
    hidden_dropout_prob: 0.1,
    attention_probs_dropout_prob: 0.1,
    max_position_embeddings: 512,
    type_vocab_size: 2,
    initializer_range: 0.02,
    layer_norm_eps: 1e-12,
  },

}
