import tensorflow as tf

def get_masked_lm_output(config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    input_tensor = gather_indexes(input_tensor, positions)

    hidden = tf.keras.layers.Dense(
        units=config.hidden_size,
        activation=tf.keras.activations.get(config.hidden_act),
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
        name="cls/predictions/transform/dense"
    )(input_tensor)

    hidden = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="cls/predictions/transform/layer_norm")(hidden)

    output_bias = tf.Variable(tf.zeros([config.vocab_size]), name="cls/predictions/output_bias")
    logits = tf.keras.layers.Dense(
      units=config.vocab_size,
      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
      name="cls/predictions/output_weights"
    )(hidden)
    logits = tf.nn.bias_add(logits, output_bias)

    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(label_ids, depth=config.vocab_size, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=-1)
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

    return (loss, per_example_loss, log_probs)

def get_next_sentence_output(config, input_tensor, labels):
    output_weights = tf.keras.layers.Dense(
        units=2,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
        use_bias=False,
        name="cls/seq_relationship/output_weights"
    )(input_tensor)

    output_bias = tf.Variable(tf.zeros([2]), name="cls/seq_relationship/output_bias")
    logits = output_weights + output_bias
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, log_probs)

def gather_indexes(sequence_tensor, positions):
    batch_size = tf.shape(sequence_tensor)[0]
    seq_length = tf.shape(sequence_tensor)[1]
    width = tf.shape(sequence_tensor)[2]

    flat_offsets = tf.reshape(tf.range(batch_size) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor
