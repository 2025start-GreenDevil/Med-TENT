import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Layer, Softmax, Reshape
from tensorflow.keras import backend as K

epsilon = 0.1  # label smoothing용

# ===================== Custom Layers =====================
class GatherIndexes(Layer):
    """주어진 인덱스에 해당하는 텐서 조각을 추출"""
    def __init__(self, **kwargs):
        super(GatherIndexes, self).__init__(**kwargs)

    def call(self, inputs):
        sequence_tensor, positions = inputs
        # tf.batch_gather를 사용하여 배치별로 인덱싱을 수행
        return tf.gather(sequence_tensor, positions, batch_dims=1)

    def compute_output_shape(self, input_shape):
        input_tensor_shape, positions_shape = input_shape
        # 출력 shape: [batch_size, num_positions, hidden_size]
        return (input_tensor_shape[0], positions_shape[1], input_tensor_shape[2])

class MaskedLMPredictionHead(Layer):
    """Masked LM을 위한 출력 헤드. 가중치 공유(weight tying)를 처리"""
    def __init__(self, config, embedding_weights, **kwargs):
        super(MaskedLMPredictionHead, self).__init__(**kwargs)
        self.config = config
        self.embedding_weights = embedding_weights

    def build(self, input_shape):
        self.bias = self.add_weight(name='bias', shape=[self.config.vocab_size], initializer='zeros', trainable=True)
        super(MaskedLMPredictionHead, self).build(input_shape)

    def call(self, inputs):
        x = tf.matmul(inputs, self.embedding_weights, transpose_b=True)
        x = tf.nn.bias_add(x, self.bias)
        return x

# ===================== Masked LM (Corrected) =====================
def get_masked_lm_output(config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    gathered_tensor = GatherIndexes()([input_tensor, positions])
    hidden = Dense(
        units=config.hidden_size,
        activation=tf.keras.activations.get(config.hidden_act),
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
        name="cls_predictions_transform_dense"
    )(gathered_tensor)
    hidden = LayerNormalization(epsilon=1e-12, name="cls_predictions_transform_layer_norm")(hidden)
    logits = MaskedLMPredictionHead(config, output_weights, name="cls_predictions_output")(hidden)

    probs = Softmax(axis=-1)(logits)
    log_probs = tf.keras.layers.Lambda(lambda x: tf.math.log(x))(probs)

    label_ids = Reshape((-1,))(label_ids)
    label_weights = Reshape((-1,))(label_weights)
    
    one_hot_labels = tf.keras.layers.Lambda(
        lambda x: tf.one_hot(x, depth=config.vocab_size, dtype=tf.float32),
        output_shape=(None, config.vocab_size)
    )(label_ids)

    per_example_loss = tf.keras.layers.Lambda(
        lambda x: -tf.reduce_sum(x[0] * x[1], axis=-1)
    )([one_hot_labels, log_probs])
    
    label_weights_float = tf.keras.layers.Lambda(
        lambda x: tf.cast(x, dtype=tf.float32)
    )(label_weights)
    
    numerator = tf.keras.layers.Lambda(
        lambda x: tf.reduce_sum(x[0] * x[1])
    )([label_weights_float, per_example_loss])
    
    denominator = tf.keras.layers.Lambda(
        lambda x: tf.reduce_sum(x) + 1e-5
    )(label_weights_float)
    
    loss = numerator / denominator
    return (loss, per_example_loss, log_probs)

# ===================== Next Sentence Prediction (Corrected) =====================
def get_next_sentence_output(config, input_tensor, labels):
    logits = Dense(units=2, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range), name="cls_seq_relationship_output")(input_tensor)
    
    probs = Softmax(axis=-1)(logits)
    log_probs = tf.keras.layers.Lambda(lambda x: tf.math.log(x))(probs)
    
    labels = Reshape((-1,))(labels)
    one_hot_labels = tf.keras.layers.Lambda(
        lambda x: tf.one_hot(x, depth=2, dtype=tf.float32),
        output_shape=(None, 2)
    )(labels)

    per_example_loss = tf.keras.layers.Lambda(
        lambda x: -tf.reduce_sum(x[0] * x[1], axis=-1)
    )([one_hot_labels, log_probs])
    
    # ✅ 수정된 부분: tf.reduce_mean을 Lambda 레이어로 변경
    loss = tf.keras.layers.Lambda(
        lambda x: tf.reduce_mean(x)
    )(per_example_loss)
    
    return (loss, per_example_loss, log_probs)