# This code is adapted from work by lrasmy (Zhilab), originally dated 2019-08-10.

import copy
import json
import tensorflow as tf
import six


class BertConfig(object):
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=3072,
                 hidden_act="gelu", hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1, max_position_embeddings=512,
                 type_vocab_size=16, initializer_range=0.02):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        config = cls(vocab_size=None)
        for key, value in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with tf.io.gfile.GFile(json_file, "r") as reader:
            return cls.from_dict(json.loads(reader.read()))

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(tf.keras.Model):
    def __init__(self, config, is_training, input_ids, input_mask=None, token_type_ids=None, **kwargs):
        super(BertModel, self).__init__(**kwargs)

        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        # batch_size, seq_length KerasTensor
        input_shape = tf.keras.layers.Lambda(lambda x: tf.shape(x))(input_ids)

        if input_mask is None:
            input_mask = tf.keras.layers.Lambda(
                lambda s: tf.ones([s[0], s[1]], dtype=tf.int32)
            )(input_shape)

        if token_type_ids is None:
            token_type_ids = tf.keras.layers.Lambda(
                lambda s: tf.zeros([s[0], s[1]], dtype=tf.int32)
            )(input_shape)

        # 단어 임베딩 레이어를 self.word_embedding_layer에 저장하여 나중에 접근할 수 있도록 함
        self.word_embedding_layer = tf.keras.layers.Embedding(
            config.vocab_size, 
            config.hidden_size,
            embeddings_initializer=create_initializer(config.initializer_range), 
            name="word_embeddings"
        )

        self.embedding_output = self._create_embedding_layer(config, input_ids, token_type_ids)
        self.sequence_output = self._create_encoder(config, self.embedding_output, input_mask)
        self.pooled_output = self._create_pooler(config, self.sequence_output)

    def _create_embedding_layer(self, config, input_ids, token_type_ids):
        word_embeddings = tf.keras.layers.Embedding(
            config.vocab_size, config.hidden_size,
            embeddings_initializer=create_initializer(config.initializer_range),
            name="word_embeddings"
        )(input_ids)

        word_embeddings = self.word_embedding_layer(input_ids)

        token_type_embeddings = tf.keras.layers.Embedding(
            config.type_vocab_size, config.hidden_size,
            embeddings_initializer=create_initializer(config.initializer_range),
            name="token_type_embeddings"
        )(token_type_ids)

        # position indices를 Lambda 안에서 생성
        position_indices = tf.keras.layers.Lambda(lambda x: tf.range(tf.shape(x)[1]))(input_ids)
        position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings, config.hidden_size,
            embeddings_initializer=create_initializer(config.initializer_range),
            name="position_embeddings"
        )(position_indices)
        position_embeddings = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 0))(position_embeddings)

        embeddings = tf.keras.layers.Add()([word_embeddings, token_type_embeddings, position_embeddings])
        embeddings = tf.keras.layers.LayerNormalization(epsilon=1.e-12)(embeddings)
        embeddings = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)(embeddings)
        return embeddings

    def _create_encoder(self, config, input_tensor, input_mask):
        attention_mask = self._create_attention_mask_from_input_mask(input_tensor, input_mask)
        layer_output = input_tensor
        for i in range(config.num_hidden_layers):
            layer_output = TransformerBlock(config, name=f"transformer_layer_{i}")(
                layer_output, attention_mask
            )
        return layer_output

    def _create_attention_mask_from_input_mask(self, from_tensor, to_mask):
        # batch_size, from_seq_length 동적 추출
        from_shape = tf.keras.layers.Lambda(lambda x: tf.shape(x))(from_tensor)

        # to_mask reshape & cast Lambda 안에서 처리
        to_mask_processed = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.reshape(x, [tf.shape(x)[0], 1, tf.shape(x)[1]]), tf.float32)
        )(to_mask)

        # broadcast ones Lambda
        broadcast_ones = tf.keras.layers.Lambda(
            lambda x: tf.ones([x[0], x[1], 1], dtype=tf.float32)
        )(from_shape)

        # multiply
        mask = tf.keras.layers.Multiply()([broadcast_ones, to_mask_processed])
        return mask

    def _create_pooler(self, config, sequence_output):
        first_token_tensor = sequence_output[:, 0]
        return tf.keras.layers.Dense(
            config.hidden_size, activation="tanh",
            kernel_initializer=create_initializer(config.initializer_range),
            name="pooler_dense"
        )(first_token_tensor)

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        return self.sequence_output

    def get_embedding_table(self):
        return self.embedding_output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=config.num_attention_heads,
            key_dim=int(config.hidden_size / config.num_attention_heads),
            dropout=config.attention_probs_dropout_prob
        )
        self.dropout1 = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.intermediate_dense = tf.keras.layers.Dense(
            config.intermediate_size,
            activation=tf.keras.activations.get(config.hidden_act),
            kernel_initializer=create_initializer(config.initializer_range)
        )
        self.output_dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=create_initializer(config.initializer_range)
        )
        self.dropout2 = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.add1 = tf.keras.layers.Add()
        self.add2 = tf.keras.layers.Add()

    def call(self, input_tensor, attention_mask):
        attention_output = self.attention(input_tensor, input_tensor, attention_mask=attention_mask)
        attention_output = self.dropout1(attention_output)
        attention_output = self.layernorm1(self.add1([input_tensor, attention_output]))
        intermediate_output = self.intermediate_dense(attention_output)
        layer_output = self.output_dense(intermediate_output)
        layer_output = self.dropout2(layer_output)
        layer_output = self.layernorm2(self.add2([layer_output, attention_output]))
        return layer_output


def create_initializer(initializer_range=0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)