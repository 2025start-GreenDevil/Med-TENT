# coding=utf-8
# Updated for TensorFlow 2.x

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf

class BertConfig(object):
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
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
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class BertModel(tf.keras.Model):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=False):
        super(BertModel, self).__init__()

        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = tf.shape(input_ids)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=(batch_size, seq_length), dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=(batch_size, seq_length), dtype=tf.int32)

        self.embedding_table = self._create_embedding_layer(config, input_ids, token_type_ids, seq_length)
        self.sequence_output = self._create_encoder(config, self.embedding_table, input_mask)
        self.pooled_output = self._create_pooler(config, self.sequence_output)

    def _create_embedding_layer(self, config, input_ids, token_type_ids, seq_length):
        word_embeddings = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            name="word_embeddings"
        )(input_ids)

        token_type_embeddings = tf.keras.layers.Embedding(
            input_dim=config.type_vocab_size,
            output_dim=config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            name="token_type_embeddings"
        )(token_type_ids)

        position_embeddings = tf.keras.layers.Embedding(
            input_dim=config.max_position_embeddings,
            output_dim=config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            name="position_embeddings"
        )(tf.range(seq_length))

        position_embeddings = tf.expand_dims(position_embeddings, 0)
        embeddings = word_embeddings + token_type_embeddings + position_embeddings
        embeddings = tf.keras.layers.LayerNormalization(epsilon=1e-12)(embeddings)
        embeddings = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)(embeddings)
        return embeddings

    def _create_encoder(self, config, input_tensor, input_mask):
        attention_mask = self._create_attention_mask(input_tensor, input_mask)
        encoder_output = input_tensor
        for i in range(config.num_hidden_layers):
            encoder_output = self._transformer_block(config, encoder_output, attention_mask, name=f"layer_{i}")
        return encoder_output

    def _create_attention_mask(self, input_tensor, input_mask):
        to_mask = tf.cast(tf.expand_dims(input_mask, axis=1), tf.float32)
        broadcast_ones = tf.ones(shape=[tf.shape(input_tensor)[0], tf.shape(input_tensor)[1], 1], dtype=tf.float32)
        return broadcast_ones * to_mask

    def _transformer_block(self, config, input_tensor, attention_mask, name):
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=config.num_attention_heads,
            key_dim=int(config.hidden_size / config.num_attention_heads),
            dropout=config.attention_probs_dropout_prob,
            name=f"{name}_attention"
        )(input_tensor, input_tensor, attention_mask=attention_mask)

        attention_output = tf.keras.layers.Dropout(config.hidden_dropout_prob)(attention_output)
        attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-12)(input_tensor + attention_output)

        intermediate_output = tf.keras.layers.Dense(
            config.intermediate_size,
            activation=tf.keras.activations.get(config.hidden_act),
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            name=f"{name}_intermediate"
        )(attention_output)

        layer_output = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            name=f"{name}_output"
        )(intermediate_output)

        layer_output = tf.keras.layers.Dropout(config.hidden_dropout_prob)(layer_output)
        layer_output = tf.keras.layers.LayerNormalization(epsilon=1e-12)(layer_output + attention_output)

        return layer_output

    def _create_pooler(self, config, sequence_output):
        first_token_tensor = sequence_output[:, 0]
        pooled_output = tf.keras.layers.Dense(
            config.hidden_size,
            activation="tanh",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            name="pooler"
        )(first_token_tensor)
        return pooled_output

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        return self.sequence_output

    def get_embedding_table(self):
        return self.embedding_table
