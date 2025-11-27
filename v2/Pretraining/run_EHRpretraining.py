# This code is adapted from work by lrasmy (Zhilab), originally dated 2019-08-10.

import os
import argparse
import tensorflow as tf
from modeling import BertConfig, BertModel
from optimization import create_optimizer
import json
from run_EHRpretraining_utils import get_masked_lm_output, get_next_sentence_output

# Utilities 
def decode_record(record, name_to_features):
    example = tf.io.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t
    return example

def create_pretrain_dataset(file_pattern, seq_length, max_predictions_per_seq, batch_size):
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "masked_lm_positions": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights": tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64),
    }

    files = tf.io.gfile.glob(file_pattern)
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(lambda r: decode_record(r, name_to_features))
    dataset = dataset.shuffle(100).repeat().batch(batch_size, drop_remainder=True) # drop_remainder 추가
    return dataset

def build_pretrain_model(config, seq_length, max_predictions_per_seq):
    input_ids = tf.keras.Input(shape=(seq_length,), dtype=tf.int32, name="input_ids")
    input_mask = tf.keras.Input(shape=(seq_length,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(seq_length,), dtype=tf.int32, name="segment_ids")
    masked_lm_positions = tf.keras.Input(shape=(max_predictions_per_seq,), dtype=tf.int32, name="masked_lm_positions")
    masked_lm_ids = tf.keras.Input(shape=(max_predictions_per_seq,), dtype=tf.int32, name="masked_lm_ids")
    masked_lm_weights = tf.keras.Input(shape=(max_predictions_per_seq,), dtype=tf.float32, name="masked_lm_weights")
    next_sentence_labels = tf.keras.Input(shape=(1,), dtype=tf.int32, name="next_sentence_labels")

    # BertModel을 먼저 빌드
    bert_model = BertModel(config, is_training=True, input_ids=input_ids,
                           input_mask=input_mask, token_type_ids=segment_ids)

    sequence_output = bert_model.get_sequence_output()
    pooled_output = bert_model.get_pooled_output()
    
    # bert_model의 word_embedding_layer 속성에 직접 접근하여 가중치를 가져옴
    embedding_table = bert_model.word_embedding_layer.embeddings


    (masked_lm_loss, _, _) = get_masked_lm_output(
        config, sequence_output, embedding_table, masked_lm_positions,
        masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, _, _) = get_next_sentence_output(
        config, pooled_output, next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss

    # 최종 모델 이름 지정
    final_model = tf.keras.Model(
        inputs=[input_ids, input_mask, segment_ids, masked_lm_positions,
                masked_lm_ids, masked_lm_weights, next_sentence_labels],
        outputs=total_loss,
        name="bert_pretrain_model")
    
    final_model.bert_model = bert_model

    return final_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bert_config_file", required=True)
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--max_predictions_per_seq", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_steps", type=int, default=100000)
    parser.add_argument("--num_warmup_steps", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.bert_config_file, "r") as f:
        bert_config_dict = json.load(f)
    bert_config = BertConfig.from_dict(bert_config_dict)

    dataset = create_pretrain_dataset(args.input_file,
                                      args.max_seq_length,
                                      args.max_predictions_per_seq,
                                      args.train_batch_size)

    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model = build_pretrain_model(bert_config,
                                     args.max_seq_length,
                                     args.max_predictions_per_seq)

        optimizer = create_optimizer(args.learning_rate,
                                     args.num_train_steps,
                                     args.num_warmup_steps)

        train_loss = tf.keras.metrics.Mean(name="train_loss")

    @tf.function
    def train_step(features):
        with tf.GradientTape() as tape:
            loss = model(features, training=True)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss.update_state(loss)

    print("***** Starting training loop *****")
    for step, batch in enumerate(dataset.take(args.num_train_steps)):
        train_step(batch)
        if step % 100 == 0:
            print(f"Step {step} - Loss: {train_loss.result().numpy():.4f}")

    print("***** Saving final model weights in both formats *****")

    # Keras HDF5 (.weights.h5) 형식으로 저장 
    h5_path = os.path.join(args.output_dir, "model.weights.h5")
    model.save_weights(h5_path)
    print(f"Weights saved in Keras HDF5 format to: {h5_path}")

    # TensorFlow Checkpoint (.ckpt) 형식으로 저장
    checkpoint = tf.train.Checkpoint(model=model)
    ckpt_path_prefix = os.path.join(args.output_dir, "model.ckpt")
    latest_ckpt = checkpoint.save(file_prefix=ckpt_path_prefix)
    print(f"Weights saved in Checkpoint format to: {latest_ckpt}")

if __name__ == "__main__":
    main()