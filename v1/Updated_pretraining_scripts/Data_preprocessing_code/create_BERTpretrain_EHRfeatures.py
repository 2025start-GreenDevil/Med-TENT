import collections
import random
import tensorflow as tf
import pickle
import argparse

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

def create_masked_EHR_predictions(input_seq, masked_lm_prob, max_predictions_per_seq, vocab, rng):
    cand_indexes = list(range(len(input_seq)))
    rng.shuffle(cand_indexes)
    output_tokens = input_seq[:]

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(input_seq) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = 0

        if rng.random() < 0.8:
            masked_token = 0
        else:
            if rng.random() < 0.5:
                masked_token = input_seq[index]
            else:
                masked_token = rng.randint(1, max(vocab.values()))

        output_tokens[index] = masked_token
        masked_lms.append(MaskedLmInstance(index=index, label=input_seq[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lm_positions = [p.index for p in masked_lms]
    masked_lm_labels = [p.label for p in masked_lms]
    return (output_tokens, masked_lm_positions, masked_lm_labels)

def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

def create_float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

def write_EHRinstance_to_example_files(seqs, max_seq_length, max_predictions_per_seq,
                                       masked_lm_prob, vocab, output_files, rng):
    writers = [tf.io.TFRecordWriter(output_file) for output_file in output_files]
    writer_index = 0
    total_written = 0

    for seq_index, seq in enumerate(seqs):
        if len(seq[-2]) > max_seq_length:
            continue
        if seq[3][0] <= 0:
            continue

        input_seq = seq[-2]
        input_mask = [1] * len(input_seq)
        segment_ids = seq[-1]

        input_ids, masked_lm_positions, masked_lm_ids = create_masked_EHR_predictions(
            input_seq, masked_lm_prob, max_predictions_per_seq, vocab, rng)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        masked_lm_weights = [1.0] * len(masked_lm_ids)
        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if max(seq[1]) > 7 else 0

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        features["next_sentence_labels"] = create_int_feature([next_sentence_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)
        total_written += 1

    for writer in writers:
        writer.close()

    print(f"Wrote {total_written} total instances")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--vocab_file", required=True)
    parser.add_argument("--max_seq_length", type=int, default=32)
    parser.add_argument("--max_predictions_per_seq", type=int, default=2)
    parser.add_argument("--masked_lm_prob", type=float, default=0.35)
    parser.add_argument("--random_seed", type=int, default=12345)
    args = parser.parse_args()

    vocab = pickle.load(open(args.vocab_file, 'rb'))
    train_data = []
    with open(args.input_file, 'rb') as f:
        try:
            while True:
                tp = pickle.load(f)
                train_data.extend(tp)
        except EOFError:
            pass

    rng = random.Random(args.random_seed)
    output_files = args.output_file.split(",")
    print("*** Writing to output files ***")
    for output_file in output_files:
        print(f"  {output_file}")

    write_EHRinstance_to_example_files(train_data,
                                       args.max_seq_length,
                                       args.max_predictions_per_seq,
                                       args.masked_lm_prob,
                                       vocab,
                                       output_files,
                                       rng)

if __name__ == "__main__":
    main()
