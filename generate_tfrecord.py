"""
disclaimer : This source is from the code of model skip thought

Converts a set of text files to TFRecord format with Example protos.

Each Example proto in the output contains the following fields:

  decode_pre: list of int64 ids corresponding to the "previous" sentence.
  encode: list of int64 ids corresponding to the "current" sentence.
  decode_post: list of int64 ids corresponding to the "post" sentence.

In addition, the following files are generated:

  vocab.txt: List of "<word> <id>" pairs, where <id> is the integer
             encoding of <word> in the Example protos.
  word_counts.txt: List of "<word> <count>" pairs, where <count> is the number
                   of occurrences of <word> in the input files.

The vocabulary of word ids is constructed from the top --num_words by word
count. All other words get the <unk> word id.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf
import data_utils 

log = tf.logging
flags = tf.flags
FLAGS = tf.flags.FLAGS

flags.DEFINE_string("input_files", None,
                       "Comma-separated list of globs matching the input "
                       "files. The format of the input files is assumed to be "
                       "a list of newline-separated sentences, where each "
                       "sentence is already tokenized.")

flags.DEFINE_string("vocab_file", "",
                       "(Optional) existing vocab file. Otherwise, a new vocab "
                       "file is created and written to the output directory. "
                       "The file format is a list of newline-separated words, "
                       "where the word id is the corresponding 0-based index "
                       "in the file.")

flags.DEFINE_string("output_dir", None, "Output directory.")

flags.DEFINE_integer("train_output_shards", 100,
                        "Number of output shards for the training set.")

flags.DEFINE_integer("validation_output_shards", 1,
                        "Number of output shards for the validation set.")

flags.DEFINE_integer("num_validation_sentences", 50000,
                        "Number of output shards for the validation set.")

flags.DEFINE_integer("num_words", 20000,
                        "Number of words to include in the output.")

flags.DEFINE_integer("max_sentences", 0,
                        "If > 0, the maximum number of sentences to output.")

flags.DEFINE_integer("max_sentence_length", 30,
                        "If > 0, exclude sentences whose encode, decode_pre OR"
                        "decode_post sentence exceeds this length.")

flags.DEFINE_boolean("case_sensitive", True,
                        "Use case sensitive vocabulary")

log.set_verbosity(log.INFO)

PAD = "<pad>"
PAD_ID = 0
UNK = "<unk>"
UNK_ID = 1
START="<s>"
START_ID = 2
END="</s>"
END_ID = 3

def build_vocab(input_files):
    """Loads or builds the model vocabulary.

    Args:
    input_files: List of pre-tokenized input .txt files.

    Returns:
    vocab: A dictionary of word to id.
    """

    if FLAGS.vocab_file:
        log.info("Loading existing vocab file")
        vocab = collections.OrderedDict()

        with tf.gfile.GFile(FLAGS.vocab_file,mode="r") as f:
            for i, line in enumerate(f):

                word = line.decode("utf-8").strip()
                if word in vocab: print("Duplicate",word)
                vocab[word] =i
        log.info("READ VOCAB SIZE %d FROM %d",len(vocab),FLAGS.vocab_file)
        return vocab
    
    
    wordcount = collections.Counter()
    for input_file in input_files:
        log.info("Processing file %s", input_file)
        for sent in tf.gfile.GFile(input_file):
            wordcount.update(sent.split())

    words = list(wordcount.keys())
    freqs = list(wordcount.values())
    sorted_indices = np.argsort(freqs)[::-1]

    vocab = collections.OrderedDict()
    vocab[UNK]=UNK_ID
    vocab[PAD]=PAD_ID
    vocab[START]=START_ID
    vocab[END]=END_ID


    for w_id, w_index in enumerate(sorted_indices[0:FLAGS.num_words -2]):
        vocab[words[w_index]] = w_id + 4

    log.info("Created vocab with %d words", len(vocab))

    vocab_file = os.path.join(FLAGS.output_dir, "vocab.txt")
    with tf.gfile.GFile(vocab_file,"w") as f:
        f.write("\n".join(vocab.keys()))
    log.info("Wrote vocab file to %s", vocab_file)

    return vocab 



def _process_input_file(filename, vocab, stats):
    """Processes the sentences in an input file.

    Args:
        filename: Path to a pre-tokenized input .txt file.
        vocab: A dictionary of word to id.

    Returns:
        processed: A list of serialized Example protos
    """
    def _create_serialized_example(current, vocab):
        """Helper for creating a serialized Example proto."""
        #key is "features", value is Int64 

        example = tf.train.Example(features=tf.train.Features(feature={
            "features": data_utils.int64_feature(data_utils.sentence_to_ids(current, vocab,FLAGS.case_sensitive)),
        }))

        return example.SerializeToString()


    log.info("Processing input file: %s", filename)
    processed = []

    for sentence_str in tf.gfile.GFile(filename):
        if stats["sentence_count"]%50000==0:
            print(sentence_str)
            print(data_utils.sentence_to_ids(sentence_str, vocab,FLAGS.case_sensitive))
        sentence_tokens = sentence_str.split()

        sentence_tokens = sentence_tokens[:FLAGS.max_sentence_length]

        serialized = _create_serialized_example(sentence_tokens, vocab) #generate serialized example of each sentence 
        processed.append(serialized)
        stats.update(["sentence_count"])

    log.info("Completed processing file %s", filename)
    return processed


def _write_dataset(name, dataset, indices, num_shards):
    """Writes a sharded TFRecord dataset.

    Args:
        name: Name of the dataset (e.g. "train").
        dataset: List of serialized Example protos.
        indices: List of indices of 'dataset' to be written.
        num_shards: The number of output shards.
    """
    def _write_shard(filename, dataset, indices):
        """Writes a TFRecord shard."""
        with tf.python_io.TFRecordWriter(filename) as writer:
            for j in indices:
                writer.write(dataset[j])

    borders = np.int32(np.linspace(0, len(indices), num_shards + 1))

    for i in range(num_shards):
        filename = os.path.join(FLAGS.output_dir, "%s-%.5d-of-%.5d" % (name, i,
                                                                    num_shards))
        shard_indices = indices[borders[i]:borders[i + 1]]
        _write_shard(filename, dataset, shard_indices)
        log.info("Wrote dataset indices [%d, %d) to output shard %s",
                        borders[i], borders[i + 1], filename)
    log.info("Finished writing %d sentences in dataset %s.",
                    len(indices), name)


def main(_):
    #get input files
    input_files=[]
    for pattern in FLAGS.input_files.split(","):
        match = tf.gfile.Glob(pattern)
        if not match: raise ValueError("no %s pattern" %pattern)
        input_files.extend(match)
    log.info("%d input files",len(input_files))

    #make vocab
    log.info("Generating vocab.")
    vocab = build_vocab(input_files)

    #make dataset
    log.info("Generating dataset.")
    stats = collections.Counter() #sentence counter 
    dataset = []
    for filename in input_files:
        dataset.extend(_process_input_file(filename, vocab, stats)) #stats called by reference
        if FLAGS.max_sentences and stats["sentence_count"] >= FLAGS.max_sentences:
            break
    log.info("Generated dataset with %d sentences.", len(dataset))

    #write dataset
    log.info("Write the dataset.")
    indices = range(len(dataset))
    val_indices = indices[:FLAGS.num_validation_sentences] #not shuffled 
    train_indices = indices[FLAGS.num_validation_sentences:] #not shuffled 
    
    _write_dataset("train", dataset, train_indices, FLAGS.train_output_shards)
    _write_dataset("validation", dataset, val_indices,
                    FLAGS.validation_output_shards)

if __name__ == "__main__":
    flags.mark_flag_as_required("input_files")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()





        



