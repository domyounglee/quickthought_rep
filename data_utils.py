import os
import tensorflow as tf


PAD = "<pad>"
PAD_ID = 0
UNK = "<unk>"
UNK_ID = 1
START="<s>"
START_ID = 2
END="</s>"
END_ID = 3

def load_vocab_file(self, filename):
    # The first three words in both vocab files are special characters:
    # <unk>: unknown word.
    # <s>: start of a sentence.
    # </s>: # end of a sentence.
    # In addition, we add <pad> as a place holder for a padding space.
    vocab_file = os.path.join( filename)

    words = list(map(lambda w: w.strip().lower(), open(vocab_file)))
    words.insert(0, '<pad>')
    words = words[:4] + list(set(words[4:]))  # Keep the special characters on top.
    word2id = {word: i for i, word in enumerate(words)}
    id2word = words

    assert id2word[PAD_ID] == '<pad>'
    assert id2word[UNK_ID] == '<unk>'
    assert id2word[START_ID] == '<s>'
    assert id2word[END_ID] == '</s>'

    return word2id, id2word

def recover_sentence(sent_ids, id2word):
    """Convert a list of word ids back to a sentence string.
    """
    words = list(map(lambda i: id2word[i] if 0 <= i < len(id2word) else '<unk>', sent_ids))

    # Then remove tailing <pad>
    i = len(words) - 1
    while i >= 0 and words[i] == '<pad>':
        i -= 1
    words = words[:i + 1]
    return ' '.join(words)

def int64_feature(value):

    """Helper for creating an Int64 Feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(
        value=[int(v) for v in value]))

def sentence_to_ids(sentence, vocab,case_sensitive):

    """Helper for converting a sentence (list of words) to a list of ids."""
    if case_sensitive:
        ids = [vocab.get(w, UNK_ID) for w in sentence]
    else:
        ids = [vocab.get(w.lower(), UNK_ID) for w in sentence]

    return [START_ID]+ids+[END_ID]

def parse_function(example_proto):
    features = {"features":tf.VarLenFeature( dtype=tf.int64)}
    parsed_features=tf.parse_single_example(example_proto,features)
    return parsed_features["features"]



