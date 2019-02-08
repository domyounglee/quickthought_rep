
import import_data

import tensorflow as tf
import numpy as np
import collections

FLAGS = tf.flags.FLAGS

#To make { word : word_vector }
def read_vocab_embs(vocabulary_file, embedding_matrix_file):
  tf.logging.info("Reading vocabulary from %s", vocabulary_file)
  with tf.gfile.GFile(vocabulary_file, mode="r") as f:
    lines = list(f.readlines())
  vocab = [line.strip() for line in lines]
  
  with open(embedding_matrix_file, "r") as f:
    embedding_matrix = np.load(f)
  tf.logging.info("Loaded embedding matrix with shape %s",
                  embedding_matrix.shape)
  word_embedding_dict = collections.OrderedDict(
      zip(vocab, embedding_matrix))
  return word_embedding_dict

#To make { word : id }
def read_vocab(vocabulary_file):
  tf.logging.info("Reading vocabulary from %s", vocabulary_file)
  with tf.gfile.GFile(vocabulary_file, mode="r") as f:
    lines = list(f.readlines())
  reverse_vocab = [line.strip() for line in lines]
  tf.logging.info("Loaded vocabulary with %d words.", len(reverse_vocab))
 
  #tf.logging.info("Loading embedding matrix from %s", embedding_matrix_file)
  # Note: tf.gfile.GFile doesn't work here because np.load() calls f.seek()
  # with 3 arguments.
  word_embedding_dict = collections.OrderedDict( # memory the save order 
      zip(reverse_vocab, list(range(len(reverse_vocab)))))
  return word_embedding_dict
    



class qt(object):
    """Quick thought model """

    def __init__(self, config, mode="train"):
        """Basic setup. The actual TensorFlow graph is constructed in build().

        Args:
        config: Object containing configuration parameters.
        mode: "train", "eval" or "encode".

        Raises:
        ValueError: If mode is invalid.
        """
        if mode not in ["train", "eval", "encode"]:
            raise ValueError("Unrecognized mode: %s" % mode)

        self.config = config
        self.mode = mode

        # Initializer used for non-recurrent weights.
        self.uniform_initializer = tf.random_uniform_initializer(
            minval=-0.1,
            maxval=0.1)

        # Input sentences represented as sequences of word ids. "encode" is the
        # source sentence, "decode_pre" is the previous sentence and "decode_post"
        # is the next sentence.
        # Each is an int64 Tensor with  shape [batch_size, padded_length].
        self.encode_ids = None

        # Boolean masks distinguishing real words (1) from padded words (0).
        # Each is an int32 Tensor with shape [batch_size, padded_length].
        self.encode_mask = None

        # Input sentences represented as sequences of word embeddings.
        # Each is a float32 Tensor with shape [batch_size, padded_length, emb_dim].
        self.encode_emb = None

        # The output from the sentence encoder.
        # A float32 Tensor with shape [batch_size, num_gru_units].
        self.thought_vectors = None

        # The total loss to optimize.
        self.total_loss = None

    def build_inputs(self):

        if self.mode == "encode":
            encode_ids = tf.placeholder(tf.int64, (None, None), name="encode_ids")
            encode_mask = tf.placeholder(tf.int8, (None, None), name="encode_mask")
        else:
            # Prefetch serialized tf.Example protos.

            encode = import_data.prefetch_input_data(FLAGS.input_file_pattern,
                                                    batch_size=FLAGS.batch_size,epochs=FLAGS.nepochs)
            encode_ids = encode.ids
            encode_mask = encode.mask

        self.encode_ids = encode_ids
        self.encode_mask = encode_mask


    def build_word_embeddings(self):

        rand_init = self.uniform_initializer
        self.word_embeddings = []
        self.encode_emb = []
        self.init = None
        for v in self.config.vocab_configs:

            if v.mode == 'fixed': #Use fixed, pre-trained embeddings
                
                if self.mode == "train": #model will be trained
                    #genereate word matrix and assign the pretrained word matrix 
                    self.word_emb = tf.get_variable(  name=v.name, shape=[v.size, v.dim],  trainable=False)
                    embedding_placeholder = tf.placeholder( tf.float32, [v.size, v.dim])
                    embedding_init = self.word_emb.assign(embedding_placeholder) # operator 

                    # 1 x v.dim uniform random vector for pad, unk ,start ,end
                    rand = np.random.rand(4, v.dim)
                    pretrained = np.load(v.embs_file) # load the pretrained vector 
                  
                    assert(pretrained.shape[0] == v.size - 4)
                    word_init = np.concatenate((rand, pretrained), axis=0) #for unknown in index 0
                    self.init = (embedding_init, embedding_placeholder, word_init) #to make a initAssign Function
                        
                else:#In 'eval' mode this is a shell to load the model
                    self.word_emb = tf.get_variable( name=v.name, shape=[v.size, v.dim])

                encode_emb = tf.nn.embedding_lookup(self.word_emb, self.encode_ids) #batch x seq_len x dim
                
                self.encode_emb.extend([encode_emb, encode_emb]) # target(in) , context (out)


            #Train word embeddings from scratch
            if v.mode == 'trained': 
                for inout in ["", "_out"]:
                    word_emb = tf.get_variable( name=v.name + inout, shape=[v.size, v.dim],initializer=rand_init)
                    if self.mode == 'train':
                        self.word_embeddings.append(word_emb)

                    encode_emb = tf.nn.embedding_lookup(word_emb, self.encode_ids)
                    self.encode_emb.append(encode_emb)

            #if you use 'expand' vocabulary in 'encode' mode
            if v.mode == 'expand':
                for inout in ["", "_out"]:
                    encode_emb = tf.placeholder(tf.float32, (
                        None, None, v.dim), v.name + inout)
                    self.encode_emb.append(encode_emb)
                    word_emb_dict = read_vocab_embs(v.vocab_file + inout + ".txt",
                        v.embs_file + inout + ".npy")
                    self.word_embeddings.append(word_emb_dict)

            #if you use 'fixed' vocabulary in 'encode' mode
            if v.mode != 'expand' and self.mode == 'encode':
                word_emb_dict = read_vocab(v.vocab_file)
                self.word_embeddings.extend([word_emb_dict, word_emb_dict])




    def rnn(self, word_embs, mask, scope, encoder_dim, cell_type="GRU"):
        def _initialize_cell(num_units, cell_type="GRU"):
            init=tf.contrib.layers.xavier_initializer()

            if cell_type == "GRU":
                return tf.contrib.rnn.GRUCell(num_units=num_units,kernel_initializer=init)
            elif cell_type == "LSTM":
                return tf.contrib.rnn.LSTMCell(num_units=num_units,initializer=init)
            else:
                raise ValueError("Invalid cell type")


        length = tf.to_int32(tf.reduce_sum(mask, 1), name="length")

        if encoder_dim % 2:
            raise ValueError(
                "encoder_dim must be even when using a bidirectional encoder.")
       

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=_initialize_cell( encoder_dim // 2, cell_type=cell_type),
            cell_bw=_initialize_cell( encoder_dim // 2, cell_type=cell_type),
            inputs=word_embs,
            sequence_length=length,
            dtype=tf.float32,
            scope=scope)
        if cell_type == "LSTM":
            states = [states[0][1], states[1][1]]
        state = tf.concat(states, 1)

        return state

    def build_encoder(self):
        """Builds the sentence encoder.

        Inputs:
        self.encode_emb
        self.encode_mask

        Outputs:
        self.thought_vectors

        Raises:
        ValueError: if config.bidirectional_encoder is True and config.encoder_dim
            is odd.
        """
        names = ["","_out"] #source & context
        self.thought_vectors = []
        print(self.config.encoder)
        for i in range(2):
            with tf.variable_scope("encoder" + names[i]) as scope:
                #sent_rep : batch x (final hidden fw + final hidden bw)
                if self.config.encoder == "gru":
                    sent_rep = self.rnn(self.encode_emb[i], self.encode_mask, scope, self.config.encoder_dim, cell_type="GRU")
                elif self.config.encoder == "lstm":
                    sent_rep = self.rnn(self.encode_emb[i], self.encode_mask, scope, self.config.encoder_dim, cell_type="LSTM")
               
                else:
                    raise ValueError("Invalid encoder")

                thought_vectors = tf.identity(sent_rep, name="thought_vectors")#Return a tensor with the same shape and contents as input.
                self.thought_vectors.append(thought_vectors)


    def build_loss(self):
        """Builds the loss Tensor.

        Outputs:
        self.total_loss
        """

        all_sen_embs = self.thought_vectors #[batch_size, dimension]
    
        if FLAGS.dropout: #drop out some vectors
            mask_shp = [1, self.config.encoder_dim]
            bin_mask = tf.random_uniform(mask_shp) > FLAGS.dropout_rate #[[True ,False ,..]] 
            bin_mask = tf.where(bin_mask, tf.ones(mask_shp), tf.zeros(mask_shp)) #select some vector to zero
            src = all_sen_embs[0] * bin_mask
            dst = all_sen_embs[1] * bin_mask
            scores = tf.matmul(src, dst, transpose_b=True) # batch x batch
        else:
            scores = tf.matmul(all_sen_embs[0], all_sen_embs[1], transpose_b=True) # batch x batch

        # Ignore source sentence
        scores = tf.matrix_set_diag(scores, np.zeros(FLAGS.batch_size)) # replace diagonal to zero

        # Targets
        targets_np = np.zeros((FLAGS.batch_size, FLAGS.batch_size))
        ctxt_sent_pos = list(range(-FLAGS.context_size, FLAGS.context_size + 1)) #[-3,-2,-1,0,1,2,3]
        ctxt_sent_pos.remove(0)
        for ctxt_pos in ctxt_sent_pos:
            targets_np += np.eye(FLAGS.batch_size, k=ctxt_pos) #identity matrix 'k' is the index of diagonal default is 0
            """
            np.eye(4,k=1)
            array(  [[0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.],
                    [0., 0., 0., 0.]])

            """
        targets_np_sum = np.sum(targets_np, axis=1, keepdims=True)
        targets_np = targets_np/targets_np_sum #divide the element by each column sum 
        targets = tf.constant(targets_np, dtype=tf.float32) #Creates a constant tensor.


        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=targets, logits=scores)
    
        loss = tf.reduce_mean(losses)

        tf.summary.scalar("losses/ent_loss", loss)
        self.total_loss = loss

        if self.mode == "eval":
            # Forward and backward scores    
            f_scores = scores[:-1] 
            b_scores = scores[1:]

            f_max = tf.to_int64(tf.argmax(f_scores, axis=1))
            b_max = tf.to_int64(tf.argmax(b_scores, axis=1))

            targets = list(range(FLAGS.batch_size - 1))
            targets = tf.constant(targets, dtype=tf.int64)
            fwd_targets = targets + 1

            names_to_values, names_to_updates = tf.contrib.slim.metrics.aggregate_metric_map({
                "Acc/Fwd Acc":  tf.contrib.slim.metrics.streaming_accuracy(f_max,fwd_targets), 
                "Acc/Bwd Acc":  tf.contrib.slim.metrics.streaming_accuracy(b_max,targets)
            })

            for name, value in names_to_values.items():
                tf.summary.scalar(name, value)

            self.eval_op = list(names_to_updates.values())

    def build(self):
        """Creates all ops for training, evaluation or encoding."""
        self.build_inputs() # connect input to TFRecord files
        self.build_word_embeddings()
        self.build_encoder()
        self.build_loss()

    def build_enc(self):
        """Creates all ops for training, evaluation or encoding."""
        self.build_inputs()
        self.build_word_embeddings()
        self.build_encoder()
