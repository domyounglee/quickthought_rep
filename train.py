"""Train the Quick-thoughts model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import json

import configuration
import qt_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", None,
                       "File pattern of sharded TFRecord files containing")
tf.flags.DEFINE_string("train_dir", None,
                       "Directory for saving and loading checkpoints.")
tf.flags.DEFINE_integer("batch_size", 400, "Batch size")
tf.flags.DEFINE_float("learning_rate", 0.0005, "Learning rate")
tf.flags.DEFINE_integer("learning_rate_decay_steps", 400000, "Learning rate decay steps")
tf.flags.DEFINE_float("clip_gradient_norm", 5.0, "Gradient clipping norm")
tf.flags.DEFINE_integer("save_model_secs", 600, "Checkpointing frequency")
tf.flags.DEFINE_integer("save_summaries_secs", 600, "Summary frequency")
tf.flags.DEFINE_integer("nepochs", 1, "Number of epochs")
tf.flags.DEFINE_integer("num_train_inst", 25813172, "Number of training instances")
tf.flags.DEFINE_integer("sequence_length", 30, "Max sentence length considered")
tf.flags.DEFINE_integer("context_size", 1, "Prediction context size")
tf.flags.DEFINE_boolean("dropout", False, "Use dropout")
tf.flags.DEFINE_float("dropout_rate", 0.3, "Dropout rate")
tf.flags.DEFINE_string("model_config", None, "Model configuration json")
tf.flags.DEFINE_integer("max_ckpts", 5, "Max number of ckpts to keep")
tf.flags.DEFINE_string("Glove_path", None, "Path to Glove dictionary")

tf.logging.set_verbosity(tf.logging.INFO)

def main(_):

    with open(FLAGS.model_config) as json_config_file:
        model_config = json.load(json_config_file)

    model_config = configuration.model_config(model_config, mode="train")
    
    tf.logging.info("Building training graph.")
    g = tf.Graph()
    with g.as_default():
        model = qt_model.qt(model_config, mode="train")
        model.build()

        train_tensor = tf.contrib.slim.learning.create_train_op(
            total_loss=model.total_loss,
            optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate),
            clip_gradient_norm=FLAGS.clip_gradient_norm)
            #global_step=model.global_step,

        if FLAGS.max_ckpts != 5:
            saver = tf.train.Saver(max_to_keep=FLAGS.max_ckpts)
        else:
            saver = tf.train.Saver()

    load_words = model.init
    if load_words:
        def InitAssignFn(sess):
            sess.run(load_words[0], {load_words[1]: load_words[2]})


    nsteps = int(FLAGS.nepochs * (FLAGS.num_train_inst / FLAGS.batch_size))
    tf.contrib.slim.learning.train(
        train_op=train_tensor,
        logdir=FLAGS.train_dir,
        graph=g,
        number_of_steps=nsteps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        saver=saver,
        save_interval_secs=FLAGS.save_model_secs, 
        init_fn=InitAssignFn if load_words else None
    )


if __name__ == "__main__":
    tf.flags.mark_flag_as_required("input_file_pattern")
    tf.flags.mark_flag_as_required("train_dir")
    tf.flags.mark_flag_as_required("model_config")
    tf.app.run()


