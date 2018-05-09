# borrowed largely from Tensorflow 2016: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/text_classification.py
"""Example of Estimator for DNN-based text classification with DBpedia data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import numpy as np
from sklearn import metrics
import tensorflow as tf

from utils import data_utils

from model import SkipLSTMCell, RecurrentResidualCell

all_models_dir = "checkpoints/"

FLAGS = None

GLOBAL_BATCH_SIZE = 16
EMBEDDING_SIZE = 256
HIDDEN_DIM = 256
VOCAB_SIZE = 0
NUM_CLASSES = 0
OVERFIT_NUM = 50

# For now, let's chop out anything that's too gigantic from the dataset.
# Later, we can figure out some kind of wise binning strategy for making more uniform batches.
max_data_length = 1000

def estimator_spec_for_softmax_classification(logits, labels, mode):
  """Returns EstimatorSpec instance for softmax classification."""
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        })

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
  
  acc = tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
  eval_metric_ops = {'accuracy': acc}
  tf.summary.scalar('accuracy', acc[1])
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def rnn_model(features, labels, mode):
  """RNN model to predict from sequence of words to a class."""

  data = features["data"]
  length = features["length"]

  # Convert indexes of words into embeddings.
  # This creates embeddings matrix of [VOCAB_SIZE, EMBEDDING_SIZE] and then
  # maps word indexes of the sequence into [batch_size, sequence_length,
  # EMBEDDING_SIZE].
  word_vectors = tf.contrib.layers.embed_sequence(
      data, vocab_size=VOCAB_SIZE, embed_dim=EMBEDDING_SIZE)

  # Create an LSTM cell with hidden size of HIDDEN_DIM.
  if (FLAGS.cell_type == 'baseline'):
    cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_DIM)
  elif (FLAGS.cell_type == 'skip'):
    cell = SkipLSTMCell(HIDDEN_DIM, n_skip=10)
  elif (FLAGS.cell_type == 'rrn'):
    cell = RecurrentResidualCell(HIDDEN_DIM, k_depth=2)
  else:
    cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_DIM)

  # Create a dynamic RNN and pass in a function to compute sequence lengths.
  _, state = tf.nn.dynamic_rnn(
                cell,
                word_vectors,
                dtype=tf.float32,
                sequence_length=length)

  encoding = state[0]

  # Given encoding of RNN, take encoding of last step (e.g hidden size of the
  # neural network of last step) and pass it as features for softmax
  # classification over output classes.
  logits = tf.layers.dense(encoding, NUM_CLASSES, activation=None)
  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)

# Given a list of sentences that form the dataset, return the vocab size
def get_vocab_size(data):
  max_word_index = 0
  for sen in data:
      for word in sen:
          max_word_index = max(max_word_index, word)
  vocab_size = max_word_index + 1
  return vocab_size

def get_num_classes(labels):
  max_label = 0
  for label in labels:
      max_label = max(max_label, label)
  num_classes = max_label + 1
  return num_classes

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  data_path = data_utils.get_data_path(FLAGS.dataset)
  print('Accessing training data at path:', data_path)
  if (FLAGS.dataset == 'PMNIST'):
    train, val, test = data_utils.load_mnist(data_path, maxlen=max_data_length)
  else:
    train, val, test = data_utils.load_data(data_path, maxlen=max_data_length)
  global VOCAB_SIZE, NUM_CLASSES

  x_train, y_train = train
  VOCAB_SIZE = get_vocab_size(x_train)
  NUM_CLASSES = get_num_classes(y_train)

  print()
  print('DATASET PARAMS\n===============')
  print('Dataset:', FLAGS.dataset)
  print('Total words:', VOCAB_SIZE)
  print('Number of classes:', NUM_CLASSES)
  print()

  print('TRAIN PARAMS\n===============')
  print('Cell type to use:', FLAGS.cell_type)
  print('Batch size:', GLOBAL_BATCH_SIZE)
  print('Embedding size:', EMBEDDING_SIZE)
  print('Hidden dimension:', HIDDEN_DIM)
  print('Model directory:', FLAGS.model_dir)
  print('Overfit sanity check:', FLAGS.overfit)
  print('Steps to train:', FLAGS.steps)
  print()

  def make_gen(dataset):

    def gen():
      for data, label in zip(*dataset):
        features = {"data": data, "length": len(data)}
        yield (features, label)

    return gen

  # Build model
  model_fn = rnn_model
  if (FLAGS.model_dir):
    final_model_path = all_models_dir + FLAGS.model_dir
  else: # use a temp folder, i.e. don't save checkpoints
    final_model_path = None

  # Set session config options to pass into Estimator
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  run_metadata = tf.RunMetadata()
  options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  estimator_config = tf.estimator.RunConfig(session_config=options, save_summary_steps=FLAGS.save_summ_steps)
  
  
    
    
  classifier = tf.estimator.Estimator(
                             model_fn=model_fn, model_dir=final_model_path,
                             config=estimator_config)

  def make_input_fn(dataset, shuffle=False, repeat=True, batch_size=GLOBAL_BATCH_SIZE, count=None, prefetch=True):

    def input_fn():
      gen = make_gen(dataset)
      ds = tf.data.Dataset.from_generator(
                    gen,
                    ({"data": tf.int64, "length": tf.int64}, tf.int64),
                    ({"data": tf.TensorShape([None]), "length": tf.TensorShape([])}, tf.TensorShape([])))
      if (count):
        ds = ds.take(count)
      if (shuffle):
        ds = ds.shuffle(10000)
      if (repeat):
        ds = ds.repeat()
      ds = ds.padded_batch(batch_size,
              padded_shapes=({"data": tf.TensorShape([None]),
                             "length": tf.TensorShape([])},
                             tf.TensorShape([])))
      if (prefetch):
        ds = ds.prefetch(buffer_size=1)
      return ds
    return input_fn

  if FLAGS.overfit: # overfit to a small subset of the data
    train_input_fn = make_input_fn(train,
                                  shuffle=False, 
                                  repeat=True, 
                                  batch_size=1, 
                                  count=OVERFIT_NUM)
    test_input_fn = make_input_fn(train, 
                                  shuffle=False,
                                  repeat=False, 
                                  batch_size=1, 
                                  count=OVERFIT_NUM)
  else: # use regular batching and all the data
    train_input_fn = make_input_fn(train, shuffle=True, repeat=True)
    test_input_fn = make_input_fn(test, shuffle=False, repeat=False)
  
  classifier.train(input_fn=train_input_fn, steps=FLAGS.steps)

  print()
  print("=================")
  print("train complete, now testing")
  print("=================")
  print()
  # # Predict.
  if (FLAGS.overfit):
    x_test = train[0][:OVERFIT_NUM]
    y_test = train[1][:OVERFIT_NUM]
  else:
    x_test, y_test = test
  predictions = classifier.predict(input_fn=test_input_fn)
  y_predicted = np.array(list(p['class'] for p in predictions))
  y_predicted = y_predicted.reshape(np.array(y_test).shape)

  # Score with sklearn.
  score = metrics.accuracy_score(y_test, y_predicted)
  print('Accuracy (sklearn): {0:f}'.format(score))

  # Score with tensorflow.
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-o',
      '--overfit',
      default=False,
      help='Overfit a small subset of the training data.',
      action='store_true')
  parser.add_argument(
      '-d',
      '--dataset',
      default='20NG',
      help='Name of the dataset to use.')
  parser.add_argument(
      '-m',
      '--model_dir',
      default=None,
      help='Model directory to store TF checkpoints')
  parser.add_argument(
      '-s',
      '--steps',
      dest='steps',
      type=int,
      default=4000,
      help='Number of steps to run.')
  parser.add_argument(
      '-c',
      '--cell_type',
      default='baseline',
      help='Type of RNN cell to test')
  parser.add_argument(
      '--save_summ_steps',
      default=50,
      help='number of steps between summary saves')
  FLAGS, unparsed = parser.parse_known_args()
  try:
    os.mkdir(all_models_dir)
  except OSError:
    print("Checkpoint directory already exists:", all_models_dir)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)