# borrowed largely from Tensorflow 2016: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/text_classification.py
"""Example of Estimator for DNN-based text classification with DBpedia data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf

from utils import data_utils

FLAGS = None

BATCH_SIZE = 8
EMBEDDING_SIZE = 50
VOCAB_SIZE = 0
NUM_CLASSES = 0


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

  eval_metric_ops = {
      'accuracy':
          tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def rnn_model(features, labels, mode):
  """RNN model to predict from sequence of words to a class."""

  def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

  # Convert indexes of words into embeddings.
  # This creates embeddings matrix of [VOCAB_SIZE, EMBEDDING_SIZE] and then
  # maps word indexes of the sequence into [batch_size, sequence_length,
  # EMBEDDING_SIZE].
  word_vectors = tf.contrib.layers.embed_sequence(
      features, vocab_size=VOCAB_SIZE, embed_dim=EMBEDDING_SIZE)

  # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
  cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)

  # Create a dynamic RNN and pass in a function to compute sequence lengths.
  _, encoding = tf.nn.dynamic_rnn(
                cell,
                word_vectors,
                dtype=tf.float32,
                sequence_length=length(word_vectors))

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

  # TODO: Set VOCAB_SIZE to be equal to the vocab size
  train, val, test = data_utils.load_data("20NG/20news.pkl")
  global VOCAB_SIZE, NUM_CLASSES

  x_train, y_train = train
  VOCAB_SIZE = get_vocab_size(x_train)
  NUM_CLASSES = get_num_classes(y_train)

  print('Total words: %d' % VOCAB_SIZE)
  print('Number of classes: %d' % NUM_CLASSES)

  def gen_train():
    for data, label in zip(*train):
      yield (data,label)

  def gen_test():
    for data, label in zip(*test):
      yield (data,label)

  # Build model
  model_fn = rnn_model
  classifier = tf.estimator.Estimator(model_fn=model_fn)

  def train_input_fn():
    ds_train = tf.data.Dataset.from_generator(
                    gen_train,
                    (tf.int64, tf.int64),
                    (tf.TensorShape([None]), tf.TensorShape([])))
    # For now, we use batch size 1 to avoid the variable length problem
    # A solution is here: https://stackoverflow.com/questions/34670112/how-to-deal-with-batches-with-variable-length-sequences-in-tensorflow
    ds = ds_train.shuffle(len(train[0]))  \
            .repeat()                     \
            .padded_batch(BATCH_SIZE, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])))
    return ds

  def test_input_fn():
    ds_test = tf.data.Dataset.from_generator(
                    gen_test,
                    (tf.int64, tf.int64),
                    (tf.TensorShape([None]), tf.TensorShape([])))
    ds_test = ds_test.padded_batch(BATCH_SIZE, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])))
    return ds_test 

  classifier.train(input_fn=train_input_fn, steps=300)

  # # Predict.
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

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)