# borrowed largely from Tensorflow 2016: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/text_classification.py
"""Example of Estimator for DNN-based text classification with DBpedia data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from utils import data_utils

from model import SkipLSTMCell, RecurrentResidualCell

all_models_dir = "checkpoints/"

FLAGS = None

GLOBAL_BATCH_SIZE = 16
EMBEDDING_SIZE = 64
HIDDEN_DIM = 64
VOCAB_SIZE = 0
NUM_CLASSES = 0
OVERFIT_NUM = 50

# For now, let's chop out anything that's too gigantic from the dataset.
# Later, we can figure out some kind of wise binning strategy for making more uniform batches.
max_data_length = 1000


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

def make_gen(dataset):
  def gen():
    for data, label in zip(*dataset):
      features = {"data": data, "length": len(data)}
      yield (features, label)
  return gen

def make_input_ds(dataset, shuffle=False, repeat=True, batch_size=GLOBAL_BATCH_SIZE, count=None, prefetch=True):

  gen = make_gen(dataset)
  ds = tf.data.Dataset.from_generator(
                gen,
                ({"data": tf.int64, "length": tf.int64}, tf.int64),
                ({"data": tf.TensorShape([None]), "length": tf.TensorShape([])}, tf.TensorShape([])))
  if count:
    ds = ds.take(count)
  if shuffle:
    ds = ds.shuffle(10000)
  if repeat:
    ds = ds.repeat()
  ds = ds.padded_batch(batch_size,
          padded_shapes=({"data": tf.TensorShape([None]),
                         "length": tf.TensorShape([])},
                         tf.TensorShape([])))
  if prefetch:
    ds = ds.prefetch(buffer_size=1)

  return ds

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def build_graph(
  inputs,
  labels,
  cell_type = None,
  state_size = 64,
  num_classes = NUM_CLASSES,
  vocab_size = VOCAB_SIZE,
  batch_size = 16,
  num_steps = 200,
  num_layers = 1,
  grad_norm = 1,
  lr = 1e-3):
  
  dropout_is_train = tf.placeholder(tf.bool)
  # reset_graph()
  keep_prob = tf.cond(dropout_is_train, lambda: tf.constant(0.5), lambda: tf.constant(1.0))

  x = inputs
  y = labels
  data = x["data"]
  length = x["length"]

  word_vectors = tf.contrib.layers.embed_sequence(
    data, vocab_size=vocab_size, embed_dim=EMBEDDING_SIZE)

  word_vectors = tf.nn.dropout(word_vectors, keep_prob=keep_prob)

  if cell_type == 'baseline':
    cell = tf.contrib.rnn.LSTMCell(state_size)
  elif cell_type == 'skip':
    cell = SkipLSTMCell(state_size, n_skip=10)
  elif cell_type == 'rrn':
    cell = RecurrentResidualCell(state_size, k_depth=2)
  else:
    cell = tf.nn.rnn_cell.LSTMCell(state_size)

  init_state = cell.zero_state(batch_size, tf.float32)
  rnn_outputs, final_state = tf.nn.dynamic_rnn(
                                cell,
                                word_vectors,
                                initial_state=init_state,
                                dtype=tf.float32,
                                sequence_length=length)

  state = final_state[0]
  state = tf.nn.dropout(state, keep_prob=keep_prob)

  with tf.variable_scope('dense_output'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(state, W) + b

  predictions = tf.argmax(logits, 1)

  l2_weight = 0.01
  l2_reg_loss = l2_weight * tf.nn.l2_loss(W)

  acc = tf.reduce_mean(tf.cast(tf.equal(y, predictions), tf.float32))
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
  total_loss = loss + l2_reg_loss

  optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  gvs = optimizer.compute_gradients(total_loss)
  capped_gvs = [(tf.clip_by_norm(grad, grad_norm), var) for grad, var in gvs]
  train_step = optimizer.apply_gradients(capped_gvs, global_step=tf.train.get_global_step())

  tf.summary.scalar('loss', total_loss)
  tf.summary.scalar('accuracy', acc)

  return dict(
      x = x,
      y = y,
      dropout_is_train = dropout_is_train,
      init_state = init_state,
      final_state = final_state,
      total_loss = total_loss,
      accuracy = acc,
      train_step = train_step,
      preds = predictions,
      saver = tf.train.Saver()
  )


def write_avg_summ(writer, step, lst, name):
  lst_avg = np.mean(lst)
  summ = tf.Summary()
  summ.value.add(tag=name, simple_value=lst_avg)
  writer.add_summary(summ, step)


def train_network(g, train_init_op, val_init_op, test_init_op, data_lens, num_steps=200, batch_size=16, verbose=True, save=True):
  train_len, val_len, test_len = data_lens
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  with tf.Session(config=config) as sess:
    
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(save + '/train', sess.graph)
    val_writer = tf.summary.FileWriter(save + '/val')
    test_writer = tf.summary.FileWriter(save + '/test')

    def run_eval_and_avg(eval_init_op, summary_writer, num_eval_iters, eval_type="test"):
      sess.run(eval_init_op)
      loss_lst = []
      acc_lst = []
      for _ in range(num_eval_iters):
        summary, loss, acc = sess.run(
                [merged, g['total_loss'], g['accuracy']],
                feed_dict={dropout_is_train: False})
        summary_writer.add_summary(summary, step)
        loss_lst.append(loss)
        acc_lst.append(acc)
      write_avg_summ(summary_writer, step, loss_lst, 'loss')
      write_avg_summ(summary_writer, step, acc_lst, 'accuracy')
      print("%s loss: %.4f"%(eval_type, np.mean(loss_lst)))
      print("%s acc: %.4f"%(eval_type, np.mean(acc_lst)))
      return (np.mean(loss_lst), np.mean(acc_lst))

    sess.run(tf.global_variables_initializer())
    training_losses = []
    sess.run(train_init_op)
    print('Training...')
    tot_loss = 0
    t = time.time()
    for step in range(num_steps):

        dropout_is_train = g['dropout_is_train']
        
        # Record runtime stats twice, on step 20 and 1020
        if step in [20, 1020]:
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          train_summary, loss_value, _, _ = sess.run(
                                [merged, g['total_loss'], g['final_state'], g['train_step']],
                                feed_dict={dropout_is_train: True},
                                options=run_options,
                                run_metadata=run_metadata)
          train_writer.add_run_metadata(run_metadata, 'step%d'%step)
          train_writer.add_summary(train_summary, step)
          tot_loss += loss_value
          training_losses.append(loss_value)

          tl = timeline.Timeline(run_metadata.step_stats)
          ctf = tl.generate_chrome_trace_format()
          with open(save+'/timeline_%d.json'%step, 'w') as f:
              f.write(ctf)
          print('Adding run metadata for step %d'%step)

        # Normal train step
        else:
          train_summary, loss_value, train_state, _ = sess.run(
                  [merged, g['total_loss'], g['final_state'], g['train_step']],
                  feed_dict={dropout_is_train: True})
          train_writer.add_summary(train_summary, step) # Record train stats (loss, accuracy) at each step
          tot_loss += loss_value
          training_losses.append(loss_value)


        # Record validation stats (loss, accuracy) at every 50th step

        if step % 100 == 0:
          num_eval_iters = 4

          # evaluate on validation and test sets for num_eval_iters batches
          run_eval_and_avg(val_init_op, val_writer, num_eval_iters, eval_type="val")
          run_eval_and_avg(test_init_op, test_writer, num_eval_iters, eval_type="test")
          sess.run(train_init_op) # reset back to train dataset
        
        # Print train loss every 100 steps
        if step%100 == 0:
          if step == 0:
            steps_taken = 1
          else:
            steps_taken = 100
          print("Step: %d, Loss: %.4f (%.3fs)}"%(step, tot_loss/steps_taken, time.time()-t))
          tot_loss=0
          t = time.time()

    print()
    print("===================================")
    print("train complete, testing results:")
    print("===================================")
    print()

    # After training, calculate evaluation stats on full test set
    num_test_iters = test_len//batch_size
    test_loss, test_acc = run_eval_and_avg(test_init_op, test_writer, num_test_iters, eval_type="test")

    if isinstance(save, str):
      g['saver'].save(sess, save)
    return training_losses, test_loss, test_acc


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

  data_lens = [len(train[0]), len(val[0]), len(test[0])]


  print()
  print('DATASET PARAMS\n===============')
  print('Dataset:', FLAGS.dataset)
  print('Total words:', VOCAB_SIZE)
  print('Number of classes:', NUM_CLASSES)
  print('Number of examples in train/test/val:')
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

  

  # Build model
  if (FLAGS.model_dir):
    final_model_path = all_models_dir + FLAGS.model_dir
  else: # use a temp folder, i.e. don't save checkpoints
    final_model_path = None


  

  train_ds = make_input_ds(train, shuffle=True, repeat=True)
  val_ds = make_input_ds(val, shuffle=True, repeat=True)
  test_ds = make_input_ds(test, shuffle=False, repeat=True)

  it = tf.data.Iterator.from_structure(train_ds.output_types,
                                           train_ds.output_shapes)

  features, labels = it.get_next()

  train_init_op = it.make_initializer(train_ds)
  val_init_op = it.make_initializer(val_ds)
  test_init_op = it.make_initializer(test_ds)

  

  g = build_graph(features, labels, cell_type=FLAGS.cell_type, num_steps=FLAGS.steps,
            batch_size=GLOBAL_BATCH_SIZE, num_classes=NUM_CLASSES, vocab_size=VOCAB_SIZE, state_size=HIDDEN_DIM)

  # merged = tf.summary.merge_all()
  # train_writer = tf.summary.FileWriter(final_model_path + '/train', sess.graph)
  # test_writer = tf.summary.FileWriter(final_model_path + '/test')

  train_losses, test_loss, test_acc = train_network(g, train_init_op, val_init_op, test_init_op, data_lens, 
                                       num_steps=FLAGS.steps, batch_size=GLOBAL_BATCH_SIZE, verbose=True, save=final_model_path)


  print()
  # print("=================")
  # print("train complete, now testing")
  # print("=================")
  print()

  # # Predict.
  # x_test, y_test = test
  # predictions = classifier.predict(input_fn=test_input_fn)
  # y_predicted = np.array(list(p['class'] for p in predictions))
  # y_predicted = y_predicted.reshape(np.array(y_test).shape)

  # Score with sklearn.
  # score = metrics.accuracy_score(y_test, y_predicted)
  # print('Accuracy (sklearn): {0:f}'.format(score))

  # # Score with tensorflow.
  # scores = classifier.evaluate(input_fn=test_input_fn)
  # print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


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
      default='test_runtime',
      help='Model directory to store TF checkpoints')
  parser.add_argument(
      '-s',
      '--steps',
      dest='steps',
      type=int,
      default=1000,
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