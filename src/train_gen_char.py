from model import SkipLSTMCell
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
import math
import urllib.request

""" From https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
"""
"""
Load and process data, utility functions
"""

def ptb_iterator(raw_data, batch_size, num_steps, steps_ahead=1):
  """Iterate on the raw PTB data.
  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.
  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)

  batch_len = data_len // batch_size

  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  offset = 0
  if data_len % batch_size:
    offset = np.random.randint(0, data_len % batch_size)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i + offset:batch_len * (i + 1) + offset]

  epoch_size = (batch_len - steps_ahead) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+steps_ahead]
    yield (x, y)

  if epoch_size * num_steps < batch_len - steps_ahead:
    yield (data[:, epoch_size*num_steps : batch_len - steps_ahead], data[:, epoch_size*num_steps + 1:])

def shuffled_ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)
    r = len(raw_data) % num_steps
    if r:
        n = np.random.randint(0, r)
        raw_data = raw_data[n:n + len(raw_data) - r]

    raw_data = np.reshape(raw_data, [-1, num_steps])
    np.random.shuffle(raw_data)

    num_batches = int(np.ceil(len(raw_data) / batch_size))

    for i in range(num_batches):
        data = raw_data[i*batch_size:min(len(raw_data), (i+1)*batch_size),:]
        yield (data[:,:-1], data[:,1:])


file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
file_name = 'tinyshakespeare.txt'
if not os.path.exists(file_name):
    urllib.request.urlretrieve(file_url, file_name)

with open(file_name,'r') as f:
    raw_data = f.read()
    print("Data length:", len(raw_data))

vocab = set(raw_data)
vocab_size = len(vocab)
idx_to_vocab = dict(enumerate(vocab))
vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

data = [vocab_to_idx[c] for c in raw_data]
del raw_data

def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        yield ptb_iterator(data, batch_size, num_steps)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def build_graph(
    cell_type = None,
    skip_layers = [None],
    num_weights_for_custom_cell = 5,
    state_size = 100,
    num_classes = vocab_size,
    batch_size = 32,
    num_steps = 200,
    num_layers = 3,
    build_with_dropout=False,
    learning_rate = 1e-4):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    dropout = tf.constant(1.0)

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    if cell_type == 'SkipLSTM':
        cell = tf.nn.rnn_cell.MultiRNNCell([SkipLSTMCell(state_size, skip) for skip in skip_layers])
    elif cell_type == 'GRU':
        cell = tf.nn.rnn_cell.GRUCell(state_size)
    elif cell_type == 'LSTM':
        cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    elif cell_type == 'LN_LSTM':
        cell = LayerNormalizedLSTMCell(state_size)
    else:
        cell = tf.nn.rnn_cell.BasicRNNCell(state_size)

    if build_with_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)
    if cell_type != 'SkipLSTM':
        if cell_type == 'LSTM' or cell_type == 'LN_LSTM':
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        else:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

    if build_with_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)

    init_state = cell.zero_state(batch_size, tf.float32)

    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    #reshape rnn_outputs and y
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    predictions = tf.nn.softmax(logits)

    Y = tf.argmax(predictions, 1)                          # [ BATCHSIZE x SEQLEN ]
    Y = tf.reshape(Y, [-1], name="Y")



    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    # https://github.com/burliEnterprises/tensorflow-shakespeare-poem-generator/blob/master/rnn_train.py
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(y_reshaped, tf.uint8), tf.cast(Y, tf.uint8)), tf.float32))

    loss_summary = tf.summary.scalar("batch_loss", total_loss)
    acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
    summaries = tf.summary.merge([loss_summary, acc_summary])

    # Init Tensorboard stuff. This will save Tensorboard information into a different
    # folder at each run named 'log/<timestamp>/'. Two sets of data are saved so that
    # you can compare training and validation curves visually in Tensorboard.


    # Init for saving models. They will be saved into a directory named 'checkpoints'.
    # Only the last checkpoint is kept.
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    saver = tf.train.Saver(max_to_keep=1000)

    return dict(
        x = x,
        y = y,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        batch_accuracy = accuracy,
        train_step = train_step,
        preds = predictions,
        saver = saver,
        summaries = summaries,
    )


def train_network(g, num_epochs, num_steps = 200, batch_size = 32, verbose = True, save=False):
    timestamp = str(math.trunc(time.time()))
    # TODO:@jiana rename log to better file Name
    summary_writer = tf.summary.FileWriter("log/" + save + timestamp + "-training")
    # validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    if save:
        f = open(save + "_data.txt", "w+")
    total_steps = 0
    with tf.Session(config=session_config) as sess:
        sess.run(tf.initialize_all_variables())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            batch_accuracy = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1
                total_steps+=1

                feed_dict={g['x']: X, g['y']: Y}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_,  batch_accuracy_, training_state, smm, _  = sess.run([g['total_loss'],
                                                                                g['batch_accuracy'],
                                                                                g['final_state'],
                                                                                g['summaries'],
                                                                                g['train_step'],
                                                                                ],
                                                                                feed_dict)
                training_loss += training_loss_
                batch_accuracy += batch_accuracy_
                summary_writer.add_summary(smm, total_steps)
            if verbose:
                print("Average training loss for Epoch", idx, ":", training_loss/steps)
                print("Average accuracy for Epoch", idx, ":", batch_accuracy/steps)
            if save:
                f.write("Average training loss for Epoch" + str(idx) + ":" + str(training_loss/steps)+ "\n")
                f.write("Average accuracy for Epoch" + str(idx) + ":"+ str(batch_accuracy/steps) + "\n")

            training_losses.append(training_loss/steps)

        if isinstance(save, str):
            g['saver'].save(sess, save)
        if save:
            f.close()

    return training_losses

def generate_characters(g, checkpoint, num_chars, prompt='A', pick_top_chars=None):
    """ Accepts a current character, initial state"""

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        g['saver'].restore(sess, checkpoint)

        state = None
        current_char = vocab_to_idx[prompt]
        chars = [current_char]

        for i in range(num_chars):
            if state is not None:
                feed_dict={g['x']: [[current_char]], g['init_state']: state}
            else:
                feed_dict={g['x']: [[current_char]]}

            preds, state = sess.run([g['preds'],g['final_state']], feed_dict)

            if pick_top_chars is not None:
                p = np.squeeze(preds)
                p[np.argsort(p)[:-pick_top_chars]] = 0
                p = p / np.sum(p)
                current_char = np.random.choice(vocab_size, 1, p=p)[0]
            else:
                current_char = np.random.choice(vocab_size, 1, p=np.squeeze(preds))[0]

            chars.append(current_char)

    chars = map(lambda x: idx_to_vocab[x], chars)
    # print("".join(chars))
    text = "".join(chars)
    print(text)
    f = open(checkpoint + "_data.txt", "a+")
    f.write(text)
    f.close()

    return(text)

def arr_to_str(arr):
    return '.'.join(str(e) for e in arr)

cell_type = "SkipLSTM"
skip_layers = [5]
num_layers = 1
g = build_graph(cell_type=cell_type,
                num_steps=None,
                skip_layers=skip_layers,
                state_size = 100,
                batch_size = 32,
                num_classes=vocab_size,
                learning_rate=5e-4)
epoch_num = 2

t = time.time()

if cell_type == "SkipLSTM":

    save_file = "gen_char_saves/"+ cell_type + "/" + arr_to_str(skip_layers) + "layers_"+ str(epoch_num) + "epochs"
else:
    save_file = "gen_char_saves/"+ cell_type +"/" + str(num_layers) + "layers_"+ str(epoch_num) + "epochs"

# if not Path(save_file + ".index").is_file():
losses = train_network(g, epoch_num, num_steps=80, save=save_file)
g = build_graph(cell_type=cell_type, num_layers = num_layers,skip_layers=skip_layers, num_steps=None, batch_size=1, num_classes=vocab_size, state_size = 100)
generate_characters(g, save_file , 750, prompt='A', pick_top_chars=5)
print("It took", time.time() - t, "seconds to train for " + str(epoch_num) + " epochs.")
print("The average loss on the final epoch was:", losses[-1])

f = open(save_file + "_data.txt", "a+")
f.write("It took" + str(time.time() - t)+ "seconds to train for " + str(epoch_num) + "epochs.\n")
f.write ("The average loss on the final epoch was:"+ str(losses[-1]) + "\n")
f.close()


