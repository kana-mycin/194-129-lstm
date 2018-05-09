import os

import tensorflow as tf

import collections
from tensorflow.python.framework import ops

from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
_TRANSFORM_VARIABLE_NAME = "U"


class SkipLSTMCell(rnn_cell_impl.RNNCell):
    """ Based on the paper http://www.aclweb.org/anthology/D16-1093 """

    def __init__(self, num_units, forget_bias=1.0, activation=None, reuse=None, n_skip=None, **kwargs):
        super(SkipLSTMCell, self).__init__(_reuse=reuse)
        self._n_skip = n_skip
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh
        self._output_size = self._num_units + 1
        self._state_size = (self._num_units, self._num_units, self._num_units, 1)

    def call(self, inputs, state):
        """Run this multi-layer cell on inputs, starting from state.
        inputs: [B, I + sum(ha_l)]
        state: a list of [c_{t-1}, h_{t-1}, h_skip, h_cnt]
        """
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)

        # Parameters of gates are concatenated into one multiply for efficiency.
        c, h, h_skip, h_cnt = state
        n_skip = self._n_skip
        if n_skip:
            skip_bool = h_cnt % self._n_skip == 0

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(
            value=gate_inputs, num_or_size_splits=4, axis=one)
        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)

        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = math_ops.add
        multiply = math_ops.multiply

        first = multiply(c, sigmoid(add(f, forget_bias_tensor)))
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                    multiply(sigmoid(i), self._activation(j)))
        if n_skip:
            new_h = multiply(self._activation(new_c), sigmoid(o)) + skip_bool * 1 * h_skip
            h_skip = h_skip * (1-skip_bool) + new_h * skip_bool
        else:
            new_h = multiply(self._activation(new_c), sigmoid(o)) 
            h_skip = new_h 

        h_cnt += 1
        new_state = SCLSTMStateTuple(new_h, new_c, h_skip, h_cnt)
        return new_h, new_state
    
    @property
    def output_size(self):
        # outputs h and z
        return self._num_units

    @property
    def state_size(self):
        # the state is c, h, h_skip, h_cnt
        return (self._num_units, self._num_units, self._num_units, 1)

    def zero_state(self, batch_size, dtype):
        c = tf.zeros([batch_size, self._num_units], name='ZeroState')
        h = tf.zeros([batch_size, self._num_units], name='ZeroState2')
        h_skip = tf.zeros([batch_size, self._num_units], name='ZeroState3')
        return SCLSTMStateTuple(h, c, h_skip, tf.constant(1, dtype=tf.float32))

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)

        input_depth = inputs_shape[1].value
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, 4 * self._num_units])
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[4 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True


_SCLSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("h", "c", "s", "n"))
class SCLSTMStateTuple(_SCLSTMStateTuple):
  """
  Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores four elements: `(h, c, s, n)`, in that order. Where `h` is the hidden state
  and `c` is the cell state and `s` is the skip hidden state and `n` is a count of hidden
  states.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (h, c, s, n) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype





class RecurrentResidualCell(rnn_cell_impl.RNNCell):
    """ Based on the paper http://www.aclweb.org/anthology/D16-1093 """
    

    def __init__(self, num_units, activation=None, reuse=None, k_depth=1, **kwargs):
        super(RecurrentResidualCell, self).__init__(_reuse=reuse)
        self._k_depth = k_depth
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._output_size = self._num_units + 1
        self._state_size = (self._num_units, self._num_units, self._num_units, 1)



    def call(self, inputs, state):
        """
        Run this multi-layer cell on inputs, starting from state.
        inputs: x ; [B, H]
        state: a list of [h_{t-1}]
        """
        sigmoid = math_ops.sigmoid
        add = math_ops.add
        multiply = math_ops.multiply

        transformed_input = math_ops.matmul(inputs, self._kernel)
        transformed_input = nn_ops.bias_add(transformed_input, self._bias)
        in_all = array_ops.split(value=transformed_input, num_or_size_splits=self._k_depth, axis=1)

        h = state[0]
        y = h
        for idx in range(self._k_depth):
            hy = math_ops.matmul(y, self._U_dict[idx])
            y = sigmoid(in_all[idx] + hy)

        new_h = self._activation(add(h, y))
        new_state = RRNStateTuple(new_h)

        return new_h, new_state

    @property
    def output_size(self):
        # output h
        return self._num_units

    @property
    def state_size(self):
        # the state is h
        return RRNStateTuple(self._num_units)

    def zero_state(self, batch_size, dtype):
        h = tf.zeros([batch_size, self._num_units])
        return RRNStateTuple(h)

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)

        input_depth = inputs_shape[1]
        k = self._k_depth
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth, k * self._num_units])
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[k * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self._U_dict = {}
        for idx in range(k):
            U_idx = self.add_variable(
                _TRANSFORM_VARIABLE_NAME + '_' + str(idx+1),
                shape=[self._num_units, self._num_units])
            self._U_dict[idx] = U_idx
        self.built = True



"""
Define state tuples for the models, for consistent implementations.
"""

_SCLSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("h", "c", "s", "n"))
class SCLSTMStateTuple(_SCLSTMStateTuple):
  """
  Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores four elements: `(h, c, s, n)`, in that order. Where `h` is the hidden state
  and `c` is the cell state and `s` is the skip hidden state and `n` is a count of hidden
  states.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (h, c, s, n) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype


_RRNStateTuple = collections.namedtuple("RRNStateTuple", ("h"))
class RRNStateTuple(_RRNStateTuple):
  """
  Tuple used by RRN Cells for `state_size`, `zero_state`, and output state.
  Stores one element: `(h)`, where `h` is the hidden state
  """
  __slots__ = ()

  @property
  def dtype(self):
    (h) = self
    return h.dtype
