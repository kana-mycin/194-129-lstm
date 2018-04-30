import os

import tensorflow as tf

# Helper TensorFlow functions
from utils import maybe_download

# The encoder-decoder architecture
# from nmt.model import Model
from nmt.utils import vocab_utils
from nmt.train import train

from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops



_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

# Bogus class for testing the "custom_cell" code I inserted
class PrintLSTMCell(BasicLSTMCell):

    def call(self, inputs, state):
        x = [tf.constant(["Running a PrintLSTMCell"])]
        outputs, next_state = super(PrintLSTMCell, self).call(inputs, state)
        print_output = LSTMStateTuple(tf.Print(next_state.c, x), tf.Print(next_state.h, x))
        return outputs, print_output

# An LSTMCell class that provies L-length skip connections

L = 2

class SkipLSTMCell(rnn_cell_impl.RNNCell):
    """ Based on the paper http://www.aclweb.org/anthology/D16-1093 """
    

    def __init__(self, num_units, forget_bias=1.0, activation=None, reuse=None, n_skip=None, **kwargs):
        super(SkipLSTMCell, self).__init__(_reuse=reuse)
        self._n_skip = n_skip
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh
        self._output_size = self._num_units + 1
        self._state_size = (self._num_units, self._num_units, 1)



    def call(self, inputs, state):
        """Run this multi-layer cell on inputs, starting from state.
        inputs: [B, I + sum(ha_l)]
        state: a list of [c_{t-1}, h_{t-1}, h_skip, h_cnt]
        """

        x = [tf.constant(["Running a SkipLSTMCell"])]

        
        sigmoid = math_ops.sigmoid

        one = constant_op.constant(1, dtype=dtypes.int32)
        # Parameters of gates are concatenated into one multiply for efficiency.
        # if self._state_is_tuple:
        c, h, h_skip, h_cnt = state
        n_skip = self._n_skip
        if n_skip:
            skip_bool = h_cnt % self._n_skip == 0

        # else:
        #   c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

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

        # c: [B, num_units]
        # f: [B, num_units/4]

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

        new_state = [new_h, new_c, h_skip, h_cnt]


        # outputs, next_state = super(SkipLSTMCell, self).call(inputs, state)
        # print_output = LSTMStateTuple(tf.Print(next_state.c, x), tf.Print(next_state.h, x))
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
        c = tf.zeros([batch_size, self._num_units])
        h = tf.zeros([batch_size, self._num_units])
        return [h, c, h, 0]

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


out_dir = os.path.join('datasets', 'nmt_data_vi')
site_prefix = "https://nlp.stanford.edu/projects/nmt/data/"

# maybe_download(site_prefix + 'iwslt15.en-vi/train.en', out_dir, 13603614)
# maybe_download(site_prefix + 'iwslt15.en-vi/train.vi', out_dir, 18074646)

# maybe_download(site_prefix + 'iwslt15.en-vi/tst2012.en', out_dir, 140250)
# maybe_download(site_prefix + 'iwslt15.en-vi/tst2012.vi', out_dir, 188396)

# maybe_download(site_prefix + 'iwslt15.en-vi/tst2013.en', out_dir, 132264)
# maybe_download(site_prefix + 'iwslt15.en-vi/tst2013.vi', out_dir, 183855)

# maybe_download(site_prefix + 'iwslt15.en-vi/vocab.en', out_dir, 139741)
# maybe_download(site_prefix + 'iwslt15.en-vi/vocab.vi', out_dir, 46767)

def create_standard_hparams(data_path, out_dir):
    
    hparams = tf.contrib.training.HParams(
        
        # Data
        src="vi",
        tgt="en",
        train_prefix=os.path.join(data_path, "train"),
        dev_prefix=os.path.join(data_path, "tst2012"),
        test_prefix=os.path.join(data_path, "tst2013"),
        vocab_prefix="",
        embed_prefix="",
        out_dir=out_dir,
        src_vocab_file=os.path.join(data_path, "vocab.vi"),
        tgt_vocab_file=os.path.join(data_path, "vocab.en"),
        src_embed_file="",
        tgt_embed_file="",
        src_file=os.path.join(data_path, "train.vi"),
        tgt_file=os.path.join(data_path, "train.en"),
        dev_src_file=os.path.join(data_path, "tst2012.vi"),
        dev_tgt_file=os.path.join(data_path, "tst2012.en"),
        test_src_file=os.path.join(data_path, "tst2013.vi"),
        test_tgt_file=os.path.join(data_path, "tst2013.en"),

        # Networks
        num_units=512,
        num_layers=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_encoder_residual_layers=0,
        num_decoder_residual_layers=0,
        dropout=0.2,
        encoder_type="uni",
        residual=False,
        time_major=True,
        num_embeddings_partitions=0,

        unit_type="custom",
        custom_cell=SkipLSTMCell,

        # Train
        optimizer="adam",
        batch_size=128,
        init_op="uniform",
        init_weight=0.1,
        max_gradient_norm=100.0,
        learning_rate=0.001,
        warmup_steps=0,
        warmup_scheme="t2t",
        decay_scheme="luong234",
        colocate_gradients_with_ops=True,
        num_train_steps=12000,

        # Data constraints
        num_buckets=5,
        max_train=0,
        src_max_len=25,
        tgt_max_len=25,
        src_max_len_infer=0,
        tgt_max_len_infer=0,

        # Data format
        sos="<s>",
        eos="</s>",
        subword_option="",
        check_special_token=True,

        # Misc
        forget_bias=1.0,
        num_gpus=1,
        epoch_step=0,  # record where we were within an epoch.
        steps_per_stats=10,
        steps_per_eval=10,
        steps_per_external_eval=500,
        share_vocab=False,
        metrics=["bleu"],
        log_device_placement=False,
        random_seed=None,
        # only enable beam search during inference when beam_width > 0.
        beam_width=0,
        length_penalty_weight=0.0,
        override_loaded_hparams=True,
        num_keep_ckpts=5,
        avg_ckpts=False,
        num_intra_threads=1,
        num_inter_threads=8,

        # For inference
        inference_indices=None,
        infer_batch_size=32,
        sampling_temperature=0.0,
        num_translations_per_input=1,
        
    )
    
    src_vocab_size, _ = vocab_utils.check_vocab(hparams.src_vocab_file, hparams.out_dir)
    tgt_vocab_size, _ = vocab_utils.check_vocab(hparams.tgt_vocab_file, hparams.out_dir)
    hparams.add_hparam('src_vocab_size', src_vocab_size)
    hparams.add_hparam('tgt_vocab_size', tgt_vocab_size)
    
    out_dir = hparams.out_dir
    if not tf.gfile.Exists(out_dir):
        tf.gfile.MakeDirs(out_dir)
         
    for metric in hparams.metrics:
        hparams.add_hparam("best_" + metric, 0)  # larger is better
        best_metric_dir = os.path.join(hparams.out_dir, "best_" + metric)
        hparams.add_hparam("best_" + metric + "_dir", best_metric_dir)
        tf.gfile.MakeDirs(best_metric_dir)

        if hparams.avg_ckpts:
            hparams.add_hparam("avg_best_" + metric, 0)  # larger is better
            best_metric_dir = os.path.join(hparams.out_dir, "avg_best_" + metric)
            hparams.add_hparam("avg_best_" + metric + "_dir", best_metric_dir)
            tf.gfile.MakeDirs(best_metric_dir)

    return hparams

# If desired as a baseline, train a vanilla LSTM model without attention
hparams = create_standard_hparams(
    data_path=os.path.join("datasets", "nmt_data_vi"), 
    out_dir="nmt_model_test"
)

hparams.add_hparam("attention", "")
train(hparams)
