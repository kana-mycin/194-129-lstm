import os

import tensorflow as tf

# Helper TensorFlow functions
from utils import maybe_download

# The encoder-decoder architecture
# from nmt.model import Model
from nmt.utils import vocab_utils
from nmt.train import train

from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple

# Bogus class for testing the "custom_cell" code I inserted
class PrintLSTMCell(BasicLSTMCell):

    def call(self, inputs, state):
        x = [tf.constant(["Running a PrintLSTMCell"])]
        outputs, next_state = super(PrintLSTMCell, self).call(inputs, state)
        print_output = LSTMStateTuple(tf.Print(next_state.c, x), tf.Print(next_state.h, x))
        return outputs, print_output

# An LSTMCell class that provies L-length skip connections

L = 2

class SkipLSTMCell(BasicLSTMCell):

    def __init__(self, num_units, **kwargs):
        super(SkipLSTMCell, self).__init__(num_units, **kwargs)

    def call(self, inputs, state):
        x = [tf.constant(["Running a SkipLSTMCell"])]

        outputs, next_state = super(SkipLSTMCell, self).call(inputs, state)
        print_output = LSTMStateTuple(tf.Print(next_state.c, x), tf.Print(next_state.h, x))
        return outputs, print_output

out_dir = os.path.join('datasets', 'nmt_data_vi')
site_prefix = "https://nlp.stanford.edu/projects/nmt/data/"

maybe_download(site_prefix + 'iwslt15.en-vi/train.en', out_dir, 13603614)
maybe_download(site_prefix + 'iwslt15.en-vi/train.vi', out_dir, 18074646)

maybe_download(site_prefix + 'iwslt15.en-vi/tst2012.en', out_dir, 140250)
maybe_download(site_prefix + 'iwslt15.en-vi/tst2012.vi', out_dir, 188396)

maybe_download(site_prefix + 'iwslt15.en-vi/tst2013.en', out_dir, 132264)
maybe_download(site_prefix + 'iwslt15.en-vi/tst2013.vi', out_dir, 183855)

maybe_download(site_prefix + 'iwslt15.en-vi/vocab.en', out_dir, 139741)
maybe_download(site_prefix + 'iwslt15.en-vi/vocab.vi', out_dir, 46767)

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