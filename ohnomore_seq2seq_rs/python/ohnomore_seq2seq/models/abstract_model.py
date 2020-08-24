import tensorflow as tf
import os
from tensorflow.python.ops.lookup_ops import HashTable, KeyValueTensorInitializer,index_table_from_tensor
from abc import ABC, abstractmethod
from enum import Enum

from ohnomore_seq2seq import Attentions

class Mode(Enum):
    Train = 0
    Validation = 1
    Predict = 2


class AbstractModel(ABC):
    def __init__(self, mode, config):
        self._mode = mode
        self.config = config

        if self._mode != Mode.Predict:
            self.input = tf.placeholder(name='inputs', dtype=tf.int32, shape=[None, None])
            self.pos_input = tf.placeholder(name='pos_input', dtype=tf.int32, shape=[None])
            self.morph_input = tf.placeholder(name='morph_input', dtype=tf.int32, shape=[None, config.max_morph_tags])
        else:
            k, v = list(zip(*config.dec_char_numberer.value2num.items()))
            i = KeyValueTensorInitializer(v, k,key_dtype=tf.int64,value_dtype=tf.string)
            self.dec2char = HashTable(i, default_value="<UNK>")
            self.enc_chars = self.lookup(config.enc_char_numberer)
            self.morph_tags = self.lookup(config.morph_numberer)
            self.pos_tags = self.lookup(config.pos_numberer)

            self.input_plc =tf.placeholder(name='inputs', dtype=tf.string, shape=[None, None])
            self.pos_plc = tf.placeholder(name='pos_input', dtype=tf.string, shape=[None])
            self.morph_plc = tf.placeholder(name='morph_input', dtype=tf.string, shape=[None, config.max_morph_tags])
            self.input = self.enc_chars.lookup(self.input_plc)
            self.pos_input = self.pos_tags.lookup(self.pos_plc)
            self.morph_input = self.morph_tags.lookup(self.morph_plc)


        self.input_lengths = tf.placeholder(name='input_lengths', dtype=tf.int32, shape=[None])



        self.enc_character_embeddings = tf.get_variable('enc_character_embeddings',
                                                        shape=[self.config.enc_vocab_size,
                                                               self.config.char_embedding_size])
        self.dec_character_embeddings = tf.get_variable('dec_character_embeddings',
                                                        shape=[self.config.dec_vocab_size,
                                                               self.config.char_embedding_size])

        self.pos_embeddings = tf.get_variable('pos_embeddings',
                                              shape=[self.config.pos_vocab_size,
                                                     self.config.pos_embedding_size])



        self.morph_embeddings = tf.get_variable('morph_embeddings',
                                              shape=[self.config.morph_vocab_size,
                                                     self.config.morph_embedding_size])

        if self._mode != Mode.Train:
            self.start_tokens = tf.tile([config.start_idx], [tf.shape(self.input)[0]])

        if self._mode != Mode.Predict:
            # length +2 since lengths are <bow> + word + <eow>
            self.dec_input = tf.placeholder(name='dec_in', dtype=tf.int32, shape=[None, None])
            # exclude start token from targets for loss-computations since we feed the start token and don't want to
            # predict it
            self.decoder_targets = self.dec_input[:, 1:]
            self.dec_lengths = tf.placeholder(name='dec_lengths', dtype=tf.int32, shape=[None])

    def lookup(self, numberer):
        k, v = list(zip(*numberer.value2num.items()))
        i = KeyValueTensorInitializer(k, v)
        return HashTable(i, default_value=numberer.unknown_idx)

    @abstractmethod
    def encode(self):
        """All sequence to sequence models should implement an encode function."""
        return

    @abstractmethod
    def decode(self, **kwargs):
        """All sequence to sequence models should implement a decode function."""
        return

    @property
    def mode(self):
        return self._mode

    def compute_loss(self, logits):
        """
        Computes cross entropy between targets and predictions.

        :param logits: shape=[batch, ?, n_decoder_symbols]
        :return: x_ent: masked loss: shape=[batch, max_timesteps]
        """
        # pad logits to max sequence length of targets
        logits = tf.pad(logits,
                        [[0, 0], [0, tf.maximum(tf.shape(self.decoder_targets)[1] - tf.shape(logits)[1], 0)],
                         [0, 0]])

        if self._mode == Mode.Validation:
            # validation lemmas can get longer than actual targets -> pad labels to max length of logits
            self.labels = tf.pad(self.decoder_targets,
                                 [[0, 0], [0, tf.maximum(tf.shape(logits)[1] - tf.shape(self.decoder_targets)[1], 0)]],
                                 constant_values=0)
        else:
            self.labels = self.decoder_targets

        if self._mode == Mode.Validation:
            x_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.labels,
                logits=logits)
        else:
            x_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.decoder_targets,
                logits=logits)
        # mask losses
        mask = tf.sequence_mask(self.dec_lengths, tf.shape(logits)[1], dtype=tf.float32)
        loss = x_ent * mask
        return loss

    def perf_metrics(self):
        """
        Computes the following performance metrics:

            - true_predictions
            - accuracy
            - expected length mean
            - predicted length means divided by true/false

        :return: two tuples: (accuracy, per_example_accuracy),
                             (correct_length_mean, incorrect_length_mean, expected_lengths_mean)
        """
        # pad predictions since they can become shorter than targets
        preds = tf.pad(self.predictions,
                       [[0, 0],
                        [0, tf.maximum(tf.shape(self.decoder_targets)[1] - tf.shape(self.predictions)[1], 0)]])
        eq = tf.cast(tf.equal(self.labels, preds), dtype=tf.int32)

        per_example_acc = tf.floor(tf.reduce_mean(tf.cast(eq, tf.float32), axis=1))
        incorrect = tf.ones_like(per_example_acc) - per_example_acc
        acc = tf.reduce_mean(per_example_acc)

        expected_lengths = tf.cast(self.dec_lengths, tf.float32)
        float_lengths = tf.cast(self.seq_lengths, tf.float32)

        expected_lengths_mean = tf.reduce_mean(expected_lengths)
        correct_length_mean = tf.reduce_sum(float_lengths * per_example_acc) / tf.reduce_sum(per_example_acc)
        incorrect_length_mean = tf.reduce_sum(float_lengths * incorrect) / tf.reduce_sum(incorrect)

        per_example_acc = tf.cast(per_example_acc, tf.bool)

        return (acc, per_example_acc), (correct_length_mean , incorrect_length_mean, expected_lengths_mean)

    def create_train_op(self, losses):
        """
        Creates the train op, computes gradients and applies gradient clipping.

        :param losses: losses for which to calculate gradients
        :return: the train_op
        """
        learning_rate = self.config.learning_rate

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if self.config.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate)
        elif self.config.optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise NotImplementedError("Optimizer: {} is not supported".format(self.config.optimizer))

        gradients, variables = zip(*opt.compute_gradients(losses))
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(
            gradients, self.config.max_gradient_norm)

        return opt.apply_gradients(zip(clipped_gradients, variables), global_step=self.global_step)

    def make_rnn_cell(self):
        assert self.config.layers > 0

        cell_type = self.config.cell_type
        cell_list = []

        for _ in range(self.config.layers):
            if cell_type == 'gru':
                cell = tf.nn.rnn_cell.GRUCell(self.config.hsize)
            elif cell_type == 'lstm':
                cell = tf.nn.rnn_cell.LSTMCell(self.config.hsize, cell_clip=self.config.encoder_cellclip)
            else:
                raise NotImplementedError("Cell: {} is not supported".format(cell_type))

            if self._mode == Mode.Train:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.config.encoder_dropout)
            cell_list.append(cell)

        return tf.nn.rnn_cell.MultiRNNCell(cell_list) if self.config.layers > 1 else cell

    def wrap_attention(self, cell, outputs):
        """
        Wraps cell with attention mechanism specified in the config.

        :param cell: some rnn cell
        :param outputs: the memory to be handed to the attention mechanism, usually the output of a rnn
        :return: the attention wrapped cell.
        """
        if self.config.attention_kind == Attentions.luong:
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.config.hsize, outputs, scale=True,
                                                                    memory_sequence_length=self.input_lengths)
        elif self.config.attention_kind == Attentions.bahdanau:
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.config.attention_size, outputs,
                                                                       memory_sequence_length=self.input_lengths)
        elif self.config.attention_kind == Attentions.luong_monotonic:
            attention_mechanism = tf.contrib.seq2seq.LuongMonotonicAttention(self.config.hsize, outputs, scale=True,
                                                                    memory_sequence_length=self.input_lengths)
        elif self.config.attention_kind == Attentions.bahdanau_monotonic:
            attention_mechanism = tf.contrib.seq2seq.BahdanauMonotonicAttention(self.config.attention_size, outputs,
                                                                       memory_sequence_length=self.input_lengths)

        return tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=self.config.hsize, alignment_history=True)
