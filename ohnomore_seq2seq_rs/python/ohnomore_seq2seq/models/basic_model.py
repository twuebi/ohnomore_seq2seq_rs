from . import AbstractModel
from . import Mode

import tensorflow as tf


class BasicModel(AbstractModel):
    def __init__(self, mode, config):
        super().__init__(mode, config)

        outputs, state = self.encode()

        state = self.feedforward(encoder_state=state, scope="morph_ff")

        logits, self.predictions, self.alignment_history = self.decode(encoder_state=state, outputs=outputs, scope="decoder")
        if self._mode == Mode.Predict:
            self.char_preds = self.dec2char.lookup(tf.cast(self.predictions,tf.int64))

        if self._mode != Mode.Predict:
            # loss calculation
            losses = self.compute_loss(logits)
            self.loss = tf.reduce_sum(losses)

            # performance metrics
            accuracy_metrics, length_metrics = self.perf_metrics()
            self.acc, self.per_example_acc = accuracy_metrics

            self.correct_lengths_mean, self.incorrect_lengths_mean, self.expected_lengths_mean = length_metrics
            self.length_diffs = self.incorrect_lengths_mean - self.expected_lengths_mean


            if self.config.save_path != '':
                self.merged = self.create_tensorboard_summaries()
            else:
                self.merged = tf.no_op()

        if self._mode == Mode.Train:
            self.train_op = self.create_train_op(losses)

    def encode(self):
        if self.config.encoder == 'bi':
            return self.encode_bi_dir('bi_encoder')
        elif self.config.encoder == 'uni':
            return self.encode_uni_dir('uni_encoder')
        else:
            raise NotImplementedError('Encoder {} not supported.'.format(self.config.encoder))

    def encode_uni_dir(self, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            input_embeddings = tf.nn.embedding_lookup(self.enc_character_embeddings, self.input)
            cell = self.make_rnn_cell()

            if self._mode == Mode.Train:
                input_embeddings = tf.nn.dropout(input_embeddings, self.config.input_dropout)

            return tf.nn.dynamic_rnn(cell, input_embeddings, self.input_lengths, dtype=tf.float32)

    def encode_bi_dir(self, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            input_embeddings = tf.nn.embedding_lookup(self.enc_character_embeddings, self.input)

            cell_fw = self.make_rnn_cell()
            cell_bw = self.make_rnn_cell()

            if self._mode == Mode.Train:
                input_embeddings = tf.nn.dropout(input_embeddings, self.config.input_dropout)

            outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                            cell_bw=cell_bw,
                                                                            inputs=input_embeddings,
                                                                            sequence_length=self.input_lengths,
                                                                            dtype=tf.float32)

            # destructure bi-rnn output, concatenate them along last dimension and project them to hsize
            hidden_projection_layer = tf.layers.Dense(self.config.hsize, activation=tf.nn.selu)
            output_projection_layer = tf.layers.Dense(self.config.hsize, activation=tf.nn.selu)
            if self.config.cell_type == 'lstm':
                c_comb_layer = tf.layers.Dense(self.config.hsize, activation=tf.nn.selu)

            outputs = output_projection_layer(tf.concat(outputs, axis=-1))

            if self.config.cell_type == 'lstm':
                state_fw_h = state_fw.h
                state_bw_h = state_bw.h
                new_h = hidden_projection_layer(tf.concat([state_fw_h, state_bw_h], -1))

                state_fw_c = state_fw.c
                state_bw_c = state_bw.c
                new_c = c_comb_layer(tf.concat([state_fw_c, state_bw_c], -1))

                states = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            else:
                states = hidden_projection_layer(tf.concat([state_fw, state_bw], -1))

            return outputs, states

    def decode(self, encoder_state, outputs, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            projection_layer = tf.layers.Dense(self.config.dec_vocab_size, use_bias=False)

            cell = self.make_rnn_cell()

            if self.config.attention:
                cell = self.wrap_attention(cell, outputs)
                encoder_state = cell.zero_state(tf.shape(self.input)[0], tf.float32).clone(cell_state=encoder_state)

            if self._mode == Mode.Train:
                # exclude last timestep since there are no further predictions hence it will not be input to the next
                # state
                input_embeddings = tf.nn.embedding_lookup(self.dec_character_embeddings, self.dec_input[:, :-1])
                helper = tf.contrib.seq2seq.TrainingHelper(input_embeddings, self.dec_lengths)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, encoder_state, projection_layer)
                final_outputs, self.final_state, self.seq_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True)
            else:
                inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.dec_character_embeddings,
                                                                            start_tokens=self.start_tokens,
                                                                            end_token=self.config.end_idx)

                inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell, inference_helper, encoder_state,
                                                                    projection_layer)

                final_outputs, self.final_state, self.seq_lengths = tf.contrib.seq2seq.dynamic_decode(
                    inference_decoder, maximum_iterations=self.config.max_timesteps * 2, impute_finished=True)

            logits = final_outputs.rnn_output
            predictions = final_outputs.sample_id
            alignments = self.final_state.alignment_history.stack() if self.config.attention else tf.no_op()

            return logits, predictions, alignments

    def feedforward(self, encoder_state, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

            morph = tf.nn.embedding_lookup(self.morph_embeddings, self.morph_input)
            morph = tf.reshape(morph, shape=[-1, self.config.max_morph_tags*self.config.morph_embedding_size])

            pos = tf.nn.embedding_lookup(self.pos_embeddings, self.pos_input)


            if self._mode == Mode.Train:
                morph = tf.nn.dropout(morph, self.config.input_dropout)
                pos = tf.nn.dropout(pos, self.config.input_dropout)

            h_dense = tf.layers.Dense(self.config.hsize, activation=tf.nn.selu)
            c_dense = tf.layers.Dense(self.config.hsize, activation=tf.nn.selu)

            h = encoder_state.h
            c = encoder_state.c

            h = h_dense(tf.concat([h, pos, morph] , axis=-1))
            c = c_dense(tf.concat([c, pos, morph], axis=-1))

            return tf.nn.rnn_cell.LSTMStateTuple(c,h)


    def create_tensorboard_summaries(self):
        # accuracy and loss summaries
        acc_sum = tf.summary.scalar("accuracy", self.acc)
        loss_sum = tf.summary.histogram("loss_histogram", self.loss)
        scalar_loss = tf.summary.scalar("loss", self.loss)

        # mean length scalars
        correct_lengths_summary = tf.summary.scalar("correct_lengths_mean", self.correct_lengths_mean)
        incorrect_lengths_summary = tf.summary.scalar("incorrect_lengths_mean", self.incorrect_lengths_mean)
        expected_lengths_summary = tf.summary.scalar("expected_lengths_mean", self.expected_lengths_mean)
        lengths_diff_summary = tf.summary.scalar('predicted_expected_lengths_diff', self.length_diffs)

        # length histograms
        pred_lengths_hist = tf.summary.histogram('predicted_length_hist', self.seq_lengths)
        expected_lengths_hist = tf.summary.histogram('expected_length_hist', self.dec_lengths)
        input_length_hist = tf.summary.histogram('input_length_hist', self.input_lengths)

        # pos histogram
        posses = self.dec_input[:, 0] if self.mode == Mode.Train else self.start_tokens
        pos_hist = tf.summary.histogram('pos_hist', posses)

        summary_list = [input_length_hist, expected_lengths_hist, pred_lengths_hist, pos_hist, acc_sum,
                                loss_sum, scalar_loss, correct_lengths_summary, incorrect_lengths_summary,
                                expected_lengths_summary, lengths_diff_summary]
        # attention images
        if self.config.attention:
            attention_summary = self.create_attention_images_summary(alignments=self.alignment_history)
            summary_list.append(attention_summary)

        return tf.summary.merge(summary_list)

    def create_attention_images_summary(self, alignments):
        """
        taken from https://github.com/tensorflow/nmt/blob/master/nmt/attention_model.py
        create attention image and attention summary.
        """

        # Transpose to (batch, src_seq_len, tgt_seq_len,1)
        alignments = tf.expand_dims(tf.transpose(alignments, [1, 2, 0]), -1)[:,:tf.reduce_max(self.input_lengths)]

        attention_summary = tf.summary.image("attention_images", alignments)
        return attention_summary
