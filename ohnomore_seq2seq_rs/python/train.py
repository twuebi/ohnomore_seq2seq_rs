import argparse
import os

import toml
import tensorflow as tf

import numpy as np
from ohnomore_seq2seq import read_config
from ohnomore_seq2seq import Numberer
from ohnomore_seq2seq import read_unique_tokens
from ohnomore_seq2seq import py2numpy, sample_n_batches, get_batches
from ohnomore_seq2seq import Mode, BasicModel


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

    parser = argparse.ArgumentParser(
        description='Reads conll-x and stores lemmatization relevant features (Form, Lemma, CPos, Pos)')

    parser.add_argument('val_input', type=argparse.FileType('r'))
    parser.add_argument('train_input', type=argparse.FileType('r'))
    parser.add_argument('config', type=argparse.FileType('r'))
    parser.add_argument('-word_filter', type=argparse.FileType('r'))
    parser.add_argument('-pos_filter', type=argparse.FileType('r'))
    
    args = parser.parse_args()
    config = toml.load(args.config)

    validation_input = args.val_input
    training_input = args.train_input

    pos_filter = None
    word_filter = None

    if args.word_filter:
        word_filter = {w.strip().lower() for w in args.word_filter}
    if args.pos_filter:
        pos_filter = {pos.strip() for pos in args.pos_filter}

    # pad = 0
    enc_char_numberer = Numberer(first_elements=['<PAD>'])
    # pad = 0, bow = 1, eow = 2
    dec_char_numberer = Numberer(first_elements=['<PAD>', '<BOW>', '<EOW>'])
    pos_numberer = Numberer()
    # pad = 0
    morph_numberer = Numberer(first_elements=['<PAD>'])

    py_train_data = read_unique_tokens(file=training_input,
                                       enc_numberer=enc_char_numberer,
                                       dec_numberer=dec_char_numberer,
                                       pos_numberer=pos_numberer,
                                       morph_numberer=morph_numberer,
                                       pos_filter=pos_filter,
                                       word_filter=word_filter,
                                       reverse=False,
                                       train=True)

    py_validation_data = read_unique_tokens(file=validation_input,
                                            enc_numberer=enc_char_numberer,
                                            dec_numberer=dec_char_numberer,
                                            pos_numberer=pos_numberer,
                                            morph_numberer=morph_numberer,
                                            pos_filter=pos_filter,
                                            word_filter=word_filter,
                                            reverse=False,
                                            train=False)

    train_file_name = os.path.splitext(os.path.split(training_input.name)[-1])[0]

    config = read_config(config, enc_char_numberer, dec_char_numberer, pos_numberer, morph_numberer, len(py_train_data))

    numpy_train_data = py2numpy(py_train_data, config)
    numpy_validation_data = py2numpy(py_validation_data, config)
    
    train_batches_per_epoch = int(np.ceil(len(py_train_data) / config.batch_size))
    print("Train shape: ",numpy_train_data[0].shape,"Batches per epoch: ", train_batches_per_epoch)
    print("Val Shape: ", numpy_validation_data[0].shape)
    assert config.sampling_strategy in ['linear', 'random']

    if config.sampling_strategy == 'linear':
        train_batches = get_batches(numpy_train_data, config.batch_size)

    validation_batches = get_batches(numpy_validation_data, config.batch_size)
    val_posses = np.unique(numpy_validation_data[-2])

    conf = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_frac,allow_growth=True)
                                   ,device_count={'GPU': 1})
    best_accuracy = -1
    patience = 0
    with tf.Session(config=conf) as sess:
        with tf.name_scope("train"):
            with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
                m = BasicModel(Mode.Train, config)
        with tf.name_scope("val"):
            with tf.variable_scope("model", reuse=True):
                v = BasicModel(Mode.Validation, config)
        with tf.device("cpu:0"):
            with tf.name_scope("pred"):
                with tf.variable_scope("model", reuse=True):
                    p = BasicModel(Mode.Predict, config)
        sess.run([tf.global_variables_initializer(), tf.tables_initializer(name="init_tables")])
        
        if config.save_path != '':
            if not os.path.isdir(config.save_path):
                os.mkdir(config.save_path)
            saver = tf.train.Saver(max_to_keep=0)
            tensor2op = lambda x : {k:v.split(":")[0] for k,v in x.items()}
            inputs = tensor2op({"input": p.input_plc.name, "input_lengths": p.input_lengths.name,
                      "morph": p.morph_plc.name,"pos": p.pos_plc.name})
            outputs = tensor2op({"ids": p.predictions.name, "chars": p.char_preds.name,"seq_length":p.seq_lengths.name})
            init_ops = {"table": "init_tables"}
            ops = {"input": inputs, "output": outputs, "init_ops": init_ops}
            morph_feats = morph_numberer.num2value
            save_cfg = {"ops":ops, "max_morph_tags": config.max_morph_tags, "pad_symbol": config.pad_sym, "morph_feats":morph_feats}

            with open(os.path.join(config.save_path,"inference_config.toml"), "w") as o:
                toml.dump(save_cfg, o)

        validation_batch_counter = 0
        while patience < config.patience:
            train_ep_loss = 0

            if config.sampling_strategy == 'random':
                train_batches = sample_n_batches(numpy_train_data, train_batches_per_epoch, config.batch_size)

            for x, (train_forms, train_form_lengths, train_lemmas, train_lemma_lengths, train_pos, train_morph) in enumerate(
                    train_batches, start=1):
                _, train_loss, global_step = sess.run([m.train_op, m.loss, m.global_step],
                                                      {m.input: train_forms,
                                                       m.input_lengths: train_form_lengths,
                                                       m.morph_input: train_morph,
                                                       m.pos_input: train_pos,
                                                       m.dec_input: train_lemmas,
                                                       m.dec_lengths: train_lemma_lengths})
                train_ep_loss += train_loss

            print('Evaluating after step {}'.format(global_step))
            val_ep_loss = 0
            val_ep_acc = 0
            val_ep_preds = []

            for n, (val_forms, val_form_lengths, val_lemmas, val_lemma_lengths, val_pos, val_morph) in enumerate(
                    get_batches(numpy_validation_data, config.batch_size, config.batching_policy), start=1):
                preds, val_loss, acc, per_ex = sess.run([v.predictions, v.loss, v.acc, v.per_example_acc],
                                                            {v.input: val_forms,
                                                             v.morph_input: val_morph,
                                                             v.pos_input: val_pos,
                                                             v.input_lengths: val_form_lengths,
                                                             v.dec_input: val_lemmas,
                                                             v.dec_lengths: val_lemma_lengths})
                validation_batch_counter += 1

                val_ep_loss += val_loss

                val_ep_acc += acc

            print('Patience at: {} of {}'.format(patience, config.patience),
                  'Training loss: {:.2f}'.format(train_ep_loss / x),
                  'Validation acc: {:.2f}'.format((val_ep_acc / n) * 100),
                  'Validation loss: {:.2f}'.format(val_ep_loss / n), sep=' ::: ')

            if best_accuracy - val_ep_acc < 0:
                patience = 0
                best_accuracy = val_ep_acc
                if config.save_path != '':
                    if not os.path.isdir(config.save_path):
                        os.mkdir(config.save_path)
                    x = outputs.values()
                    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                        sess,
                        sess.graph_def,
                        list(x)+list(init_ops.values()))
                    save_name = "ohnomore_{}.pb".format(train_file_name)
                    with open(os.path.join(config.save_path, save_name), 'wb') as f:
                        f.write(frozen_graph_def.SerializeToString())

            else:
                patience += 1
