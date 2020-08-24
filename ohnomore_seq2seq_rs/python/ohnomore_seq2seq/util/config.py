from collections import namedtuple
from .attentions import Attentions

Config = namedtuple("Config",
                    "save_path gpu_frac batching_policy sampling_strategy truncate patience "
                    "batch_size max_timesteps pos_numberer enc_char_numberer morph_numberer dec_char_numberer start_sym "
                    "pad_sym end_sym pad_idx start_idx end_idx "
                    "enc_vocab_size dec_vocab_size pos_vocab_size morph_vocab_size max_morph_tags char_embedding_size "
                    "morph_embedding_size pos_embedding_size optimizer learning_rate max_gradient_norm "
                    "cell_type layers hsize encoder encoder_cellclip decoder_cellclip "
                    "attention attention_kind attention_size "
                    "input_dropout encoder_dropout decoder_dropout projection_dropout "
                    )


def read_config(config, enc_char_numberer, dec_char_numberer, pos_numberer, morph_numberer, num_examples):
    try:
        attention_kind = Attentions[config['attention_kind']]
    except:
        raise NotImplementedError("Attention mechanism '{}' is not supported.".format(config['attention_kind']))
    return Config(gpu_frac=config['gpu_frac'],
                  save_path=config['save_path'],
                  batch_size=min(config['batch_size'], num_examples),
                  patience=config['patience'],
                  sampling_strategy=config['sampling_strategy'],
                  batching_policy=config['batching_policy'],
                  truncate=config['truncate'],
                  max_timesteps=config['max_timesteps'],
                  pad_sym=config['pad_sym'],
                  start_sym=config['start_sym'],
                  end_sym=config['end_sym'],
                  pad_idx=enc_char_numberer[config['pad_sym']],
                  pos_numberer=pos_numberer,
                  enc_char_numberer=enc_char_numberer,
                  morph_numberer=morph_numberer,
                  dec_char_numberer=dec_char_numberer,
                  start_idx=dec_char_numberer[config['start_sym']],
                  end_idx=dec_char_numberer.number('<EOW>', train=False),
                  enc_vocab_size=enc_char_numberer.max,
                  dec_vocab_size=dec_char_numberer.max,
                  morph_vocab_size=morph_numberer.max,
                  pos_vocab_size=pos_numberer.max,
                  max_morph_tags=config['max_morph_tags'],
                  char_embedding_size=config['char_embedding_size'],
                  pos_embedding_size=config['pos_embedding_size'],
                  morph_embedding_size=config['morph_embedding_size'],
                  optimizer=config['optimizer'],
                  learning_rate=config['learning_rate'],
                  max_gradient_norm=config['max_gradient_norm'],
                  cell_type=config['cell_type'],
                  layers=config['layers'],
                  hsize=config['hsize'],
                  encoder=config['encoder'],
                  encoder_cellclip=config['encoder_cellclip'],
                  decoder_cellclip=config['decoder_cellclip'],
                  attention=config['attention'],
                  attention_kind=attention_kind,
                  attention_size=config['attention_size'],
                  input_dropout=config['input_dropout'],
                  encoder_dropout=config['encoder_dropout'],
                  decoder_dropout=config['decoder_dropout'],
                  projection_dropout=config['projection_dropout'])