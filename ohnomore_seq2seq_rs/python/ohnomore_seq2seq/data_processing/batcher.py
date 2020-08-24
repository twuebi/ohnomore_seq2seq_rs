import numpy as np
import random
import sys

def py2numpy(tokens, config):
    """
    Iterates over tokens and converts the un-fixed size representation to fixed size ones. Prepends pos and appends
    <eow> to lemma which is a requirement of decoder.

    :param tokens: list like sequence of tuples with tuple such that tuple[0] = form, tuple[1] = lemma, tuple[2] = pos
    :param config: named tuple containing fields max_timesteps, end_idx and truncate
    :return: tuple with (ndarray: forms, ndarray: form_lenghts, ndarray: lemmas, ndarray: lemma_lengths, ndarray: pos)
    """
    max_timesteps = config.max_timesteps
    eow_idx = config.end_idx
    truncate = config.truncate
    max_morph_tags = config.max_morph_tags
    number_of_instances = len(tokens)

    if number_of_instances == 0:
        print('Received empty token collection. Did you proide an empty input file? Exitting!', file=sys.stderr)
        exit(1)

    forms = np.zeros(shape=(number_of_instances,
                            max_timesteps),
                     dtype=np.uint16)
    form_lengths = np.zeros(shape=(number_of_instances),
                            dtype=np.uint16)

    lemmas = np.zeros(shape=(number_of_instances,
                             max_timesteps + 2),  # + 2 for <bow> and <eow>
                      dtype=np.uint16)
    lemma_lengths = np.zeros(shape=(number_of_instances),
                             dtype=np.uint16)

    pos = np.zeros(shape=(number_of_instances),
                   dtype=np.uint16)

    morph = np.zeros(shape=(number_of_instances,max_morph_tags),dtype=np.uint16)

    for num, example in enumerate(tokens):
        form_example, lemma_example, pos_example, morph_example = example

        form = truncate_sequence(form_example, max_timesteps, truncate)
        form_length = len(form)

        form_lengths[num] = form_length
        forms[num, :form_length] = form

        # prepend pos_tag
        lemmas[num, 0] = pos_example
        # actual lemma
        lemma = truncate_sequence(lemma_example, max_timesteps, truncate)

        lemma_length = len(lemma)

        lemmas[num, 1:1 + lemma_length] = lemma
        # append <eow>
        lemmas[num, 1 + lemma_length] = eow_idx

        # + 1 since either <bow> (training / inference) or <eow> (target loss) needs to be processed
        lemma_lengths[num] = lemma_length + 1

        pos[num] = pos_example

        morph[num, :len(morph_example)] = morph_example

    return forms, form_lengths, lemmas, lemma_lengths, pos, morph


def truncate_sequence(example, max_timesteps, truncation_mode):
    """
    Truncates example if its lengths is greater than max_timesteps. Will perform truncation depending on truncation_mode.
    Truncation_mode is either 'split' or 'suffix'.

    'split': returns max_timesteps // 2 of the first half, skips max_timesteps - length and adds max_timesteps // 2 of
    the last half.
    'suffix': returns sequence up to max_timesteps

    :param example: sequence to truncate
    :param max_timesteps: maximum length of the truncated example
    :param truncation_mode: one of ['suffix', 'split']
    :return: truncated example
    """
    assert truncation_mode in ['suffix', 'split']
    example_length = len(example)

    if example_length <= max_timesteps:
        return example

    if truncation_mode == 'suffix':
        truncated_example = example[:max_timesteps]
    elif truncation_mode == 'split':
        half = example_length // 2
        diff = max(len(example) - max_timesteps, 0)
        truncated_example = example[:half] + example[half + diff:example_length + diff]

    return truncated_example


def get_batches(data, batch_size, mode='use_all'):
    """
    Returns generator batches. Exact batch size and number of batches depends on mode.

    'use_all': uses np.array_split, will return all data but not always with same batch size
    'smaller_last': will return len(data) // batch_size batches with size batch_size and last batch with the rest
    'discard_last': will return len(data) // batch_size batches with size batch_size

    :param data: tuple containing numpy arrays (form, form_lengths, lemmas, lemma_lengths, pos)
    :param batch_size: batch size
    :param mode: one of ['use_all', 'smaller_last', 'discard_last']
    :return: batches
    """
    forms, form_lengths, lemmas, lemma_lengths, pos, morph = data


    keys = list(range(len(forms)))
    if mode == 'use_all':
        num_batches = np.ceil(len(forms) / batch_size)
        batched_keys = np.array_split(keys, num_batches)
    elif mode == 'smaller_last':
        num_batches = int(np.ceil(len(forms) / batch_size))
        batched_keys = [keys[n * batch_size:(1 + n) * batch_size] for n in range(num_batches)]
    elif mode == 'discard_last':
        num_batches = len(forms) // batch_size
        batched_keys = [keys[n * batch_size:(1 + n) * batch_size] for n in range(num_batches)]

    for getters in batched_keys:
        yield forms[getters], form_lengths[getters], lemmas[getters], lemma_lengths[getters], pos[
            getters], morph[getters]


def sample_batch(data, batch_size):
    """
    Samples randomly one batch from data
    """
    forms, form_lengths, lemmas, lemma_lengths, pos, morph = data

    keys = list(range(len(forms)))

    getters = random.sample(keys, batch_size)

    return forms[getters], form_lengths[getters], lemmas[getters], lemma_lengths[getters], pos[getters], morph[getters]


def sample_n_batches(data, n, batch_size):
    """
    Randomly samples n batches
    """
    for _ in range(n):
        yield sample_batch(data, batch_size)