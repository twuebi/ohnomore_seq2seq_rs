def read_conllx(file, enc_numberer=None, dec_numberer=None, pos_numberer=None, morph_numberer=None, reverse=False,
                train=False, vectorize=True, word_filter=None, pos_filter=None):
    """
    :param file: file or list like sequence
    :param enc_numberer: an instance of Numberer, used to assign numbers to characters of the encoder
    :param dec_numberer: an instance of Numberer, used to assign numbers to symbols of the decoder
    :param reverse: boolean, if true input forms get reversed
    :param train: boolean, if true unseen entities get unknown idx or unqiue one
    :param vectorize: boolean, if true returned tuples will be numbered
    :param word_filter: contains words to exclude
    :param pos_filter: contains part of speeches to exclude
    :return: iterator yielding tuples: (form, lemma, pos)
    """
    for line in file:
        line = line.strip('\n')
        parts = line.split('\t')

        if len(parts) == 10:
            form = parts[1].lower()
            lemma = parts[2].lower()
            pos = parts[4]

            morph = parts[5].split("|")

            def check(form, lemma, pos):
                if pos_filter and pos in pos_filter:
                    return True
                if word_filter and form in word_filter:
                    return True
                if word_filter and lemma in word_filter:
                    return True
                if word_filter and (lemma.replace("ß", "ss") in word_filter or lemma.replace("ss", "ß") in word_filter):
                    return True
                if word_filter and (lemma.replace("ü", "ue") in word_filter or lemma.replace("ue", "ü") in word_filter):
                    return True
                if word_filter and (lemma.replace("ö", "oe") in word_filter or lemma.replace("oe", "ö") in word_filter):
                    return True
                if word_filter and (lemma.replace("ä", "ae") in word_filter or lemma.replace("ae", "ö") in word_filter):
                    return True

            if vectorize:
                vectorized_form = enc_numberer.number_sequence(form, train)

                if reverse:
                    vectorized_form = tuple(reversed(vectorized_form))

                vectorized_lemma = dec_numberer.number_sequence(lemma, train)
                vectorized_pos = pos_numberer.number(pos, train)
                vectorized_morph = morph_numberer.number_sequence(morph, train)

                token = (vectorized_form,
                         vectorized_lemma,
                         vectorized_pos,
                         vectorized_morph)

            else:
                # no vectorization -> plain text
                token = (parts[1].lower(),
                         parts[2].lower(),
                         parts[4],
                         parts[5])

            yield token

def read_unique_tokens(file, enc_numberer, dec_numberer, reverse, train, pos_numberer=None, morph_numberer=None,
                       word_filter=None, pos_filter=None):
    """
    :param file: conll-x file to be read
    :param enc_numberer: an instance of Numberer, used to assign numbers to characters
    :param dec_numberer: an instance of Numberer, used to assign numbers to parts of speech
    :param reverse: boolean, if true input sequences are reversed
    :param train: boolean, if true unseen entities get unknown idx else unique one
    :param word_filter: contains words to exclude
    :param pos_filter: contains part of speeches to exclude
    :return: set containing unique tokens of file
    """
    tokens = read_conllx(file=file,
                         enc_numberer=enc_numberer,
                         dec_numberer=dec_numberer,
                         pos_numberer=pos_numberer,
                         morph_numberer=morph_numberer,
                         reverse=reverse,
                         train=train,
                         word_filter=word_filter,
                         pos_filter=pos_filter)

    return filter_uniq_tokens(tokens)


def filter_uniq_tokens(tokens):
    return set(tokens)


def write_tokens_to_file(tokens, outp):
    """
    Writes tokens to  file, only writes form, lemma and pos, replaces other conllx fields with '_'
    :param tokens: list of tuples with length 3, (form, lemma, pos)
    :param outp: file to write to
    """
    underscore = '_'
    for token in tokens:
        pseudo_conll_x = '{und}\t{form}\t{lemma}\t{und}\t{pos}\t{morph}\t{und}\t{und}\t{und}\t{und}'\
                                    .format(form=token[0],lemma=token[1],pos=token[2],morph=token[3],und=underscore)
        print(pseudo_conll_x, file=outp)
