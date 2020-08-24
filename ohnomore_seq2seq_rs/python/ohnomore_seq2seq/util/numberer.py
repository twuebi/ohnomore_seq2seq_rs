class Numberer:
    def __init__(self, unknown_token="<UNK>", first_elements=None, all_elements=None):
        """
        :param unknown_token: The representation of unknown entities, will be indexed after first elements
        :param first_elements: First elements to initialize the numberer with. Will be inserted before unknown tokens
        :param all_elements: used to initialize numberer with list, assumes that unknown token is in all_elements
        """
        if all_elements:
            self.num2value = all_elements
            self.value2num = dict(zip(self.num2value, range(len(self.num2value))))
            self.idx = len(self.num2value)
        elif first_elements:
            self.num2value = [*first_elements, unknown_token]
            self.value2num = dict(zip(self.num2value, range(len(self.num2value))))
            self.idx = len(self.num2value)
        else:
            self.num2value = [unknown_token]
            self.value2num = {unknown_token: 0}
            self.idx = 1
        self.unknown_idx = self.value2num[unknown_token]

    def number(self, value, train):
        if train and value not in self.value2num:
            self.value2num[value] = self.idx
            self.num2value.append(value)
            self.idx += 1
        return self.value2num.get(value, self.unknown_idx)

    def __getitem__(self, item):
        return self.value2num[item]

    @property
    def max(self):
        """Returns the number of elements, not the index of the last added
        item"""
        return self.idx

    def value(self, num):
        return self.num2value[num]

    def number_sequence(self, sequence, train):
        return tuple(self.number(element, train=train) for element in sequence)

    def decode_sequence(self, sequence, stop_sym=None):
        dec_seq = "".join([self.num2value[n] for n in sequence])
        if stop_sym:
            dec_seq = dec_seq.split(stop_sym)[0]
        return dec_seq

    def to_file(self, file):
        print("\n".join(self.num2value), file=file)


def load_numberer_from_file(file):
    num2val = list(map(lambda x: x.strip('\n'), file.readlines()))
    return Numberer(all_elements=num2val)