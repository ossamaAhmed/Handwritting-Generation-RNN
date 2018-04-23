import numpy as np


def data_generator(inputs, batch_size=1):
    for n in range(0, len(inputs[0]) - batch_size + 1, batch_size):
        result = []
        for j in range(0, len(inputs)):
            result.append(inputs[j][n:n + batch_size])
        yield result


def get_standard_normalization_params():
    x_std = 2.0943575
    x_mean = 0.41900063
    y_std = 1.8706665
    y_mean = -0.006098041
    return x_mean, x_std, y_mean, y_std


def convert_sentence_to_one_hot_encoding(alphabet, sentence, max_num_of_chars):
    result = []
    for char in sentence:
        result.append([0 if char != alphabet_char else 1 for alphabet_char in alphabet])
    #for the rest of the max_num_of_chars fill it with zeros? -> double check in paper
    for i in range(max_num_of_chars - len(sentence)):
        result.append([0 for _ in alphabet])
    res = np.stack(result)
    return res


def convert_one_hot_encoding_to_sentence(alphabet, one_hot_vectors):
    result = []
    for vector in one_hot_vectors:
        char = alphabet[np.where(vector == 1)]
        if len(char) == 1:
            result.append(char[0])
    res = ''.join(result)
    return res


def define_alphabet():
    alphabet = []
    for i in range(ord('a'), ord('z') + 1):
        alphabet.append(chr(i))
    for i in range(ord('A'), ord('Z') + 1):
        alphabet.append(chr(i))
    # for i in range(ord('0'), ord('9') + 1):
    #     alphabet.append(chr(i))
    # alphabet.extend([' ', '"', "\'", '(', ')', ',', '#', '-', '?', '!', ';', ':'])
    alphabet = np.array(alphabet)
    return alphabet


def get_corresponding_chars_in_sentence(sentence, start_index, end_index, strokes_sentence):
    skip_cuts = 0
    include_cuts = 0
    total_cuts = 0
    for i in range(0, len(strokes_sentence)):
        if strokes_sentence[i, 0] == 1:
            if i < start_index:
                skip_cuts += 1
            elif start_index <= i < end_index:
                include_cuts += 1
            total_cuts += 1
    #calculate on average for this person hand writting how many cuts per char
    cuts_per_char = total_cuts / float(len(sentence))
    print(cuts_per_char)
    skip_chars = int(skip_cuts / cuts_per_char)
    include_chars = int(include_cuts / cuts_per_char)
    # # approximatly how many characters before start_index
    # skip_chars = 0
    # for i in range(0, start_index):
    #     if strokes_sentence[i, 0] == 1:
    #         skip_chars += 1
    # #we should have a prior about how many cut per char in alphabet and use this info
    # #approximatly how many character between start_index and end_index
    # include_chars = 0
    # for i in range(start_index, end_index):
    #     if strokes_sentence[i, 0] == 1:
    #         include_chars += 1
    if skip_chars + include_chars <= len(strokes_sentence):
        return sentence[skip_chars: skip_chars + include_chars]
    else:
        return sentence[skip_chars:]