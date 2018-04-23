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
    for i in range(ord('0'), ord('9') + 1):
        alphabet.append(chr(i))
    alphabet.extend([' ', '"', "\'", '(', ')', ',', '#', '-', '?', '!', ';', ':'])
    alphabet = np.array(alphabet)
    return alphabet