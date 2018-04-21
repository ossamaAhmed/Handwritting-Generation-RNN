import numpy as np


def data_generator(inputs, batch_size=1):
    for n in range(0, len(inputs[0]) - batch_size + 1, batch_size):
        result = []
        for j in range(0, len(inputs)):
            result.append(inputs[j][n:n + batch_size])
        yield result


def convert_sentence_to_one_hot_encoding(alphabet, sentence, max_num_of_chars):
    result = []
    for char in sentence:
        if char in alphabet:
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
