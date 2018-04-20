from io import BytesIO
import numpy as np
from tensorflow.python.lib.io import file_io
import itertools
#np.random.rand(0)


class DataGenerator(object):
    def __init__(self, strokes_file_path, labels_file_path):
        self.strokes_file_path = strokes_file_path
        self.labels_file_path = labels_file_path
        self.train_strokes = None
        self.train_sentences = None
        self.validation_strokes = None
        self.validation_sentences = None
        self.alphabet = None
        self.read_data()

    def read_data(self):
        self.strokes = np.load(BytesIO(file_io.read_file_to_string(self.strokes_file_path, binary_mode=True)),
                               encoding='latin1')
        with file_io.FileIO(self.labels_file_path, 'r') as f:
            self.texts = np.array(f.readlines())
        # with open(self.labels_file_path) as f:
        #     self.texts = f.readlines()
        #divide dataset into validation and training
        indices = np.arange(len(self.strokes))
        np.random.shuffle(indices)
        data_size = len(indices)
        train_size = int(0.65 * data_size)
        self.train_strokes = self.strokes[indices[0:train_size]]
        self.train_sentences = self.texts[indices[0:train_size]]
        self.validation_strokes = self.strokes[indices[train_size:]]
        self.validation_sentences = self.texts[indices[train_size:]]
        #constructing the alphabet commas, hyphens dot ' ..etc
        self.alphabet = []
        for i in range(ord('a'), ord('z') + 1):
            self.alphabet.append(chr(i))
        for i in range(ord('A'), ord('Z') + 1):
            self.alphabet.append(chr(i))
        for i in range(ord('0'), ord('9') + 1):
            self.alphabet.append(chr(i))
        self.alphabet.append(' ')
        self.alphabet.append('"')
        self.alphabet.append('\'')

    def preprocess_data(self, batch_size, sequence_length=1):
        training_input, training_target= self.data_preprocessing_strokes_to_strokes(self.train_strokes,
                                                                                    sequence_length)
        validation_input, validation_target = self.data_preprocessing_strokes_to_strokes(self.validation_strokes,
                                                                                         sequence_length)
        training_data_generator = self.training_data_generator([training_input, training_target],
                                                               batch_size=batch_size)
        validation_data_generator = self.validation_data_generator([validation_input, validation_target],
                                                                   batch_size=batch_size)

        return training_data_generator, validation_data_generator

    def preprocess_data_conditional(self, batch_size, max_num_of_chars, sequence_length=1):
        training_input, training_target, training_sentences_input = self.data_preprocessing_chars_to_strokes(
            self.train_strokes,
            self.train_sentences,
            max_num_of_chars,
            sequence_length)

        validation_input, validation_target, validation_sentences_input = self.data_preprocessing_chars_to_strokes(
            self.validation_strokes,
            self.validation_sentences,
            max_num_of_chars,
            sequence_length)
        training_data_generator = self.training_data_generator(
            [training_input, training_target, training_sentences_input],
            batch_size=batch_size)
        validation_data_generator = self.validation_data_generator(
            [validation_input, validation_target,validation_sentences_input],
            batch_size=batch_size)

        return training_data_generator, validation_data_generator

    def data_preprocessing_strokes_to_strokes(self, strokes, sequence_length=1):
        inputs = []
        targets = []
        strokes = strokes.tolist()
        for i in range(0, len(strokes)):
            strokes_sentence = strokes[i]
            if len(strokes_sentence) < sequence_length + 1: # skip sequences less than sequence length + 1
                continue
            number_of_samples = int(np.round(len(strokes_sentence) / float(sequence_length)))
            #print(number_of_samples)
            for j in range(0, number_of_samples):
                # choose a random num between 0 and len(strokes_sentence) - sequence_length - 1
                # later have a restriction on the intersection of the chosen sets to limit a
                # sequence chosen more than once
                start_idx = np.random.randint(0, len(strokes_sentence) - sequence_length) #to account for samples < sequence
                inputs.append(strokes_sentence[start_idx:start_idx + sequence_length])
                targets.append(strokes_sentence[start_idx + 1:start_idx + sequence_length + 1])
        #wrong because of bernouli (fixed)
        inputs = np.array(inputs)
        targets = np.array(targets)
        x_mean = inputs[:, :, 1].mean()
        x_std = inputs[:, :, 1].std()
        y_mean = inputs[:, :, 2].mean()
        y_std = inputs[:, :, 2].std()
        inputs[:, :, 1] = (inputs[:, :, 1] - x_mean) / x_std
        inputs[:, :, 2] = (inputs[:, :, 2] - y_mean) / y_std
        targets[:, :, 1] = (targets[:, :, 1] - x_mean) / x_std
        targets[:, :, 2] = (targets[:, :, 2] - y_mean) / y_std
        #further scaling for numerical stability
        # inputs[:, :, 1] = inputs[:, :, 1] / (inputs[:, :, 1].max() - inputs[:, :, 1].min())
        # inputs[:, :, 2] = inputs[:, :, 2] / (inputs[:, :, 2].max() - inputs[:, :, 2].min())
        # targets[:, :, 1] = targets[:, :, 1] / (targets[:, :, 1].max() - targets[:, :, 1].min())
        # targets[:, :, 2] = targets[:, :, 2] / (targets[:, :, 2].max() - targets[:, :, 2].min())
        return inputs, targets

    def data_preprocessing_chars_to_strokes(self, strokes, sentences, max_num_of_chars, sequence_length=1):
        inputs = []
        targets = []
        sentences_input = []
        strokes = strokes.tolist()
        for i in range(0, len(strokes)):
            strokes_sentence = strokes[i]
            char_sentence = sentences[i]
            if len(strokes_sentence) < sequence_length + 1:# skip sequences less than sequence length + 1
                continue
            start_idx = 0
            inputs.append(strokes_sentence[start_idx:start_idx + sequence_length])
            targets.append(strokes_sentence[start_idx + 1:start_idx + sequence_length + 1])
            sentences_input.append(self.convert_sentence_to_one_hot_encoding(char_sentence, max_num_of_chars)
                                   [:max_num_of_chars])
        #wrong because of bernouli (fixed)
        inputs = np.array(inputs)
        targets = np.array(targets)
        x_mean = inputs[:, :, 1].mean()
        x_std = inputs[:, :, 1].std()
        y_mean = inputs[:, :, 2].mean()
        y_std = inputs[:, :, 2].std()
        inputs[:, :, 1] = (inputs[:, :, 1] - x_mean) / x_std #should we do this for validation set?
        inputs[:, :, 2] = (inputs[:, :, 2] - y_mean) / y_std
        targets[:, :, 1] = (targets[:, :, 1] - x_mean) / x_std
        targets[:, :, 2] = (targets[:, :, 2] - y_mean) / y_std
        #further scaling for numerical stability
        # inputs[:, :, 1] = inputs[:, :, 1] / (inputs[:, :, 1].max() - inputs[:, :, 1].min())
        # inputs[:, :, 2] = inputs[:, :, 2] / (inputs[:, :, 2].max() - inputs[:, :, 2].min())
        # targets[:, :, 1] = targets[:, :, 1] / (targets[:, :, 1].max() - targets[:, :, 1].min())
        # targets[:, :, 2] = targets[:, :, 2] / (targets[:, :, 2].max() - targets[:, :, 2].min())
        return inputs, targets, sentences_input

    @staticmethod
    def training_data_generator(inputs, batch_size=1):
        for n in range(0, len(inputs[0]) - batch_size + 1, batch_size):
            result = []
            for j in range(0, len(inputs)):
                result.append(inputs[j][n:n + batch_size])
            yield result

    def convert_sentence_to_one_hot_encoding(self, sentence, max_num_of_chars):
        result = []
        for char in sentence:
            result.append([0 if char != alphabet_char else 1 for alphabet_char in self.alphabet])
        #for the rest of the max_num_of_chars fill it with zeros? -> double check in paper
        for i in range(max_num_of_chars - len(sentence)):
            result.append([0 for _ in self.alphabet])
        res = np.stack(result)
        # print(np.shape(res))
        return res

    @staticmethod
    def validation_data_generator(inputs, batch_size=1): #keep feeding validation sets since its smaller than the training one
        while True:
            for n in range(0, len(inputs[0]) - batch_size + 1, batch_size):
                result = []
                for j in range(0, len(inputs)):
                    result.append(inputs[j][n:n + batch_size])
                yield result


# datagen = DataGenerator(strokes_file_path='./data/strokes.npy', labels_file_path='./data/sentences.txt')
# # train_gen, valid_gen = datagen.preprocess_data(batch_size=20, sequence_length=167)
# c_train_gen, c_valid_gen = datagen.preprocess_data_conditional(batch_size=20, max_num_of_chars=50, sequence_length=250)
# for batch in c_train_gen:
#     stroke_t, stroke_t_plus_one, sentence = batch
#     print(np.amax(stroke_t[:,:,0]))
#     print(np.amin(stroke_t[:,:,0]))
#     print(np.shape(sentence)) # [BATCH, U, #max_chars]
#     break
# print(np.shape(DataGenerator.convert_sentence_to_one_hot_encoding('abc')))
