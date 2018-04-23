from io import BytesIO
import numpy as np
from tensorflow.python.lib.io import file_io
from utils.data_utils import convert_sentence_to_one_hot_encoding, data_generator, define_alphabet, \
    get_standard_normalization_params
#np.random.rand(0)
#Standard Normalization to be used in testing and inference as well

x_mean, x_std, y_mean, y_std = get_standard_normalization_params()


class DataGenerator(object):
    def __init__(self, strokes_file_path, labels_file_path):
        self.strokes_file_path = strokes_file_path
        self.labels_file_path = labels_file_path
        self.raw_strokes = None
        self.raw_sentences = None
        self.strokes = None
        self.sentences = None
        self.train_strokes = None
        self.train_sentences = None
        self.validation_strokes = None
        self.validation_sentences = None
        self.data_partition_factor = 0.8
        self.read_data()
        self.alphabet = define_alphabet()

    def read_data(self):
        self.raw_strokes = np.load(BytesIO(file_io.read_file_to_string(self.strokes_file_path, binary_mode=True)),
                                   encoding='latin1')
        with file_io.FileIO(self.labels_file_path, 'r') as f:
            self.raw_sentences = np.array(f.readlines())
        #remove data with zeros as data points in strokes (will confuse the network)
        #TODO:replace some chars to a non-char refer to graves paper page 31
        #TODO: more processing to be done, the data seems to be noisy
        strokes = np.copy(self.raw_strokes)
        texts = np.copy(self.raw_sentences)
        for i in range(0, len(strokes)):
            indicies_to_delete = []
            for j in range(0, len(strokes[i])):
                if strokes[i][j][0] == 0. and strokes[i][j][1] == 0. and strokes[i][j][2] == 0.:
                    indicies_to_delete.append(j)
            strokes[i] = np.delete(strokes[i], indicies_to_delete, axis=0)
        self.strokes = np.copy(strokes)
        indices = np.arange(len(strokes))
        np.random.shuffle(indices)
        data_size = len(indices)
        train_size = int(self.data_partition_factor * data_size)
        self.train_strokes = strokes[indices[0:train_size]]
        self.train_sentences = texts[indices[0:train_size]]
        self.validation_strokes = strokes[indices[train_size:]]
        self.validation_sentences = texts[indices[train_size:]]

    def generate_unconditional_dataset(self, batch_size=30, sequence_length=300):
        training_input, training_target = self.data_preprocessing_strokes_to_strokes(self.train_strokes,
                                                                                     sequence_length)
        validation_input, validation_target = self.data_preprocessing_strokes_to_strokes(self.validation_strokes,
                                                                                         sequence_length)
        training_data_generator = data_generator([training_input, training_target], batch_size=batch_size)
        validation_data_generator = data_generator([validation_input, validation_target], batch_size=batch_size)

        return training_data_generator, validation_data_generator

    def generate_conditional_dataset(self, batch_size=30, max_num_of_chars=15, sequence_length=300):
        training_input, training_target, training_sentences_input = self.data_preprocessing_sentence_to_strokes(
            self.alphabet,
            self.train_strokes,
            self.train_sentences,
            max_num_of_chars,
            sequence_length)

        validation_input, validation_target, validation_sentences_input = self.data_preprocessing_sentence_to_strokes(
            self.alphabet,
            self.validation_strokes,
            self.validation_sentences,
            max_num_of_chars,
            sequence_length)
        training_data_generator = data_generator([training_input, training_target, training_sentences_input],
                                                 batch_size=batch_size)
        validation_data_generator = data_generator([validation_input, validation_target,validation_sentences_input],
                                                   batch_size=batch_size)

        return training_data_generator, validation_data_generator

    @staticmethod
    def data_preprocessing_strokes_to_strokes(strokes, sequence_length=300):
        inputs = []
        targets = []
        strokes = strokes.tolist()
        for i in range(0, len(strokes)):
            strokes_sentence = strokes[i]
            if len(strokes_sentence) < sequence_length + 1: # skip sequences less than sequence length + 1
                continue
            number_of_samples = int(np.round(len(strokes_sentence) / float(sequence_length)))
            for j in range(0, number_of_samples):
                # choose a random num between 0 and len(strokes_sentence) - sequence_length - 1
                # later have a restriction on the intersection of the chosen sets to limit a
                # sequence chosen more than once
                start_idx = np.random.randint(0, len(strokes_sentence) - sequence_length)
                inputs.append(strokes_sentence[start_idx:start_idx + sequence_length])
                targets.append(strokes_sentence[start_idx + 1:start_idx + sequence_length + 1])
        inputs = np.array(inputs)
        targets = np.array(targets)
        # TODO: standard normalization
        inputs[:, :, 1] = (inputs[:, :, 1] - x_mean) / x_std
        inputs[:, :, 2] = (inputs[:, :, 2] - y_mean) / y_std
        targets[:, :, 1] = (targets[:, :, 1] - x_mean) / x_std
        targets[:, :, 2] = (targets[:, :, 2] - y_mean) / y_std
        return inputs, targets

    @staticmethod
    def data_preprocessing_sentence_to_strokes(alphabet, strokes, sentences, max_num_of_chars, sequence_length=300):
        inputs = []
        targets = []
        sentences_input = []
        strokes = strokes.tolist()
        for i in range(0, len(strokes)):
            strokes_sentence = strokes[i]
            char_sentence = sentences[i]
            if len(strokes_sentence) < sequence_length + 1:# skip sequences less than sequence length + 1
                continue
            # TODO: generate more data maybe according to the average number of strokes, the new data might be more noisy
            #TODO: count the number of ones in the strokes_sentence and cut the sentence according to this

            start_idx = 0
            inputs.append(strokes_sentence[start_idx:start_idx + sequence_length])
            targets.append(strokes_sentence[start_idx + 1:start_idx + sequence_length + 1])
            sentences_input.append(convert_sentence_to_one_hot_encoding(alphabet, char_sentence, max_num_of_chars)
                                   [:max_num_of_chars])
        inputs = np.array(inputs)
        targets = np.array(targets)
        # TODO: standard normalization
        inputs[:, :, 1] = (inputs[:, :, 1] - x_mean) / x_std
        inputs[:, :, 2] = (inputs[:, :, 2] - y_mean) / y_std
        targets[:, :, 1] = (targets[:, :, 1] - x_mean) / x_std
        targets[:, :, 2] = (targets[:, :, 2] - y_mean) / y_std
        return inputs, targets, sentences_input
