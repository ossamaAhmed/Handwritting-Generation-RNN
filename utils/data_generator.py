from io import BytesIO
import numpy as np
from tensorflow.python.lib.io import file_io
#np.random.rand(0)


class DataGenerator(object):
    def __init__(self, strokes_file_path, labels_file_path):
        self.strokes_file_path = strokes_file_path
        self.labels_file_path = labels_file_path
        self.strokes = None
        self.texts = None
        self.read_data()

    def read_data(self):
        self.strokes = np.load(BytesIO(file_io.read_file_to_string(self.strokes_file_path, binary_mode=True)),
                               encoding='latin1')
        with file_io.FileIO(self.labels_file_path, 'r') as f:
            self.texts = f.readlines()
        # with open(self.labels_file_path) as f:
        #     self.texts = f.readlines()

    def generate_strokes_to_strokes_batch(self, batch_size=1, sequence_length=1):
        inputs = []
        targets = []
        # divide data to [BATCH, SEQUENCE_LENGTH, 3]
        # first for each sentence divide it up to sequence_length
        for i in range(0, len(self.strokes)):
            strokes_sentence = self.strokes[i]
            number_of_samples = int(np.round(len(strokes_sentence) / float(sequence_length)))
            for j in range(0, number_of_samples):
                # choose a random num between 0 and len(strokes_sentence) - sequence_length - 1
                # later have a restriction on the intersection of the chosen sets to limit a
                # sequence chosen more than once
                start_idx = np.random.randint(0, len(strokes_sentence) - sequence_length)
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
        inputs[:, :, 1] = inputs[:, :, 1] / (inputs[:, :, 1].max() - inputs[:, :, 1].min())
        inputs[:, :, 2] = inputs[:, :, 2] / (inputs[:, :, 2].max() - inputs[:, :, 2].min())
        targets[:, :, 1] = targets[:, :, 1] / (targets[:, :, 1].max() - targets[:, :, 1].min())
        targets[:, :, 2] = targets[:, :, 2] / (targets[:, :, 2].max() - targets[:, :, 2].min())
        return self.shuffle_batches(inputs, targets, batch_size)

    @staticmethod
    def shuffle_batches(inputs, targets, batch_size):
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        for n in range(0, len(inputs) - batch_size + 1, batch_size):
            batch = indices[n:n + batch_size]
            yield inputs[batch], targets[batch]


#test
# datagen = DataGenerator(strokes_file_path='../data/strokes.npy', labels_file_path='../data/sentences.txt')
# batch_gen = datagen.generate_strokes_to_strokes_batch(batch_size=20, sequence_length=167)
# for batch in batch_gen:
#     inputs, targets = batch
#     print(np.amax(inputs[:,:,0]))
#     print(np.amin(inputs[:,:,0]))
#     break
