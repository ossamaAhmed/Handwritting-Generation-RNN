import tensorflow as tf

import numpy as np
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_generator import DataGenerator
from models.unconditional_model import Model as UnConditionalModel
#tf.set_random_seed(0)
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'data', 'Input Directory.')
flags.DEFINE_string('output_dir', 'output', 'Output Directory.')
experiment_name = 'high_learning_rate_high_dropout_data_normalized'


class TrainModel(object):
    def __init__(self, strokes_file_path, labels_file_path):
        #load data
        self.strokes_file_path = strokes_file_path
        self.labels_file_path = labels_file_path
        self.datagen = None
        self.model = None
        self.EPOCHS = 100
        self.learning_rate = 5e-4
        self.decay_rate = 0.95


    def load_data(self):
        self.datagen = DataGenerator(strokes_file_path=self.strokes_file_path, labels_file_path=self.labels_file_path)

    def train(self):
        self.load_data()
        #self.model = Model()
        self.model = UnConditionalModel()
        self.model.build_model()
        saver = tf.train.Saver()
        summary_proto = tf.Summary()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            tf.global_variables_initializer().run()
            writer_file_path = os.path.join(FLAGS.output_dir, experiment_name, 'improved_graph')
            checkpoint_file = os.path.join(FLAGS.output_dir, experiment_name, 'unconditional_model')
            writer = tf.summary.FileWriter(writer_file_path, sess.graph)
            for epoch in range(0, self.EPOCHS):
                sess.run(tf.assign(self.model.learning_rate,
                                   self.learning_rate * (self.decay_rate ** epoch)))
                print("Epoch number " + str(epoch))
                batch_generator = self.datagen.generate_strokes_to_strokes_batch(batch_size=30,
                                                                                 sequence_length=300)

                batch_idx = 0
                average_loss = 0.0
                for batch in batch_generator:
                    stroke_t, stroke_t_plus_one = batch
                    feed_dict = {self.model.stroke_t: stroke_t,
                                 self.model.stroke_t_plus_one: stroke_t_plus_one}
                    global_step, summary_train, _, network_loss = sess.run([self.model.global_step,
                                                                            self.model.summary_op,
                                                                            self.model.train_op,
                                                                            self.model.network_loss],
                                                                           feed_dict=feed_dict)
                    average_loss += network_loss
                    writer.add_summary(summary_train, global_step=global_step)
                    if batch_idx % 50 == 0:
                        saver.save(sess, checkpoint_file, global_step=global_step)
                        summary_proto.ParseFromString(summary_train)
                        print("Batch Number ", batch_idx , " | Average Loss is: ", (average_loss / (batch_idx + 1)))
                        print(summary_proto)
                    batch_idx += 1
                print("Epoch ", str(epoch), " has an average loss of ", (average_loss /batch_idx))


def main(_):
    training = TrainModel(strokes_file_path=os.path.join(FLAGS.data_dir, 'strokes.npy'),
                          labels_file_path=os.path.join(FLAGS.data_dir, 'sentences.txt'))
    training.train()


if __name__ == '__main__':
    tf.app.run()


