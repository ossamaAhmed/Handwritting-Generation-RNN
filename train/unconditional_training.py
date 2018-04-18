import tensorflow as tf

import numpy as np
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_generator import DataGenerator
from models.unconditional_model import Model as UnConditionalModel
from configs.config import TrainingConfig
#tf.set_random_seed(0)
flags = tf.app.flags
FLAGS = flags.FLAGS
train_config = TrainingConfig()
flags.DEFINE_string('data_dir', 'data', 'Input Directory.')
flags.DEFINE_string('output_dir', 'output', 'Output Directory.')
experiment_name = 'high_learning_rate_high_dropout_data_normalized'


class TrainModel(object):
    def __init__(self, strokes_file_path, labels_file_path):
        #load data
        self.strokes_file_path = strokes_file_path
        self.labels_file_path = labels_file_path
        self.datagen = None
        self.train_model = None

    def load_data(self):
        self.datagen = DataGenerator(strokes_file_path=self.strokes_file_path, labels_file_path=self.labels_file_path)

    def train(self):
        self.load_data()
        self.train_model = UnConditionalModel(train_config)
        self.train_model.build_model()
        saver = tf.train.Saver(max_to_keep=50)
        summary_proto = tf.Summary()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            tf.global_variables_initializer().run()
            writer_file_path = os.path.join(FLAGS.output_dir, experiment_name, 'improved_graph')
            checkpoint_file = os.path.join(FLAGS.output_dir, experiment_name, 'unconditional_model')
            writer = tf.summary.FileWriter(writer_file_path, sess.graph)
            valid_loss = 0
            for epoch in range(0, train_config.EPOCHS):
                sess.run(tf.assign(self.train_model.learning_rate,
                                   train_config.learning_rate * (train_config.decay_rate ** epoch)))
                print("Epoch number " + str(epoch))
                batch_generator, validation_set = self.datagen.preprocess_data(batch_size=train_config.BATCH_SIZE,
                                                                               sequence_length=
                                                                               train_config.SEQUENCE_LENGTH)

                batch_idx = 0
                average_loss = 0.0
                state = sess.run(self.train_model.initial_state)
                val_x, val_y = next(validation_set)
                validation_feed = {self.train_model.initial_state: state, self.train_model.stroke_t: val_x,
                                   self.train_model.stroke_t_plus_one: val_y}
                for batch in batch_generator:
                    stroke_t, stroke_t_plus_one = batch
                    feed_dict = {self.train_model.initial_state: state, self.train_model.stroke_t: stroke_t,
                                 self.train_model.stroke_t_plus_one: stroke_t_plus_one}
                    #ADD STATE
                    state, global_step, summary_train, _, network_loss = sess.run([self.train_model.final_state,
                                                                                   self.train_model.global_step,
                                                                                   self.train_model.summary_op,
                                                                                   self.train_model.train_op,
                                                                                   self.train_model.network_loss],
                                                                                  feed_dict=feed_dict)
                    average_loss += network_loss
                    writer.add_summary(summary_train, global_step=global_step)
                    valid_loss_summary, valid_loss, = sess.run([self.train_model.validation_summary,
                                                                self.train_model.network_loss],
                                                               validation_feed)
                    writer.add_summary(valid_loss_summary, global_step=global_step)

                    print('Epoch ', epoch, ' and Batch ', batch_idx + 1, ' | training loss is ',
                          average_loss / (batch_idx + 1), ' | validation loss is ', valid_loss)

                    if batch_idx % 2 == 0:
                        saver.save(sess, checkpoint_file, global_step=global_step)
                        summary_proto.ParseFromString(summary_train)
                    batch_idx += 1
                print("Finished Epoch ", str(epoch), " which has an average training loss of ", (average_loss / batch_idx),
                      ' and validation loss of ', valid_loss)


def main(_):
    training = TrainModel(strokes_file_path=os.path.join(FLAGS.data_dir, 'strokes.npy'),
                          labels_file_path=os.path.join(FLAGS.data_dir, 'sentences.txt'))
    training.train()


if __name__ == '__main__':
    tf.app.run()


