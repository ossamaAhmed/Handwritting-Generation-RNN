import tensorflow as tf
import sys
import os.path
from utils.data_generator import DataGenerator
from models.unconditional_model import Model as UnConditionalModel
from configs.unconditional_config import TrainingConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

flags = tf.app.flags
FLAGS = flags.FLAGS
train_config = TrainingConfig()
flags.DEFINE_string('data_dir', 'data', 'Input Directory.')
flags.DEFINE_string('output_dir', 'output', 'Output Directory.')
flags.DEFINE_string('experiment_name', 'unconditional_model_standard_data_w_10_clipping', 'Output Directory.')


class TrainModel(object):
    def __init__(self, strokes_file_path, labels_file_path):
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
        saver = tf.train.Saver(max_to_keep=10)
        summary_proto = tf.Summary()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            tf.global_variables_initializer().run()
            writer_file_path = os.path.join(FLAGS.output_dir, FLAGS.experiment_name, 'improved_graph')
            checkpoint_file = os.path.join(FLAGS.output_dir, FLAGS.experiment_name, 'unconditional_model')
            writer = tf.summary.FileWriter(writer_file_path, sess.graph)
            for epoch in range(0, train_config.EPOCHS):
                sess.run(tf.assign(self.train_model.learning_rate,
                                   train_config.learning_rate * (train_config.decay_rate ** epoch)))
                print("Epoch number " + str(epoch))
                batch_generator, validation_generator = self.datagen.generate_unconditional_dataset(
                    batch_size=train_config.BATCH_SIZE,
                    sequence_length=train_config.SEQUENCE_LENGTH)
                batch_idx = 0
                training_loss = 0.0
                for batch in batch_generator:
                    stroke_t, stroke_t_plus_one = batch
                    feed_dict = {self.train_model.stroke_point_t: stroke_t,
                                 self.train_model.stroke_point_t_plus_one: stroke_t_plus_one}
                    global_step, summary_train, _, network_loss = sess.run([self.train_model.global_step,
                                                                            self.train_model.summary_ops,
                                                                            self.train_model.train_op,
                                                                            self.train_model.network_loss],
                                                                           feed_dict=feed_dict)
                    training_loss += network_loss
                    batch_idx += 1
                    writer.add_summary(summary_train, global_step=global_step)
                    writer.add_summary(summary_train, global_step=global_step)
                    if batch_idx % 10 == 0:
                        print('Epoch ', epoch, ' and Batch ', batch_idx, ' | training loss is ',
                              training_loss / batch_idx)
                        saver.save(sess, checkpoint_file, global_step=global_step)
                        summary_proto.ParseFromString(summary_train)
                num_of_training_batches = batch_idx
                validation_loss = 0.
                batch_idx = 0
                for batch in validation_generator:
                    stroke_t, stroke_t_plus_one = batch
                    validation_feed = {self.train_model.stroke_point_t: stroke_t,
                                       self.train_model.stroke_point_t_plus_one: stroke_t_plus_one}
                    valid_loss_summary, valid_loss = sess.run([self.train_model.validation_summary,  # move this outside, not vorrect
                                                               self.train_model.network_loss],
                                                              feed_dict=validation_feed)
                    validation_loss += valid_loss
                    batch_idx += 1
                writer.add_summary(valid_loss_summary, global_step=global_step)
                print("Finished Epoch ", str(epoch), " which has an average training loss of ",
                      training_loss / num_of_training_batches, ' and validation loss of ',
                      validation_loss / batch_idx)


def main(_):
    training = TrainModel(strokes_file_path=os.path.join(FLAGS.data_dir, 'strokes.npy'),
                          labels_file_path=os.path.join(FLAGS.data_dir, 'sentences.txt'))
    training.train()


if __name__ == '__main__':
    tf.app.run()


