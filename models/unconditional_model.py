import tensorflow as tf
from models.customized_layers import build_lstm_layers, build_mixture_density_network, \
    get_mixture_density_network_loss
from models.checkpoint_utils import add_names_for_inference, create_summaries


class Model(object):
    def __init__(self, config):
        # define args here for now
        self.config = config
        #model variables to be accessed from trianing
        # accessible variables for training:
        self.learning_rate = None
        self.initial_state = None
        self.stroke_point_t = None
        self.stroke_point_t_plus_one = None
        self.summary_ops = None
        self.global_step = None
        self.validation_summary = None
        self.optimizer = None
        self.train_op = None
        self.network_loss = None
        # variables needed for inference
        self.predicted_cuts = None
        self.weights = None
        self.mean_x = None
        self.mean_y = None
        self.std_x = None
        self.std_y = None
        self.correlation = None
        # LSTM outputs
        self.final_state = None

    def build_model(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        with tf.name_scope('data'):
            self.stroke_point_t, self.stroke_point_t_plus_one = self.build_data_scope(self.config.TIME_STEP_INPUT_SIZE)
        with tf.name_scope('lstm_layers'):
            lstm_outputs, self.initial_state, self.final_state = build_lstm_layers(self.stroke_point_t,
                                                                                   self.config.NUM_OF_HIDDEN_LAYERS,
                                                                                   self.config.NUM_OF_LSTM_CELLS,
                                                                                   self.config.BATCH_SIZE,
                                                                                   self.config.SEQUENCE_LENGTH)
        with tf.name_scope('mixture_density_network'):
            self.predicted_cuts, self.weights, self.mean_x, self.mean_y, self.std_x, self.std_y, self.correlation = \
                build_mixture_density_network(lstm_outputs, self.config.NUM_OF_MIXTURES)
        with tf.name_scope('network_loss'):
            self.network_loss = get_mixture_density_network_loss(self.predicted_cuts, self.weights, self.mean_x,
                                                                 self.mean_y, self.std_x, self.std_y,
                                                                 self.correlation, self.stroke_point_t_plus_one,
                                                                 self.config.epsilon_regularizer)
        with tf.name_scope("summaries"):
            summaries_dict = {
                'network_loss': self.network_loss,
                'lstm_output': lstm_outputs,
                'predicted_cuts': self.predicted_cuts,
                'weights': self.weights,
                'mean_x': self.mean_x,
                'mean_y': self.mean_y,
                'std_x': self.std_x,
                'std_y': self.std_y,
                'correlation': self.correlation
            }
            self.summary_ops = create_summaries(summaries_dict)
            self.validation_summary = tf.summary.scalar('validation_loss', self.network_loss)
        with tf.name_scope("inference"):
            inference_dict = {
                'initial_state': self.initial_state,
                'predicted_cuts': self.weights,
                'weights': self.mean_x,
                'mean_x': self.mean_y,
                'mean_y': self.mean_y,
                'std_x': self.std_x,
                'std_y': self.std_y,
                'correlation': self.correlation,
                'final_state': self.final_state
            }
            inference_ops = add_names_for_inference(inference_dict)
        if self.config.inference:
            return

        #training mode
        with tf.name_scope("train"):
            self.learning_rate = tf.Variable(0.0, trainable=False)
            # grads = tf.gradients(self.network_loss, tf.trainable_variables())
            # grads, _ = tf.clip_by_global_norm(grads, self.config.grad_clip)
            # for i in range(0, len(grads)):
            #     if 'mixture_density_network' in grads[i].name:
            #         grads[i] = tf.clip_by_value(grads[i], -10.0, 10.0)
            #     else:
            #         grads[i] = tf.clip_by_value(grads[i], -.0, 100.0)
            # grads_and_vars = list(zip(grads, tf.trainable_variables()))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.network_loss, global_step=self.global_step)
            # self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        return

    def build_data_scope(self, stroke_point_len):
        stroke_t = tf.placeholder(tf.float32,
                                  shape=[None, None, stroke_point_len],
                                  name='input')
        stroke_t_plus_one = tf.placeholder(tf.float32,
                                           shape=[None, None, stroke_point_len],
                                           name='target')
        return stroke_t, stroke_t_plus_one
