import tensorflow as tf
from models.customized_layers import build_lstm_layers_with_window, build_mixture_density_network, \
    get_mixture_density_network_loss
from models.checkpoint_utils import add_names_for_inference, create_summaries


class Model(object):
    def __init__(self, config):
        self.config = config
        #accessible variables for training:
        self.learning_rate = None
        self.initial_states = None
        self.stroke_point_t = None
        self.character_sequence = None
        self.stroke_point_t_plus_one = None
        self.summary_ops = None
        self.global_step = None
        self.validation_summary = None
        self.optimizer = None
        self.train_op = None
        self.network_loss = None
        #variables needed for inference
        self.init_window = None
        self.init_window_location_params = None
        self.predicted_cuts = None
        self.weights = None
        self.mean_x = None
        self.mean_y = None
        self.std_x = None
        self.std_y = None
        self.correlation = None
        #LSTM outputs
        self.final_window = None
        self.final_states = None
        self.final_window_location_params = None

    def build_model(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        with tf.name_scope('data'):
            self.stroke_point_t, self.character_sequence, self.stroke_point_t_plus_one = self.build_data_layer(
                self.config.TIME_STEP_INPUT_SIZE,
                self.config.LEN_OF_ALPHABET)
        with tf.name_scope('lstm_layers_with_window'):
            stacked_lstms, self.initial_states, self.final_states, self.init_window, self.final_window, \
                self.init_window_location_params, self.final_window_location_params, rnn_output = \
                build_lstm_layers_with_window(self.stroke_point_t, self.config.LEN_OF_ALPHABET,
                                              self.config.U,
                                              self.config.NUM_OF_HIDDEN_LAYERS, self.config.NUM_OF_LSTM_CELLS,
                                              self.config.BATCH_SIZE, self.config.W_NUM_OF_GAUSSIANS,
                                              self.character_sequence, self.config.SEQUENCE_LENGTH)
        with tf.name_scope('mixture_density_network'):
            self.predicted_cuts, self.weights, self.mean_x, self.mean_y, self.std_x, self.std_y, self.correlation =\
                build_mixture_density_network(rnn_output, self.config.NUM_OF_MIXTURES)
        with tf.name_scope('network_loss'):
            self.network_loss = get_mixture_density_network_loss(self.predicted_cuts, self.weights, self.mean_x,
                                                                 self.mean_y, self.std_x, self.std_y,
                                                                 self.correlation, self.stroke_point_t_plus_one,
                                                                 self.config.epsilon_regularizer)

        with tf.name_scope("summaries"):
            summaries_dict = {
                'network_loss': self.network_loss,
                'rnn_output': rnn_output,
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
                'init_window': self.init_window,
                'init_window_location_params': self.init_window_location_params,
                'initial_states': self.initial_states,
                'predicted_cuts': self.weights,
                'weights': self.mean_x,
                'mean_x': self.mean_y,
                'mean_y': self.mean_y,
                'std_x': self.std_x,
                'std_y': self.std_y,
                'correlation': self.correlation,
                'final_window': self.final_window,
                'final_states': self.final_states,
                'final_window_location_params': self.final_window_location_params,
            }
            inference_ops = add_names_for_inference(inference_dict)
        if self.config.inference:
            return

        #training mode
        with tf.name_scope("train"):
            self.learning_rate = tf.Variable(0.0, trainable=False) #change to steps
            grads = tf.gradients(self.network_loss, tf.trainable_variables())
            grads, _ = tf.clip_by_global_norm(grads, self.config.grad_clip)
            # for i in range(0, len(grads)):
            #     if 'mixture_density_network' in grads[i].name:
            #         grads[i] = tf.clip_by_value(grads[i], -10.0, 10.0)
            #     else:
            #         grads[i] = tf.clip_by_value(grads[i], -.0, 100.0)
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        return

    def build_data_layer(self, stroke_point_len, num_of_chars):
        stroke_point_t = tf.placeholder(tf.float32,
                                        shape=[None, None, stroke_point_len],
                                        name='stroke_input')
        character_sequence = tf.placeholder(tf.float32,
                                            shape=[None, None, num_of_chars],
                                            name='char_sequence')
        stroke_point_t_plus_one = tf.placeholder(tf.float32,
                                                 shape=[None, None, stroke_point_len],
                                                 name='stroke_target')
        return stroke_point_t, character_sequence, stroke_point_t_plus_one


