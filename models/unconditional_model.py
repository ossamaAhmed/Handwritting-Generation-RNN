import numpy as np
import tensorflow as tf


class Model(object):
    def __init__(self, config):
        # define args here for now
        self.BATCH_SIZE = config.BATCH_SIZE
        self.SEQUENCE_LENGTH = config.SEQUENCE_LENGTH
        self.NUM_OF_LSTM_CELLS = config.NUM_OF_LSTM_CELLS
        self.NUM_OF_HIDDEN_LAYERS = config.NUM_OF_HIDDEN_LAYERS
        self.NUM_OF_MIXTURES = config.NUM_OF_MIXTURES
        self.TIME_STEP_INPUT_SIZE = config.TIME_STEP_INPUT_SIZE  # first being the cut boolean
        self.TIME_STEP_TARGET_SIZE = config.TIME_STEP_TARGET_SIZE  # same as input
        self.output_keep_prob = config.output_keep_prob
        self.eps_lower = config.eps_lower
        self.eps_upper = config.eps_upper
        self.momentum = config.momentum
        self.decay = config.decay
        self.inference = config.inference
        self.grad_clip = config.grad_clip
        #model variables to be accessed from trianing
        self.learning_rate = None# to be trainable afterwards and early stopping
        self.network_loss = None
        self.optimizer = None
        self.train_op = None
        self.stroke_t = None
        self.stroke_t_plus_one = None
        self.global_step = None
        self.summary_op = None
        self.initial_state = None
        self.final_state = None
        self.validation_summary = None
        self.z = None
        self.gaussian_distribution = None
        self.intermediate1 = None
        self.intermediate2 = None

    def build_model(self):
        #with tf.device('/device:GPU:0'):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        with tf.name_scope('data'):
            self.stroke_t, self.stroke_t_plus_one = self.build_data_scope()
        with tf.name_scope('lstm_layers'):
            lstm_outputs, self.initial_state, self.final_state = self.build_lstm_layers(self.stroke_t)
        with tf.name_scope('mixture_density_network'):
            predicted_cuts, mixtures_weights, mixtures_mean, mixtures_std, mixtures_correlation = \
                self.build_mixture_density_network(lstm_outputs)
        # get the loss function
        with tf.name_scope('mixture_of_gaussian_loss'):
            loss_gaussian = self.get_mixture_density_network_loss(mixtures_weights,
                                                                  mixtures_mean,
                                                                  mixtures_std,
                                                                  mixtures_correlation,
                                                                  self.stroke_t_plus_one)
        with tf.name_scope('predicted_cuts_loss'):
            predicted_cuts_loss = self.get_cuts_prediction_loss_function(self.stroke_t_plus_one, predicted_cuts)

        with tf.name_scope('network_loss'):
            self.network_loss = self.get_network_loss(predicted_cuts_loss, loss_gaussian)
            self.learning_rate = tf.Variable(0.0, trainable=False)
        with tf.name_scope("summaries"):
            self.create_summaries(loss_gaussian,
                                  predicted_cuts_loss,
                                  lstm_outputs,
                                  self.final_state, predicted_cuts,
                                  mixtures_weights,
                                  mixtures_mean,
                                  mixtures_std,
                                  mixtures_correlation)
        self.validation_summary = tf.summary.scalar('validation_loss', self.network_loss)
        inference_dict = {'network_loss': self.network_loss, 'lstm_initial_state': self.initial_state,
                          'lstm_final_state': self.final_state, 'predicted_cuts': predicted_cuts,
                          'mixtures_weights': mixtures_weights, 'mixtures_mean': mixtures_mean,
                          'mixtures_std': mixtures_std, 'mixtures_correlation': mixtures_correlation}
        self.add_names_for_inference(inference_dict)

        if self.inference: #training_mode
            return

        with tf.name_scope("train"):
            grads = tf.gradients(self.network_loss, tf.trainable_variables())
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
            # for i in range(0, len(grads)):
            #     if 'mixture_density_network' in grads[i].name:
            #         grads[i] = tf.clip_by_value(grads[i], -10.0, 10.0)
            #     else:
            #         grads[i] = tf.clip_by_value(grads[i], -.0, 100.0)
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            #self.train_op = self.optimizer.minimize(self.loss)
            self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        return

    def build_data_scope(self):
        stroke_t = tf.placeholder(tf.float32,
                                  shape=[None, self.SEQUENCE_LENGTH, self.TIME_STEP_INPUT_SIZE],
                                  name='input')
        stroke_t_plus_one = tf.placeholder(tf.float32,
                                           shape=[None, self.SEQUENCE_LENGTH, self.TIME_STEP_INPUT_SIZE],
                                           name='target')
        return stroke_t, stroke_t_plus_one

    def build_lstm_layers(self, inputs):
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([self.lstm_cell(self.NUM_OF_LSTM_CELLS) \
                                                    for _ in range(self.NUM_OF_HIDDEN_LAYERS)])
        initial_state = stacked_lstm.zero_state(self.BATCH_SIZE, tf.float32)
        if not self.inference:# training mode
            stacked_lstm = tf.contrib.rnn.DropoutWrapper(stacked_lstm, output_keep_prob=self.output_keep_prob)
        inputs = tf.unstack(inputs, num=self.SEQUENCE_LENGTH, axis=1)
        # inputs_sequences = np.array([BATCH_SIZE, SEQUENCE_LENGTH])
        # inputs_sequences.fill(200)
        outputs, final_state = tf.nn.static_rnn(stacked_lstm, inputs, initial_state=initial_state)
        outputs = tf.transpose(outputs, perm=[1, 0, 2])
        outputs = tf.reshape(outputs, [-1, self.NUM_OF_LSTM_CELLS])
        return outputs, initial_state, final_state

    def lstm_cell(self, lstm_size):
        # added clipping to resolve nans while trainings
        #try with dynamic rnn next time
        return tf.contrib.rnn.LSTMCell(lstm_size, initializer=tf.contrib.layers.xavier_initializer())

    def build_mixture_density_network(self, inputs):
        number_of_MDN_inputs = 1 + self.NUM_OF_MIXTURES * (1 + 2 + 2 + 1)  # 1 weight, 2D mean, 2D STD, 1 correlation
        w_mdn = tf.Variable(tf.truncated_normal([self.NUM_OF_LSTM_CELLS, number_of_MDN_inputs], 0.0, 0.075,
                                                dtype=tf.float32))  # initial values might change in training
        b_mdn = tf.Variable(tf.truncated_normal([number_of_MDN_inputs], 0.0, 0.075,
                                                dtype=tf.float32))
        mdn_coeffs = tf.matmul(inputs, w_mdn) + b_mdn
        # now split all the parameters into weight, 2D mean, 2D STD, 1 correlation
        mixtures_params, cuts_param = tf.split(mdn_coeffs, [-1, 1], 1)
        mixtures_weights, mixtures_mean, mixtures_std, mixtures_correlation = \
            tf.split(mixtures_params,
                     [self.NUM_OF_MIXTURES * 1, self.NUM_OF_MIXTURES * 2,
                      self.NUM_OF_MIXTURES * 2, self.NUM_OF_MIXTURES * 1], 1)
        #predicted_cuts = 1. / (1. + tf.exp(cuts_param))
        predicted_cuts = tf.sigmoid(cuts_param)
        #mixtures_weights = tf.exp(mixtures_weights)
        #mixtures_weights = mixtures_weights / tf.reshape(tf.reduce_sum(mixtures_weights, 1), shape=[-1, 1])
        mixtures_weights = tf.nn.softmax(self.eps_lower + mixtures_weights) #stability
        mixtures_weights = tf.reshape(mixtures_weights, shape=[-1, self.NUM_OF_MIXTURES])
        mixtures_std = tf.exp(mixtures_std)
        mixtures_std = tf.reshape(mixtures_std, shape=[-1, self.NUM_OF_MIXTURES, 2])
        mixtures_correlation = tf.tanh(self.eps_lower + mixtures_correlation)
        mixtures_correlation = tf.reshape(mixtures_correlation, shape=[-1, self.NUM_OF_MIXTURES])
        mixtures_mean = tf.reshape(mixtures_mean, shape=[-1, self.NUM_OF_MIXTURES, 2])
        return predicted_cuts, mixtures_weights, mixtures_mean, mixtures_std, mixtures_correlation

    def get_mixture_density_network_loss(self, mixtures_weights,
                                         mixtures_mean, mixtures_std,
                                         mixtures_correlation, target):
        target = tf.reshape(target, [-1, 3])
        target_cuts, target_coord_x, target_coord_y = tf.unstack(target, axis=1)
        target_coord_x = tf.reshape(target_coord_x, shape=[-1, 1])
        target_coord_x = tf.tile(target_coord_x, [1, self.NUM_OF_MIXTURES])
        target_coord_y = tf.reshape(target_coord_y, shape=[-1, 1])
        target_coord_y = tf.tile(target_coord_y, [1, self.NUM_OF_MIXTURES])
        mixtures_mean_coord_x, mixtures_mean_coord_y = tf.unstack(mixtures_mean, axis=2)
        mixtures_std_coord_x, mixtures_std_coord_y = tf.unstack(mixtures_std, axis=2)
        intermediate1 = (tf.square((target_coord_x - mixtures_mean_coord_x) / mixtures_std_coord_x)) + \
            (tf.square((target_coord_y - mixtures_mean_coord_y) / mixtures_std_coord_y))
        intermediate2 = tf.abs((target_coord_x - mixtures_mean_coord_x) / mixtures_std_coord_x)
        # z = intermediate1 - \
        #     (((target_coord_x - mixtures_mean_coord_x) * \
        #       (target_coord_y - mixtures_mean_coord_y) * 2 * \
        #       mixtures_correlation) / (mixtures_std_coord_x * mixtures_std_coord_y)) #-52 - 50
        z = (tf.square(target_coord_x - mixtures_mean_coord_x / mixtures_std_coord_x)) + \
            (tf.square(target_coord_y - mixtures_mean_coord_y / mixtures_std_coord_y)) - \
            (((target_coord_x - mixtures_mean_coord_x) * \
              (target_coord_y - mixtures_mean_coord_y) * 2 * \
              mixtures_correlation) / (mixtures_std_coord_x * mixtures_std_coord_y))
        gaussian_distribution = (tf.exp(-z / (2 * (1 - tf.square(mixtures_correlation))))) / \
                                ((2 * np.pi) * mixtures_std_coord_y * mixtures_std_coord_x *
                                 tf.sqrt(1 - tf.square(mixtures_correlation)))
        gaussian_distribution = mixtures_weights * gaussian_distribution
        gaussian_distribution = tf.reshape(gaussian_distribution, shape=[self.BATCH_SIZE, self.SEQUENCE_LENGTH, -1])
        loss_gaussian = tf.reduce_sum(-tf.log(tf.clip_by_value(tf.reduce_sum(gaussian_distribution, 2),
                                                               self.eps_lower, self.eps_upper)), 1)
        self.z = z
        self.gaussian_distribution = gaussian_distribution
        self.intermediate1 = intermediate1
        self.intermediate2 = intermediate2
        return loss_gaussian

    def get_cuts_prediction_loss_function(self, target, predicted_cuts):
        target = tf.reshape(target, [-1, 3])
        target_cuts, target_coord_x, target_coord_y = tf.unstack(target, axis=1)  # REPETITION
        target_cuts = tf.reshape(target_cuts, shape=[self.BATCH_SIZE, self.SEQUENCE_LENGTH])
        predicted_cuts = tf.reshape(predicted_cuts, shape=[self.BATCH_SIZE, self.SEQUENCE_LENGTH])
        loss_bernouli = ((1 - target_cuts) * (-tf.log(tf.clip_by_value(1 - predicted_cuts, self.eps_lower, self.eps_upper)))) + \
                        ((target_cuts) * (-tf.log(tf.clip_by_value(predicted_cuts, self.eps_lower, self.eps_upper))))
        loss_bernouli = tf.reduce_sum(loss_bernouli, 1)
        return loss_bernouli

    def get_network_loss(self, predicted_cuts_loss, loss_gaussian):
        network_loss = predicted_cuts_loss + loss_gaussian
        mean_network_loss = tf.div(tf.reduce_sum(network_loss, 0), (self.BATCH_SIZE * self.SEQUENCE_LENGTH)) # to be revised
        return mean_network_loss

    def add_names_for_inference(self, vars_dict): #not efficient for sure will change it later
        for name, var in vars_dict.items():
            tf.identity(var, name=name)
        return

    def create_summaries(self, loss_gaussian, predicted_cuts_loss, lstm_outputs, state, predicted_cuts,
                         mixtures_weights, mixtures_mean, mixtures_std, mixtures_correlation):
        tf.summary.scalar("input_minumum", tf.reduce_min(self.stroke_t))
        tf.summary.scalar("Input_maximum", tf.reduce_max(self.stroke_t))
        tf.summary.scalar("LSTM_Output_minimum", tf.reduce_min(lstm_outputs))
        tf.summary.scalar("LSTM_Output_maximum", tf.reduce_max(lstm_outputs))
        tf.summary.scalar("LSTM_State_minimum", tf.reduce_min(state))
        tf.summary.scalar("LSTM_State_maximum", tf.reduce_max(state))
        tf.summary.scalar("predicted_cuts_minimum", tf.reduce_min(predicted_cuts))
        tf.summary.scalar("predicted_cuts_maximum", tf.reduce_max(predicted_cuts))
        tf.summary.scalar("mixtures_weights_minimum", tf.reduce_min(mixtures_weights))
        tf.summary.scalar("mixtures_weights_maximum", tf.reduce_max(mixtures_weights))
        tf.summary.scalar("mixtures_mean_minimum", tf.reduce_min(mixtures_mean))
        tf.summary.scalar("mixtures_mean_maximum", tf.reduce_max(mixtures_mean))
        tf.summary.scalar("mixtures_std_minimum", tf.reduce_min(mixtures_std))
        tf.summary.scalar("mixtures_std_maximum", tf.reduce_max(mixtures_std))
        tf.summary.scalar("mixtures_correlation_minimum", tf.reduce_min(mixtures_correlation))
        tf.summary.scalar("mixtures_correlation_maximum", tf.reduce_max(mixtures_correlation))
        tf.summary.scalar("loss_gaussian_minimum", tf.reduce_min(loss_gaussian))
        tf.summary.scalar("loss_gaussian_maximum", tf.reduce_max(loss_gaussian))
        tf.summary.scalar("predicted_cuts_loss_minimum", tf.reduce_min(predicted_cuts_loss))
        tf.summary.scalar("predicted_cuts_loss_maximum", tf.reduce_max(predicted_cuts_loss))
        tf.summary.scalar("gaussian_distribution_z__minimum", tf.reduce_min(self.z))
        tf.summary.scalar("gaussian_distribution_z_maximum", tf.reduce_max(self.z))
        tf.summary.scalar("intermediate1_minimum", tf.reduce_min(self.intermediate1))
        tf.summary.scalar("intermediate1_maximum", tf.reduce_max(self.intermediate1))
        tf.summary.scalar("intermediate2_minimum", tf.reduce_min(self.intermediate2))
        tf.summary.scalar("intermediate2_maximum", tf.reduce_max(self.intermediate2))
        tf.summary.scalar("gaussian_distribution_minimum", tf.reduce_min(self.gaussian_distribution))
        tf.summary.scalar("gaussian_distribution_maximum", tf.reduce_max(self.gaussian_distribution))
        tf.summary.scalar("training_loss", self.network_loss)
        self.summary_op = tf.summary.merge_all()
        return



#test
# model = Model()
# model.build_model()
