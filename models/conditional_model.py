import numpy as np
import tensorflow as tf


#important: add dropout here and compare
class Model(object):
    def __init__(self, config):
        # define args here for now
        self.BATCH_SIZE = config.BATCH_SIZE
        self.SEQUENCE_LENGTH = config.SEQUENCE_LENGTH
        self.NUM_OF_LSTM_CELLS = config.NUM_OF_LSTM_CELLS
        self.NUM_OF_HIDDEN_LAYERS = config.NUM_OF_HIDDEN_LAYERS
        self.NUM_OF_MIXTURES = config.NUM_OF_MIXTURES
        self.NUM_OF_CHARS = config.NUM_OF_CHARS
        self.TIME_STEP_INPUT_SIZE = config.TIME_STEP_INPUT_SIZE
        self.TIME_STEP_TARGET_SIZE = config.TIME_STEP_TARGET_SIZE  # prediction so MDN as unconditional
        self.output_keep_prob = config.output_keep_prob #not sure if we need it (double check dropout at inference)
        self.inference = config.inference
        self.grad_clip = config.grad_clip
        self.LEN_OF_ALPHABET = config.LEN_OF_ALPHABET
        self.W_NUM_OF_GAUSSIANS = config.W_NUM_OF_GAUSSIANS

        #accessible variables for training:
        self.learning_rate = None
        self.initial_states = None #double check this !!!!!
        self.stroke_point_t = None
        self.character_sequence = None
        self.stroke_point_t_plus_one = None
        self.summary_ops = None
        self.global_step = None
        self.validation_summary = None
        self.optimizer = None
        self.train_op = None
        self.network_loss = None


    def build_model(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        with tf.name_scope('data'):
            stroke_point_t, character_sequence, stroke_point_t_plus_one = self.build_data_layer(
                self.TIME_STEP_INPUT_SIZE,
                self.LEN_OF_ALPHABET)
            self.stroke_point_t = stroke_point_t
            self.character_sequence = character_sequence
            self.stroke_point_t_plus_one = stroke_point_t_plus_one
        with tf.name_scope('lstm_layers_with_window'):
            stacked_lstms, initial_states, final_states, \
            init_window, final_window, init_window_location_params, final_window_location_params, rnn_output = \
                self.build_lstm_layers(stroke_point_t, self.LEN_OF_ALPHABET, self.SEQUENCE_LENGTH,
                                       self.NUM_OF_HIDDEN_LAYERS, self.NUM_OF_LSTM_CELLS, self.BATCH_SIZE,
                                       self.W_NUM_OF_GAUSSIANS, character_sequence)
            self.initial_states = initial_states
        with tf.name_scope('mixture_density_network'):
            predicted_cuts, weights, mean_x, mean_y, std_x, std_y, correlation =\
                self.build_mixture_density_network(rnn_output, self.NUM_OF_MIXTURES)
        with tf.name_scope('network_loss'):
            network_loss = self.get_mixture_density_network_loss(predicted_cuts, weights, mean_x, mean_y,
                                                                 std_x, std_y, correlation, stroke_point_t_plus_one)
            self.network_loss = network_loss
        with tf.name_scope("summaries"):
            self.summary_ops = self.create_summaries(network_loss, rnn_output, predicted_cuts, weights, mean_x,
                                                     mean_y, std_x, std_y, correlation)
        self.validation_summary = tf.summary.scalar('validation_loss', network_loss)
        # inference_dict = {'network_loss': network_loss, 'lstm_initial_state': initial_state,
        #                   'lstm_final_state': self.final_state, 'predicted_cuts': predicted_cuts,
        #                   'mixtures_weights': mixtures_weights, 'mixtures_mean': mixtures_mean,
        #                   'mixtures_std': mixtures_std, 'mixtures_correlation': mixtures_correlation}
        if self.inference: #training_mode
            return
        with tf.name_scope("train"):
            self.learning_rate = tf.Variable(0.0, trainable=False) #change to steps
            grads = tf.gradients(network_loss, tf.trainable_variables())
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
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

    def build_lstm_layers(self, input_stroke, num_of_chars, sequence_len, num_of_layers, lstm_size,
                          batch_size, num_of_gaussian_functions, char_sequence):
        #cant do it like the unconditional one because of the window
        stacked_lstms = [self.lstm_cell(lstm_size) for _ in range(num_of_layers)]
        initial_states = [stacked_lstms[i].zero_state(batch_size=batch_size, dtype=tf.float32) for i in range(num_of_layers)]
        #first cell
        #initialize random window for the first time step
        init_window = tf.zeros([batch_size, num_of_chars]) #shape [BATCH, NUM_OF_CHARS] turn to a var later
        init_window_location_parms = tf.zeros([batch_size, num_of_gaussian_functions]) #double check shape?
        input_stroke = tf.unstack(input_stroke, num=self.SEQUENCE_LENGTH, axis=1) #shape [SEQ, BATCH, 3]
        #now compute the output of the cells at each time step
        prev_window_output = init_window
        prev_window_location_params = init_window_location_parms
        cell_0_state = initial_states[0]
        cell_1_state = initial_states[1]
        cell_2_state = initial_states[2]
        rnn_output = []
        for i in range(0, len(input_stroke)):
            cell_0_input = tf.concat([input_stroke[i], prev_window_output], axis=1)
            with tf.variable_scope("cell0", reuse=tf.AUTO_REUSE):
                output0, cell_0_state = tf.nn.static_rnn(stacked_lstms[0], [cell_0_input], initial_state=cell_0_state)
            #now the output goes to a window layer
            window, location_params = self.build_window_layer(prev_window_location_params, num_of_gaussian_functions,
                                                              output0[0], sequence_len, char_sequence)
            #connect cell 2 and cell 3 below (to be done in a for loop with skip connections)
            cell_1_input = tf.concat([input_stroke[i], output0[0], window], axis=1)
            with tf.variable_scope("cell1", reuse=tf.AUTO_REUSE):
                output1, cell_1_state = tf.nn.static_rnn(stacked_lstms[1], [cell_1_input], initial_state=cell_1_state)
            cell_2_input = tf.concat([input_stroke[i], output0[0], output1[0], window], axis=1)
            with tf.variable_scope("cell2", reuse=tf.AUTO_REUSE): # should we calculate another window here ? not clear
                output2, cell_2_state = tf.nn.static_rnn(stacked_lstms[2], [cell_2_input], initial_state=cell_2_state)
            #assign new values
            rnn_output.append(output2[0])
            prev_window_output = window
            prev_window_location_params = location_params
        #shape at this point is [BATCH_SIZE, LSTM_SIZE] of len SEQ_LENGTH
        rnn_output = tf.transpose(tf.stack(rnn_output), perm=[1, 0, 2]) #maybe its the other way around? not sure
        rnn_output = tf.concat(rnn_output, 1) #[BATCH_SIZE, SEQ_LENGTH, LSTM_SIZE]
        print(rnn_output.shape)
        return stacked_lstms, initial_states, [cell_0_state, cell_1_state, cell_2_state], \
                init_window, prev_window_output, init_window_location_parms, prev_window_location_params, rnn_output

    def lstm_cell(self, lstm_size):
        #try with dynamic rnn next time
        return tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer())

    def build_window_layer(self, previous_location_parms, num_of_gaussian_functions, input, sequence_len, char_sequence):
        #calculate window weights first
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.075) #according to grave
        location_params = tf.layers.dense(input, num_of_gaussian_functions,
                                          activation=tf.exp,
                                          kernel_initializer=kernel_initializer, use_bias=False) # need to be added to location of previous window
        width_params = tf.layers.dense(input, num_of_gaussian_functions,
                                       kernel_initializer=kernel_initializer,
                                       activation=tf.exp,
                                       use_bias=False)  # bias might change
        importance_params = tf.layers.dense(input, num_of_gaussian_functions,
                                            kernel_initializer=kernel_initializer,
                                            activation=tf.exp,
                                            use_bias=False)# [BATCH, #ofGaussians]
        location_params = location_params + previous_location_parms
        #now lets generate the character indicies U
        character_indicies = tf.range(0., sequence_len)  # not sure if its starting at 0 or 1? double check
        # use broacasting with each gaussian so should be in position 2 and expand the window parameters
        location_params = tf.expand_dims(location_params, 2)
        width_params = tf.expand_dims(width_params, 2)
        importance_params = tf.expand_dims(importance_params, 2) # [BATCH, 1, #ofGaussians]
        # character_indicies = tf.expand_dims(tf.expand_dims(character_indicies, 0), 2)#maybe replace by fill afterwards
        #eq 46 and 47
        window_weights = tf.reduce_sum(tf.multiply(importance_params,
                                       tf.exp(tf.multiply(-width_params,
                                                          tf.square(tf.subtract(location_params,
                                                                                character_indicies))))), axis=1,
                                       keepdims=True) #[BATCH, 1, seq_length]
        #CHAR SEQUENCE is [BATCH, SEQ_LENGTH, ONE-HOT]
        window = tf.squeeze(tf.matmul(window_weights, char_sequence))#[BATCH, num_of_chars]
        return window, tf.squeeze(location_params)

    def build_mixture_density_network(self, inputs, num_of_mixtures):
        predicted_cuts = tf.layers.dense(inputs, 1,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                                         activation=tf.sigmoid,
                                         use_bias=False)
        weights = tf.layers.dense(inputs, num_of_mixtures,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                                  use_bias=False)
        mean_x = tf.layers.dense(inputs, num_of_mixtures,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                                 use_bias=False)
        mean_y = tf.layers.dense(inputs, num_of_mixtures,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                                 use_bias=False)
        std_x = tf.layers.dense(inputs, num_of_mixtures,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                                activation=tf.sigmoid,
                                use_bias=False)
        std_y = tf.layers.dense(inputs, num_of_mixtures,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                                activation=tf.sigmoid,
                                use_bias=False)
        correlation = tf.layers.dense(inputs, num_of_mixtures,
                                      activation=tf.tanh,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                                      use_bias=False)
        weights = tf.nn.softmax(weights)
        print(weights.shape)
        return predicted_cuts, weights, mean_x, mean_y, std_x, std_y, correlation

    def get_mixture_density_network_loss(self, predicted_cuts, weights, mean_x, mean_y,
                                         std_x, std_y, correlation, target):
        target_cuts, target_coord_x, target_coord_y = tf.unstack(tf.expand_dims(target, axis=-1), axis=-2)
        # target_cuts, target_coord_x, target_coord_y = tf.expand_dims(target_cuts, -1), \
        #                                               tf.expand_dims(target_coord_x, -1), \
        #                                               tf.expand_dims(target_coord_y, -1)
        print('target_cuts', target_coord_x.shape)
        x_norm = tf.div(tf.subtract(target_coord_x, mean_x), std_x)
        y_norm = tf.div(tf.subtract(target_coord_y, mean_y), std_y)
        z = tf.square(x_norm) + tf.square(y_norm) - (2. * correlation * x_norm * y_norm)
        one_minus_correlation_sq = 1. - tf.square(correlation)
        gaussian_distribution = tf.div(tf.exp(tf.div(-z, 2. * one_minus_correlation_sq)),
                                       2. * np.pi * std_x * std_y * one_minus_correlation_sq)
        #now calculate loss
        print('mean_x', mean_x.shape)
        print('gaussian', gaussian_distribution.shape)
        bernouli_loss = -tf.log(target_cuts * predicted_cuts + ((1. - target_cuts) * (1. - predicted_cuts))) #clip it here
        print('predicted_cuts', predicted_cuts.shape)
        gaussian_loss = -tf.log(tf.reduce_sum(weights * gaussian_distribution, axis=2))
        print(gaussian_loss.shape)
        print(bernouli_loss.shape)
        network_loss = tf.reduce_mean(tf.squeeze(bernouli_loss) + gaussian_loss)
        print(network_loss)
        return network_loss

    def add_names_for_inference(self, vars_dict): #not efficient for sure will change it later
        for name, var in vars_dict.items():
            tf.identity(var, name=name)
        return

    def create_summaries(self, network_loss, rnn_output, predicted_cuts, weights, mean_x, mean_y, std_x,
                         std_y,  correlation):
        tf.summary.scalar("LSTM_Output_minimum", tf.reduce_min(rnn_output))
        tf.summary.scalar("LSTM_Output_maximum", tf.reduce_max(rnn_output))
        tf.summary.scalar("predicted_cuts_minimum", tf.reduce_min(predicted_cuts))
        tf.summary.scalar("predicted_cuts_maximum", tf.reduce_max(predicted_cuts))
        tf.summary.scalar("mixtures_weights_minimum", tf.reduce_min(weights))
        tf.summary.scalar("mixtures_weights_maximum", tf.reduce_max(weights))
        tf.summary.scalar("mixtures_mean_x_minimum", tf.reduce_min(mean_x))
        tf.summary.scalar("mixtures_mean_x_maximum", tf.reduce_max(mean_x))
        tf.summary.scalar("mixtures_mean_y_minimum", tf.reduce_min(mean_y))
        tf.summary.scalar("mixtures_mean_y_maximum", tf.reduce_max(mean_y))
        tf.summary.scalar("mixtures_std_x_minimum", tf.reduce_min(std_x))
        tf.summary.scalar("mixtures_std_x_maximum", tf.reduce_max(std_x))
        tf.summary.scalar("mixtures_std_y_minimum", tf.reduce_min(std_y))
        tf.summary.scalar("mixtures_std_y_maximum", tf.reduce_max(std_y))
        tf.summary.scalar("mixtures_correlation_minimum", tf.reduce_min(correlation))
        tf.summary.scalar("mixtures_correlation_maximum", tf.reduce_max(correlation))
        tf.summary.scalar("training_loss", network_loss)
        return tf.summary.merge_all()
