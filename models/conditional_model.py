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
        self.NUM_OF_CHARS = config.NUM_OF_CHARS
        self.TIME_STEP_INPUT_SIZE = config.TIME_STEP_INPUT_SIZE
        self.TIME_STEP_TARGET_SIZE = config.TIME_STEP_TARGET_SIZE  # prediction so MDN as unconditional
        self.output_keep_prob = config.output_keep_prob #not sure if we need it (double check dropout at inference)

    def build_model(self):
        stroke_point_t, character_sequence = self.build_data_layer(3, 57)
        self.build_lstm_layers(stroke_point_t, 57, 300, 3, 400,
                               20, 10, character_sequence)

    def build_data_layer(self, stroke_point_len, num_of_chars):
        stroke_point_t = tf.placeholder(tf.float32,
                                        shape=[None, None, stroke_point_len],
                                        name='stroke_point')
        character_sequence = tf.placeholder(tf.float32,
                                            shape=[None, None, num_of_chars],
                                            name='char_sequence')
        return stroke_point_t, character_sequence

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
                init_window, init_window_location_parms, rnn_output

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