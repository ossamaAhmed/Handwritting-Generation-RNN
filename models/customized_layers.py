import tensorflow as tf
import numpy as np


def build_lstm_layers(input_stroke, num_of_layers, lstm_size, batch_size, sequence_length):
    # cant do it like the unconditional one because of the window
    # TODO: try with a dynamic rnn cell (variable sequence length)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(lstm_size) for _ in range(num_of_layers)])
    initial_state = stacked_lstm.zero_state(batch_size, tf.float32)
    #TODO: maybe add a dropout layer and compare
    input_stroke = tf.unstack(input_stroke, num=sequence_length, axis=1)  # shape [SEQ, BATCH, 3]
    # now compute the output of the cells at each time step
    outputs, final_state = tf.nn.static_rnn(stacked_lstm, input_stroke, initial_state=initial_state)
    # shape at this point is [BATCH_SIZE, LSTM_SIZE] of len SEQ_LENGTH
    rnn_output = tf.transpose(tf.stack(outputs), perm=[1, 0, 2])  # maybe its the other way around? not sure
    print('rnn_output_pre', rnn_output.shape)
    rnn_output = tf.concat(rnn_output, 1)  # [BATCH_SIZE, SEQ_LENGTH, LSTM_SIZE]
    print('rnn_output_after_concat', rnn_output.shape)
    return outputs, initial_state, final_state


def build_lstm_layers_with_window(input_stroke, num_of_chars, max_num_of_chars, num_of_layers, lstm_size,
                                  batch_size, num_of_gaussian_functions, char_sequence, sequence_length):
    # cant do it like the unconditional one because of the window
    stacked_lstms = [lstm_cell(lstm_size) for _ in range(num_of_layers)]
    initial_states = [stacked_lstms[i].zero_state(batch_size=batch_size, dtype=tf.float32) for i in
                      range(num_of_layers)]
    # first cell
    # initialize random window for the first time step
    init_window = tf.zeros([batch_size, num_of_chars])  # shape [BATCH, NUM_OF_CHARS] turn to a var later
    init_window_location_parms = tf.zeros([batch_size, num_of_gaussian_functions])  # double check shape?
    input_stroke = tf.unstack(input_stroke, num=sequence_length, axis=1)  # shape [SEQ, BATCH, 3]
    # now compute the output of the cells at each time step
    prev_window_output = init_window
    prev_window_location_params = init_window_location_parms
    cell_0_state = initial_states[0]
    cell_1_state = initial_states[1]
    cell_2_state = initial_states[2]
    rnn_output = []
    for i in range(0, len(input_stroke)):
        with tf.variable_scope("cell0", reuse=tf.AUTO_REUSE):
            cell_0_input = tf.concat([input_stroke[i], prev_window_output], axis=1)
            output0, cell_0_state = tf.nn.static_rnn(stacked_lstms[0], [cell_0_input], initial_state=cell_0_state)
        # now the output goes to a window layer
        with tf.variable_scope("window", reuse=tf.AUTO_REUSE):
            window, location_params = build_window_layer(prev_window_location_params, num_of_gaussian_functions,
                                                         output0[0], max_num_of_chars, char_sequence)
            # connect cell 2 and cell 3 below (to be done in a for loop with skip connections)
            cell_1_input = tf.concat([input_stroke[i], output0[0], window], axis=1)
        with tf.variable_scope("cell1", reuse=tf.AUTO_REUSE):
            output1, cell_1_state = tf.nn.static_rnn(stacked_lstms[1], [cell_1_input], initial_state=cell_1_state)
        cell_2_input = tf.concat([input_stroke[i], output0[0], output1[0], window], axis=1)
        with tf.variable_scope("cell2", reuse=tf.AUTO_REUSE):  # should we calculate another window here ? not clear
            output2, cell_2_state = tf.nn.static_rnn(stacked_lstms[2], [cell_2_input], initial_state=cell_2_state)
        # assign new values
        rnn_output.append(output2[0])
        prev_window_output = window
        prev_window_location_params = location_params
    # shape at this point is [BATCH_SIZE, LSTM_SIZE] of len SEQ_LENGTH
    rnn_output = tf.transpose(tf.stack(rnn_output), perm=[1, 0, 2])  # maybe its the other way around? not sure
    rnn_output = tf.concat(rnn_output, 1)  # [BATCH_SIZE, SEQ_LENGTH, LSTM_SIZE]
    return stacked_lstms, initial_states, [cell_0_state, cell_1_state, cell_2_state], \
        init_window, prev_window_output, init_window_location_parms, prev_window_location_params, rnn_output


def lstm_cell(lstm_size):
    # try with dynamic rnn next time
    return tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer())


def build_window_layer(previous_location_parms, num_of_gaussian_functions, inputs, max_num_of_chars,
                       char_sequence):
    # calculate window weights first
    kernel_initializer = tf.truncated_normal_initializer(stddev=0.075)  # according to grave
    location_params = tf.layers.dense(inputs, num_of_gaussian_functions,
                                      activation=tf.exp,
                                      kernel_initializer=kernel_initializer,
                                      use_bias=True,
                                      name='location_params')  # need to be added to location of previous window
    width_params = tf.layers.dense(inputs, num_of_gaussian_functions,
                                   kernel_initializer=kernel_initializer,
                                   activation=tf.exp,
                                   use_bias=True,
                                   name='width_params')  # bias might change
    importance_params = tf.layers.dense(inputs, num_of_gaussian_functions,
                                        kernel_initializer=kernel_initializer,
                                        activation=tf.exp,
                                        use_bias=True,
                                        name='importance_params')  # [BATCH, #ofGaussians]
    location_params = location_params + previous_location_parms
    # now lets generate the character indicies U
    character_indicies = tf.range(0., max_num_of_chars)  # not sure if its starting at 0 or 1? double check
    # use broacasting with each gaussian so should be in position 2 and expand the window parameters
    location_params = tf.expand_dims(location_params, 2)
    width_params = tf.expand_dims(width_params, 2)
    importance_params = tf.expand_dims(importance_params, 2)  # [BATCH, 1, #ofGaussians]
    # character_indicies = tf.expand_dims(tf.expand_dims(character_indicies, 0), 2)#maybe replace by fill afterwards
    # eq 46 and 47
    window_weights = tf.reduce_sum(tf.multiply(importance_params,
                                               tf.exp(tf.multiply(-width_params,
                                                                  tf.square(tf.subtract(location_params,
                                                                                        character_indicies))))), axis=1,
                                   keepdims=True)  # [BATCH, 1, seq_length]
    # CHAR SEQUENCE is [BATCH, SEQ_LENGTH, ONE-HOT]
    window = tf.squeeze(tf.matmul(window_weights, char_sequence), axis=1)  # [BATCH, num_of_chars]
    return window, tf.squeeze(location_params, axis=-1)


def build_mixture_density_network(inputs, num_of_mixtures):
    # so apparently we need to name these inputs!!!! BUUUUUGS
    predicted_cuts = tf.layers.dense(inputs, 1,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                                     activation=tf.sigmoid,
                                     use_bias=True,
                                     name='predicted_cuts')
    weights = tf.layers.dense(inputs, num_of_mixtures,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                              use_bias=True,
                              name='weights')
    mean_x = tf.layers.dense(inputs, num_of_mixtures,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                             use_bias=True,
                             name='mean_x')
    mean_y = tf.layers.dense(inputs, num_of_mixtures,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                             use_bias=True,
                             name='mean_y')
    std_x = tf.layers.dense(inputs, num_of_mixtures,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                            activation=tf.exp,
                            use_bias=True,
                            name='std_x')
    std_y = tf.layers.dense(inputs, num_of_mixtures,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                            activation=tf.exp,
                            use_bias=True,
                            name='std_y')
    correlation = tf.layers.dense(inputs, num_of_mixtures,
                                  activation=tf.tanh,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.075),
                                  use_bias=True,
                                  name='correlation')
    weights = tf.nn.softmax(weights)
    return predicted_cuts, weights, mean_x, mean_y, std_x, std_y, correlation


def get_mixture_density_network_loss(predicted_cuts, weights, mean_x, mean_y,
                                     std_x, std_y, correlation, target, epsilon_regularizer):
    target_cuts, target_coord_x, target_coord_y = tf.unstack(tf.expand_dims(target, axis=-1), axis=-2)
    x_norm = tf.div(tf.subtract(target_coord_x, mean_x), std_x)
    y_norm = tf.div(tf.subtract(target_coord_y, mean_y), std_y)
    z = tf.square(x_norm) + tf.square(y_norm) - (2. * correlation * x_norm * y_norm)
    one_minus_correlation_sq = 1. - tf.square(correlation)
    gaussian_distribution = tf.div(tf.exp(tf.div(-z, 2. * one_minus_correlation_sq)),
                                   2. * np.pi * std_x * std_y * one_minus_correlation_sq)
    # now calculate loss
    #prefer regularizer over clipping (smooth func for grad)
    bernouli_loss = -tf.log(
        (target_cuts * predicted_cuts + ((1. - target_cuts) * (1. - predicted_cuts))) + epsilon_regularizer)
    gaussian_loss = -tf.log(tf.reduce_sum(weights * gaussian_distribution, axis=2) + epsilon_regularizer)
    network_loss = tf.reduce_mean(tf.squeeze(bernouli_loss, axis=-1) + gaussian_loss)
    return network_loss
