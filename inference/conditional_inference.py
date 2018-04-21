import tensorflow as tf
import numpy as np
from models.conditional_model import Model
from models import gaussian_sample
from configs.config import InferenceConfig
validation_config = InferenceConfig()
trained_model_path = './trained_models/high_learning_rate_data_normalized/'


def generate_conditionaly(sentence='hello'):
    inference_model = Model(validation_config)
    inference_model.build_model()
    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(trained_model_path)
    print("loading model: ", ckpt.model_checkpoint_path)
    # metagraph = tf.train.export_meta_graph(clear_devices=True)
    # saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path  + '.meta', clear_devices=True)
    saver.restore(sess, ckpt.model_checkpoint_path)
    strokes = np.reshape(predict(sentence, sess), [-1, 3])
    tf.reset_default_graph()
    sess.close()
    return []


def predict(seq_length, sess):
    current_stroke = np.zeros([1, 1, 3], dtype=np.float32)  # first input
    current_stroke[0, 0, :] = [1, 0.78, 2]
    sampled_strokes = []  # resulting strokes
    # add the first stroke
    sampled_strokes.append(np.copy(current_stroke))
    # getting the models variables
    data_input = tf.get_default_graph().get_tensor_by_name("data/input:0")
    # stacked_lstm = tf.get_default_graph().get_tensor_by_name("stacked_lstm:0")
    lstm_initial_state = tf.get_default_graph().get_tensor_by_name("lstm_initial_state:0")
    lstm_final_state = tf.get_default_graph().get_tensor_by_name("lstm_final_state:0")
    predicted_cuts = tf.get_default_graph().get_tensor_by_name("predicted_cuts:0")
    mixtures_weights = tf.get_default_graph().get_tensor_by_name("mixtures_weights:0")
    mixtures_mean = tf.get_default_graph().get_tensor_by_name("mixtures_mean:0")
    mixtures_std = tf.get_default_graph().get_tensor_by_name("mixtures_std:0")
    mixtures_correlation = tf.get_default_graph().get_tensor_by_name("mixtures_correlation:0")
    # feed it in the model to get the second stroke in a while loop
    state = np.zeros(lstm_initial_state.shape, dtype=np.float32)  # one batch initial state for initial state lstm
    for i in range(0, seq_length):
        feed_dict = {data_input: current_stroke, lstm_initial_state: state}
        fetch_list = [predicted_cuts, mixtures_weights, mixtures_mean, mixtures_std, mixtures_correlation,
                      lstm_final_state]
        predicted_cuts_v, mixtures_weights_v, mixtures_mean_v, \
        mixtures_std_v, mixtures_correlation_v, state = sess.run(fetch_list, feed_dict=feed_dict)
        # choose one randomly
        current_stroke[0, 0, 1:] = \
            gaussian_sample.sample(mixtures_weights_v, mixtures_std_v, mixtures_correlation_v, mixtures_mean_v)
        # cut or no?
        prob = np.random.rand()
        if prob < predicted_cuts_v:
            current_stroke[0, 0, 0] = 1
        else:
            current_stroke[0, 0, 0] = 0
        sampled_strokes.append(np.copy(current_stroke))
    return sampled_strokes


def sample(seq_length, sess):
    current_stroke = np.zeros([1, 1, 3], dtype=np.float32)  # first input
    current_stroke[0, 0, :] = [1, 0.78, 2]
    sampled_strokes = []  # resulting strokes
    # add the first stroke
    sampled_strokes.append(np.copy(current_stroke))
    # getting the models variables
    data_input = tf.get_default_graph().get_tensor_by_name("data/input:0")
    # stacked_lstm = tf.get_default_graph().get_tensor_by_name("stacked_lstm:0")
    lstm_initial_state = tf.get_default_graph().get_tensor_by_name("lstm_initial_state:0")
    lstm_final_state = tf.get_default_graph().get_tensor_by_name("lstm_final_state:0")
    predicted_cuts = tf.get_default_graph().get_tensor_by_name("predicted_cuts:0")
    mixtures_weights = tf.get_default_graph().get_tensor_by_name("mixtures_weights:0")
    mixtures_mean = tf.get_default_graph().get_tensor_by_name("mixtures_mean:0")
    mixtures_std = tf.get_default_graph().get_tensor_by_name("mixtures_std:0")
    mixtures_correlation = tf.get_default_graph().get_tensor_by_name("mixtures_correlation:0")
    # feed it in the model to get the second stroke in a while loop
    state = np.zeros(lstm_initial_state.shape, dtype=np.float32)  # one batch initial state for initial state lstm
    for i in range(0, seq_length):
        feed_dict = {data_input: current_stroke, lstm_initial_state: state}
        fetch_list = [predicted_cuts, mixtures_weights, mixtures_mean, mixtures_std, mixtures_correlation,
                      lstm_final_state]
        predicted_cuts_v, mixtures_weights_v, mixtures_mean_v, \
        mixtures_std_v, mixtures_correlation_v, state = sess.run(fetch_list, feed_dict=feed_dict)
        # choose one randomly
        current_stroke[0, 0, 1:] = \
            gaussian_sample.sample(mixtures_weights_v, mixtures_std_v, mixtures_correlation_v, mixtures_mean_v)
        # cut or no?
        prob = np.random.rand()
        if prob < predicted_cuts_v:
            current_stroke[0, 0, 0] = 1
        else:
            current_stroke[0, 0, 0] = 0
        sampled_strokes.append(np.copy(current_stroke))
    return sampled_strokes
