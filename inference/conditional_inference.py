import tensorflow as tf
import numpy as np
from models.conditional_model import Model
from models import gaussian_sample
from configs.conditional_config import InferenceConfig
from utils.data_utils import convert_sentence_to_one_hot_encoding, define_alphabet
alphabet = define_alphabet()
validation_config = InferenceConfig()
trained_model_path = './experiments/conditional_experiments/conditional_model_clipping_paper_scaled_data/'


def generate_conditionaly(sentence='hello'):
    inference_model = Model(validation_config)
    inference_model.build_model()
    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(trained_model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    strokes = np.reshape(predict(sentence, sess, inference_model), [-1, 3])
    tf.reset_default_graph()
    sess.close()
    return strokes


def predict(sentence, sess, inference_model, seq_length=300):
    current_stroke = np.zeros([1, 1, 3], dtype=np.float32)
    char_sequence = [convert_sentence_to_one_hot_encoding(alphabet, sentence,
                                                          validation_config.U)]
    current_stroke[0, 0, :] = [0, 0, 0]
    sampled_strokes = []  # resulting strokes
    sampled_strokes.append(np.copy(current_stroke))
    # feed it in the model to get the second stroke in a while loop
    lstm_states = []
    for i in range(len(inference_model.initial_states)):
        lstm_states.append(np.zeros(inference_model.initial_states[i][0].shape, dtype=np.float32))
    window = np.zeros(inference_model.init_window.shape, dtype=np.float32)
    window_location_params = np.zeros(inference_model.init_window_location_params.shape, dtype=np.float32)
    for i in range(0, seq_length):
        # inputs: initial states, stroke_t, chars, prev_window, prev_k
        feed_dict = {inference_model.stroke_point_t: current_stroke,
                     inference_model.character_sequence: char_sequence,
                     inference_model.init_window: window,
                     inference_model.init_window_location_params: window_location_params}
        for j in range(len(inference_model.initial_states)):
            feed_dict[inference_model.initial_states[j]]: lstm_states[j]
        fetch_list = [inference_model.predicted_cuts,
                      inference_model.weights,
                      inference_model.mean_x,
                      inference_model.mean_y,
                      inference_model.std_x,
                      inference_model.std_y,
                      inference_model.correlation,
                      inference_model.final_window,
                      inference_model.final_states,
                      inference_model.final_window_location_params]
        predicted_cuts, weights, mean_x, mean_y, std_x, std_y, \
            correlation, window, lstm_states, window_location_params = sess.run(fetch_list, feed_dict=feed_dict)
        current_stroke[0, 0, 1:] = \
            gaussian_sample.sample(weights[0, 0, :], std_x[0, 0, :], std_y[0, 0, :], correlation[0, 0, :],
                                   mean_x[0, 0, :], mean_y[0, 0, :])
        # cut or no?
        current_stroke[0, 0, 0] = np.random.binomial(1, predicted_cuts[0, 0, 0])
        sampled_strokes.append(np.copy(current_stroke))
    return sampled_strokes
