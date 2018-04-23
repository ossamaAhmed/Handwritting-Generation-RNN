import tensorflow as tf
import numpy as np
from models.unconditional_model import Model
from models import gaussian_sample
from configs.unconditional_config import InferenceConfig
validation_config = InferenceConfig()
trained_model_path = './experiments/unconditional_experiments/unconditional_model_standard_data_no_clipping'


def generate_unconditionaly(seq=100):
    inference_model = Model(validation_config)
    inference_model.build_model()
    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(trained_model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    samples = np.reshape(sample(seq, sess, inference_model), [-1, 3])
    tf.reset_default_graph()
    sess.close()
    return samples


def sample(seq_length, sess, inference_model):
    current_stroke = np.zeros([1, 1, 3], dtype=np.float32)  # first input
    current_stroke[0, 0, :] = [1, 0, 1]
    sampled_strokes = [] # resulting strokes
    sampled_strokes.append(np.copy(current_stroke))
    # feed it in the model to get the second stroke in a while loop
    state = []
    for i in range(len(inference_model.initial_state)):
        for j in range(len(inference_model.initial_state[i])):
            state.append(np.zeros(inference_model.initial_state[i][j].shape, dtype=np.float32))
    for i in range(0, seq_length):
        feed_dict = {inference_model.stroke_point_t: current_stroke,
                     inference_model.initial_state: state}
        fetch_list = [inference_model.predicted_cuts,
                      inference_model.weights,
                      inference_model.mean_x,
                      inference_model.mean_y,
                      inference_model.std_x,
                      inference_model.std_y,
                      inference_model.correlation,
                      inference_model.final_state]
        predicted_cuts, weights, mean_x, mean_y, std_x, std_y, correlation, final_state = \
            sess.run(fetch_list, feed_dict=feed_dict)
        # choose one randomly
        current_stroke[0, 0, 1:] = \
            gaussian_sample.sample(weights[0, 0, :], std_x[0, 0, :], std_y[0, 0, :], correlation[0, 0, :],
                                   mean_x[0, 0, :], mean_y[0, 0, :])
        current_stroke[0, 0, 0] = np.random.binomial(1, predicted_cuts[0, 0, 0])
        sampled_strokes.append(np.copy(current_stroke))
    return sampled_strokes
