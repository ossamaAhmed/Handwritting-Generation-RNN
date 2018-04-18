import tensorflow as tf
import numpy as np
from models.unconditional_model import Model
from configs.config import InferenceConfig
validation_config = InferenceConfig()


def choose_the_mixture(prob, weights):
    amount = 0
    for i in range(0, len(weights)):
        amount += weights[i]
        if amount >= prob:
            return i
    return -1


def sample(weights, std_deviations, correlations, means):
    chosen = choose_the_mixture(np.random.random(), weights[0])
    # print(np.shape(mixtures_weights_v))
    mean = [means[0, chosen, 0], means[0, chosen, 1]]
    covariance = [[np.square(std_deviations[0, chosen, 0]),
                   correlations[0, chosen] * std_deviations[0, chosen, 0] * std_deviations[0, chosen, 1]],
                  [correlations[0, chosen] * std_deviations[0, chosen, 0] * std_deviations[0, chosen, 1],
                   np.square(std_deviations[0, chosen, 1])]]
    return np.random.multivariate_normal(mean, covariance)
