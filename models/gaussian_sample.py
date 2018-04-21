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


def sample(weights, std_x, std_y, correlations, mean_x, meany):
    chosen = choose_the_mixture(np.random.random(), weights)
    # print(np.shape(mixtures_weights_v))
    mean = [mean_x[chosen], meany[chosen]]
    covariance = [[np.square(std_x[chosen]),
                   correlations[chosen] * std_x[chosen] * std_y[chosen]],
                  [correlations[chosen] * std_x[chosen] * std_y[chosen],
                   np.square(std_y[chosen])]]
    return np.random.multivariate_normal(mean, covariance)
