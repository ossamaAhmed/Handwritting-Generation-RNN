import numpy as np


def choose_the_mixture(prob, weights):
    amount = 0
    for i in range(0, len(weights)):
        amount += weights[i]
        if amount >= prob:
            return i
    return -1


def sample(weights, std_x, std_y, correlations, mean_x, mean_y):
    chosen = choose_the_mixture(np.random.random(), weights)
    mean = [mean_x[chosen], mean_y[chosen]]
    covariance = [[np.square(std_x[chosen]),
                   correlations[chosen] * std_x[chosen] * std_y[chosen]],
                  [correlations[chosen] * std_x[chosen] * std_y[chosen],
                   np.square(std_y[chosen])]]
    return np.random.multivariate_normal(mean, covariance)
