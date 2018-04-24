import numpy as np


def choose_the_mixture(weights):
    return np.random.choice(np.arange(weights.shape[0]), p=weights)


def sample(weights, std_x, std_y, correlations, mean_x, mean_y):
    result = 0
    for chosen in range(20):
        mean = np.array([mean_x[chosen], mean_y[chosen]])
        covariance = np.array([[np.square(std_x[chosen]), correlations[chosen] * std_x[chosen] * std_y[chosen]],
                               [correlations[chosen] * std_x[chosen] * std_y[chosen], np.square(std_y[chosen])]])
        result += weights[chosen] * np.random.multivariate_normal(mean, covariance)
    return result

