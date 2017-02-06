import numpy as np
import sklearn.metrics


def groupers(column):
    choices = np.unique(column)
    result = []
    for choice in choices:
        result.append((choice, column == choice))
    return result


def sample(data, sample_n):
    """Chooses sample_n rows from data"""
    indices = np.random.choice(len(data), sample_n, replace=False)
    return data[indices]


def dist(datapoints):
    return sklearn.metrics.euclidean_distances(datapoints)


