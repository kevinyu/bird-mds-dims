from itertools import groupby

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np


def plotting_tools(table):
    stim_to_stim_type = sorted(set(zip(table["stim"], table["stim_type"])))
    stim_types = np.array([_[1] for _ in stim_to_stim_type])
    sorter = np.argsort(stim_types)
    data = stim_types[sorter]

    dividers = [len(list(group)) for _, group in (groupby(data))]
    for i in range(1, len(dividers)):
        dividers[i] += dividers[i-1]
    dividers = map(lambda x: x-0.5, dividers)

    lengths = [len(list(group)) for _, group in groupby(data)]

    midpoints = []
    for i in range(len(lengths)):
        midpoints.append(dividers[i] - 0.5 * lengths[i])
    return dividers, midpoints


def plot_confusion_matrix(conf, table=None, labels=None):
    im = plt.imshow(conf.p_xy, interpolation="none")

    if table is not None:
        dividers, midpoints = plotting_tools(table)
        plt.hlines(dividers, -0.5, conf.n - 0.5, "r")
        plt.vlines(dividers, -0.5, conf.n - 0.5, "r")
        if labels is not None:
            plt.xticks(midpoints, labels, rotation="vertical")
            plt.yticks(midpoints, labels, rotation="horizontal")

    return im


