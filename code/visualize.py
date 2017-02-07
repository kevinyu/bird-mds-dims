from itertools import groupby

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np


def plot_gaussian_contour(xlim, ylim, mu, sig, delta=0.025, n=5, cmap=None):
    x = np.arange(xlim[0], xlim[1], delta)
    y = np.arange(ylim[0], ylim[1], delta)
    X, Y = np.meshgrid(x, y)

    Z = mlab.bivariate_normal(
        X,
        Y,
        np.sqrt(sig[0,0]),
        np.sqrt(sig[1,1]),
        mu[0],
        mu[1],
        sig[0,1])

    return plt.contour(X, Y, Z, n, cmap=cmap)


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


