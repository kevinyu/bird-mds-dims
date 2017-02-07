import numpy as np

import scipy.stats


def fit_gaussians(table, x_arr, key="stim"):
    if key not in {"stim", "stim_type"}:
        raise Exception("key must be either stim or stim_type")

    dists = []
    p = []
    for val in np.unique(table[key]):
        x_stim = x_arr[table[key] == val]
        mu = np.mean(x_stim, axis=0)
        cov = np.cov(x_stim.T)
        dists.append(scipy.stats.multivariate_normal(mu, cov, allow_singular=True))
        p.append(float(len(x_stim)))
    p = np.array(p) / np.sum(p)

    return dists, p
