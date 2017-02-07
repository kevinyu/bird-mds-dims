"""
Functions for calculating mutual information over distributions
"""

import numpy as np


def monte_carlo_mutual_information(dists, n, p=None, anthropic=False):
    """Monte carlo estimate of mutual information of mixture model, weighted by p
   
    Returns monte carlo estimate of the mutual information with estimated error
    """
    if p is None:
        p = np.ones(len(dists)) / float(len(dists))
    sampled_dists = np.random.choice(dists, n, replace=True, p=p)

    total = []
    for idx, dist in enumerate(sampled_dists):
        r = dist.rvs(1)
        p_r_cond = dist.pdf(r)

        if anthropic:
            p_anthropic = np.ma.array(p, mask=False)
            p_anthropic.mask[idx] = True
            p_anthropic = p_anthropic / np.sum(p_anthropic)

            p_r_mixture = np.sum([p_anthropic[i] * dists[i].pdf(r)
                for i, d in enumerate(dists)
                if d is not dist])
        else:
            p_r_mixture = np.sum([p[i] * dists[i].pdf(r) for i in range(len(dists))])

        if p_r_cond > 1e-10:
            log_term = np.log2(p_r_cond / p_r_mixture)
            total.append(log_term)

    return np.sum(total) / n, np.std(total, ddof=1) / np.sqrt(n)

