from __future__ import print_function

import time 

import numpy as np

from util import groupers, sample


class CachedTemplate(object):
    """Cache array lookups to make template generation more efficient"""

    def __init__(self):
        self.cache = {}

    def summarize(self, x_arr):
        return np.mean(x_arr, axis=0)

    def _distance(self, x, template):
        return np.linalg.norm(template - x)

    def distances(self, x, y_arr):
        distances = []
        for y in y_arr:
            distances.append(self._distance(x, y))
        return distances

    def _cached_select(self, x_arr, selector):
        key = tuple(selector)
        if tuple(selector) not in self.cache:
            self.cache[key] = x_arr[selector]
        return self.cache[key]

    def compute_templates(self, x_arr, selectors, sample_n=None, equal_sample=True):
        row_templates = []
        for selector in selectors:
            template_x = self._cached_select(x_arr, selector)
            if sample_n:
                template_x = sample(template_x, sample_n)
            template = self.summarize(template_x)
            row_templates.append(template)
        return row_templates


def template_selectors(table, key="stim"):
    """Return array filters that will select the pool of datapoints
    from which to build templates

    Usage:
    selectors, stims = template_selectors(table)
    table[

    """
    selectors = []

    # iterate over all data points
    for row in table:
        # compute templates to each other stim
        templates = []
        for stim, grouper in groupers(table[key]):
            # select all of the elements that are either
            #  - when the same stim, exclude the trial
            selector = (grouper * 
                    np.logical_or(
                        table["stim"] != row["stim"],
                        table["trial"] != row["trial"]))
            templates.append(selector)
        selectors.append(templates)

    return np.array(selectors), np.unique(table[key])


def compute_distances_to_all_templates(x_arr, template_selectors, TemplateClass, verbose=False):
    sample_n = np.min([[np.sum(selector)
        for selectors in template_selectors
        for selector in selectors]])

    distance_arr = []
    templates_cache = {}  # dont bother doing template computation if we've done it before
    template = TemplateClass()
    if verbose:
        print("Starting distance computations")
        start = time.time()
    for i, (x, selectors) in enumerate(zip(x_arr, template_selectors)):
        if verbose and i % 10 == 0:
            print("Completed {}/{} rows in {}".format(i, len(x_arr), time.time() - start))
        distances = []

        templates = template.compute_templates(x_arr, selectors, sample_n=sample_n, equal_sample=True)
        distances = template.distances(x, templates)
        distance_arr.append(distances)
    return np.array(distance_arr)


def compute_euclidian_distances_to_all_templates(x_arr, template_selectors, verbose=False):
    # TODO can refactor this now maybe to stop having to pass CachedTemplate class
    return compute_distances_to_all_templates(x_arr, template_selectors, CachedTemplate, verbose=verbose)


def compute_nearest_template_with_ties(distance_arr, stims):
    nearest_stims = []
    for distances in distance_arr:
        nearest_stims.append(np.where(distances == np.min(distances))[0])
    return nearest_stims

