import numpy as np


class ConfusionMatrix(object):
    def __init__(self, actual, predicted, sorter=None):
        self.actual = np.array(actual)
        self.predicted = predicted  # predicted can be a list of lists

        self.labels = np.unique(self.actual)
        self.n = len(self.labels)
        self._matrix = np.zeros((self.n, self.n))
        for a, p in zip(self.actual, self.predicted):
            try:
                tied = float(len(p))
            except:
                tied = 1.0
            self._matrix[a][p] += 1.0 / tied
        self._sorter = sorter

    def set_sorter(self, sorter):
        self._sorter = sorter

    @property
    def matrix(self):
        return (self._matrix if
                self._sorter is None else
                self._matrix[:, self._sorter][self._sorter])

    @property
    def p_xy(self):
        """Return joint probability matrix (normalized)"""
        return self.matrix / np.sum(self.matrix)

    @property
    def p_x(self):
        return np.sum(self.p_xy, axis=1)

    @property
    def p_y(self):
        return np.sum(self.p_xy, axis=0)

    def p_x_cond(self, i=None):
        """Conditional probabilities for x"""
        if i is None:
            return # normalized 
        return self.p_xy[i] / np.sum(self.p_xy[i])

    def p_y_cond(self, i=None):
        """Conditional probabilities for y"""
        if i is None:
            return
        return self.p_xy[:, i] / np.sum(self.p_xy[:, i])

    def accuracy(self):
        return np.sum(np.diag(self.p_xy))

    def information(self):
        nonzero = self.p_xy >= 1e-12
        return np.sum(self.p_xy[nonzero] *
            np.log2(self.p_xy[nonzero] / np.outer(self.p_x, self.p_y)[nonzero]))

