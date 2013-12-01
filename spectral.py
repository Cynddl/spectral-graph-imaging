try:
    import prettyplotlib as ppl
except Exception, e:
    pass

from prettyplotlib import plt

from graph_tool.all import *
import graph_tool as gt

import math
import numpy as np
from scipy import linalg as LA
from scipy.sparse import linalg as SLA


class ImageGraph:
    """Graph constructed by an image using neighbors relations."""

    def __init__(self, img):
        self.img = img
        self.width = len(img)
        self.height = len(img[0])
        self.shape = (self.width, self.height)

        self.g = gt.Graph(directed=False)

        self.index = [(i, j) for i in range(self.width) for j in range(self.height)]
        self.to_index = [[self.g.add_vertex() for i in range(self.width)] for j in range(self.height)]

        self.weight = self.g.new_edge_property('double')

    def apply_weight(self, sigma=0.2):

        def try_add_edge(i, j, i2, j2):
            try:
                if i >= 0 and j >= 0 and i2 >= 0 and j2 >= 0:
                    e = self.g.add_edge(self.to_index[i][j], self.to_index[i2][j2])
                    first_px = self.img[i][j]
                    last_px = self.img[i2][j2]
                    self.weight[e] = math.exp(-np.abs(first_px - last_px)**2 / (2 * sigma**2))  # if abs(first_px - last_px) < 10 else 0
            except IndexError:
                pass

        for i, j in self.index:
            try_add_edge(i, j, i, j + 1)
            try_add_edge(i, j, i + 1, j)
            try_add_edge(i, j, i + 1, j + 1)
            try_add_edge(i, j, i + 1, j - 1)

    def compute_spectral(self):
        L = gt.spectral.laplacian(self.g, weight=self.weight, normalized=True)
        eigenvalues, eigenvectors = LA.eigh(L.todense(), type=3)

        # Order eigenvalues and eigenvectors
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Normalize the eigenvectors
        eigenvectors[:, 0] = eigenvectors[:, 0] / LA.norm(eigenvectors[:, 0])
        eigenvectors[:, 1] = eigenvectors[:, 1] / LA.norm(eigenvectors[:, 1])

        return eigenvalues, eigenvectors

    def compute_sparse_spectral(self, k=10):
        L = gt.spectral.laplacian(self.g, weight=self.weight, normalized=True)
        eigenvalues, eigenvectors = SLA.eigsh(L, k=k, which='SA')

        # Order eigenvalues and eigenvectors
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Normalize the eigenvectors
        eigenvectors[:, 0] = eigenvectors[:, 0] / LA.norm(eigenvectors[:, 0])
        eigenvectors[:, 1] = eigenvectors[:, 1] / LA.norm(eigenvectors[:, 1])

        return eigenvalues, eigenvectors


def plot_signal(signal, shape=None, name=None):
    """
    Plot a numeric signal as an image.
    """
    s = np.copy(signal)
    if s.max() != 0.:
        s *= 255.0/s.max()
    s = s.astype(int)
    if shape:
        s = s.reshape(shape)

    plt.figure()
    plt.imshow(s, cmap=plt.cm.gray, interpolation='nearest')
    if name:
        plt.savefig(name)
    plt.close()


## Fourier transform

def fourier(signal, eigenvectors):
    return np.dot(signal, eigenvectors)


def inverse_fourier(f_signal, eigenvectors):
    return sum(f_signal[i] * eigenvectors[:, i] for i in range(len(f_signal)))


def inverse_fourier_filter(f_signal, eigenvectors, eigenvalues, gamma=10.):
    return sum(f_signal[i] * eigenvectors[:, i] / (1. + gamma * eigenvalues[i]) for i in range(len(f_signal)))
