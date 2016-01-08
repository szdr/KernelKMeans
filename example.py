# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
from KernelKMeans import KernelKMeans


def make_dataset(N):
    X = X = np.zeros((N, 2))
    X[: N / 2, 0] = 10 * np.cos(np.linspace(0.1 * np.pi, N / 2, num=N / 2))
    X[N / 2:, 0] = np.random.randn(N / 2)
    X[: N / 2, 1] = 10 * np.sin(np.linspace(0.1 * np.pi, N / 2, num=N / 2))
    X[N / 2:, 1] = np.random.randn(N / 2)
    return X


if __name__ == '__main__':

    X = make_dataset(500)
    kernel = lambda X: pairwise.rbf_kernel(X, gamma=0.1)
    kernel = pairwise.linear_kernel
    kkm = KernelKMeans(n_clusters=2, max_iter=100, kernel=kernel)
    y = kkm.fit_predict(X)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c="blue")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c="red")
    plt.show()
