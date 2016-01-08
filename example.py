# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
from KernelKMeans import KernelKMeans


def make_dataset(N):
    X = X = np.zeros((N, 2))
    X[: N / 2, 0] = 10 * np.cos(np.linspace(0.2 * np.pi, N / 2, num=N / 2))
    X[N / 2:, 0] = np.random.randn(N / 2)
    X[: N / 2, 1] = 10 * np.sin(np.linspace(0.2 * np.pi, N / 2, num=N / 2))
    X[N / 2:, 1] = np.random.randn(N / 2)
    return X


if __name__ == '__main__':

    X = make_dataset(500)

    # kernel k-means with linear kernel
    kkm_linear = KernelKMeans(
        n_clusters=2, max_iter=100, kernel=pairwise.linear_kernel)
    y_linear = kkm_linear.fit_predict(X)

    # kernel k-means with rbf kernel
    kkm_rbf = KernelKMeans(
        n_clusters=2, max_iter=100,
        kernel=lambda X: pairwise.rbf_kernel(X, gamma=0.1))
    y_rbf = kkm_rbf.fit_predict(X)

    plt.subplot(121)
    plt.scatter(X[y_linear == 0][:, 0], X[y_linear == 0][:, 1], c="blue")
    plt.scatter(X[y_linear == 1][:, 0], X[y_linear == 1][:, 1], c="red")
    plt.title("linear kernel")
    plt.axis("scaled")
    plt.subplot(122)
    plt.scatter(X[y_rbf == 0][:, 0], X[y_rbf == 0][:, 1], c="blue")
    plt.scatter(X[y_rbf == 1][:, 0], X[y_rbf == 1][:, 1], c="red")
    plt.title("rbf kernel")
    plt.axis("scaled")

    plt.show()
