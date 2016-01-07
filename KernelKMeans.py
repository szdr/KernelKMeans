# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt


class KernelKMeans(object):

    def __init__(self,
                 n_clusters=8,
                 max_iter=300,
                 kernel=pairwise.linear_kernel):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.kernel = kernel

    def _initialize_cluster(self, X):
        self.N = np.shape(X)[0]
        self.y = np.random.randint(low=0, high=self.n_clusters, size=self.N)
        self.K = self.kernel(X)

    def predict(self, X):
        self._initialize_cluster(X)
        for _ in range(self.max_iter):
            obj = np.tile(np.diag(self.K).reshape((-1, 1)), self.n_clusters)
            print(np.shape(obj))
            N_c = np.bincount(self.y)
            print(N_c)
            for c in range(self.n_clusters):
                obj[:, c] -= 2 * \
                    np.sum((self.K)[:, self.y == c], axis=1) / N_c[c]
                obj[:, c] += np.sum((self.K)[self.y == c][:, self.y == c]) / \
                    (N_c[c] ** 2)
            print(obj)
            self.y = np.argmin(obj, axis=1)
        return self.y

if __name__ == '__main__':
    N = 500
    X = X = np.zeros((N, 2))
    X[: N / 2, 0] = 5 * np.cos(np.linspace(0.2 * np.pi, N / 2, num=N / 2))
    X[N / 2:, 0] = np.random.randn(N / 2)
    X[: N / 2, 1] = 5 * np.sin(np.linspace(0.2 * np.pi, N / 2, num=N / 2))
    X[N / 2:, 1] = np.random.randn(N / 2)

    kkm = KernelKMeans(n_clusters=2, max_iter=10, kernel=lambda X: pairwise.rbf_kernel(X, gamma=0.05))
    y = kkm.predict(X)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c="blue")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c="red")
    plt.show()
