"""
script to fit gaussian mixture models
numpy practice
matthew farrugia-roberts
"""

import numpy as np
from scipy.stats import multivariate_normal as mvnorm
import matplotlib.pyplot as plt

class gmm:
    def __init__(self, n_classes, n_dimensions, random_state=None):
        self.K = n_classes
        self.D = n_dimensions
        self.rng = np.random.default_rng(seed=random_state)
    def fit(self, X, tol=1e-10, verbose=False):
        N, D = X.shape
        K = self.K
        assert D == self.D
        def Mstep(Z):
            totals = Z.sum(axis=0)
            priors = totals / N
            means  = (Z.T @ X) / totals[:, np.newaxis]
            covars = np.ndarray(shape=(K, D, D))
            for k in range(K):
                X_centered = X - means[k]
                covars[k] = (Z[:, k]*X_centered.T) @ X_centered / totals[k]
            return priors, means, covars
        def Estep(priors, means, covars):
            resps = np.ndarray((N, K))
            for k in range(K):
                resps[:, k] = priors[k] * mvnorm.pdf(X, means[k], covars[k])
            return resps / resps.sum(axis=1)[:, np.newaxis]
        # init:
        Z = np.zeros((N, K))
        Z[np.arange(N), self.rng.choice(self.K, N)] = 1
        T = Mstep(Z)
        if verbose:
            plt.ion()
            fig, ax = plt.subplots()
            xs = ax.scatter(*X.T, c=Z, marker="x")
            cs = ax.scatter(*T[1].T, c=["red", "green", "blue"])
            ax.set_title("performing EM algorithm...")
            fig.canvas.draw()
        # loop
        converged = False
        while not converged:
            T_old = T
            Z = Estep(*T)
            T = Mstep(Z)
            if verbose:
                cs.set_offsets(T[1])
                xs.set_color(Z)
                fig.canvas.draw()
                plt.pause(1e-10)
            converged = np.sum(np.square(T_old[1]-T[1])) < tol
        self.priors = T[0]
        self.means  = T[1]
        self.covars = T[2]
        return self





# make some random data
np.random.seed(42)
data = np.random.random((40, 2))
G = gmm(3, 2, random_state=42).fit(data, verbose=True)
print(G.priors)
print(G.means)
print([np.linalg.det(cv) for cv in G.covars])
