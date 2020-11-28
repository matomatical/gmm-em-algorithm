"""
script to fit gaussian mixture models
numpy practice
disclaimer: there may be errors
Matthew Farrugia-Roberts, 2020.04
"""

import numpy as np
from scipy.stats import multivariate_normal as mv_normal
import matplotlib.pyplot as plt

class gmm:
    def __init__(self, n_clusters, n_dimensions, random_state=None):
        self.n_clusters = n_clusters
        self.n_dimensions = n_dimensions
        self.rng = np.random.default_rng(seed=random_state)
    def fit(self, X, tol=1e-10, plotting=False):
        n, d = X.shape
        k = self.n_clusters
        # check dimensionality makes sense
        assert d == self.n_dimensions
        # we only have 2 dimensions and 3 colours 
        assert (d == 2 and k == 3) or not plotting

        # parameter estimation formulas (fn of X)
        def e_step(priors, means, covars):
            resps = np.ndarray((n, k))
            for i in range(k):
                resps[:, i] = priors[i] * mv_normal.pdf(X,means[i],covars[i])
            return resps / resps.sum(axis=1)[:, np.newaxis]
        def m_step(resps):
            totals = resps.sum(axis=0)
            priors = totals / n
            means  = (resps.T @ X) / totals[:, np.newaxis]
            covars = np.ndarray(shape=(k, d, d))
            for i in range(k):
                X_centered = X - means[i]
                covars[i] = (resps[:, i]*X_centered.T) @ X_centered/totals[i]
            return priors, means, covars
        
        # init:
        # random hard cluster assignments, corresponding empirical parameters
        Z = np.zeros((n, k))
        Z[np.arange(n), self.rng.choice(k, n)] = 1
        T = m_step(Z)
        if plotting:
            plt.ion()
            fig, ax = plt.subplots()
            xs = ax.scatter(*X.T, c=Z, marker="x")
            cs = ax.scatter(*T[1].T, c=["red", "green", "blue"])
            fig.canvas.draw()

        # loop until convergence:
        converged = False
        while not converged:
            T0 = T
            Z  = e_step(*T)
            T  = m_step(Z)
            # check sum of squared mean distances to detect convergence
            converged = np.sum(np.square(T0[1]-T[1])) < tol
            if plotting:
                plt.pause(1e-10)
                cs.set_offsets(T[1])
                xs.set_color(Z)
                fig.canvas.draw()
        
        # save learned parameters
        self.priors = T[0]
        self.means  = T[1]
        self.covars = T[2]
        return self


# make some random data
np.random.seed(42)
data = np.random.random((40, 2))

# fit the model (with animated plot)
G = gmm(3, 2, random_state=42).fit(data, plotting=True)

# print the resulting parameters
print('priors', G.priors, sep='\n')
print('means',  G.means, sep='\n')
print('covars', *[covar for covar in G.covars], sep='\n')
