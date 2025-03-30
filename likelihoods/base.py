# likelihoods/base.py

import numpy as np

class BaseLikelihood:
    """Interface for likelihood functions."""
    def __init__(self, N):
        self.N = N
        self.mu = np.zeros(N)

    def neg_loglike(self, x):
        """Compute the negative log-likelihood for input x."""
        raise NotImplementedError("Must be implemented in subclass.")
