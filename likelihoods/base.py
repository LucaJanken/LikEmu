# likelihoods/base.py

import numpy as np

class BaseLikelihood:
    """Interface for likelihood functions."""
    def __init__(self, N):
        self.N = N
        self.mu = np.zeros(N)

        # ensure prior_lower / prior_upper always exist
        self.prior_lower = -np.inf * np.ones(N)
        self.prior_upper =  np.inf * np.ones(N)


    def neg_loglike(self, x):
        """Compute the negative log-likelihood for input x."""
        raise NotImplementedError("Must be implemented in subclass.")
