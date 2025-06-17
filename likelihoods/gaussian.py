# likelihoods/gaussian.py

import numpy as np
from .base import BaseLikelihood

class GaussianLikelihood(BaseLikelihood):
    """Gaussian likelihood with a randomly generated covariance matrix."""
    def __init__(self, N):
        super().__init__(N)
        self.Sigma = self.generate_covariance(N)
    
    @staticmethod
    def generate_covariance(N):
        # Generate moderate eigenvalues (e.g., between 0.5 and 1.5)
        eigvals = np.exp(np.random.uniform(np.log(0.01), np.log(1000), size=N))
        A = np.random.randn(N, N)
        Q, _ = np.linalg.qr(A)
        # return diagonal covariance matrix for testing
        # return np.diag(eigvals) 
        return Q @ np.diag(eigvals) @ Q.T

    def neg_loglike(self, x):
        diff = x - self.mu
        inv_Sigma = np.linalg.inv(self.Sigma)
        quad = np.sum(diff * (np.dot(diff, inv_Sigma)), axis=1)
        log_det = np.log(np.linalg.det(self.Sigma))
        return 0.5 * (np.log((2 * np.pi) ** self.N) + log_det + quad)
