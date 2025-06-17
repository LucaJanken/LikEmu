# likelihoods/planck_gaussian.py

import os
import numpy as np
from .base import BaseLikelihood


def load_full_planck_chain(chains_dir, dim, skip_rows=500, max_rows_per_file=None):
    """
    Load and concatenate Planck MCMC chain files (*.txt) from a directory.

    Returns
    -------
    np.ndarray
        Combined chain array of shape (total_samples, 2 + dim), where:
        - Column 0: multiplicity (number of times sample is repeated)
        - Column 1: log-likelihood
        - Columns 2 to 2+dim: parameter values
    """
    chain_files = [
        os.path.join(chains_dir, f)
        for f in os.listdir(chains_dir)
        if f.endswith(".txt")
    ]
    data_list = []
    for file in chain_files:
        data = np.loadtxt(
            file,
            skiprows=skip_rows,
            max_rows=max_rows_per_file,
            usecols=range(2 + dim)
        )
        data_list.append(data)
    return np.concatenate(data_list, axis=0)


class PlanckGaussianLikelihood(BaseLikelihood):
    """
    Gaussian likelihood using the empirical mean and covariance from full-Planck MCMC chains,
    accounting for sample multiplicities (weights).
    """
    def __init__(
        self,
        N,
        chains_dir,
        skip_rows=500,
        max_rows_per_file=None
    ):
        super().__init__(N)

        # Load chain data
        combined_chain = load_full_planck_chain(
            chains_dir, N, skip_rows, max_rows_per_file
        )

        multiplicities = combined_chain[:, 0]
        X_full = combined_chain[:, 2 : 2 + N]

        # Normalize weights
        weights = multiplicities / np.sum(multiplicities)

        # Weighted mean
        mu = np.average(X_full, axis=0, weights=weights)

        # Weighted covariance
        diff = X_full - mu
        cov = np.cov(diff.T, aweights=weights, ddof=0)

        # Store for likelihood evaluations
        self.mu = mu
        self.Sigma = cov
        self.inv_Sigma = np.linalg.inv(cov)
        self.log_det = np.log(np.linalg.det(cov))

    def neg_loglike(self, x):
        diff = x - self.mu
        if diff.ndim == 1:
            quad = diff @ self.inv_Sigma @ diff
            return 0.5 * (np.log((2 * np.pi) ** self.N) + self.log_det + quad)
        quad = np.sum(diff * (diff @ self.inv_Sigma), axis=1)
        return 0.5 * (np.log((2 * np.pi) ** self.N) + self.log_det + quad)
