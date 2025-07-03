import numpy as np
from .planck_gaussian import PlanckGaussianLikelihood
from .base import BaseLikelihood

class ExtendedPlanckGaussianLikelihood(BaseLikelihood):
    """
    Gaussian likelihood on the first N_main parameters (using full-Planck chains)
    plus an extra penalty exp[-½ (x_{N+1} · x_{N+2})²],
    with a full (N_main+2)-dimensional covariance (block-diagonal).
    """
    def __init__(
        self,
        N_main,
        chains_dir,
        skip_rows=500,
        max_rows_per_file=None
    ):
        # total dimension is N_main + 2
        super().__init__(N_main + 2)
        self.N_main = N_main

        # reuse PlanckGaussianLikelihood for the first N_main dims
        gaussian = PlanckGaussianLikelihood(
            N_main,
            chains_dir,
            skip_rows,
            max_rows_per_file
        )
        # store the Gaussian sub-likelihood
        self.gaussian = gaussian

        # build full mean: [mu_main; 0; 0]
        mu_main = gaussian.mu
        self.mu = np.concatenate([mu_main, np.zeros(2)])

        # build block-diagonal covariance:
        # [ Sigma_main    0      ]
        # [    0        I_{2×2} ]
        Sigma_main = gaussian.Sigma
        D = N_main + 2
        Sigma_full = np.zeros((D, D))
        Sigma_full[:N_main, :N_main] = Sigma_main
        Sigma_full[N_main:, N_main:] = np.eye(2)
        self.Sigma = Sigma_full
        self.inv_Sigma = np.linalg.inv(Sigma_full)
        self.log_det = np.log(np.linalg.det(Sigma_full))

    def neg_loglike(self, x):
        # ensure (n_samples, D)
        x = np.atleast_2d(x)
        # split
        x_main = x[:, : self.N_main]
        x1 = x[:, self.N_main]
        x2 = x[:, self.N_main + 1]

        # 1) Gaussian part on first N_main dims
        gauss_nll = self.gaussian.neg_loglike(x_main)

        # 2) penalty term on extras
        penalty = 0.5 * (x1 * x2) ** 2

        return gauss_nll + penalty
