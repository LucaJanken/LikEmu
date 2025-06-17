#!/usr/bin/env python3
"""
Standalone script to evaluate the “fake” AnalyticEmulator network
at (a) the analytical best‐fit (mu_true) and (b) the MCMC best‐fit found
in the chain file.
"""

import os
import sys

# add project root so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
import yaml
from sklearn.preprocessing import StandardScaler
from likelihoods.planck_gaussian import PlanckGaussianLikelihood


class AnalyticEmulator(tf.keras.Model):
    def __init__(self, likelihood):
        super().__init__()
        self.likelihood = likelihood

    def call(self, x_std, training=False):
        # Ask for float64 from numpy_function, then cast to float32
        neg_ll64 = tf.numpy_function(
            func=lambda x: self.likelihood.neg_loglike(x),
            inp=[x_std],
            Tout=tf.float64
        )
        neg_ll32 = tf.cast(neg_ll64, tf.float32)
        return tf.expand_dims(neg_ll32, axis=1)

def main():
    # 1. Load configuration
    cfg_path = os.path.join("config", "default.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    N = int(cfg['model']['dimension'])
    chain_file = os.path.join(cfg['paths']['mcmc_chains'], "chain_planck_test.txt")

    # 2. Instantiate PlanckGaussianLikelihood
    target = PlanckGaussianLikelihood(
        N=N,
        chains_dir="class_chains/27_param",
        skip_rows=500,
        max_rows_per_file=None
    )
    mu_true  = target.mu          # analytical mean, shape (N,)
    cov_true = target.Sigma       # analytical covariance

    # 3. Build the fake “network” and identity scalers
    model = AnalyticEmulator(target)
    scaler_x = StandardScaler()
    scaler_x.mean_  = np.zeros(N)
    scaler_x.scale_ = np.ones(N)
    scaler_y = StandardScaler()
    scaler_y.mean_  = np.zeros(1)
    scaler_y.scale_ = np.ones(1)

    # 4. Evaluate at mu_true
    x_mu = mu_true.reshape(1, -1).astype(np.float32)
    neg_ll_mu = model(tf.constant(x_mu), training=False).numpy().reshape(-1)[0]
    direct_mu = target.neg_loglike(mu_true)
    print(f"AnalyticEmulator at mu_true: {neg_ll_mu:.6f}")
    print(f"Direct neg_loglike(mu_true): {direct_mu:.6f}")

    # 5. Load chain, find best‐fit (lowest -loglike)
    if not os.path.exists(chain_file):
        print(f"Chain file not found: {chain_file}")
        sys.exit(1)

    chain = np.loadtxt(chain_file, skiprows=1)
    # columns: [multiplicity, -loglike, θ₁, θ₂, …, θN]
    best_idx   = np.argmin(chain[:,1])
    theta_best = chain[best_idx, 2:]
    print(f"MCMC best‐fit θ (first 3 dims): {theta_best[:3]} ...")

    # 6. Evaluate at θ_best
    x_best = theta_best.reshape(1, -1).astype(np.float32)
    neg_ll_best = model(tf.constant(x_best), training=False).numpy().reshape(-1)[0]
    direct_best = target.neg_loglike(theta_best)
    print(f"AnalyticEmulator at θ_best: {neg_ll_best:.6f}")
    print(f"Direct neg_loglike(θ_best): {direct_best:.6f}")

if __name__ == "__main__":
    main()
