#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import yaml
from likelihoods.gaussian import GaussianLikelihood
from likelihoods.planck_gaussian import PlanckGaussianLikelihood
from plotting.plot_utils import compute_levels

def plot_iterative_chains(
    chain_data,
    analytic_mu,
    analytic_cov,
    param_names=None,
    idx_pair=(0, 1),
    output_filename="plots/iterative_chain_plots.pdf"
):
    """
    Generate 2D credible interval plots for the MCMC chains of each iteration.

    Parameters:
      chain_data: List of (samples, weights) per iteration
      analytic_mu: Mean vector of the reference Gaussian
      analytic_cov: Covariance matrix of the reference
      param_names: List of parameter names
      idx_pair: Tuple (i, j) of the two dimensions to plot
      output_filename: Path for saving the PDF
    """
    x_idx, y_idx = idx_pair
    n_iterations = len(chain_data)

    # Prepare analytic reference samples in chosen dims
    analytic_samples = np.random.multivariate_normal(analytic_mu, 10 * analytic_cov, size=1_000_000)
    ref2d = analytic_samples[:, [x_idx, y_idx]]
    bins = 50
    H_ref, xe_ref, ye_ref = np.histogram2d(ref2d[:,0], ref2d[:,1], bins=bins, density=True)
    X_ref = 0.5 * (xe_ref[:-1] + xe_ref[1:])
    Y_ref = 0.5 * (ye_ref[:-1] + ye_ref[1:])
    levels_ref = compute_levels(H_ref, levels=[0.68, 0.95])

    fig, axes = plt.subplots(1, n_iterations, figsize=(5 * n_iterations, 5), squeeze=False)

    if param_names is None:
        param_names = [f"Param {i}" for i in (x_idx, y_idx)]

    for i, (samples, weights) in enumerate(chain_data):
        ax = axes[0, i]

        # extract chosen dims
        chain2d = samples[:, [x_idx, y_idx]]
        H_net, xe_net, ye_net = np.histogram2d(
            chain2d[:,0], chain2d[:,1],
            bins=bins, density=True, weights=weights
        )
        X_net = 0.5 * (xe_net[:-1] + xe_net[1:])
        Y_net = 0.5 * (ye_net[:-1] + ye_net[1:])
        levels_net = compute_levels(H_net, levels=[0.68, 0.95])

        # reference contours (green)
        for lvl, ls in zip(levels_ref, ['solid', 'dashed']):
            ax.contour(X_ref, Y_ref, H_ref.T, levels=[lvl],
                       colors='green', linestyles=ls, linewidths=2)

        # chain contours (blue)
        for lvl, ls in zip(levels_net, ['solid', 'dashed']):
            ax.contour(X_net, Y_net, H_net.T, levels=[lvl],
                       colors='blue', linestyles=ls)

        ax.set_xlabel(param_names[x_idx])
        ax.set_ylabel(param_names[y_idx])
        ax.set_title(f"Iteration {i+1}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=300)
    plt.show()


if __name__ == '__main__':
    # load config
    config_path = os.path.join("config", "default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    N = int(config['model']['dimension'])
    n_iterations = int(config['iterative']['n_iterations'])
    np.random.seed(int(config['misc']['random_seed']))

    # load chains
    chain_data = []
    for it in range(1, n_iterations + 1):
        fname = os.path.join("mcmc", "chains", f"chain_iter{it}.txt")
        if os.path.exists(fname):
            data = np.loadtxt(fname)
            weights = data[:, 0]
            samples = data[:, 2:]
            chain_data.append((samples, weights))
        else:
            print(f"{fname} not found.")

    # target likelihood
    chains_dir = "class_chains/27_param"
    target = PlanckGaussianLikelihood(
        N=N,
        chains_dir=chains_dir,
        skip_rows=500,
        max_rows_per_file=None
    )

    param_names = [f"Param {i+1}" for i in range(N)]
    # e.g.
    plot_iterative_chains(
        chain_data,
        analytic_mu=target.mu,
        analytic_cov=target.Sigma,
        param_names=param_names,
        idx_pair=(27, 28),
        output_filename="plots/iterative_chain_plots.pdf"
    )
