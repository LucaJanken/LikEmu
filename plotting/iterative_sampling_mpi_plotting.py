#!/usr/bin/env python3
import os
import sys
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from likelihoods.planck_gaussian import PlanckGaussianLikelihood
from plotting.plot_utils import compute_levels

def load_training_sets(path):
    """
    Load training_sets.pkl which may be either:
      - A list of (X, y) tuples, one per iteration, or
      - A single (X, y) tuple (final aggregate).
    Returns a list of (X, y).
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, tuple) and len(data) == 2:
        return [data]
    else:
        raise ValueError(f"Unrecognized format in {path}")

def plot_iterative_sampling(training_sets, analytic_mu, analytic_cov,
                            param_names=None, output_filename="plots/iterative_sampling.pdf"):
    n_iterations = len(training_sets)

    # Draw a huge cloud from 3× covariance for stable contour estimates
    analytic_samples = np.random.multivariate_normal(
        analytic_mu, 3*analytic_cov, size=1_000_000
    )[:, :2]
    H_ref, xedges_ref, yedges_ref = np.histogram2d(
        analytic_samples[:,0], analytic_samples[:,1],
        bins=50, density=True
    )
    X_ref = 0.5*(xedges_ref[:-1] + xedges_ref[1:])
    Y_ref = 0.5*(yedges_ref[:-1] + yedges_ref[1:])
    levels_ref = compute_levels(H_ref, levels=[0.68, 0.95])

    fig, axes = plt.subplots(1, n_iterations,
                             figsize=(5*n_iterations, 5),
                             squeeze=False)

    if param_names is None:
        param_names = [f"Param {i+1}" for i in range(analytic_mu.shape[0])]

    for i, (Xagg, yagg) in enumerate(training_sets):
        ax = axes[0, i]
        samples_2d = Xagg[:, :2]

        # network‐based density contours
        H_net, xedges_net, yedges_net = np.histogram2d(
            samples_2d[:,0], samples_2d[:,1],
            bins=50, density=True
        )
        X_net = 0.5*(xedges_net[:-1] + xedges_net[1:])
        Y_net = 0.5*(yedges_net[:-1] + yedges_net[1:])
        levels_net = compute_levels(H_net, levels=[0.68, 0.95])

        # plot reference
        for lvl, ls in zip(levels_ref, ['solid','dashed']):
            ax.contour(X_ref, Y_ref, H_ref.T,
                       levels=[lvl],
                       colors='green',
                       linestyles=ls,
                       linewidths=2,
                       zorder=2)

        # scatter of samples
        sc = ax.scatter(samples_2d[:,0], samples_2d[:,1],
                        s=2, alpha=0.6, color='blue', label='Training')
        sc.set_rasterized(True); sc.set_zorder(0)

        ax.set_title(f"Iteration {i+1}")
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename)
    print(f"Saved iterative‐sampling plot to {output_filename}")

if __name__ == "__main__":
    # load config to get dimensions & paths
    cfg = yaml.safe_load(open("config/default.yaml"))
    N = int(cfg['model']['dimension'])
    paths = cfg.get('paths', {})
    iterative_data_dir = paths.get('iterative_data', "iterative_data")
    chains_dir = paths.get('class_chains_dir', "class_chains/27_param")

    # instantiate the same likelihood you used in MPI run
    lik = PlanckGaussianLikelihood(
        N=N,
        chains_dir=chains_dir,
        skip_rows=500,
        max_rows_per_file=None
    )
    analytic_mu  = lik.mu
    analytic_cov = lik.Sigma

    # load your saved training‐sets
    ts_file = os.path.join(iterative_data_dir, "training_sets.pkl")
    training_sets = load_training_sets(ts_file)

    # make param names
    param_names = [f"Param {i+1}" for i in range(N)]

    # do the plotting
    plot_iterative_sampling(
        training_sets,
        analytic_mu,
        analytic_cov,
        param_names=param_names,
        output_filename="plots/iterative_sampling.pdf"
    )
