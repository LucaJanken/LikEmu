#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pickle
import yaml
from likelihoods.gaussian import GaussianLikelihood
from likelihoods.planck_gaussian import PlanckGaussianLikelihood
from plotting.plot_utils import compute_levels

def plot_iterative_sampling(
    training_sets,
    analytic_mu,
    analytic_cov,
    param_names=None,
    idx_pair=(0, 1),
    output_filename="plots/iterative_sampling.pdf",
    n_chains=None
):
    """
    training_sets: list of (agg_X, agg_y, agg_chain_ids)
      - agg_chain_ids uses -1 for initial samples, 0..n_chains-1 for each MCMC chain
    idx_pair: tuple of two ints giving which parameters to plot on x and y axes
    """
    x_idx, y_idx = idx_pair
    n_iter = len(training_sets)

    # Color setup
    n_colors = n_chains + 1
    if n_colors <= 10:
        cmap = plt.colormaps['tab10'].resampled(n_colors)
    elif n_colors <= 20:
        cmap = plt.colormaps['tab20'].resampled(n_colors)
    else:
        cmap = plt.colormaps['gist_ncar'].resampled(n_colors)

    boundaries = np.arange(-1.5, n_chains - 0.5 + 1.0, 1.0)
    norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=n_colors)

    # Analytic reference contours for chosen dims
    ref_samples = np.random.multivariate_normal(analytic_mu, 10*analytic_cov, size=1_000_000)
    ref2d = ref_samples[:, [x_idx, y_idx]]
    H_ref, xedges, yedges = np.histogram2d(ref2d[:,0], ref2d[:,1], bins=50, density=True)
    X_ref = 0.5*(xedges[:-1] + xedges[1:])
    Y_ref = 0.5*(yedges[:-1] + yedges[1:])
    levels_ref = compute_levels(H_ref, levels=[0.68, 0.95])

    fig, axes = plt.subplots(1, n_iter, figsize=(5*n_iter, 5), squeeze=False)

    if param_names is None:
        # fallback generic names if none provided
        param_names = [f"Param {i}" for i in range(max(x_idx, y_idx)+1)]

    for i, (Xagg, _, cids) in enumerate(training_sets):
        ax = axes[0, i]
        samp2d = Xagg[:, [x_idx, y_idx]]
        sc = ax.scatter(
            samp2d[:,0], samp2d[:,1],
            c=cids, cmap=cmap, norm=norm,
            s=4, alpha=0.6
        )
        sc.set_rasterized(True)

        # overlay analytic contours
        for lvl, ls in zip(levels_ref, ['solid','dashed']):
            ax.contour(X_ref, Y_ref, H_ref.T, levels=[lvl],
                       colors='green', linestyles=ls, linewidths=2, zorder=2)

        ax.set_xlabel(param_names[x_idx])
        ax.set_ylabel(param_names[y_idx])
        ax.set_title(f"Iteration {i+1}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename)


if __name__ == '__main__':
    cfg_path = os.path.join("config", "default.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    N = int(cfg['model']['dimension'])
    n_chains = int(cfg['mcmc']['n_chains'])

    training_sets_file = os.path.join("iterative_data", "training_sets.pkl")
    if not os.path.exists(training_sets_file):
        print(f"{training_sets_file} not found.")
        sys.exit(1)

    with open(training_sets_file, "rb") as f:
        training_sets = pickle.load(f)

    # choose Planck-based target
    target = PlanckGaussianLikelihood(
        N=N,
        chains_dir="class_chains/27_param",
        skip_rows=500
    )

    # e.g. plot
    plot_iterative_sampling(
        training_sets,
        analytic_mu=target.mu,
        analytic_cov=target.Sigma,
        param_names=[f"Param {i+1}" for i in range(N)],
        idx_pair=(27, 28),  # second and fifth parameter
        output_filename="plots/iterative_sampling.pdf",
        n_chains=n_chains
    )
