#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml
from likelihoods.gaussian import GaussianLikelihood
from likelihoods.planck_gaussian import PlanckGaussianLikelihood
from plotting.plot_utils import compute_levels

def plot_iterative_sampling(training_sets, analytic_mu, analytic_cov,
                            param_names=None, output_filename="plots/iterative_sampling.pdf"):
    """
    Generate 2D plots showing the evolution of training sample coverage over iterations.
    
    Each subplot overlays:
      - Reference credible contours (from a known multivariate Gaussian) in green.
      - A scatter plot of the aggregated training samples (projected to 2D) in blue.
    
    Parameters:
      training_sets: List of tuples, one per iteration, each containing:
          (aggregated_X, aggregated_y) where aggregated_X are the training samples in real space.
      analytic_mu: Mean vector of the reference multivariate Gaussian.
      analytic_cov: Covariance matrix of the reference distribution.
      param_names: Optional list of two parameter names for labeling the x/y axes.
      output_filename: File path to save the final figure.
    
    Note: Training sets are assumed to be in real space.
    """
    n_iterations = len(training_sets)
    
    # Generate analytic samples for the reference contours.
    analytic_samples = np.random.multivariate_normal(analytic_mu, 3*analytic_cov, size=1000000)
    analytic_samples_2d = analytic_samples[:, :2]
    
    bins = 50
    H_ref, xedges_ref, yedges_ref = np.histogram2d(analytic_samples_2d[:, 0],
                                                   analytic_samples_2d[:, 1],
                                                   bins=bins, density=True)
    X_ref = 0.5 * (xedges_ref[:-1] + xedges_ref[1:])
    Y_ref = 0.5 * (yedges_ref[:-1] + yedges_ref[1:])
    levels_ref = compute_levels(H_ref, levels=[0.68, 0.95])
    
    fig, axes = plt.subplots(1, n_iterations, figsize=(5 * n_iterations, 5), squeeze=False)
    
    if param_names is None:
        param_names = ["Param 1", "Param 2"]
    
    for i, ts in enumerate(training_sets):
        ax = axes[0, i]
        
        # In each training set tuple, ts[0] is aggregated_X (already in real space).
        aggregated_samples = ts[0]
        samples_2d = aggregated_samples[:, :2]
        
        H_net, xedges_net, yedges_net = np.histogram2d(samples_2d[:, 0],
                                                       samples_2d[:, 1],
                                                       bins=bins, density=True)
        X_net = 0.5 * (xedges_net[:-1] + xedges_net[1:])
        Y_net = 0.5 * (yedges_net[:-1] + yedges_net[1:])
        levels_net = compute_levels(H_net, levels=[0.68, 0.95])
        
        # Plot analytic (reference) contours (vector graphics).
        for lvl, ls in zip(levels_ref, ['solid', 'dashed']):
            ax.contour(X_ref, Y_ref, H_ref.T, levels=[lvl],
                       colors='green', linestyles=ls, linewidths=2, zorder=2)
        
        # Create scatter plot without rasterization flag first.
        sc = ax.scatter(samples_2d[:, 0], samples_2d[:, 1], color='blue', s=2, alpha=0.6, 
                   label='Training Samples')
        # Rasterize the scatter plot for better rendering.
        sc.set_rasterized(True)
        sc.set_zorder(0)
        
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])
        ax.set_title(f"Iteration {i+1}")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_filename)

if __name__ == '__main__':
    # Load configuration from YAML.
    config_path = os.path.join("config", "default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    N = int(config['model']['dimension'])

    training_sets_file = os.path.join("iterative_data", "training_sets.pkl")
    if os.path.exists(training_sets_file):
        with open(training_sets_file, "rb") as f:
            training_sets = pickle.load(f)
    else:
        print(f"{training_sets_file} not found. Please provide a valid training_sets file.")
        training_sets = []
    
    # Set random seed for reproducibility.
    np.random.seed(int(config['misc']['random_seed']))

    target_likelihood = GaussianLikelihood(N)
    #'''
    chains_dir = "class_chains/27_param"  # where your Planck .txt files live

    # Instantiate the Planck‐based Gaussian likelihood
    target_likelihood = PlanckGaussianLikelihood(
        N=N,
        chains_dir=chains_dir,
        skip_rows=500,            # burn-in rows to skip (optional)
        max_rows_per_file=None    # or an int if you only want a subset
    )
    #'''
    analytic_mu = target_likelihood.mu
    analytic_cov = target_likelihood.Sigma
    param_names = [f"Param {i+1}" for i in range(N)]
    
    plot_iterative_sampling(training_sets, analytic_mu, analytic_cov,
                            param_names=param_names,
                            output_filename="plots/iterative_sampling.pdf")
