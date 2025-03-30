#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import yaml
from likelihoods.gaussian import GaussianLikelihood
from plotting.plot_utils import compute_levels

def plot_iterative_chains(chain_data, analytic_mu, analytic_cov,
                          param_names=None, output_filename="plots/iterative_chain_plots.pdf"):
    """
    Generate 2D credible interval plots for the MCMC chains of each iteration.

    Parameters:
      chain_data: List of tuples, one per iteration. Each tuple is (samples, weights) where:
                   - samples: array of shape (N_samples, D) containing the MCMC parameter samples.
                   - weights: array of shape (N_samples,) with the multiplicity weights.
      analytic_mu: Mean vector of the reference multivariate Gaussian.
      analytic_cov: Covariance matrix of the reference distribution.
      param_names: Optional list of two parameter names for labeling the x/y axes.
      output_filename: File path to save the final figure (PDF).
    """
    n_iterations = len(chain_data)
    
    # Generate analytic samples for the reference contours.
    analytic_samples = np.random.multivariate_normal(analytic_mu, analytic_cov, size=100000)
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
    
    for i, (samples, weights) in enumerate(chain_data):
        ax = axes[0, i]
        
        # Extract the first two dimensions.
        chain_2d = samples[:, :2]
        
        # Compute weighted 2D histogram.
        H_net, xedges_net, yedges_net = np.histogram2d(chain_2d[:, 0],
                                                       chain_2d[:, 1],
                                                       bins=bins, density=True, weights=weights)
        X_net = 0.5 * (xedges_net[:-1] + xedges_net[1:])
        Y_net = 0.5 * (yedges_net[:-1] + yedges_net[1:])
        levels_net = compute_levels(H_net, levels=[0.68, 0.95])
        
        # Plot analytic (reference) contours: green (68% solid, 95% dashed)
        for lvl, ls in zip(levels_ref, ['solid', 'dashed']):
            ax.contour(X_ref, Y_ref, H_ref.T, levels=[lvl], colors='green', linestyles=ls, linewidths=2)
        
        # Plot chain contours: blue (68% solid, 95% dashed)
        for lvl, ls in zip(levels_net, ['solid', 'dashed']):
            ax.contour(X_net, Y_net, H_net.T, levels=[lvl], colors='blue', linestyles=ls)
        
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])
        ax.set_title(f"Iteration {i+1}")
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi = 300)
    plt.show()

if __name__ == '__main__':
    # Load configuration from YAML.
    config_path = os.path.join("config", "default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    N = int(config['model']['dimension'])
    n_iterations = int(config['iterative']['n_iterations'])
    
    # Set random seed for reproducibility.
    np.random.seed(int(config['misc']['random_seed']))
    
    # Load MCMC chain samples from the chain files.
    # Each chain file is expected to have columns:
    # [multiplicity, -log(L), param_1, param_2, ..., param_D]
    chain_data = []
    for it in range(1, n_iterations + 1):
        chain_file = os.path.join("mcmc", "chains", f"chain_iter{it}.txt")
        if os.path.exists(chain_file):
            chain = np.loadtxt(chain_file, skiprows=1)
            weights = chain[:, 0]      # Extract weights.
            samples = chain[:, 2:]     # Extract parameter samples.
            chain_data.append((samples, weights))
        else:
            print(f"{chain_file} not found.")
    
    target_likelihood = GaussianLikelihood(N)
    analytic_mu = np.zeros(N)
    analytic_cov = target_likelihood.Sigma
    param_names = [f"Param {i+1}" for i in range(N)]
    
    plot_iterative_chains(chain_data, analytic_mu, analytic_cov,
                          param_names=param_names,
                          output_filename="plots/iterative_chain_plots.pdf")
