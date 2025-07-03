import sys
import os

# Update the Python path to include project directories.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt
import configparser
from likelihoods.gaussian import GaussianLikelihood
from likelihoods.planck_gaussian import PlanckGaussianLikelihood
from likelihoods.planck_gaussian_extended import ExtendedPlanckGaussianLikelihood

def compute_levels(H, levels=[0.68, 0.95]):
    """
    Compute density contour levels corresponding to the given cumulative probability levels.
    
    Parameters:
      H: 2D histogram array (density).
      levels: List of cumulative probability thresholds.
    
    Returns:
      A list with histogram value thresholds that enclose the given probability mass,
      sorted in increasing order.
    """
    H_flat = H.flatten()
    idx = np.argsort(H_flat)[::-1]
    H_sorted = H_flat[idx]
    cumsum = np.cumsum(H_sorted)
    cumsum /= cumsum[-1]  # Normalize cumulative sum.
    contour_levels = []
    for lev in levels:
        try:
            threshold = H_sorted[np.where(cumsum >= lev)[0][0]]
        except IndexError:
            threshold = H_sorted[-1]
        contour_levels.append(threshold)
    # Sort levels in increasing order as required by contour()
    return sorted(contour_levels)

def corner_plot(network_params, analytic_samples, bestfit, param_names, network_weights=None, pdf_filename="corner_plot.pdf"):
    """
    Create a corner plot comparing the MCMC (network) samples and analytic samples.
    
    Parameters:
      network_params: Array of shape (N, D) containing MCMC samples.
      analytic_samples: Array of shape (M, D) sampled from an analytic Gaussian.
      bestfit: 1D array of shape (D,) with the best-fit parameter values.
      param_names: List of D parameter names.
      network_weights: Optional array of shape (N,) with multiplicity weights.
      pdf_filename: Filename to save the PDF plot.
    """
    D = network_params.shape[1]
    fig, axes = plt.subplots(D, D, figsize=(3*D, 3*D), squeeze=False)
    
    # Compute axis limits for each parameter using percentiles.
    combined = np.vstack([network_params, analytic_samples])
    limits = []
    for d in range(D):
        lo, hi = np.percentile(combined[:, d], [2.5, 97.5])
        pad = 0.5 * (hi - lo)
        limits.append((lo - pad, hi + pad))
    
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            # Upper triangle: hide.
            if i < j:
                ax.set_visible(False)
                continue
                
            # Diagonal panels: 1D histograms.
            if i == j:
                # Plot weighted network histogram.
                ax.hist(network_params[:, i], bins=30, density=True, alpha=0.5,
                        color='blue', weights=network_weights, label='Network')
                # Plot analytic histogram.
                ax.hist(analytic_samples[:, i], bins=30, density=True, alpha=0.5,
                        color='green', label='Analytic')
                # Vertical line for best-fit.
                ax.axvline(bestfit[i], color='red', linestyle='-', lw=2, label='Best-fit')
                ax.set_xlim(limits[i])
                # Only label the x-axis on the bottom row.
                if i < D - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(param_names[i])
                # Only label the y-axis on the left column.
                if j > 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel("Density")
                # Add legend on the first diagonal.
                if i == 0:
                    ax.legend(fontsize='small')
                    
            # Lower triangle: 2D contours.
            if i > j:
                x_min, x_max = limits[j]
                y_min, y_max = limits[i]
                
                # Compute 2D histogram for network samples with weights.
                H_net, xedges, yedges = np.histogram2d(
                    network_params[:, j], network_params[:, i],
                    bins=50, range=[[x_min, x_max], [y_min, y_max]], density=True, 
                    weights=network_weights)
                # Compute bin centers from the edges.
                xcenters = 0.5 * (xedges[:-1] + xedges[1:])
                ycenters = 0.5 * (yedges[:-1] + yedges[1:])
                xx, yy = np.meshgrid(xcenters, ycenters)
                levels_net = compute_levels(H_net, levels=[0.68, 0.95])
                
                # Compute 2D histogram for analytic samples.
                H_an, xedges_an, yedges_an = np.histogram2d(
                    analytic_samples[:, j], analytic_samples[:, i],
                    bins=50, range=[[x_min, x_max], [y_min, y_max]], density=True)
                # Compute bin centers (should match xcenters, ycenters if range and bins are same).
                xcenters_an = 0.5 * (xedges_an[:-1] + xedges_an[1:])
                ycenters_an = 0.5 * (yedges_an[:-1] + yedges_an[1:])
                levels_an = compute_levels(H_an, levels=[0.68, 0.95])
                
                # Plot contours: solid for network, dashed for analytic.
                cs_net = ax.contour(xx, yy, H_net.T, levels=levels_net, colors='blue', linestyles='solid')
                cs_an = ax.contour(xx, yy, H_an.T, levels=levels_an, colors='green', linestyles='dashed')
                # Mark best-fit with a red dot.
                ax.plot(bestfit[j], bestfit[i], 'ro')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                # Only label the x-axis on the bottom row.
                if i < D - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(param_names[j])
                # Only label the y-axis on the left column.
                if j > 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel(param_names[i])
    
    plt.suptitle("Corner Plot: 68% and 95% Credible Regions", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plots/{pdf_filename}")
    plt.show()

if __name__ == "__main__":
    # Load configuration from YAML.
    config_path = os.path.join("config", "default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set random seed for reproducibility.
    np.random.seed(int(config['misc']['random_seed']))
    tf.random.set_seed(int(config['misc']['random_seed']))
    
    # Get the number of parameters (dimension).
    N = int(config['model']['dimension'])
    
    # Load the compressed chain and best-fit file.
    iter = 1
    chain_file = f"mcmc/chains/chain_iter{iter}.txt"
    bestfit_file = f"mcmc/bestfit/best_iter{iter}.txt"
    
    # The chain file is expected to have a header and rows of the form:
    # [multiplicity, predicted_loglike, param_1, param_2, ..., param_D]
    chain = np.loadtxt(chain_file, skiprows=1)
    # Extract weights (multiplicity) and parameter samples.
    network_weights = chain[:, 0]
    network_params = chain[:, 2:]  # shape (N_chain, D)
    
    # Load best-fit values.
    bestfit = np.loadtxt(bestfit_file).flatten()
    
    # Create the target likelihood to sample from the analytic Gaussian.
    # (Assumes that the likelihood's parameters are the same as during training.)
    #target_likelihood = GaussianLikelihood(N)

    chains_dir = "class_chains/27_param"  # where your Planck .txt files live

    # Instantiate the Planck‐based Gaussian likelihood
    target_likelihood = PlanckGaussianLikelihood(
        N=N,
        chains_dir=chains_dir,
        skip_rows=500,            # burn-in rows to skip (optional)
        max_rows_per_file=None    # or an int if you only want a subset
    )
    
    # Sample from the analytic multivariate Gaussian.
    M_analytic = 1000000
    analytic_samples = np.random.multivariate_normal(target_likelihood.mu, target_likelihood.Sigma * 3, size=M_analytic)
    
    # Parameter names (either from config or generic).
    param_names = [f"Param {i+1}" for i in range(N)]
    
    # Generate the corner plot.
    corner_plot(network_params, analytic_samples, bestfit, param_names,
                network_weights=network_weights, pdf_filename="corner_plot.pdf")
