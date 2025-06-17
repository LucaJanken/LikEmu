#!/usr/bin/env python3
"""
test_planck_corner.py

1. Run adaptive Metropolis–Hastings on the analytic PlanckGaussianLikelihood.
2. Load the compressed chain + best‐fit.
3. Draw a full D×D corner plot:
     - Diagonals: 1D histograms (posterior vs analytic).
     - Lower triangle: 2D 68%/95% contours.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcmc.sampler import adaptive_metropolis_hastings
from likelihoods.planck_gaussian import PlanckGaussianLikelihood

def compute_levels(H, levels=[0.68, 0.95]):
    """Compute histogram‐value thresholds for given cumulative-prob levels."""
    flat = H.flatten()
    idx = np.argsort(flat)[::-1]
    sorted_vals = flat[idx]
    cumsum = np.cumsum(sorted_vals)
    cumsum /= cumsum[-1]
    thresh = []
    for lev in levels:
        i = np.searchsorted(cumsum, lev)
        thresh.append(sorted_vals[i])
    return sorted(thresh)

class AnalyticEmulator(tf.keras.Model):
    """Wrap PlanckGaussianLikelihood.neg_loglike as a Keras Model."""
    def __init__(self, like):
        super().__init__()
        self.like = like

    def call(self, x, training=False):
        # get float64 then cast
        neg64 = tf.numpy_function(
            func=lambda xx: self.like.neg_loglike(xx),
            inp=[x],
            Tout=tf.float64
        )
        neg32 = tf.cast(neg64, tf.float32)
        return tf.expand_dims(neg32, axis=1)

def corner_plot(params, weights, analytic_samples, bestfit, names, out_pdf):
    D = params.shape[1]
    fig, axes = plt.subplots(D, D, figsize=(2.5*D, 2.5*D), squeeze=False)

    # compute limits
    all_data = np.vstack([params, analytic_samples])
    limits = []
    for d in range(D):
        lo, hi = np.percentile(all_data[:,d], [1,99])
        pad = 0.2*(hi-lo)
        limits.append((lo-pad, hi+pad))

    for i in range(D):
        for j in range(D):
            ax = axes[i,j]
            # hide upper triangle
            if i < j:
                ax.axis('off')
                continue

            # diagonal: 1D hist
            if i == j:
                ax.hist(analytic_samples[:,i], bins=40, density=True,
                        alpha=0.4, color='green', label='Analytic')
                ax.hist(params[:,i], bins=40, density=True, weights=weights,
                        alpha=0.4, color='blue', label='MCMC')
                ax.axvline(bestfit[i], color='red', lw=2)
                ax.set_xlim(limits[i])
                if j==0: ax.legend(fontsize='small')
                if i==D-1: ax.set_xlabel(names[i])
                if j==0: ax.set_ylabel("Density")

            # lower triangle: 2D contours
            if i > j:
                xlo,xhi = limits[j]
                ylo,yhi = limits[i]
                Hm, xe, ye = np.histogram2d(
                    params[:,j], params[:,i],
                    bins=40, range=[[xlo,xhi],[ylo,yhi]],
                    density=True, weights=weights
                )
                Ha, _, _ = np.histogram2d(
                    analytic_samples[:,j], analytic_samples[:,i],
                    bins=40, range=[[xlo,xhi],[ylo,yhi]],
                    density=True
                )
                xc = 0.5*(xe[:-1]+xe[1:])
                yc = 0.5*(ye[:-1]+ye[1:])
                Xc, Yc = np.meshgrid(xc, yc)
                lv_m = compute_levels(Hm)
                lv_a = compute_levels(Ha)
                ax.contour(Xc, Yc, Hm.T, levels=lv_m, colors='blue')
                ax.contour(Xc, Yc, Ha.T, levels=lv_a, colors='green', linestyles='dashed')
                ax.plot(bestfit[j], bestfit[i], 'ro')
                ax.set_xlim(xlo,xhi)
                ax.set_ylim(ylo,yhi)
                if i==D-1: ax.set_xlabel(names[j])
                if j==0: ax.set_ylabel(names[i])

    plt.tight_layout()
    plt.suptitle("Corner Plot: MCMC (blue) vs Analytic (green)", y=1.02, fontsize=16)
    fig.savefig(out_pdf, dpi=200, bbox_inches='tight')
    print(f"Saved corner plot to {out_pdf}")

def main():
    # load config
    cfg = yaml.safe_load(open("config/default.yaml"))
    N = int(cfg['model']['dimension'])
    mcmc_cfg = cfg['mcmc']

    # instantiate likelihood
    target = PlanckGaussianLikelihood(
        N=N,
        chains_dir="class_chains/27_param",
        skip_rows=500,
        max_rows_per_file=None
    )
    mu, Sigma = target.mu, target.Sigma

    # build fake network + identity scalers
    model = AnalyticEmulator(target)
    sx = StandardScaler(); sx.mean_=np.zeros(N); sx.scale_=np.ones(N)
    sy = StandardScaler(); sy.mean_=np.zeros(1); sy.scale_=np.ones(1)

    # run MCMC
    np.random.seed(int(cfg['misc']['random_seed']))
    tf.random.set_seed(int(cfg['misc']['random_seed']))
    start = np.random.multivariate_normal(mu, Sigma)
    chains_dir = cfg['paths']['mcmc_chains']
    os.makedirs(chains_dir, exist_ok=True)
    chfile = os.path.join(chains_dir, "chain_planck_corner.txt")
    bffile = os.path.join(cfg['paths']['mcmc_bestfit'], "best_planck_corner.txt")
    os.makedirs(cfg['paths']['mcmc_bestfit'], exist_ok=True)

    adaptive_metropolis_hastings(
        keras_model=model,
        initial_point=start,
        n_steps=1_000_000,
        temperature=1.0,
        base_cov=Sigma,
        scaler_x=sx,
        scaler_y=sy,
        adapt_interval=int(mcmc_cfg['adapt_interval']),
        cov_update_interval=int(mcmc_cfg['cov_update_interval']),
        chain_file=chfile,
        bestfit_file=bffile,
        epsilon=float(mcmc_cfg['epsilon']),
        burn_in_fraction=0.3
    )

    # load chain + best-fit
    chain = np.loadtxt(chfile, skiprows=1)
    weights = chain[:,0]
    samples = chain[:,2:]
    bestfit = np.loadtxt(bffile).reshape(-1)

    # analytic draws for contour/hist
    M = 1000000
    analytic = np.random.multivariate_normal(mu, Sigma, size=M)

    names = [f"θ{i+1}" for i in range(N)]
    os.makedirs("plots", exist_ok=True)
    corner_plot(samples, weights, analytic, bestfit, names, out_pdf="plots/planck_corner.pdf")

if __name__ == "__main__":
    main()
