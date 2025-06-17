#!/usr/bin/env python3
"""
Benchmarking Script for Evaluating a Surrogate Model against an Analytic Gaussian Likelihood,
with nested-sampling evidence (logZ ± err) via PyPolyChord.
"""

# Standard Library Imports
import os
import sys
import pickle
import yaml

# Third-Party Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp

# PolyChord Imports
from pypolychord import PolyChordSettings, run_polychord

# Internal Module Imports (update PYTHONPATH if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from getdist import MCSamples, ParamNames
from likelihoods.gaussian import GaussianLikelihood
from likelihoods.planck_gaussian import PlanckGaussianLikelihood
from keras.models import load_model
from models.activations import CustomTanh
from models.quadratic_decomposition import QuadraticDecomposition

# ------------------------------
# Utility Functions
# ------------------------------

def load_chain(chain_file):
    chain = np.loadtxt(chain_file, skiprows=1)
    return chain

def compute_getdist_stats(samples, weights, param_names, labels):
    names_obj = ParamNames(names=param_names, labels=labels)
    mc_samples = MCSamples(samples=samples, names=names_obj, weights=weights)
    stats = mc_samples.getMargeStats()
    credible_intervals = {}
    for name in param_names:
        par_info = next((p for p in stats.names if p.name == name), None)
        if par_info is None:
            raise RuntimeError(f"Parameter {name} not found in GetDist stats")
        cred68 = (par_info.limits[0].lower, par_info.limits[0].upper)
        cred95 = (par_info.limits[1].lower, par_info.limits[1].upper) if len(par_info.limits) > 1 else (None, None)
        credible_intervals[name] = {"68": cred68, "95": cred95}
    return credible_intervals

def analytic_density_grid(mu, cov, grid_points=100, range_factor=3):
    std = np.sqrt(np.diag(cov))
    x = np.linspace(mu[0] - range_factor*std[0], mu[0] + range_factor*std[0], grid_points)
    y = np.linspace(mu[1] - range_factor*std[1], mu[1] + range_factor*std[1], grid_points)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[...,0], pos[...,1] = X, Y
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    norm_const = 1.0/(2*np.pi*np.sqrt(det_cov))
    diff = pos - mu
    exponent = np.einsum('...i,ij,...j->...', diff, inv_cov, diff)
    density = norm_const * np.exp(-0.5 * exponent)
    return x, y, density

def compute_levels(H, levels=[0.68, 0.95]):
    H_flat = H.flatten()
    idx = np.argsort(H_flat)[::-1]
    H_sorted = H_flat[idx]
    cumsum = np.cumsum(H_sorted)
    cumsum /= cumsum[-1]
    contour_levels = []
    for lev in levels:
        i = np.where(cumsum >= lev)[0]
        threshold = H_sorted[i[0]] if i.size else H_sorted[-1]
        contour_levels.append(threshold)
    return sorted(contour_levels)

def evaluate_model_at_point(model, scaler_x, scaler_y, point):
    full_dim = scaler_x.mean_.shape[0]
    p = np.asarray(point)
    if p.shape[0] < full_dim:
        fp = scaler_x.mean_.copy()
        fp[:p.shape[0]] = p
    else:
        fp = p.copy()
    x_std = (fp - scaler_x.mean_) / scaler_x.scale_
    y_std = model.predict(x_std[np.newaxis,:], verbose=0)
    return (y_std * scaler_y.scale_ + scaler_y.mean_).item()

# ------------------------------
# Nested-Sampling Evidence via PolyChord
# ------------------------------

def prior_transform(u, bounds):
    theta = np.empty_like(u)
    for i, ui in enumerate(u):
        low, high = bounds[i]
        theta[i] = low + ui*(high - low)
    return theta

def compute_evidence_polychord(loglikelihood_fn, bounds, n_live=500, settings_kwargs=None):
    D = len(bounds)
    nDerived = 0
    settings = PolyChordSettings(D, nDerived)
    settings.nlive = n_live
    settings.precision_criterion = 0.001
    settings.max_ndead = 10000

    # this is the new way to set your output basename
    settings.file_root = "poly_out"

    if settings_kwargs:
        for k, v in settings_kwargs.items():
            setattr(settings, k, v)

    # call with the positional signature
    result = run_polychord(
        loglikelihood_fn,  # your callable
        D,                 # number of parameters
        nDerived,          # how many derived quantities
        settings,          # PolyChordSettings
        lambda u: prior_transform(u, bounds)
    )
    return result.logZ, result.logZerr


# ------------------------------
# TensorFlow LBFGS & Profile Likelihood
# ------------------------------

def tf_lbfgs_minimize(objective_fn, initial_position, tolerance=1e-8, max_iterations=100):
    x0 = tf.convert_to_tensor(initial_position, dtype=tf.float32)

    @tf.function
    def value_and_gradients_fn(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = objective_fn(x)
        grads = tape.gradient(loss, x)
        return loss, grads

    result = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=value_and_gradients_fn,
        initial_position=x0,
        tolerance=tolerance,
        max_iterations=max_iterations
    )
    return result.position.numpy(), result.objective_value.numpy()

def profile_likelihood_1d_tf(likelihood_fn, D, fixed_idx, grid, init_guess, bounds):
    prof = []
    for val in tqdm(grid, desc=f"1D profile idx={fixed_idx}", leave=True):
        def obj(free):
            full = []
            j = 0
            for i in range(D):
                if i == fixed_idx:
                    full.append(tf.constant(val, dtype=tf.float32))
                else:
                    full.append(free[j]); j += 1
            full = tf.stack(full)[tf.newaxis,:]
            return tf.squeeze(likelihood_fn(full))
        free0 = np.array([init_guess[i] for i in range(D) if i != fixed_idx], dtype=np.float32)
        _, loss = tf_lbfgs_minimize(obj, free0)
        prof.append(loss)
    return np.array(prof)

def profile_likelihood_2d_tf(likelihood_fn, D, idx_pair, grid_x, grid_y, init_guess, bounds):
    prof = np.zeros((len(grid_x), len(grid_y)))
    for ix, vx in enumerate(tqdm(grid_x, desc="2D profile", leave=True)):
        for iy, vy in enumerate(grid_y):
            def obj(free):
                full = []
                j = 0
                for i in range(D):
                    if i == idx_pair[0]:
                        full.append(tf.constant(vx, dtype=tf.float32))
                    elif i == idx_pair[1]:
                        full.append(tf.constant(vy, dtype=tf.float32))
                    else:
                        full.append(free[j]); j += 1
                full = tf.stack(full)[tf.newaxis,:]
                return tf.squeeze(likelihood_fn(full))
            free0 = np.array([init_guess[i] for i in range(D) if i not in idx_pair], dtype=np.float32)
            _, loss = tf_lbfgs_minimize(obj, free0)
            prof[ix,iy] = loss
    return prof

# ------------------------------
# Main Benchmark & Plotting
# ------------------------------

def plot_benchmark(chain_file, output_pdf, model_file, scalers_file,
                   iteration, sel_idx, sel_names, sel_labels,
                   compute_profile=False, include_2d=False, compute_evidence=False):

    results = []

    # Load the MCMC chain
    chain = load_chain(chain_file)
    D_full = chain.shape[1] - 2
    chain_params_all = chain[:, 2:]
    chain_params     = chain_params_all[:, sel_idx]
    weights          = chain[:, 0]

    # GetDist credible intervals on network chain
    chain_cred = compute_getdist_stats(chain_params, weights, sel_names, sel_labels)
    results.append("Network Chain Credible Intervals:")
    for p in sel_names:
        lo68, hi68 = chain_cred[p]['68']
        lo95, hi95 = chain_cred[p]['95']
        results.append(f" {p}  68%: ({lo68:.3f}, {hi68:.3f}), 95%: ({lo95:.3f}, {hi95:.3f})")

    # Best-fit from chain
    idx_best_net   = np.argmin(chain[:,1])
    net_best_val   = chain[idx_best_net,1]
    net_best_proj  = chain_params_all[idx_best_net, sel_idx]

    # Analytic Gaussian setup
    #true_like      = GaussianLikelihood(D_full)
    #'''
    chains_dir = "class_chains/27_param"  # where your Planck .txt files live

    # Instantiate the Planck‐based Gaussian likelihood
    true_like = PlanckGaussianLikelihood(
        N=27,
        chains_dir=chains_dir,
        skip_rows=500,            # burn-in rows to skip (optional)
        max_rows_per_file=None    # or an int if you only want a subset
    )
    #'''
    mu, Sigma      = true_like.mu, true_like.Sigma
    true_best_val  = true_like.neg_loglike(mu[np.newaxis,:])[0]
    true_best_proj = mu[sel_idx]

    # Analytic credible intervals via random sampling
    analytic_samples_full = np.random.multivariate_normal(mu, Sigma, size=100000)
    analytic_samples      = analytic_samples_full[:, sel_idx]
    analytic_cred = compute_getdist_stats(analytic_samples, None, sel_names, sel_labels)
    results.append("\nAnalytic True Likelihood Credible Intervals:")
    for p in sel_names:
        lo68, hi68 = analytic_cred[p]['68']
        lo95, hi95 = analytic_cred[p]['95']
        results.append(f" {p}  68%: ({lo68:.3f}, {hi68:.3f}), 95%: ({lo95:.3f}, {hi95:.3f})")

    # Summary of best fits
    results.append("\nBest-fit Values:")
    results.append(f" Analytic    log-L minimum: {true_best_val:.6f}")
    results.append(f" Network     log-L minimum: {net_best_val:.6f}")

    # Full-dimensional coordinates
    true_best_full    = mu
    network_best_full = chain_params_all[idx_best_net]

    results.append(f" Analytic full coordinate: {np.array2string(true_best_full, precision=5, separator=', ')}")
    results.append(f" Network  full coordinate: {np.array2string(network_best_full, precision=5, separator=', ')}")


    # Load surrogate & scalers
    model = load_model(model_file, custom_objects={"CustomTanh": CustomTanh,
                                                   "QuadraticDecomposition": QuadraticDecomposition})
    with open(scalers_file, "rb") as f:
        scalers = pickle.load(f)
    scaler_x, scaler_y = (scalers[iteration] if iteration < len(scalers)
                          else scalers[-1])
    net_pred_true = evaluate_model_at_point(model, scaler_x, scaler_y, mu)
    results.append(f"Network prediction @ true best-fit: {net_pred_true:.6f}")

    # Define integration bounds & initial guess
    bounds_full  = [(mu[i] - 5*np.sqrt(Sigma[i,i]), mu[i] + 5*np.sqrt(Sigma[i,i]))
                    for i in range(D_full)]
    init_full    = mu.copy()

    # Set up corner‐style figure
    bins = 50
    fig, axes = plt.subplots(2,2, figsize=(10,10))
    ax1, ax2     = axes[0,0], axes[1,1]
    ax_joint     = axes[1,0]
    axes[0,1].axis('off')

    # --- 1D histograms for param 1 ---
    ax1.hist(chain_params[:,0], bins=bins, density=True, alpha=0.6,
             label='Network')
    ax1.hist(analytic_samples[:,0], bins=bins, density=True, alpha=0.4,
             label='Analytic')
    for style, level in [('--','68'),(':','95')]:
        ax1.axvline(chain_cred[sel_names[0]][level][0], color='C0',
                    linestyle=style)
        ax1.axvline(chain_cred[sel_names[0]][level][1], color='C0',
                    linestyle=style)
        ax1.axvline(analytic_cred[sel_names[0]][level][0], color='C1',
                    linestyle=style)
        ax1.axvline(analytic_cred[sel_names[0]][level][1], color='C1',
                    linestyle=style)
    ax1.set_xlabel(sel_labels[0])
    ax1.set_ylabel("Density")
    ax1.legend(fontsize='small')

    # --- 1D histograms for param 2 ---
    ax2.hist(chain_params[:,1], bins=bins, density=True, alpha=0.6,
             label='Network')
    ax2.hist(analytic_samples[:,1], bins=bins, density=True, alpha=0.4,
             label='Analytic')
    for style, level in [('--','68'),(':','95')]:
        ax2.axvline(chain_cred[sel_names[1]][level][0], color='C0',
                    linestyle=style)
        ax2.axvline(chain_cred[sel_names[1]][level][1], color='C0',
                    linestyle=style)
        ax2.axvline(analytic_cred[sel_names[1]][level][0], color='C1',
                    linestyle=style)
        ax2.axvline(analytic_cred[sel_names[1]][level][1], color='C1',
                    linestyle=style)
    ax2.set_xlabel(sel_labels[1])
    ax2.set_ylabel("Density")
    ax2.legend(fontsize='small')

    # --- Joint 2D contours ---
    Hnet, xedges, yedges = np.histogram2d(chain_params[:,0], chain_params[:,1],
                                          bins=bins, density=True, weights=weights)
    xcent = 0.5*(xedges[:-1]+xedges[1:])
    ycent = 0.5*(yedges[:-1]+yedges[1:])
    lev_net = compute_levels(Hnet, [0.68, 0.95])
    mu_sel  = mu[sel_idx]
    cov_sel = Sigma[np.ix_(sel_idx, sel_idx)]
    xg, yg, Hanalytic = analytic_density_grid(mu_sel, cov_sel, grid_points=100)
    lev_ana = compute_levels(Hanalytic, [0.68, 0.95])
    Xn, Yn = np.meshgrid(xcent, ycent)
    Xa, Ya = np.meshgrid(xg, yg)
    ax_joint.contour(Xn, Yn, Hnet.T, levels=lev_net, colors=['C0','C0'],
                     linestyles=['solid','dashed'])
    ax_joint.contour(Xa, Ya, Hanalytic, levels=lev_ana, colors=['C1','C1'],
                     linestyles=['solid','dashed'])
    ax_joint.plot(net_best_proj[0], net_best_proj[1], 'ro', label='Net best-fit')
    ax_joint.plot(true_best_proj[0], true_best_proj[1], 'ms', label='True best-fit')
    ax_joint.set_xlabel(sel_labels[0])
    ax_joint.set_ylabel(sel_labels[1])
    ax_joint.legend(fontsize='small')

    # --- Optional profile overlays ---
    if compute_profile:
        # define analytic vs network nll in TF
        def analytic_nll_tf(x):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            delta = x - tf.constant(mu, dtype=tf.float32)
            invS  = tf.linalg.inv(tf.constant(Sigma, dtype=tf.float32))
            quad  = tf.reduce_sum(delta * tf.matmul(delta, invS), axis=1)
            ld    = tf.math.log(tf.linalg.det(tf.constant(Sigma, dtype=tf.float32)))
            N     = tf.cast(tf.shape(delta)[1], tf.float32)
            return 0.5*(N*tf.math.log(2*np.pi) + ld + quad)

        def network_nll_tf(x):
            x_std = (x - scaler_x.mean_) / scaler_x.scale_
            ystd  = model(x_std, training=False)
            y     = ystd*scaler_y.scale_ + scaler_y.mean_
            return tf.squeeze(y, axis=-1)

        prof_a_1d = {}
        prof_n_1d = {}
        grid_dict = {}
        for idx in sel_idx:
            lo, hi       = bounds_full[idx]
            grid         = np.linspace(lo, hi, 50)
            pa           = profile_likelihood_1d_tf(analytic_nll_tf, D_full,
                                                   idx, grid, init_full, bounds_full)
            pn           = profile_likelihood_1d_tf(network_nll_tf, D_full,
                                                   idx, grid, init_full, bounds_full)
            prof_a_1d[idx] = pa
            prof_n_1d[idx] = pn
            grid_dict[idx] = grid

        # overlay 1D profiles
        for i, idx in enumerate(sel_idx):
            pa_nll = prof_a_1d[idx]
            pa_like = np.exp(-(pa_nll - pa_nll.min()))
            h_ana, _ = np.histogram(analytic_samples[:,i], bins=bins, density=True)
            pa_like *= (h_ana.max()/pa_like.max())

            pn_nll  = prof_n_1d[idx]
            pn_like = np.exp(-(pn_nll - pn_nll.min()))
            h_net, _= np.histogram(chain_params[:,i], bins=bins, density=True)
            pn_like *= (h_net.max()/pn_like.max())

            ax = ax1 if i==0 else ax2
            ax.plot(grid_dict[idx], pa_like, 'k-', label='Analytic prof')
            ax.plot(grid_dict[idx], pn_like, 'r-', label='Network prof')
            ax.legend(fontsize='x-small')

        # overlay 2D if requested
        if include_2d == True:
            gx = np.linspace(bounds_full[sel_idx[0]][0], bounds_full[sel_idx[0]][1], 50)
            gy = np.linspace(bounds_full[sel_idx[1]][0], bounds_full[sel_idx[1]][1], 50)
            pa2 = profile_likelihood_2d_tf(analytic_nll_tf, D_full, sel_idx, gx, gy, init_full, bounds_full)
            pn2 = profile_likelihood_2d_tf(network_nll_tf, D_full, sel_idx, gx, gy, init_full, bounds_full)
            pa2_like = np.exp(-(pa2 - pa2.min()))
            pn2_like = np.exp(-(pn2 - pn2.min()))
            scale2    = Hanalytic.max()/pa2_like.max()
            pa2_like *= scale2; pn2_like *= scale2
            X2, Y2 = np.meshgrid(gx, gy)
            cs_a = ax_joint.contour(X2, Y2, pa2_like.T, levels=5,
                                    colors='black', linestyles='solid')
            cs_n = ax_joint.contour(X2, Y2, pn2_like.T, levels=5,
                                    colors='red',   linestyles='dashed')
            proxy_a = mlines.Line2D([],[],color='black',linestyle='solid',
                                    label='Ana prof 2D')
            proxy_n = mlines.Line2D([],[],color='red',  linestyle='dashed',
                                    label='Net prof 2D')
            ax_joint.legend(handles=[proxy_a, proxy_n], fontsize='x-small')

    # --- PolyChord evidence ---
    def analytic_logL(theta):
        delta   = theta - mu
        invS    = np.linalg.inv(Sigma)
        exponent= delta @ invS @ delta
        logdet  = np.log(np.linalg.det(Sigma))
        N       = len(theta)
        return -0.5*(N*np.log(2*np.pi) + logdet + exponent)
    
    @tf.function
    def fast_network_logL(theta: tf.Tensor) -> tf.Tensor:
        # theta: shape (D,)
        x_std = (theta - tf.constant(scaler_x.mean_, dtype=tf.float32)) \
                / tf.constant(scaler_x.scale_, dtype=tf.float32)
        y_std = model(x_std[tf.newaxis, :], training=False)
        y     = y_std * tf.constant(scaler_y.scale_, dtype=tf.float32) \
                + tf.constant(scaler_y.mean_, dtype=tf.float32)
        # return negative log-likelihood
        return -y[0]

    def network_logL(theta):
        # theta: numpy array of shape (D,)
        t = tf.constant(theta, dtype=tf.float32)
        return float(fast_network_logL(t))

    # Optional: Compute PolyChord evidence
    if compute_evidence == True:
        # analytic
        logZ_a, err_a = compute_evidence_polychord(
            analytic_logL,
            bounds_full,
            n_live=675,
            settings_kwargs={"file_root": "poly_analytic"}
        )

        # network
        logZ_n, err_n = compute_evidence_polychord(
            network_logL,
            bounds_full,
            n_live=675,
            settings_kwargs={"file_root": "poly_network"}
        )

        results.append("\nNested-Sampling Evidence via PolyChord:")
        results.append(f" Analytic logZ = {logZ_a:.5f} ± {err_a:.5f}")
        results.append(f" Network  logZ = {logZ_n:.5f} ± {err_n:.5f}")

    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf')
    results.append(f"\nSaved corner/profile plot to {output_pdf}")

    plt.show()
    return "\n".join(results)

def main():
    # Load config
    cfg = yaml.safe_load(open(os.path.join("config","default.yaml")))
    np.random.seed(int(cfg['misc']['random_seed']))

    iter_val    = 1
    chain_file  = f"mcmc/chains/chain_iter{iter_val}.txt"
    model_file  = f"trained_models/model_iter{iter_val}.h5"
    scalers_file= "iterative_data/scalers.pkl"
    output_pdf  = "plots/benchmark_corner.pdf"

    sel_idx     = [0,1]
    sel_names   = [f"param{i+1}" for i in sel_idx]
    sel_labels  = [f"Param {i+1}" for i in sel_idx]

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    bench_text = plot_benchmark(chain_file, output_pdf,
                                model_file, scalers_file,
                                iter_val, sel_idx,
                                sel_names, sel_labels,
                                compute_profile=False,
                                include_2d=False,
                                compute_evidence=False)

    with open("plots/benchmark.out","w") as f:
        f.write(bench_text)

    print(bench_text)

if __name__ == "__main__":
    main()
