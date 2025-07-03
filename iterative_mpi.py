#!/usr/bin/env python3
"""
MPI-parallelized iterative training + MCMC sampler in one file.
Now tracks chain IDs so that samples can be color‐coded by chain.
Rank 0 handles data aggregation, surrogate training, resampling.
All ranks run one MCMC chain each, with a distributed Gelman–Rubin check.
"""
import os
import sys
import pickle
import yaml

try:
    from pypolychord import PolyChordSettings, run_polychord
except Exception:  # pragma: no cover - library may be missing
    PolyChordSettings = None
    run_polychord = None
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from likelihoods.planck_gaussian import PlanckGaussianLikelihood
from likelihoods.planck_gaussian_extended import ExtendedPlanckGaussianLikelihood
from data.data_generator import generate_data
from models.emulation_model import build_emulation_model, Alsing
from training.train import train_model
from mcmc.sampler import adaptive_metropolis_hastings_mpi

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def resample_chain_points(chain, new_sample_size, temperature, strategy="flat"):
    """
    chain: ndarray with columns [count, logl, param0, ..., paramD-1, chain_id]
    returns: (samples[new_sample_size, D], chain_ids[new_sample_size])
    """
    if isinstance(chain, str):
        chain = np.loadtxt(chain, skiprows=0)
    mult = chain[:, 0]
    logl = chain[:, 1]
    samples = chain[:, 2:-1]
    cids = chain[:, -1].astype(int)
    n_avail = len(samples)
    size = min(new_sample_size, n_avail)

    if strategy == "flat":
        lw = np.log(mult) + logl / temperature
        lw -= lw.max()
        w = np.exp(lw)
        w /= w.sum()
        idx = np.random.choice(n_avail, size=size, p=w, replace=False)
    elif strategy == "random":
        idx = np.random.choice(n_avail, size=size, replace=False)
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    return samples[idx], cids[idx]

def retention_probability(ages, sample_iters, current_iter,
                          base_probability, decay_rate,
                          min_probability, control_coef):
    eff = decay_rate * (1 - control_coef * (sample_iters / current_iter))
    probs = np.where(
        ages <= 1,
        base_probability,
        np.maximum(min_probability, base_probability - eff * (ages - 1))
    )
    return probs

def compute_temperature(it, n_iter, T0, TF, schedule="linear"):
    frac = (it - 1) / (n_iter - 1)
    if schedule == "linear":
        return T0 + (TF - T0) * frac
    elif schedule == "exponential":
        return T0 * (TF / T0) ** frac
    else:
        raise ValueError(f"Unknown schedule {schedule}")

# -----------------------------------------------------------------------------
# MPI setup
# -----------------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -----------------------------------------------------------------------------
# Main MPI workflow
# -----------------------------------------------------------------------------
with open("config/default.yaml") as f:
    cfg = yaml.safe_load(f)

# Unpack config
n_iter = int(cfg['iterative']['n_iterations'])
rb = float(cfg['iterative']['retention']['base_probability'])
rd = float(cfg['iterative']['retention']['decay_rate'])
rmin = float(cfg['iterative']['retention']['min_probability'])
cc = float(cfg['iterative']['retention']['control_coef'])

scaler_t = cfg['data']['scaler']
n0 = int(cfg['data']['initial_sample_size'])
inc = int(cfg['data']['sample_size_increment'])
init_str = cfg['data']['initial_sampling_strategy']
it_str = cfg['data']['iterative_sampling_strategy']
off = float(cfg['data']['offset_scale'])
nstd = float(cfg['data']['nstd'])
covs = float(cfg['data']['cov_scale'])

sampler_method = cfg.get('sampling', {}).get('method', 'mcmc')

if sampler_method == 'mcmc':
    mc_cfg = cfg['sampling']['mcmc']
    nc = int(mc_cfg['n_chains'])
    r0 = float(mc_cfg['r_start'])
    r1 = float(mc_cfg['r_stop'])
    rc = float(mc_cfg['r_converged'])
    T0 = float(mc_cfg['initial_temperature'])
    TF = float(mc_cfg['final_temperature'])
    sched = mc_cfg['annealing_schedule']
    ai = int(mc_cfg['adapt_interval'])
    ci = int(mc_cfg['cov_update_interval'])
    eps = float(mc_cfg['epsilon'])
    max_steps = int(mc_cfg['max_steps'])
else:
    pc_cfg = cfg['sampling']['polychord']
    nc = int(pc_cfg.get('nprocs', size))
    pc_nlive = int(pc_cfg.get('nlive', 500))
    pc_num_repeats = int(pc_cfg.get('num_repeats', 1))
    pc_prec = float(pc_cfg.get('precision_criterion', 0.01))
    pc_feedback = int(pc_cfg.get('feedback', 1))
    pc_out = pc_cfg.get('output_dir', 'polychord_output')
    pc_mpi = bool(pc_cfg.get('mpi', True))
    T0 = float(cfg['sampling']['mcmc']['initial_temperature'])
    TF = float(cfg['sampling']['mcmc']['final_temperature'])
    sched = cfg['sampling']['mcmc']['annealing_schedule']
    ai = ci = eps = max_steps = r0 = r1 = rc = None

paths = cfg['paths']
dirs = {
    'trained_models': paths['trained_models'],
    'iterative_data': paths['iterative_data'],
    'mcmc_chains': paths['mcmc_chains'],
    'mcmc_bestfit': paths['mcmc_bestfit'],
}

# Check MPI size
if size != nc:
    if rank == 0:
        raise RuntimeError(f"MPI size {size} != required processes {nc}")
    sys.exit(1)

# Seeds per rank
dt = int(cfg['misc']['random_seed'])
np.random.seed(dt + rank)
tf.random.set_seed(dt + rank)

# Likelihood selection
liketype = cfg['likelihood']['type']
if liketype == 'planck_gaussian':
    LikelihoodClass = PlanckGaussianLikelihood
    total_dim = int(cfg['model']['dimension'])
    ctor_args = dict(
        N=total_dim,
        chains_dir=cfg['likelihood'].get('chains_dir', "class_chains/27_param"),
        skip_rows=cfg['likelihood'].get('skip_rows', 500)
    )
elif liketype == 'planck_gaussian_extended':
    LikelihoodClass = ExtendedPlanckGaussianLikelihood
    total_dim = int(cfg['model']['dimension'])
    base_dim = total_dim - 2
    ctor_args = dict(
        N_main=base_dim,
        chains_dir=cfg['likelihood'].get('chains_dir', "class_chains/27_param"),
        skip_rows=cfg['likelihood'].get('skip_rows', 500)
    )
else:
    raise ValueError(f"Unknown likelihood type {liketype!r}")

tl = LikelihoodClass(**ctor_args)
N = tl.N

# Base covariance
if sampler_method == 'mcmc':
    bcs = cfg['sampling']['mcmc']['base_cov']
    if bcs == 'target':
        base_cov = tl.Sigma
    elif bcs == 'identity':
        base_cov = np.eye(N)
    elif bcs == 'empirical':
        if rank == 0:
            X0s, _, sx0, _ = generate_data(
                tl, num_samples=n0,
                sampling_strategy=init_str,
                offset_scale=off, nstd=nstd, cov_scale=covs
            )
            X0 = sx0.inverse_transform(X0s)
            local = np.cov(X0, rowvar=False)
        else:
            local = None
        base_cov = comm.bcast(local, root=0)
    else:
        if rank == 0:
            raise ValueError(f"Unknown base_cov {bcs}")
        sys.exit(1)
else:
    base_cov = None

# Initial data on rank 0, now with chain_ids
if rank == 0:
    X0s, _, sx, sy = generate_data(
        tl, num_samples=n0,
        scaler=scaler_t,
        sampling_strategy=init_str,
        offset_scale=off, nstd=nstd, cov_scale=covs
    )
    aggX = sx.inverse_transform(X0s)
    aggy = tl.neg_loglike(aggX)
    iters = np.zeros(len(aggy), dtype=int)
    chain_ids = np.full(len(aggy), -1, dtype=int)   # -1 = initial samples
    training_sets = [(aggX.copy(), aggy.copy(), chain_ids.copy())]
    scalers = [(sx, sy)]
else:
    aggX = aggy = iters = chain_ids = None
    training_sets = scalers = None

# Broadcast prior bounds
if rank == 0:
    lb = tl.prior_lower
    ub = tl.prior_upper
else:
    lb = ub = None
lb = comm.bcast(lb, root=0)
ub = comm.bcast(ub, root=0)
tl.prior_lower = lb
tl.prior_upper = ub

# Iterative loop
for it in range(1, n_iter + 1):
    model_file = os.path.join(dirs['trained_models'], f"model_iter{it}.h5")

    if rank == 0:
        print(f"Iteration {it}")
        # prune old samples
        ages = it - iters
        keep = np.random.rand(len(ages)) < retention_probability(
            ages, iters, it, rb, rd, rmin, cc
        )
        aggX, aggy, iters, chain_ids = (
            aggX[keep], aggy[keep], iters[keep], chain_ids[keep]
        )

        # refit scalers
        if scaler_t == 'minmax':
            sx = MinMaxScaler().fit(aggX)
            sy = MinMaxScaler().fit(aggy.reshape(-1,1))
        else:
            sx = StandardScaler().fit(aggX)
            sy = StandardScaler().fit(aggy.reshape(-1,1))
        scalers.append((sx, sy))

        # train surrogate
        Xs = sx.transform(aggX)
        ys = sy.transform(aggy.reshape(-1,1))
        model = build_emulation_model(
            N,
            num_hidden_layers=int(cfg['model']['num_hidden_layers']),
            neurons=int(cfg['model']['neurons']),
            activation=cfg['model']['activation'],
            use_gaussian_decomposition=cfg['model']['use_gaussian_decomposition'],
            learning_rate=float(cfg['model']['learning_rate'])
        )
        train_model(
            model, Xs, ys,
            epochs=int(cfg['training']['epochs']),
            batch_size=int(cfg['training']['batch_size']),
            val_split=float(cfg['training']['val_split']),
            patience=int(cfg['training']['patience'])
        )

        os.makedirs(dirs['trained_models'], exist_ok=True)
        model.save(model_file)

        # save scalers
        os.makedirs(dirs['iterative_data'], exist_ok=True)
        with open(os.path.join(dirs['iterative_data'], "scalers.pkl"), 'wb') as f:
            pickle.dump(scalers, f)

    comm.Barrier()

    # load model & scalers on all ranks
    model = tf.keras.models.load_model(
        model_file,
        custom_objects={'Alsing': Alsing}
    )
    if rank != 0:
        with open(os.path.join(dirs['iterative_data'], "scalers.pkl"), 'rb') as f:
            scalers = pickle.load(f)
    sx, sy = scalers[it]

    # attach stats & likelihood
    model.sxm, model.sxs = sx.mean_, sx.scale_
    model.sym, model.sys = sy.mean_[0], sy.scale_[0]
    model.likelihood = tl

    temp = compute_temperature(it, n_iter, T0, TF, sched)

    if sampler_method == 'mcmc':
        # broadcast initial points
        if rank == 0:
            pts = np.random.uniform(lb, ub, size=(nc, N))
        else:
            pts = None
        pts = comm.bcast(pts, root=0)

        local_chain, local_cov = adaptive_metropolis_hastings_mpi(
            keras_model=model,
            init_pt=pts[rank].reshape(1, -1),
            temperature=temp,
            base_cov=base_cov,
            scaler_x=sx,
            scaler_y=sy,
            adapt_interval=ai,
            cov_update_interval=ci,
            epsilon=eps,
            r_start=r0,
            r_stop=r1,
            r_converged=rc,
            max_steps=max_steps,
            rank=rank
        )

        all_ch = comm.gather(local_chain, root=0)
        all_cov = comm.gather(local_cov, root=0)

        if rank == 0:
            labeled = []
            for r, ch in enumerate(all_ch):
                cid_col = np.full((ch.shape[0], 1), r, dtype=int)
                labeled.append(np.hstack((ch, cid_col)))
            combined = np.vstack(labeled)

            os.makedirs(dirs['mcmc_chains'], exist_ok=True)
            np.savetxt(
                os.path.join(dirs['mcmc_chains'], f"chain_iter{it}.txt"),
                combined
            )

            new_n = n0 + (it - 1) * inc
            newX, new_chain_ids = resample_chain_points(
                combined, new_n, temp, strategy=it_str
            )
            newY = tl.neg_loglike(newX)
        else:
            newX = newY = new_chain_ids = None
    else:
        if PolyChordSettings is None or run_polychord is None:
            raise ImportError('pypolychord is required for PolyChord sampling')

        settings = PolyChordSettings(N, 0)
        settings.nlive = pc_nlive
        settings.num_repeats = pc_num_repeats
        settings.precision_criterion = pc_prec
        settings.feedback = pc_feedback
        settings.base_dir = pc_out
        settings.file_root = f"iter{it}"

        if rank == 0:
            os.makedirs(settings.base_dir, exist_ok=True)

        def loglike_fn(theta):
            x = np.asarray(theta)
            xstd = (x - model.sxm) / model.sxs
            ystd = model(xstd.reshape(1, -1), training=False)
            nll = ystd.numpy().item() * model.sys + model.sym
            return -nll, []

        run_polychord(
            loglike_fn,
            N,
            0,
            settings,
            lambda u: lb + (ub - lb) * np.asarray(u)
        )

        if rank == 0:
            chain_file = os.path.join(settings.base_dir, settings.file_root + '.txt')
            pc_chain = np.loadtxt(chain_file)
            weight = pc_chain[:, 0]
            neg_ll = 0.5 * pc_chain[:, 1]
            samples = pc_chain[:, 2:]
            cid = np.zeros(len(samples), dtype=int)
            combined = np.hstack([
                weight.reshape(-1, 1),
                neg_ll.reshape(-1, 1),
                samples,
                cid.reshape(-1, 1)
            ])

            os.makedirs(dirs['mcmc_chains'], exist_ok=True)
            np.savetxt(
                os.path.join(dirs['mcmc_chains'], f"chain_iter{it}.txt"),
                combined
            )

            new_n = n0 + (it - 1) * inc
            newX, new_chain_ids = resample_chain_points(
                combined, new_n, temp, strategy=it_str
            )
            newY = tl.neg_loglike(newX)
        else:
            newX = newY = new_chain_ids = None

    # broadcast new draws
    newX = comm.bcast(newX, root=0)
    newY = comm.bcast(newY, root=0)
    new_chain_ids = comm.bcast(new_chain_ids, root=0)

    if rank == 0:
        # append new
        aggX = np.vstack([aggX, newX])
        aggy = np.concatenate([aggy, newY])
        chain_ids = np.concatenate([chain_ids, new_chain_ids])
        iters = np.concatenate([iters, np.full(len(newY), it, dtype=int)])

        training_sets.append((aggX.copy(), aggy.copy(), chain_ids.copy()))
        with open(os.path.join(dirs['iterative_data'], "training_sets.pkl"), 'wb') as f:
            pickle.dump(training_sets, f)

        if sampler_method == 'mcmc':
            base_cov = sum(all_cov) / len(all_cov)
    if sampler_method == 'mcmc':
        base_cov = comm.bcast(base_cov, root=0)

if rank == 0:
    print("All iterations complete.")
