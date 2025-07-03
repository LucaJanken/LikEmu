#!/usr/bin/env python3
"""
MPI-parallelized MCMC sampler using an analytical surrogate model.
Rank 0 handles data aggregation and resampling.
All ranks run one MCMC chain each, with a distributed Gelman–Rubin check.
"""
import os
import sys
import pickle
import yaml
import numpy as np
from mpi4py import MPI
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from likelihoods.planck_gaussian import PlanckGaussianLikelihood
from likelihoods.planck_gaussian_extended import ExtendedPlanckGaussianLikelihood
from data.data_generator import generate_data

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def resample_chain_points(chain, new_sample_size, temperature, strategy="flat"):
    if isinstance(chain, str):
        chain = np.loadtxt(chain, skiprows=0)
    samples = chain[:, 2:]
    n_avail = len(samples)
    size = min(new_sample_size, n_avail)
    if strategy == "flat":
        mult, logl = chain[:, 0], chain[:, 1]
        lw = np.log(mult) + logl / temperature
        lw -= lw.max()
        w = np.exp(lw)
        w /= w.sum()
        idx = np.random.choice(n_avail, size=size, p=w, replace=False)
    elif strategy == "random":
        idx = np.random.choice(n_avail, size=size, replace=False)
    else:
        raise ValueError(f"Unknown strategy {strategy}")
    return samples[idx]

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
# Distributed Gelman–Rubin (R-hat)
# -----------------------------------------------------------------------------
def dist_gelman_rubin(local_chain):
    # local_chain: ndarray of shape (n_samples, n_dim)
    gathered = comm.allgather(local_chain)    # -> List of 2D arrays
    m = len(gathered)
    if m < 2:
        return np.full(local_chain.shape[1], np.inf)

    # truncate all to same length
    n = min(c.shape[0] for c in gathered)
    if n < 2:
        return np.full(local_chain.shape[1], np.inf)

    # stack into shape (m, n, dim)
    chains = np.stack([c[:n] for c in gathered], axis=0)

    # within‐chain and between‐chain variances
    means = chains.mean(axis=1)            # (m, dim)
    vars_ = chains.var(axis=1, ddof=1)     # (m, dim)

    W = vars_.mean(axis=0)                 # (dim,)
    B = n * ((means - means.mean(axis=0))**2).sum(axis=0) / (m - 1)

    V_hat = (n - 1)/n * W + B/n
    return np.sqrt(V_hat / W)



# -----------------------------------------------------------------------------
# Analytical surrogate “model” wrapper
# -----------------------------------------------------------------------------
class AnalyticalModel:
    def __init__(self, likelihood, scaler_x, scaler_y):
        self.likelihood = likelihood
        self.sxm = scaler_x.mean_
        self.sxs = scaler_x.scale_
        self.sym = scaler_y.mean_[0]
        self.sys = scaler_y.scale_[0]
        self.prior_lower = likelihood.prior_lower
        self.prior_upper = likelihood.prior_upper

    def __call__(self, xstd):
        x = xstd * self.sxs + self.sxm
        y = self.likelihood.neg_loglike(x)
        return (y - self.sym) / self.sys

# -----------------------------------------------------------------------------
# MPI-aware Adaptive Metropolis–Hastings sampler
# -----------------------------------------------------------------------------
def adaptive_metropolis_hastings_mpi(
    model, init_pt, temperature, base_cov,
    scaler_x, scaler_y, adapt_interval,
    cov_update_interval, epsilon,
    r_start, r_stop, r_converged
):
    D = init_pt.shape[1]
    current = init_pt.reshape(-1).copy()
    BIG_PENALTY = 1e120

    def safe_cholesky(C, eps):
        jitter = eps
        while True:
            try:
                return np.linalg.cholesky(C + jitter * np.eye(D))
            except np.linalg.LinAlgError:
                jitter *= 10

    proposal_cov = base_cov + epsilon * np.eye(D)
    L = safe_cholesky(proposal_cov, epsilon)
    jump_scale = 2.38 / np.sqrt(D)

    def loglike(x):
        if np.any(x < model.prior_lower) or np.any(x > model.prior_upper):
            return BIG_PENALTY
        xstd = (x - model.sxm) / model.sxs
        ystd = model(xstd.reshape(1, -1))[0]
        return ystd * model.sys + model.sym

    chain, llchain, history = [current.copy()], [loglike(current)], [current.copy()]
    recent = []
    step = 0
    adapting = False

    while True:
        step += 1
        z = np.random.randn(D)
        prop = current + jump_scale * (L @ z)
        pll = loglike(prop)
        delta = pll - llchain[-1]
        alpha = np.exp(np.clip(-delta / temperature, -700, 700))
        if np.random.rand() < alpha:
            current, ll = prop, pll
            recent.append(1)
        else:
            ll = llchain[-1]
            recent.append(0)
        chain.append(current.copy())
        llchain.append(ll)
        history.append(current.copy())

        if step % cov_update_interval == 0:
            R = dist_gelman_rubin(np.array(chain))
            max_r = np.max(R - 1)
            if rank == 0:
                print(f"[MCMC] Step {step:6d} ▶ max(R-1) = {max_r:.5f}")
            if not adapting and max_r < r_start:
                adapting = True
            if adapting:
                if step % adapt_interval == 0:
                    acc_rate = np.mean(recent)
                    jump_scale *= 1.1 if acc_rate > 0.25 else 0.9
                    recent = []
                if step > D:
                    emp = np.cov(np.array(history).T)
                    proposal_cov = emp + epsilon * np.eye(D)
                    L = safe_cholesky(proposal_cov, epsilon)
            if adapting and max_r < r_stop:
                adapting = False
            if max_r < r_converged and step >= cov_update_interval:
                break

    n_half = len(chain) // 2
    samples = np.vstack(chain[n_half:])
    lls = np.array(llchain[n_half:])
    compressed = []
    count = 1
    for i in range(1, len(samples)):
        if np.allclose(samples[i], samples[i - 1]):
            count += 1
        else:
            compressed.append(np.hstack(([count, lls[i - 1]], samples[i - 1])))
            count = 1
    compressed.append(np.hstack(([count, lls[-1]], samples[-1])))
    return np.array(compressed), np.cov(samples.T)

# -----------------------------------------------------------------------------
# Main MPI workflow
# -----------------------------------------------------------------------------
with open("config/default.yaml") as f:
    cfg = yaml.safe_load(f)

# Unpack parameters
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

nc = int(cfg['mcmc']['n_chains'])
r0 = float(cfg['mcmc']['r_start'])
r1 = float(cfg['mcmc']['r_stop'])
rc = float(cfg['mcmc']['r_converged'])
T0 = float(cfg['mcmc']['initial_temperature'])
TF = float(cfg['mcmc']['final_temperature'])
sched = cfg['mcmc']['annealing_schedule']
ai = int(cfg['mcmc']['adapt_interval'])
ci = int(cfg['mcmc']['cov_update_interval'])
eps = float(cfg['mcmc']['epsilon'])

paths = cfg['paths']
dirs = {
    'iterative_data': paths['iterative_data'],
    'mcmc_chains': paths['mcmc_chains'],
    'mcmc_bestfit': paths['mcmc_bestfit'],
}

# Check MPI size
if rank == 0 and size != nc:
    raise RuntimeError(f"MPI size {size} != n_chains {nc}")

# Seeds per rank
dt = int(cfg['misc']['random_seed'])
np.random.seed(dt + rank)

# Likelihood selection (with original defaults restored!)
liketype = cfg['likelihood']['type']
if liketype == 'planck_gaussian':
    LikelihoodClass = PlanckGaussianLikelihood
    ctor_args = dict(
        N=int(cfg['model']['dimension']),
        chains_dir=cfg.get('likelihood', {}).get('chains_dir', "class_chains/27_param"),
        skip_rows=cfg.get('likelihood', {}).get('skip_rows', 500)
    )
elif liketype == 'planck_gaussian_extended':
    LikelihoodClass = ExtendedPlanckGaussianLikelihood
    ctor_args = dict(
        N_main=int(cfg['model']['dimension']) - 2,
        chains_dir=cfg.get('likelihood', {}).get('chains_dir', "class_chains/27_param"),
        skip_rows=cfg.get('likelihood', {}).get('skip_rows', 500)
    )
else:
    raise ValueError(f"Unknown likelihood type {liketype!r}")

tl = LikelihoodClass(**ctor_args)
N = tl.N

# Base covariance
bcs = cfg['mcmc']['base_cov']
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
        raise ValueError(f"Unknown base_cov {bcs!r}")
    else:
        sys.exit(1)

# Initial data (rank 0)
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
    training_sets = [(aggX.copy(), aggy.copy())]
    scalers = [(sx, sy)]
else:
    aggX = aggy = iters = None
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
    if rank == 0:
        print(f"Iteration {it}")
        ages = it - iters
        keep = np.random.rand(len(ages)) < retention_probability(
            ages, iters, it, rb, rd, rmin, cc
        )
        aggX = aggX[keep]
        aggy = aggy[keep]
        iters = iters[keep]

        # re-fit scalers
        if scaler_t == 'minmax':
            sx = MinMaxScaler().fit(aggX)
            sy = MinMaxScaler().fit(aggy.reshape(-1, 1))
        else:
            sx = StandardScaler().fit(aggX)
            sy = StandardScaler().fit(aggy.reshape(-1, 1))

        scalers.append((sx, sy))
        training_sets.append((aggX.copy(), aggy.copy()))

        os.makedirs(dirs['iterative_data'], exist_ok=True)
        with open(os.path.join(dirs['iterative_data'], "scalers.pkl"), 'wb') as f:
            pickle.dump(scalers, f)
        with open(os.path.join(dirs['iterative_data'], "training_sets.pkl"), 'wb') as f:
            pickle.dump(training_sets, f)

    comm.Barrier()

    if rank != 0:
        with open(os.path.join(dirs['iterative_data'], "scalers.pkl"), 'rb') as f:
            scalers = pickle.load(f)

    sx, sy = scalers[it]
    model = AnalyticalModel(tl, sx, sy)
    temp = compute_temperature(it, n_iter, T0, TF, sched)

    if rank == 0:
        pts = np.random.uniform(lb, ub, size=(nc, N))
    else:
        pts = None
    pts = comm.bcast(pts, root=0)

    local_chain, local_cov = adaptive_metropolis_hastings_mpi(
        model=model,
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
        r_converged=rc
    )

    all_ch = comm.gather(local_chain, root=0)
    all_cov = comm.gather(local_cov, root=0)

    if rank == 0:
        base_cov = sum(all_cov) / len(all_cov)
    base_cov = comm.bcast(base_cov, root=0)

    if rank == 0:
        combined = np.vstack(all_ch)
        os.makedirs(dirs['mcmc_chains'], exist_ok=True)
        np.savetxt(
            os.path.join(dirs['mcmc_chains'], f"chain_iter{it}.txt"),
            combined
        )
        new_n = n0 + (it - 1) * inc
        newX = resample_chain_points(combined, new_n, temp, strategy=it_str)
        newY = tl.neg_loglike(newX)
    else:
        newX = newY = None

    newX = comm.bcast(newX, root=0)
    newY = comm.bcast(newY, root=0)

    if rank == 0:
        aggX = np.vstack([aggX, newX])
        aggy = np.concatenate([aggy, newY])
        training_sets.append((aggX.copy(), aggy.copy()))
        with open(os.path.join(dirs['iterative_data'], "training_sets.pkl"), 'wb') as f:
            pickle.dump(training_sets, f)

        new_iters = np.full(len(newY), it, dtype=int)
        iters = np.concatenate([iters, new_iters])

        os.makedirs(dirs['mcmc_bestfit'], exist_ok=True)
        bi = np.argmin(newY)
        bf = newX[bi]
        np.savetxt(
            os.path.join(dirs['mcmc_bestfit'], f"best_iter{it}.txt"),
            bf.reshape(1, -1)
        )

if rank == 0:
    print("All iterations complete.")
