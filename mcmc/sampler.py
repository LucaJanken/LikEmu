import numpy as np
import tensorflow as tf
from mpi4py import MPI
from tqdm import tqdm


def gelman_rubin(chains):
    """Compute Gelman--Rubin R statistic for a list of chains."""
    m = len(chains)
    if m < 2:
        return np.full(chains[0].shape[1], np.inf)
    n = min(len(c) for c in chains)
    if n < 4:
        return np.full(chains[0].shape[1], np.inf)
    halves = [c[n//2:n] for c in chains]
    n = halves[0].shape[0]
    means = np.array([h.mean(axis=0) for h in halves])
    vars_ = np.array([h.var(axis=0, ddof=1) for h in halves])
    W = vars_.mean(axis=0)
    B_over_n = ((means - means.mean(axis=0))**2).sum(axis=0) / (m - 1)
    var_hat = (n - 1)/n * W + B_over_n
    R = np.sqrt(var_hat / (W + 1e-16))
    return R

def dist_gelman_rubin(comm, local_chains):
    gathered = comm.allgather(local_chains)
    flat = [c for sub in gathered for c in sub]
    m = len(flat)
    if m < 2:
        return np.full(flat[0].shape[1], np.inf)
    n = min(c.shape[0] for c in flat)
    if n < 4:
        return np.full(flat[0].shape[1], np.inf)
    halves = [c[n // 2:] for c in flat]
    W = np.mean([h.var(axis=0, ddof=1) for h in halves], axis=0)
    means = np.array([h.mean(axis=0) for h in halves])
    B_over_n = ((means - means.mean(axis=0))**2).sum(axis=0) / (m - 1)
    var_hat = (n - 1)/n * W + B_over_n
    return np.sqrt(var_hat / (W + 1e-16))

def adaptive_metropolis_hastings(
    keras_model,
    initial_points,
    n_chains,
    temperature,
    base_cov,
    scaler_x,
    scaler_y,
    adapt_interval,
    cov_update_interval,
    chain_file,
    bestfit_file,
    epsilon,
    r_start,
    r_stop,
    r_converged
):
    """
    Adaptive Metropolis-Hastings with automatic jittering
    to enforce strict positive-definiteness on proposal covariances.
    Returns final empirical covariance of post–burn-in samples.
    """
    D = initial_points.shape[1]

    # ensure shape matches
    initial_points = np.atleast_2d(initial_points)
    if initial_points.shape[0] != n_chains:
        raise ValueError(
            f"initial_points.shape[0]={initial_points.shape[0]} must equal n_chains={n_chains}"
        )

    def extract_scaler_params(scaler):
        if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):  # StandardScaler
            return scaler.mean_, scaler.scale_
        elif hasattr(scaler, 'min_') and hasattr(scaler, 'scale_'):
            # true inverse-transform parameters
            scale = 1.0 / scaler.scale_
            mean  = - scaler.min_ * scale
            return mean, scale
        else:
            raise ValueError("Unsupported scaler type.")


    def safe_cholesky(C, eps):
        # keep adding diagonal jitter until it succeeds
        jitter = eps
        while True:
            try:
                return np.linalg.cholesky(C + jitter * np.eye(D))
            except np.linalg.LinAlgError:
                jitter *= 10

    # initial proposal covariance & factor
    proposal_cov = base_cov + epsilon * np.eye(D)
    L = safe_cholesky(proposal_cov, epsilon)
    jump_scale = 2.38 / np.sqrt(D)

    # tf constants for scalers
    x_mean, x_scale = extract_scaler_params(scaler_x)
    y_mean, y_scale = extract_scaler_params(scaler_y)

    scaler_x_mean  = tf.constant(x_mean,  dtype=tf.float32)
    scaler_x_scale = tf.constant(x_scale, dtype=tf.float32)
    scaler_y_mean  = tf.constant(y_mean,  dtype=tf.float32)
    scaler_y_scale = tf.constant(y_scale, dtype=tf.float32)



    @tf.function
    def predict_loglike_tf(x_tensor):
        x_std      = (x_tensor - scaler_x_mean) / scaler_x_scale
        y_pred_std = keras_model(x_std, training=False)
        return y_pred_std * scaler_y_scale + scaler_y_mean

    def predict_loglike(x):
        x_tensor = tf.convert_to_tensor(x.reshape(1, -1), dtype=tf.float32)
        return predict_loglike_tf(x_tensor).numpy().item()

    current_points = initial_points.copy()
    current_lls = np.array([predict_loglike(p) for p in current_points])

    chains = [[p.copy()] for p in current_points]
    llchains = [[ll] for ll in current_lls]
    accepted = np.zeros(n_chains, dtype=int)
    recent = [[] for _ in range(n_chains)]
    history = [p.copy() for p in current_points]

    step = 0
    adapting = False
    while True:
        step += 1
        for c in range(n_chains):
            z = np.random.randn(D)
            proposal = current_points[c] + jump_scale * (L @ z)
            prop_ll = predict_loglike(proposal)
            delta = prop_ll - current_lls[c]
            alpha = np.exp(np.clip(-delta / temperature, -700, 700))
            if np.random.rand() < alpha:
                current_points[c] = proposal
                current_lls[c] = prop_ll
                accepted[c] += 1
                recent[c].append(1)
            else:
                recent[c].append(0)
            chains[c].append(current_points[c].copy())
            llchains[c].append(current_lls[c])
            history.append(current_points[c].copy())

        if step % cov_update_interval == 0:
            R = gelman_rubin([np.array(ch) for ch in chains])
            max_r = np.max(R - 1)
            print(f"[MCMC] Step {step:6d} ▶ max(R-1) = {max_r:.5f}")
            if not adapting and max_r < r_start:
                adapting = True
            if adapting:
                if step % adapt_interval == 0:
                    acc = [np.mean(r[-adapt_interval:]) if len(r) >= adapt_interval else np.mean(r) for r in recent]
                    if len(acc) > 0:
                        mean_acc = np.mean(acc)
                        jump_scale *= 1.1 if mean_acc > 0.25 else 0.9
                    recent = [[] for _ in range(n_chains)]
                if step > D:
                    hist = np.array(history)
                    emp_cov = np.cov(hist.T)
                    proposal_cov = emp_cov + epsilon * np.eye(D)
                    L = safe_cholesky(proposal_cov, epsilon)
            if adapting and max_r < r_stop:
                adapting = False
            if max_r < r_converged and step >= cov_update_interval:
                break

    chains_np = [np.array(ch) for ch in chains]
    lls_np = [np.array(ll) for ll in llchains]
    halves = [len(c)//2 for c in chains_np]
    samples_post = np.vstack([c[h:] for c, h in zip(chains_np, halves)])
    loglikes_post = np.concatenate([l[h:] for l, h in zip(lls_np, halves)])

    best_idx = np.argmin(loglikes_post)
    best_sample = samples_post[best_idx]
    best_loglike = loglikes_post[best_idx]

    # compress chain
    compressed, count = [], 1
    for i in range(1, len(samples_post)):
        if np.allclose(samples_post[i], samples_post[i-1]):
            count += 1
        else:
            compressed.append(np.hstack(([count, loglikes_post[i-1]], samples_post[i-1])))
            count = 1
    compressed.append(np.hstack(([count, loglikes_post[-1]], samples_post[-1])))
    compressed = np.array(compressed)

    # save outputs
    np.savetxt(chain_file, compressed)
    np.savetxt(bestfit_file, best_sample.reshape(1, -1))

    acc_rate = accepted.sum() / (step * n_chains)
    print(f"Total steps: {step}, Acceptance rate: {acc_rate*100:.1f}%")
    print(f"Best predicted -loglike: {best_loglike}")

    final_emp_cov = np.cov(samples_post.T)
    return compressed, final_emp_cov

def adaptive_metropolis_hastings_mpi(
    keras_model, init_pt, temperature, base_cov,
    scaler_x, scaler_y, adapt_interval,
    cov_update_interval, epsilon,
    r_start, r_stop, r_converged, max_steps, rank
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

    x_mean, x_scale = scaler_x.mean_, scaler_x.scale_
    y_mean, y_scale = scaler_y.mean_.reshape(-1), scaler_y.scale_.reshape(-1)
    sxm = tf.constant(x_mean, dtype=tf.float32)
    sxs = tf.constant(x_scale, dtype=tf.float32)
    sym = tf.constant(y_mean, dtype=tf.float32)
    sys = tf.constant(y_scale, dtype=tf.float32)

    @tf.function
    def pred_tf(x):
        xstd = (x - sxm) / sxs
        ystd = keras_model(xstd, training=False)
        return ystd * sys + sym

    def loglike(x):
        lb, ub = keras_model.likelihood.prior_lower, keras_model.likelihood.prior_upper
        if np.any(x < lb) or np.any(x > ub):
            return BIG_PENALTY
        t = tf.convert_to_tensor(x.reshape(1, -1), dtype=tf.float32)
        return pred_tf(t).numpy().item()

    chain = [current.copy()]
    llchain = [loglike(current)]
    history = [current.copy()]
    recent = []
    step = 0
    adapting = False

    while True:
        if max_steps is not None and step >= max_steps:
            if rank == 0:
                print(f"[MCMC] Reached max_steps={max_steps} without convergence.")
            break
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
            R = dist_gelman_rubin([np.array(chain)])
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

    # compress chain
    n_half = len(chain) // 2
    samples = np.vstack(chain[n_half:])
    lls = np.array(llchain[n_half:])
    compressed = []
    count = 1
    for i in range(1, len(samples)):
        if np.allclose(samples[i], samples[i-1]):
            count += 1
        else:
            compressed.append(
                np.hstack(([count, lls[i-1]], samples[i-1]))
            )
            count = 1
    compressed.append(np.hstack(([count, lls[-1]], samples[-1])))
    return np.array(compressed), np.cov(samples.T)
