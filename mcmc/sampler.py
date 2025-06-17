import numpy as np
import tensorflow as tf
from tqdm import tqdm

def adaptive_metropolis_hastings(
    keras_model,
    initial_point,
    n_steps,
    temperature,
    base_cov,
    scaler_x,
    scaler_y,
    adapt_interval,
    cov_update_interval,
    chain_file,
    bestfit_file,
    epsilon,
    burn_in_fraction
):
    """
    Adaptive Metropolis-Hastings with automatic jittering
    to enforce strict positive-definiteness on proposal covariances.
    Returns final empirical covariance of post–burn-in samples.
    """
    D = initial_point.shape[0]

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

    # jump scale
    s = 2.38 / np.sqrt(D)

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

    # initialize chain
    current_point   = initial_point.copy()
    current_loglike = predict_loglike(current_point)
    samples, loglikes = [current_point.copy()], [current_loglike]
    accepted = 0
    recent_accepts = []
    sample_history = [current_point.copy()]

    burn_in_steps = int(burn_in_fraction * n_steps)

    for step in tqdm(range(1, n_steps+1), desc='MCMC Progress', unit='step'):
        # propose
        z        = np.random.randn(D)
        proposal = current_point + s * (L @ z)
        prop_ll  = predict_loglike(proposal)

        # MH acceptance
        delta        = prop_ll - current_loglike
        delta_scaled = -delta / temperature
        alpha        = np.exp(np.clip(delta_scaled, -700, 700))
        if np.random.rand() < alpha:
            current_point, current_loglike = proposal.copy(), prop_ll
            accepted += 1
            recent_accepts.append(1)
        else:
            recent_accepts.append(0)

        samples.append(current_point.copy())
        loglikes.append(current_loglike)
        sample_history.append(current_point.copy())

        # adaptation during burn-in
        if step <= burn_in_steps:
            if step % adapt_interval == 0:
                acc_rate = np.mean(recent_accepts)
                s *= 1.1 if acc_rate > 0.25 else 0.9
                recent_accepts = []

            if step % cov_update_interval == 0 and step > D:
                hist      = np.array(sample_history)
                emp_cov   = np.cov(hist.T)
                proposal_cov = emp_cov + epsilon * np.eye(D)
                L = safe_cholesky(proposal_cov, epsilon)

    samples    = np.array(samples)
    loglikes   = np.array(loglikes)
    samples_post  = samples[burn_in_steps:]
    loglikes_post = loglikes[burn_in_steps:]

    # best fit
    best_idx     = np.argmin(loglikes_post)
    best_sample  = samples_post[best_idx]
    best_loglike = loglikes_post[best_idx]

    # compress chain
    compressed, count = [], 1
    for i in range(1, len(samples_post)):
        if np.allclose(samples_post[i], samples_post[i-1]):
            count += 1
        else:
            compressed.append(
                np.hstack(([count, loglikes_post[i-1]], samples_post[i-1]))
            )
            count = 1
    compressed.append(
        np.hstack(([count, loglikes_post[-1]], samples_post[-1]))
    )
    compressed = np.array(compressed)

    # save outputs
    np.savetxt(chain_file, compressed)
    np.savetxt(bestfit_file, best_sample.reshape(1, -1))

    print(f"Total steps: {n_steps}, Acceptance rate: {accepted/n_steps*100:.1f}%")
    print(f"Best predicted -loglike (post burn-in): {best_loglike}")

    final_emp_cov = np.cov(samples_post.T)
    return final_emp_cov
