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
    burn_in_fraction  # Fraction of samples to discard as burn-in (0 <= burn_in_fraction < 1)
):
    """
    Adaptive Metropolis-Hastings sampler that uses the provided Keras model directly
    (without TFLite conversion) for computing the (unnormalized) negative log-likelihood.
    
    Parameters:
      keras_model: A trained Keras model.
      initial_point: Initial point in D-dimensional space (numpy array of shape (D,)).
      n_steps: Total number of MCMC iterations.
      temperature: Temperature T used in the acceptance probability.
      base_cov: A D x D covariance matrix (user provided or empirical) to initialize the proposal covariance.
      scaler_x: A fitted StandardScaler for inputs.
      scaler_y: A fitted StandardScaler for outputs (to unscale the model predictions).
      adapt_interval: Number of steps between adaptations of the jump scale.
      cov_update_interval: Number of steps between updating the proposal covariance from sample history.
      chain_file: Filename where the compressed chain will be saved.
      bestfit_file: Filename where the best sample (lowest -loglike) is saved.
      epsilon: Small constant for numerical stability.
      burn_in_fraction: Fraction of initial samples to discard as burn-in (0 <= burn_in_fraction < 1).
    
    The function saves the compressed chain and best sample to disk and prints a summary.
    """
    
    D = initial_point.shape[0]
    
    # Pre-compute the scaled proposal covariance and its Cholesky factor.
    proposal_cov = (2.38**2 / D) * base_cov + epsilon * np.eye(D)
    L = np.linalg.cholesky(proposal_cov)
    
    # Jump factor s = 2.4 / sqrt(D)
    s = 2.4 / np.sqrt(D)
    
    # Extract scaling parameters as tensors for speed.
    scaler_x_mean = tf.constant(scaler_x.mean_, dtype=tf.float32)
    scaler_x_scale = tf.constant(scaler_x.scale_, dtype=tf.float32)
    scaler_y_mean = tf.constant(scaler_y.mean_, dtype=tf.float32)
    scaler_y_scale = tf.constant(scaler_y.scale_, dtype=tf.float32)
    
    # Compile the inference function with tf.function for optimization.
    @tf.function
    def predict_loglike_tf(x_tensor):
        # Scale input: (x - mean) / scale
        x_std = (x_tensor - scaler_x_mean) / scaler_x_scale
        # Model inference (assumes keras_model's output is in standardized space)
        y_pred_std = keras_model(x_std, training=False)
        # Inverse scale: y_pred * scale + mean
        y_pred = y_pred_std * scaler_y_scale + scaler_y_mean
        return y_pred

    def predict_loglike(x):
        # Convert input to tensor and ensure proper shape: (1, D)
        x_tensor = tf.convert_to_tensor(x.reshape(1, -1), dtype=tf.float32)
        # Call the compiled function; the first call may have some overhead due to tracing
        y_pred = predict_loglike_tf(x_tensor)
        # Convert result to numpy and return the scalar value.
        return y_pred.numpy().item()
    
    # Initialize chain storage.
    current_point = initial_point.copy()
    current_loglike = predict_loglike(current_point)
    samples = [current_point.copy()]
    loglikes = [current_loglike]
    accepted = 0
    recent_accepts = []
    sample_history = [current_point.copy()]
    
    # Calculate the iteration count marking the end of burn-in.
    burn_in_steps = int(burn_in_fraction * n_steps)

    for step in tqdm(range(1, n_steps + 1), desc='MCMC Progress', unit='step'):
        # Generate proposal: current_point + s * (L @ z) with z ~ N(0, I)
        z = np.random.randn(D)
        proposal = current_point + s * (L @ z)
        
        proposal_loglike = predict_loglike(proposal)
        delta = proposal_loglike - current_loglike
        # Avoid overflow in exponentiation by clipping
        delta_scaled = -delta / temperature
        delta_clipped = np.clip(delta_scaled, a_min=-700, a_max=700)  # Prevent overflow
        alpha = min(1.0, np.exp(delta_clipped))
        
        if np.random.rand() < alpha:
            current_point = proposal.copy()
            current_loglike = proposal_loglike
            accepted += 1
            recent_accepts.append(1)
        else:
            recent_accepts.append(0)
        
        samples.append(current_point.copy())
        loglikes.append(current_loglike)
        sample_history.append(current_point.copy())
        
        # Only adapt during burn-in to preserve the Markov property post burn-in.
        if step <= burn_in_steps:
            if step % adapt_interval == 0:
                acc_rate = np.mean(recent_accepts)
                s *= 1.1 if acc_rate > 0.25 else 0.9
                recent_accepts = []
            
            if step % cov_update_interval == 0:
                hist = np.array(sample_history)
                emp_cov = np.cov(hist.T) + epsilon * np.eye(D)
                proposal_cov = (2.38**2 / D) * emp_cov + epsilon * np.eye(D)
                L = np.linalg.cholesky(proposal_cov)
    
    samples = np.array(samples)
    loglikes = np.array(loglikes)
    
    # Apply burn-in: discard the initial fraction of samples.
    samples_post = samples[burn_in_steps:]
    loglikes_post = loglikes[burn_in_steps:]
    
    # Identify best sample from the post burn-in samples.
    best_idx = np.argmin(loglikes_post)
    best_sample = samples_post[best_idx]
    best_loglike = loglikes_post[best_idx]
    
    # Compress the chain by collapsing consecutive identical points.
    compressed = []
    count = 1
    for i in range(1, len(samples_post)):
        if np.allclose(samples_post[i], samples_post[i-1]):
            count += 1
        else:
            compressed.append(np.hstack(([count, loglikes_post[i-1]], samples_post[i-1])))
            count = 1
    compressed.append(np.hstack(([count, loglikes_post[-1]], samples_post[-1])))
    compressed = np.array(compressed)
    
    np.savetxt(chain_file, compressed)
    np.savetxt(bestfit_file, best_sample.reshape(1, -1))
    
    print(f"Total steps: {n_steps}, Acceptance count: {accepted}, Acceptance rate: {100 * accepted / n_steps}%")
    print(f"Best predicted -loglike (post burn-in): {best_loglike}")
