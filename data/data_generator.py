import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def generate_ellipsoid_samples(mu, cov, num_samples, offset_scale, nstd, cov_scale, batch_size=1000):
    D = mu.shape[0]
    std = np.sqrt(np.diag(cov))

    # Random shift vector
    random_offset = np.random.uniform(-offset_scale * std, offset_scale * std, size=D)
    center = mu + random_offset

    # Axis-aligned bounding box
    lower_bound = center - nstd * std
    upper_bound = center + nstd * std

    # Scale covariance and Cholesky
    cov_scaled = cov * (cov_scale ** 2)
    L = np.linalg.cholesky(cov_scaled)

    accepted_samples = []
    while len(accepted_samples) < num_samples:
        # Uniform sampling in unit hypersphere
        g = np.random.randn(batch_size, D)
        g_unit = g / np.linalg.norm(g, axis=1, keepdims=True)
        r = np.random.uniform(0, 1, size=(batch_size, 1))
        radius = r ** (1.0 / D)
        z = g_unit * radius

        # Map to ellipsoid
        x_candidates = center + np.dot(z, L.T)
        mask = np.all((x_candidates >= lower_bound) & (x_candidates <= upper_bound), axis=1)
        accepted = x_candidates[mask]
        accepted_samples.append(accepted)

    accepted_samples = np.concatenate(accepted_samples, axis=0)
    return accepted_samples[:num_samples]


def generate_lhs_samples(mu, cov, num_samples, offset_scale, nstd):
    """
    Generate samples via Latin Hypercube Sampling within an axis-aligned box.

    Parameters:
      mu: Mean vector (D,)
      cov: Covariance matrix (D, D)
      num_samples: Number of points
      offset_scale: Scale for random shift of center
      nstd: Number of std deviations for box half-width

    Returns:
      samples: numpy array (num_samples, D)
    """
    D = mu.shape[0]
    std = np.sqrt(np.diag(cov))

    # Random shift of center
    random_offset = np.random.uniform(-offset_scale * std, offset_scale * std, size=D)
    center = mu + random_offset

    # Define bounding box
    lower_bound = center - nstd * std
    upper_bound = center + nstd * std

    # Latin Hypercube Sampling
    lhs = np.zeros((num_samples, D))
    for i in range(D):
        perm = np.random.permutation(num_samples)
        # stratified samples in [0,1]
        u = (perm + np.random.rand(num_samples)) / num_samples
        # scale to [lower, upper]
        lhs[:, i] = lower_bound[i] + u * (upper_bound[i] - lower_bound[i])

    return lhs


def generate_data(likelihood, num_samples, scaler, sampling_strategy, offset_scale, nstd, cov_scale):
    """
    Generate training data using different sampling strategies.

    sampling_strategy: "gaussian", "ellipsoid", or "lhs"
    """
    # Draw samples
    if sampling_strategy == "gaussian":
        x_train = np.random.multivariate_normal(likelihood.mu, likelihood.Sigma, num_samples)
    elif sampling_strategy == "ellipsoid":
        x_train = generate_ellipsoid_samples(
            likelihood.mu, likelihood.Sigma, num_samples,
            offset_scale, nstd, cov_scale
        )
    elif sampling_strategy == "lhs":
        x_train = generate_lhs_samples(
            likelihood.mu, likelihood.Sigma, num_samples,
            offset_scale, nstd
        )
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    # Compute targets
    y_train = likelihood.neg_loglike(x_train)

    # Scale inputs and outputs
    if scaler == "minmax":
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
    elif scaler == "standard":
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler}")
    
    x_train_scaled = scaler_x.fit_transform(x_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

    return x_train_scaled, y_train_scaled, scaler_x, scaler_y
