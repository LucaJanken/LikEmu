# data/data_generator.py

import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_ellipsoid_samples(mu, cov, num_samples, offset_scale, nstd, cov_scale, batch_size = 1000):
    """
    Generate samples approximately uniformly distributed within an ellipsoid defined by a scaled covariance,
    while ensuring all samples lie inside an axis-aligned bounding box centered around a randomly shifted mean.
    
    Parameters:
      mu: Mean vector (numpy array of shape (D,))
      cov: Covariance matrix (numpy array of shape (D, D))
      num_samples: Total number of samples to generate.
      offset_scale: Factor to scale the random offset (sampled uniformly per dimension).
      nstd: Number of standard deviations for the bounding box.
      cov_scale: Scaling factor for the covariance (cov_scaled = cov * cov_scale^2).
      batch_size: Number of candidate samples to generate per iteration.
      
    Returns:
      samples: numpy array of shape (num_samples, D) containing accepted samples.
    """
    D = mu.shape[0]
    std = np.sqrt(np.diag(cov))
    
    # Step 2: Generate a random shift vector (per dimension)
    random_offset = np.random.uniform(-offset_scale * std, offset_scale * std, size=D)
    center = mu + random_offset
    
    # Step 3: Define an axis-aligned bounding box around the new center.
    lower_bound = center - nstd * std
    upper_bound = center + nstd * std
    
    # Step 4: Scale covariance and perform Cholesky decomposition.
    cov_scaled = cov * (cov_scale ** 2)
    L = np.linalg.cholesky(cov_scaled)
    
    accepted_samples = []
    
    # Step 7: Buffer-based generation – keep generating batches until we have enough samples.
    while len(accepted_samples) < num_samples:
        # Step 5: Sample uniformly from a D-dimensional hypersphere.
        # Draw standard Gaussian vectors.
        g = np.random.randn(batch_size, D)
        # Normalize each vector to lie on the unit sphere.
        g_norm = np.linalg.norm(g, axis=1, keepdims=True)
        g_unit = g / g_norm
        
        # Rescale with a random radius.
        r = np.random.uniform(0, 1, size=(batch_size, 1))
        radius = r ** (1.0 / D)
        z = g_unit * radius
        
        # Transform to ellipsoid space.
        x_candidates = center + np.dot(z, L.T)
        
        # Step 6: Filter candidates: only accept those inside the bounding box.
        mask = np.all((x_candidates >= lower_bound) & (x_candidates <= upper_bound), axis=1)
        accepted = x_candidates[mask]
        accepted_samples.append(accepted)
    
    accepted_samples = np.concatenate(accepted_samples, axis=0)
    return accepted_samples[:num_samples]


def generate_data(likelihood, num_samples, sampling_strategy, offset_scale, nstd, cov_scale):
    """
    Generate training data from a given likelihood function using the specified sampling strategy.
    
    Parameters:
      likelihood: An object implementing a 'neg_loglike(x)' method, with attributes 'mu' and 'Sigma'.
      num_samples: Total number of training samples to generate.
      sampling_strategy: Either "gaussian" for standard multivariate normal sampling or "ellipsoid" for
                         the ellipsoid-based sampling strategy.
      offset_scale: Factor to scale the random offset for shifting the mean.
      nstd: Number of standard deviations for the bounding box.
      cov_scale: Scaling factor for the covariance when computing the ellipsoid.
      
    Returns:
      x_train: Scaled training inputs.
      y_train: Scaled training targets.
      scaler_x, scaler_y: Fitted StandardScalers.
    """
    D = likelihood.mu.shape[0]
    if sampling_strategy == "gaussian":
        x_train = np.random.multivariate_normal(likelihood.mu, likelihood.Sigma, num_samples)
    elif sampling_strategy == "ellipsoid":
        x_train = generate_ellipsoid_samples(likelihood.mu, likelihood.Sigma, num_samples,
                                             offset_scale, nstd, cov_scale)
    else:
        raise ValueError("Unknown sampling strategy: {}".format(sampling_strategy))
        
    y_train = likelihood.neg_loglike(x_train)
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train_scaled = scaler_x.fit_transform(x_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    return x_train_scaled, y_train_scaled, scaler_x, scaler_y