#!/usr/bin/env python3
"""
This script implements an iterative training procedure using surrogate models and MCMC sampling.
The workflow includes:
  - Loading configuration parameters from a YAML file.
  - Generating initial training data based on a Gaussian likelihood.
  - Iteratively applying memory loss via age-dependent retention of training samples.
  - Building and training a surrogate emulation model with TensorFlow.
  - Running an adaptive Metropolis-Hastings MCMC sampler using the surrogate model.
  - Resampling new training points from MCMC chains and aggregating them.
  - Saving trained models, aggregated training sets, and scalers for later use.
Dependencies include numpy, tensorflow, scikit-learn, matplotlib, and custom modules for likelihoods, data generation,
model building, training, and MCMC sampling.
"""

import os
import pickle
import numpy as np
import yaml
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from likelihoods.gaussian import GaussianLikelihood
from data.data_generator import generate_data
from models.emulation_model import build_emulation_model
from training.train import train_model
from mcmc.sampler import adaptive_metropolis_hastings

def resample_chain_points(chain_file, new_sample_size, temperature, strategy="flat"):
    """
    Resample chain points from a file.
    For 'flat' strategy: use weighted sampling based on corrected weight calculation.
    For 'random' strategy: sample uniformly at random.
    """
    chain = np.loadtxt(chain_file, skiprows=1)
    samples = chain[:, 2:]
    if strategy == "flat":
        multiplicities = chain[:, 0]
        predicted_loglikes = chain[:, 1]
        log_weights = np.log(multiplicities) + predicted_loglikes / temperature
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)
        indices = np.random.choice(np.arange(len(samples)), size=new_sample_size, p=weights, replace=False)
    elif strategy == "random":
        indices = np.random.choice(np.arange(len(samples)), size=new_sample_size, replace=False)
    else:
        raise ValueError("Unknown sampling strategy: " + strategy)
    return samples[indices]

def retention_probability(ages, sample_iters, current_iter, base_probability, decay_rate, min_probability, control_coef):
    """
    Compute retention probability as a function of sample age and birth iteration.
    For age <= 1, retention probability = base_probability.
    For age > 1, a sample's retention probability decays linearly, but the decay is scaled by a control coefficient
    that reduces the effective decay for samples born in later iterations.
    """
    effective_decay = decay_rate * (1 - control_coef * (sample_iters / current_iter))
    ret_probs = np.where(ages <= 1,
                         base_probability,
                         np.maximum(min_probability, base_probability - effective_decay * (ages - 1)))
    return ret_probs

# Load configuration.
config_path = os.path.join("config", "default.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

n_iterations = int(config['iterative']['n_iterations'])
num_samples = int(config['data']['num_samples'])
new_sample_size = int(config['iterative']['new_sample_size'])

# Read sample size increment (for new samples each iteration)
sample_size_increment = int(config['iterative'].get('sample_size_increment', 0))

# Read MCMC step size increment from config (default 0 if not provided)
base_mcmc_steps = int(config['mcmc']['n_steps'])
mcmc_step_increment = int(config['mcmc'].get('step_increment', 0))

np.random.seed(int(config['misc']['random_seed']))
tf.random.set_seed(int(config['misc']['random_seed']))

N = int(config['model']['dimension'])
target_likelihood = GaussianLikelihood(N)
base_cov = config['mcmc']['base_cov']
if base_cov == "target":
    base_cov = target_likelihood.Sigma
else:
    print("Warning: base_cov was not defined. Using random covariance matrix.")
    base_cov = np.cov(np.random.randn(num_samples, N), rowvar=False)

X_init_scaled, _, scaler_x, scaler_y = generate_data(
    target_likelihood,
    num_samples=num_samples,
    sampling_strategy=config['data']['sampling_strategy'],
    offset_scale=float(config['data']['offset_scale']),
    nstd=float(config['data']['nstd']),
    cov_scale=float(config['data']['cov_scale'])
)
X_init = scaler_x.inverse_transform(X_init_scaled)
y_init = target_likelihood.neg_loglike(X_init)

training_sets = []
training_sets.append((X_init, y_init))

scalers = []
scaler_x_current = StandardScaler().fit(X_init)
scaler_y_current = StandardScaler().fit(y_init.reshape(-1, 1))
scalers.append((scaler_x_current, scaler_y_current))

aggregated_X = X_init
aggregated_y = y_init
sample_iters = np.zeros(aggregated_X.shape[0], dtype=int)

trained_models_dir = config['paths']['trained_models']
iterative_data_dir = config['paths']['iterative_data']
mcmc_chains_dir = config['paths']['mcmc_chains']
mcmc_bestfit_dir = config['paths']['mcmc_bestfit']

# Retrieve retention parameters from configuration.
ret_base = float(config['iterative']['retention']['base_probability'])
ret_decay = float(config['iterative']['retention']['decay_rate'])
ret_min = float(config['iterative']['retention']['min_probability'])
control_coef = float(config['iterative']['retention']['control_coef'])

# Determine the resampling strategy ("flat" for weighted, "random" for uniform random).
resample_strategy = config['iterative']['sampling_strategy']

for it in range(1, n_iterations + 1):
    print(f"Iteration {it}")
    
    # Apply retention to the aggregated samples.
    ages = it - sample_iters
    ret_probs = retention_probability(ages, sample_iters, it, ret_base, ret_decay, ret_min, control_coef)
    mask = np.random.rand(len(sample_iters)) < ret_probs
    aggregated_X = aggregated_X[mask]
    aggregated_y = aggregated_y[mask]
    sample_iters = sample_iters[mask]
    
    # Update scalers with the retained data.
    scaler_x_current = StandardScaler().fit(aggregated_X)
    scaler_y_current = StandardScaler().fit(aggregated_y.reshape(-1, 1))
    X_scaled = scaler_x_current.transform(aggregated_X)
    y_scaled = scaler_y_current.transform(aggregated_y.reshape(-1, 1))
    
    scalers.append((scaler_x_current, scaler_y_current))
    
    # Build and train the surrogate model.
    model = build_emulation_model(
        N,
        num_hidden_layers=int(config['model']['num_hidden_layers']),
        neurons=int(config['model']['neurons']),
        activation=config['model']['activation'],
        use_gaussian_decomposition=config['model']['use_gaussian_decomposition'],
        learning_rate=float(config['model']['learning_rate'])
    )
    train_model(
        model, X_scaled, y_scaled,
        epochs=int(config['training']['epochs']),
        batch_size=int(config['training']['batch_size']),
        val_split=float(config['training']['val_split']),
        patience=int(config['training']['patience'])
    )
    
    os.makedirs(trained_models_dir, exist_ok=True)
    model_path = os.path.join(trained_models_dir, f"model_iter{it}.h5")
    model.save(model_path)
    
    # Compute the current MCMC step count.
    current_mcmc_steps = base_mcmc_steps + (it - 1) * mcmc_step_increment

    # Set the MCMC temperature.
    current_temp = (float(config['iterative']['final_temperature'])
                    if it == n_iterations else float(config['mcmc']['temperature']))
    
    # Run the MCMC sampler.
    initial_point = np.random.multivariate_normal(target_likelihood.mu, target_likelihood.Sigma)
    os.makedirs(mcmc_chains_dir, exist_ok=True)
    os.makedirs(mcmc_bestfit_dir, exist_ok=True)
    chain_file = os.path.join(mcmc_chains_dir, f"chain_iter{it}.txt")
    bestfit_file = os.path.join(mcmc_bestfit_dir, f"best_iter{it}.txt")
    
    adaptive_metropolis_hastings(
        keras_model=model,
        initial_point=initial_point,
        n_steps=current_mcmc_steps,
        temperature=current_temp,
        base_cov=base_cov,
        scaler_x=scaler_x_current,
        scaler_y=scaler_y_current,
        adapt_interval=int(config['mcmc']['adapt_interval']),
        cov_update_interval=int(config['mcmc']['cov_update_interval']),
        chain_file=chain_file,
        bestfit_file=bestfit_file,
        epsilon=float(config['mcmc']['epsilon']),
        burn_in_fraction=float(config['mcmc']['burn_in_fraction'])
    )
    
    # Compute the dynamic sample size for this iteration.
    current_new_samples = new_sample_size + (it - 1) * sample_size_increment
    
    # Resample new training points using the specified strategy.
    new_X = resample_chain_points(chain_file, current_new_samples, temperature=current_temp, strategy=resample_strategy)
    new_y = target_likelihood.neg_loglike(new_X)
    new_sample_iters = np.full(new_X.shape[0], it, dtype=int)
    
    aggregated_X = np.vstack([aggregated_X, new_X])
    aggregated_y = np.concatenate([aggregated_y, new_y])
    sample_iters = np.concatenate([sample_iters, new_sample_iters])
    
    training_sets.append((aggregated_X, aggregated_y))
    
os.makedirs(iterative_data_dir, exist_ok=True)
with open(os.path.join(iterative_data_dir, "training_sets.pkl"), "wb") as f:
    pickle.dump(training_sets, f)

with open(os.path.join(iterative_data_dir, "scalers.pkl"), "wb") as f:
    pickle.dump(scalers, f)
