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
Dependencies include numpy, tensorflow, scikit-learn, matplotlib, and custom modules.
"""

import os
import pickle
import numpy as np
import yaml
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from likelihoods.gaussian import GaussianLikelihood
from likelihoods.planck_gaussian import PlanckGaussianLikelihood
from data.data_generator import generate_data
from models.emulation_model import build_emulation_model
from training.train import train_model
from mcmc.sampler import adaptive_metropolis_hastings

def resample_chain_points(chain_file, new_sample_size, temperature, strategy="flat"):
    chain = np.loadtxt(chain_file, skiprows=0)
    samples = chain[:, 2:]
    n_available = len(samples)

    sample_size = min(new_sample_size, n_available)
    if new_sample_size > n_available:
        print(f"[resample_chain_points] requested {new_sample_size} samples, "
              f"but only {n_available} available → sampling {sample_size} instead.")

    if strategy == "flat":
        multiplicities = chain[:, 0]
        predicted_loglikes = chain[:, 1]
        log_weights = np.log(multiplicities) + predicted_loglikes / temperature
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)

        indices = np.random.choice(
            np.arange(n_available),
            size=sample_size,
            p=weights,
            replace=False
        )

    elif strategy == "random":
        indices = np.random.choice(
            np.arange(n_available),
            size=sample_size,
            replace=False
        )

    else:
        raise ValueError("Unknown sampling strategy: " + strategy)

    return samples[indices]

def retention_probability(ages, sample_iters, current_iter,
                          base_probability, decay_rate,
                          min_probability, control_coef):
    effective_decay = decay_rate * (1 - control_coef * (sample_iters / current_iter))
    ret_probs = np.where(
        ages <= 1,
        base_probability,
        np.maximum(min_probability, base_probability - effective_decay * (ages - 1))
    )
    return ret_probs

def compute_temperature(it, n_iterations,
                        T0, TF,
                        schedule="linear"):
    frac = (it-1) / (n_iterations-1)
    if schedule == "linear":
        return T0 + (TF - T0) * frac
    elif schedule == "exponential":
        return T0 * (TF/T0) ** frac
    else:
        raise ValueError(f"Unknown annealing schedule: {schedule}")


config_path = os.path.join("config", "default.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

n_iterations                = int(config['iterative']['n_iterations'])
ret_base                    = float(config['iterative']['retention']['base_probability'])
ret_decay                   = float(config['iterative']['retention']['decay_rate'])
ret_min                     = float(config['iterative']['retention']['min_probability'])
control_coef                = float(config['iterative']['retention']['control_coef'])

scaler                      = config['data']['scaler']  
initial_sample_size         = int(config['data']['initial_sample_size'])
sample_size_increment       = int(config['data']['sample_size_increment'])
initial_sampling_strategy   = config['data']['initial_sampling_strategy']
iterative_sampling_strategy = config['data']['iterative_sampling_strategy']
offset_scale                = float(config['data']['offset_scale'])
nstd                        = float(config['data']['nstd'])
cov_scale                   = float(config['data']['cov_scale'])

n_steps                     = int(config['mcmc']['n_steps'])
n_steps_increment           = int(config['mcmc']['n_steps_increment'])
initial_temperature         = float(config['mcmc']['initial_temperature'])
final_temperature           = float(config['mcmc']['final_temperature'])
annealing_schedule          = config['mcmc']['annealing_schedule']
adapt_interval              = int(config['mcmc']['adapt_interval'])
cov_update_interval         = int(config['mcmc']['cov_update_interval'])
epsilon                     = float(config['mcmc']['epsilon'])
burn_in_fraction            = float(config['mcmc']['burn_in_fraction'])

trained_models_dir          = config['paths']['trained_models']
iterative_data_dir          = config['paths']['iterative_data']
mcmc_chains_dir             = config['paths']['mcmc_chains']
mcmc_bestfit_dir            = config['paths']['mcmc_bestfit']

np.random.seed(int(config['misc']['random_seed']))
tf.random.set_seed(int(config['misc']['random_seed']))

N = int(config['model']['dimension'])
#target_likelihood = GaussianLikelihood(N)

chains_dir = "class_chains/27_param"  # where your Planck .txt files live

# Instantiate the Planck‐based Gaussian likelihood
target_likelihood = PlanckGaussianLikelihood(
    N=N,
    chains_dir=chains_dir,
    skip_rows=500,            # burn-in rows to skip (optional)
    max_rows_per_file=None    # or an int if you only want a subset
)

cov_strategy = config['mcmc']['base_cov']
if cov_strategy == "target":
    base_cov = target_likelihood.Sigma
elif cov_strategy == "identity":
    base_cov = np.eye(N)
elif cov_strategy == "empirical":
    X0_s, _, scaler0_x, _ = generate_data(
        target_likelihood,
        num_samples=initial_sample_size,
        sampling_strategy=initial_sampling_strategy,
        offset_scale=offset_scale,
        nstd=nstd,
        cov_scale=cov_scale
    )
    X0 = scaler0_x.inverse_transform(X0_s)
    base_cov = np.cov(X0, rowvar=False)
else:
    raise ValueError(f"Unknown base_cov strategy: {cov_strategy}")

X_init_s, _, scaler_x, scaler_y = generate_data(
    target_likelihood,
    num_samples=initial_sample_size,
    scaler=scaler,
    sampling_strategy=initial_sampling_strategy,
    offset_scale=offset_scale,
    nstd=nstd,
    cov_scale=cov_scale
)
X_init = scaler_x.inverse_transform(X_init_s)
y_init = target_likelihood.neg_loglike(X_init)

training_sets = [(X_init, y_init)]
scalers       = [(scaler_x, scaler_y)]

aggregated_X  = X_init.copy()
aggregated_y  = y_init.copy()
sample_iters  = np.zeros(aggregated_X.shape[0], dtype=int)

os.makedirs(iterative_data_dir, exist_ok=True)

for it in range(1, n_iterations + 1):
    
    print(f"Iteration {it}")
    ages = it - sample_iters
    keep = np.random.rand(len(ages)) < retention_probability(
        ages, sample_iters, it,
        ret_base, ret_decay, ret_min, control_coef
    )
    aggregated_X = aggregated_X[keep]
    aggregated_y = aggregated_y[keep]
    sample_iters = sample_iters[keep]

    if scaler == "minmax":
        scaler_x = MinMaxScaler().fit(aggregated_X)
        scaler_y = MinMaxScaler().fit(aggregated_y.reshape(-1,1))
    elif scaler == "standard":
        scaler_x = StandardScaler().fit(aggregated_X)
        scaler_y = StandardScaler().fit(aggregated_y.reshape(-1,1))
    else:
        raise ValueError(f"Unknown scaler: {scaler}")

    scalers.append((scaler_x, scaler_y))
    X_scaled = scaler_x.transform(aggregated_X)
    y_scaled = scaler_y.transform(aggregated_y.reshape(-1,1))

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
    model.save(os.path.join(trained_models_dir, f"model_iter{it}.h5"))

    temp = compute_temperature(
        it,
        n_iterations,
        initial_temperature,
        final_temperature,
        annealing_schedule
    )
    
    initial_point = np.random.multivariate_normal(target_likelihood.mu, target_likelihood.Sigma)
    print(initial_point)
    
    os.makedirs(mcmc_chains_dir, exist_ok=True)
    os.makedirs(mcmc_bestfit_dir, exist_ok=True)
    chain_file   = os.path.join(mcmc_chains_dir,  f"chain_iter{it}.txt")
    bestfit_file = os.path.join(mcmc_bestfit_dir, f"best_iter{it}.txt")

    base_cov = adaptive_metropolis_hastings(
        keras_model=model,
        initial_point=initial_point,
        n_steps=n_steps,
        temperature=temp,
        base_cov=base_cov,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        adapt_interval=adapt_interval,
        cov_update_interval=cov_update_interval,
        chain_file=chain_file,
        bestfit_file=bestfit_file,
        epsilon=epsilon,
        burn_in_fraction=burn_in_fraction
    )

    n_steps += n_steps_increment
    new_size = initial_sample_size + (it-1)*sample_size_increment
    new_X = resample_chain_points(chain_file, new_size, temp, strategy=iterative_sampling_strategy)
    new_y = target_likelihood.neg_loglike(new_X)
    new_iters = np.full(new_X.shape[0], it, dtype=int)

    aggregated_X = np.vstack([aggregated_X, new_X])
    aggregated_y = np.concatenate([aggregated_y, new_y])
    sample_iters = np.concatenate([sample_iters, new_iters])

    training_sets.append((aggregated_X.copy(), aggregated_y.copy()))

    with open(os.path.join(iterative_data_dir, "training_sets.pkl"), "wb") as f:
        pickle.dump(training_sets, f)
    with open(os.path.join(iterative_data_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)