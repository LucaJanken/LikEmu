import sys
import os
import numpy as np
import tensorflow as tf
import configparser

# Set seeds for reproducibility.
np.random.seed(42)
tf.random.set_seed(42)

# Update the Python path to include project directories.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import custom objects and modules.
from keras.models import load_model
from mcmc.sampler import adaptive_metropolis_hastings
from likelihoods.gaussian import GaussianLikelihood
from data.data_generator import generate_data
from models.activations import CustomTanh
from models.quadratic_decomposition import QuadraticDecomposition

# Read configuration from config/default.param.
config = configparser.ConfigParser()
config_path = os.path.join("config", "default.param")
config.read(config_path)

# Model parameters.
N = int(config['model']['dimension'])
num_hidden_layers = int(config['model']['num_hidden_layers'])
neurons = int(config['model']['neurons'])
activation = config['model']['activation']
use_gaussian_decomposition = config['model'].getboolean('use_gaussian_decomposition')
learning_rate = float(config['model'].get('learning_rate', 0.001))  # fallback if not defined

# Data parameters.
num_samples = int(config['data']['num_samples'])
sampling_strategy = config['data']['sampling_strategy']
offset_scale = float(config['data']['offset_scale'])
nstd = float(config['data']['nstd'])
cov_scale = float(config['data']['cov_scale'])

# MCMC parameters.
n_steps = int(config['mcmc']['n_steps'])
temperature = float(config['mcmc']['temperature'])
base_cov = config['mcmc']['base_cov']
adapt_interval = int(config['mcmc']['adapt_interval'])
cov_update_interval = int(config['mcmc']['cov_update_interval'])
epsilon = float(config['mcmc']['epsilon'])
chain_filename = config['mcmc']['chain_filename']
bestfit_filename = config['mcmc']['bestfit_filename']
burn_in_fraction = float(config['mcmc']['burn_in_fraction'])

# Create target likelihood. Because seeds are set, this will be reproducible.
target_likelihood = GaussianLikelihood(N)
if base_cov == "target":
    base_cov = target_likelihood.Sigma
else:
    base_cov = np.cov(np.random.randn(num_samples, N), rowvar=False)
print("Target likelihood covariance matrix:")
print(base_cov)

# Load the trained model.
model_path = os.path.join("trained_models", "model.h5")
custom_objects = {
    "CustomTanhActivation": CustomTanh,
    "QuadraticDecomposition": QuadraticDecomposition
}
model = load_model(model_path, custom_objects=custom_objects)
print("Model loaded successfully from", model_path)

# Generate training data to retrieve the scalers.
# Note: the number of samples here doesn't affect the likelihood, just the scalers.
x_train, y_train, scaler_x, scaler_y = generate_data(
    target_likelihood,
    num_samples=num_samples,
    sampling_strategy=sampling_strategy,
    offset_scale=offset_scale,
    nstd=nstd,
    cov_scale=cov_scale
)

# Set an initial point for the sampler.
# Sample an initial point from a multivariate normal distribution with the same covariance as the target likelihood.
initial_point = np.random.multivariate_normal(target_likelihood.mu, base_cov)

# Filenames to store the chain and best sample.
os.makedirs("mcmc/chains", exist_ok=True)
os.makedirs("mcmc/bestfit", exist_ok=True)
chain_file = f"mcmc/chains/{chain_filename}"
bestfit_file = f"mcmc/bestfit/{bestfit_filename}"

print("Starting MCMC sampling...")
adaptive_metropolis_hastings(
    keras_model=model,
    initial_point=initial_point,
    n_steps=n_steps,
    temperature=temperature,
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
print("MCMC sampling complete.")
print(f"Chain saved to {chain_file}")
print(f"Best sample saved to {bestfit_file}")
