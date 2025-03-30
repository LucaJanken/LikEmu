#!/usr/bin/env python3
import sys
import os
import re
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt
from keras.models import load_model

# Set seeds for reproducibility.
np.random.seed(44)
tf.random.set_seed(44)

# Import custom objects.
from models.activations import CustomTanh
from models.quadratic_decomposition import QuadraticDecomposition
from likelihoods.gaussian import GaussianLikelihood
from data.data_generator import generate_data

# Load configuration from YAML.
config_path = os.path.join("config", "default.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Model parameters.
N = int(config['model']['dimension'])
num_hidden_layers = int(config['model']['num_hidden_layers'])
neurons = int(config['model']['neurons'])
activation = config['model']['activation']
use_gaussian_decomposition = config['model'].getboolean('use_gaussian_decomposition')

# Data parameters.
num_samples = int(config['data']['num_samples'])
sampling_strategy = config['data']['sampling_strategy']
offset_scale = float(config['data']['offset_scale'])
nstd = float(config['data']['nstd'])
cov_scale = float(config['data']['cov_scale'])

# Load the trained model.
model_path = os.path.join("trained_models", "model_iter1.h5")
custom_objects = {
    "CustomTanh": CustomTanh,
    "QuadraticDecomposition": QuadraticDecomposition
}
model = load_model(model_path, custom_objects=custom_objects)
print("Model loaded successfully from", model_path)

# Parse the iteration number from the model filename.
# For example, from "model_iter1.h5" we extract iteration 1.
iteration_match = re.search(r"model_iter(\d+)\.h5", os.path.basename(model_path))
if iteration_match:
    model_iteration = int(iteration_match.group(1))
    # Assuming scalers list: index 0 is initial, index 1 corresponds to iteration 1, etc.
    scaler_index = model_iteration
else:
    print("Could not determine model iteration from filename. Defaulting to last scalers.")
    scaler_index = -1

# Load scalers from iterative_data.
scalers_file = os.path.join("iterative_data", "scalers.pkl")
if os.path.exists(scalers_file):
    with open(scalers_file, "rb") as f:
         scalers_list = pickle.load(f)
    if scaler_index < len(scalers_list):
        scaler_x, scaler_y = scalers_list[scaler_index]
    else:
        print(f"Scaler for iteration {model_iteration} not found. Using last available scalers.")
        scaler_x, scaler_y = scalers_list[-1]
else:
    print(f"{scalers_file} not found. Generating new scalers from training data.")
    # Fallback: generate scalers (this may not exactly match training).
    _, _, scaler_x, scaler_y = generate_data(
        GaussianLikelihood(N),
        num_samples=num_samples,
        sampling_strategy=sampling_strategy,
        offset_scale=offset_scale,
        nstd=nstd,
        cov_scale=cov_scale
    )

# Choose which two dimensions to visualize (default dims 0 and 1).
plot_dims = [0, 1]

# Recover the original training inputs (for grid domain determination).
# Here we generate temporary training data and invert the scaling.
x_train_scaled_temp, _, _, _ = generate_data(
    GaussianLikelihood(N),
    num_samples=num_samples,
    sampling_strategy=sampling_strategy,
    offset_scale=offset_scale,
    nstd=nstd,
    cov_scale=cov_scale
)
x_train_orig = scaler_x.inverse_transform(x_train_scaled_temp)
x1_min, x1_max = np.min(x_train_orig[:, plot_dims[0]]), np.max(x_train_orig[:, plot_dims[0]])
x2_min, x2_max = np.min(x_train_orig[:, plot_dims[1]]), np.max(x_train_orig[:, plot_dims[1]])

# Optionally, expand the grid domain a little bit.
padding1 = 2 * (x1_max - x1_min)
padding2 = 2 * (x2_max - x2_min)
x1_min -= padding1
x1_max += padding1
x2_min -= padding2
x2_max += padding2

# Create a grid over the chosen two dimensions.
num_points = 100
x1 = np.linspace(x1_min, x1_max, num_points)
x2 = np.linspace(x2_min, x2_max, num_points)
X1, X2 = np.meshgrid(x1, x2)
grid_partial = np.stack([X1.ravel(), X2.ravel()], axis=-1)  # shape (num_points^2, 2)

# For full-dimensional evaluation, fix the remaining dimensions at target_likelihood.mu.
target_likelihood = GaussianLikelihood(N)
full_grid = np.tile(target_likelihood.mu, (grid_partial.shape[0], 1))
full_grid[:, plot_dims[0]] = grid_partial[:, 0]
full_grid[:, plot_dims[1]] = grid_partial[:, 1]

# Compute the true negative log-likelihood on the full grid.
true_vals = target_likelihood.neg_loglike(full_grid)
true_vals = true_vals.reshape(X1.shape)

# Compute the network's predicted function.
# First, scale the full grid points using the loaded scaler.
full_grid_scaled = scaler_x.transform(full_grid)
predicted_scaled = model.predict(full_grid_scaled)
if predicted_scaled.ndim > 2:
    predicted_scaled = np.squeeze(predicted_scaled, axis=-1)
predicted = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1))
predicted = predicted.reshape(X1.shape)

# Create contour plots for the true and emulated functions.
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# True function contour.
cs1 = axs[0].contourf(X1, X2, true_vals, levels=100, cmap='viridis')
axs[0].set_title('True Negative Log-Likelihood')
axs[0].set_xlabel(f"Dim {plot_dims[0] + 1}")
axs[0].set_ylabel(f"Dim {plot_dims[1] + 1}")
fig.colorbar(cs1, ax=axs[0])

# Emulated function contour.
cs2 = axs[1].contourf(X1, X2, predicted, levels=100, cmap='viridis')
axs[1].set_title('Emulated Negative Log-Likelihood')
axs[1].set_xlabel(f"Dim {plot_dims[0] + 1}")
axs[1].set_ylabel(f"Dim {plot_dims[1] + 1}")
fig.colorbar(cs2, ax=axs[1])

# Difference between the two functions.
cs3 = axs[2].contourf(X1, X2, predicted - true_vals, levels=100, cmap='viridis')
axs[2].set_title('Difference: Emulated - True')
axs[2].set_xlabel(f"Dim {plot_dims[0] + 1}")
axs[2].set_ylabel(f"Dim {plot_dims[1] + 1}")
fig.colorbar(cs3, ax=axs[2])

plt.suptitle("2D Contour Comparison: True vs Emulated Gaussian Negative Log-Likelihood")
plt.savefig("plots/heatmap_true_vs_emulated.png")
plt.show()
