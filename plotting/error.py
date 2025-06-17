#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Import your custom activation ---
# adjust this to where `custom_tanh` is defined:
from models.activations import CustomTanh
from models.quadratic_decomposition import QuadraticDecomposition

# --- Paths (adjust if needed) ---
data_dir        = "iterative_data"
models_dir      = "trained_models"
training_file   = os.path.join(data_dir, "training_sets.pkl")
scalers_file    = os.path.join(data_dir, "scalers.pkl")
first_model     = os.path.join(models_dir, "model_iter1.h5")

# --- Load initial data and scalers ---
with open(training_file, "rb") as f:
    training_sets = pickle.load(f)
X_init, y_true = training_sets[0]

with open(scalers_file, "rb") as f:
    scalers = pickle.load(f)
scaler_x, scaler_y = scalers[0]

# --- Prepare inputs ---
X_scaled = scaler_x.transform(X_init)

# --- Load model with custom_objects ---
model = load_model(
    first_model,
    compile=False,
    custom_objects={"CustomTanh": CustomTanh,
                    "QuadraticDecomposition": QuadraticDecomposition}
)

# --- Predict and invert scaling ---
y_scaled_pred = model.predict(X_scaled, batch_size=128, verbose=1)
y_pred = scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1)).flatten()

# --- Compute error ---
error = y_pred - y_true
relative_error = error / np.abs(y_true)

# --- Plot absolute error ---
plt.figure(figsize=(8,6))
plt.scatter(y_true, error, alpha=0.6, s=10)
plt.axhline(0, linestyle="--", linewidth=0.8)
plt.xlabel(r"True $-\log\mathcal{L}$")
plt.ylabel("Prediction error (predicted − true)")
plt.title("Iteration 1 Network Error vs True $-\\log\\mathcal{L}$")
plt.grid(linestyle=":")
plt.tight_layout()
plt.savefig("plots/initial_network_error.png", dpi=300)
plt.show()

# --- Plot relative error ---
plt.figure(figsize=(8,6))
plt.scatter(y_true, relative_error, alpha=0.6, s=10)
plt.axhline(0, linestyle="--", linewidth=0.8)
plt.xlabel(r"True $-\log\mathcal{L}$")
plt.ylabel("Relative error (predicted − true) / true")
plt.title("Iteration 1 Network Relative Error vs True $-\\log\\mathcal{L}$")
plt.grid(linestyle=":")
plt.tight_layout()
plt.savefig("plots/initial_network_relative_error.png", dpi=300)
plt.show()
