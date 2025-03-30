# plotting/plot_utils.py

import numpy as np

def compute_levels(H, levels=[0.68, 0.95]):
    """
    Compute density contour levels corresponding to the given cumulative probability thresholds.
    
    Parameters:
      H (ndarray): 2D histogram array (density).
      levels (list): List of cumulative probability thresholds (e.g., [0.68, 0.95]).
    
    Returns:
      list: Sorted list of histogram value thresholds that enclose the given probability mass.
    """
    H_flat = H.flatten()
    sorted_idx = np.argsort(H_flat)[::-1]
    H_sorted = H_flat[sorted_idx]
    cumsum = np.cumsum(H_sorted)
    cumsum /= cumsum[-1]  # Normalize cumulative sum.
    contour_levels = []
    for lev in levels:
        # Find the smallest histogram value where the cumulative sum exceeds the threshold.
        threshold = H_sorted[np.where(cumsum >= lev)[0][0]]
        contour_levels.append(threshold)
    return sorted(contour_levels)