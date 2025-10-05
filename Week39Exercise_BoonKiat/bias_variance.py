import numpy as np

def bias_variance_curves(x):
    """Return illustrative bias^2, variance, noise, and total error curves."""
    bias2 = 0.6 * (1 - x)**2 + 0.05
    variance = 0.7 * x**2 + 0.02
    noise = np.full_like(x, 0.12)
    total_error = bias2 + variance + noise
    return bias2, variance, noise, total_error
