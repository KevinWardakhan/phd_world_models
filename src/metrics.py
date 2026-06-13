"""Sample-based metrics for the 2D toy problem.

M0 uses the mode-coverage metrics below to sanity-check the target sampler.
The two-sample distances (energy distance, MMD) are implemented and verified
here, but their final use is in M1+.
"""

import numpy as np


def assign_modes(samples, centers):
    """Nearest-center assignment. Returns an index per sample."""
    d2 = ((samples[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1)


def mode_fractions(samples, centers):
    """Fraction of samples assigned to each mode."""
    idx = assign_modes(samples, centers)
    counts = np.bincount(idx, minlength=len(centers))
    return counts / counts.sum()


def recovered_mode_count(samples, centers, threshold=0.01):
    """
    Number of modes holding at least `threshold` of the mass.
    Useful to check for mode dropping.
    """
    fracs = mode_fractions(samples, centers)
    return int((fracs >= threshold).sum())


def normalized_mode_entropy(samples, centers):
    """Entropy of the mode distribution, normalized to [0, 1].

    1.0 means perfectly balanced coverage, lower values indicate mode dropping.
    """
    fracs = mode_fractions(samples, centers)
    nz = fracs[fracs > 0]
    entropy = -(nz * np.log(nz)).sum()
    return float(entropy / np.log(len(centers)))


def off_support_fraction(samples, centers, side=1.0):
    """
    Fraction of samples outside every checkerboard square.
    Useful to check for mode dropping.

    A sample is on-support if it lies within the side x side box of its
    nearest mode center.
    """
    idx = assign_modes(samples, centers)
    nearest = centers[idx]
    inside = (np.abs(samples - nearest) <= side / 2.0).all(axis=1)
    return float((~inside).mean())


def gaussian_moments(samples):
    """Empirical mean and covariance, for the pi_0 sanity check."""
    return {
        "mean": samples.mean(axis=0).tolist(),
        "cov": np.cov(samples, rowvar=False).tolist(),
    }


def _pairwise_sq_dists(a, b):
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)


def energy_distance(x, y):
    """Energy distance between two sample sets."""
    dxy = np.sqrt(_pairwise_sq_dists(x, y)).mean()
    dxx = np.sqrt(_pairwise_sq_dists(x, x)).mean()
    dyy = np.sqrt(_pairwise_sq_dists(y, y)).mean()
    return float(2.0 * dxy - dxx - dyy)


def mmd_rbf(x, y, sigma=1.0):
    """Squared MMD with an RBF kernel."""
    gamma = 1.0 / (2.0 * sigma ** 2)
    kxx = np.exp(-gamma * _pairwise_sq_dists(x, x)).mean()
    kyy = np.exp(-gamma * _pairwise_sq_dists(y, y)).mean()
    kxy = np.exp(-gamma * _pairwise_sq_dists(x, y)).mean()
    return float(kxx + kyy - 2.0 * kxy)


def evaluate_samples(samples, reference, centers, side=1.0):
    """Bundle the sample-based metrics used to compare a generator to pi_1.

    `samples` are evaluated for mode coverage against the known `centers`, and
    compared to `reference` (true pi_1 draws) via the two-sample distances.
    Reused across M1-M4.
    """
    return {
        "recovered_mode_count": recovered_mode_count(samples, centers),
        "normalized_mode_entropy": normalized_mode_entropy(samples, centers),
        "off_support_fraction": off_support_fraction(samples, centers, side=side),
        "energy_distance": energy_distance(samples, reference),
        "mmd_rbf": mmd_rbf(samples, reference),
    }
