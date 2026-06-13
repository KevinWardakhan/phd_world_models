"""2D toy distributions.

pi_0: isotropic Gaussian source.
pi_1: 8-mode checkerboard target on a 4x4 grid (cells where (col + row) is even),
      the classic 2D checkerboard pattern.
"""

import numpy as np

# 4x4 grid; keep cells where (col + row) is even -> 8 cells in a checkerboard.
_GRID_CELLS = [(i, j) for i in range(4) for j in range(4) if (i + j) % 2 == 0]


def sample_pi0(n, rng, std=1.0):
    """Isotropic 2D Gaussian, mean 0."""
    return rng.normal(loc=0.0, scale=std, size=(n, 2))


def mode_centers(side=1.0, gap=0.5):
    """Centers of the 8 checkerboard modes.

    Cells sit on a 4x4 grid centered at the origin with pitch (side + gap), so
    the squares are separated and the modes stay distinct for metrics.
    """
    pitch = side + gap
    return np.array(
        [((i - 1.5) * pitch, (j - 1.5) * pitch) for (i, j) in _GRID_CELLS],
        dtype=float,
    )


def sample_pi1(n, rng, side=1.0, gap=0.5):
    """8-mode checkerboard with equal weight per mode.

    Each sample picks one of the 8 squares uniformly, then a uniform point
    inside that square.
    """
    centers = mode_centers(side=side, gap=gap)
    idx = rng.integers(0, len(centers), size=n)
    offsets = rng.uniform(-side / 2.0, side / 2.0, size=(n, 2))
    return centers[idx] + offsets
