"""Private pure-array routines shared by Python and optional Numba dispatch."""

import numpy as np

_EQUATOR_GRADIENT = np.array([1, 1j, 0])
_POLAR_GRADIENT = np.array([0, 0, 1])
_EQUATOR_GRADIENT.setflags(write=False)
_POLAR_GRADIENT.setflags(write=False)


def solid_harmonics_numeric(position, normalization, gradients=True):
    x, y, z = position
    r2 = float(position @ position)
    degree = len(normalization) - 1
    h = np.zeros((degree + 1, degree + 1), np.complex128)
    g = np.zeros((degree + 1, degree + 1, 3), np.complex128) if gradients else None
    h[0, 0] = 1
    for m in range(degree + 1):
        if m:
            h[m, m] = (2 * m - 1) * complex(x, y) * h[m - 1, m - 1]
            if gradients:
                g[m, m] = (2 * m - 1) * (complex(x, y) * g[m - 1, m - 1] + _EQUATOR_GRADIENT * h[m - 1, m - 1])
        if m < degree:
            h[m + 1, m] = (2 * m + 1) * z * h[m, m]
            if gradients:
                g[m + 1, m] = (2 * m + 1) * (z * g[m, m] + _POLAR_GRADIENT * h[m, m])
        for n in range(m + 2, degree + 1):
            h[n, m] = ((2 * n - 1) * z * h[n - 1, m] - (n + m - 1) * r2 * h[n - 2, m]) / (n - m)
            if gradients:
                g[n, m] = (
                    (2 * n - 1) * (z * g[n - 1, m] + _POLAR_GRADIENT * h[n - 1, m])
                    - (n + m - 1) * (r2 * g[n - 2, m] + 2 * position * h[n - 2, m])
                ) / (n - m)
    for n, row in enumerate(normalization):
        for m in range(n + 1):
            scale = row[m]
            h[n, m] *= scale
            if gradients:
                g[n, m] *= scale
    return h, g
