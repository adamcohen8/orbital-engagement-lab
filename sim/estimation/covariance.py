from __future__ import annotations

import numpy as np


def covariance_from_jacobian(jacobian: np.ndarray, residual: np.ndarray, *, dof: int | None = None) -> np.ndarray:
    """Approximate parameter covariance from a whitened residual Jacobian."""

    j = np.asarray(jacobian, dtype=float)
    r = np.asarray(residual, dtype=float).reshape(-1)
    if j.ndim != 2:
        raise ValueError("jacobian must be a 2D array.")
    if j.shape[0] != r.size:
        raise ValueError("jacobian row count must match residual length.")
    n_params = j.shape[1]
    degrees = int(dof if dof is not None else max(r.size - n_params, 1))
    variance = max(float(np.dot(r, r) / max(degrees, 1)), 1.0)
    normal = j.T @ j
    return variance * np.linalg.pinv(normal)


def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    diag = np.sqrt(np.maximum(np.diag(cov), 0.0))
    denom = np.outer(diag, diag)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0.0)
    np.fill_diagonal(corr, 1.0)
    return corr
