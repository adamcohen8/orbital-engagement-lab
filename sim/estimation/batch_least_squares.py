from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from sim.estimation.covariance import correlation_from_covariance, covariance_from_jacobian
from sim.estimation.parameters import ParameterSet
from sim.estimation.weighting import (
    prepare_covariance_whitener,
    robust_weights,
    whiten_residual_with_factor,
)


@dataclass(frozen=True)
class BatchLeastSquaresResult:
    parameter_names: list[str]
    x_native: np.ndarray
    x_scaled: np.ndarray
    residual: np.ndarray
    cost: float
    initial_cost: float
    success: bool
    message: str
    nfev: int
    jacobian: np.ndarray | None = None
    covariance_native: np.ndarray | None = None
    raw_residual: np.ndarray | None = None
    residual_weights: np.ndarray | None = None
    rejected_indices: tuple[int, ...] = ()
    decision_records: tuple[dict[str, Any], ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def rms_residual(self) -> float:
        # ``residual`` is the solver objective and therefore contains zeros for
        # sigma-clipped components. Reporting must use the actual post-fit
        # residuals so rejected observations remain visible rather than
        # appearing to be exact fits.
        reported = self.raw_residual if self.raw_residual is not None else self.residual
        return float(np.sqrt(np.mean(np.asarray(reported, dtype=float) ** 2)))


def solve_batch_least_squares(
    parameters: ParameterSet,
    residual_fn_native: Callable[[np.ndarray], np.ndarray],
    *,
    max_nfev: int = 40,
    xtol: float = 1.0e-8,
    ftol: float = 1.0e-8,
    gtol: float = 1.0e-8,
    robust_loss: str = "linear",
    robust_f_scale: float = 1.0,
    sigma_clip_threshold: float | None = None,
    prior_mean_native: np.ndarray | None = None,
    prior_covariance_native: np.ndarray | None = None,
    prior_parameter_names: Sequence[str] | None = None,
) -> BatchLeastSquaresResult:
    """Solve a scaled nonlinear least-squares problem with auditable weighting."""

    x0_native = parameters.initial_native()
    x0_scaled = parameters.to_scaled(x0_native)
    lower_scaled = parameters.lower_scaled()
    upper_scaled = parameters.upper_scaled()

    loss_key = str(robust_loss or "linear").strip().lower()
    robust_weights(np.zeros(1), loss=loss_key, f_scale=robust_f_scale)
    clip_threshold = None if sigma_clip_threshold is None else float(sigma_clip_threshold)
    if clip_threshold is not None and (not np.isfinite(clip_threshold) or clip_threshold <= 0.0):
        raise ValueError("sigma_clip_threshold must be finite and positive when provided.")

    # SciPy commonly asks for the residual at the same point more than once
    # (initialization, termination, and post-solve reporting).  OD residuals
    # can materialize and propagate a complete deterministic scenario, so a
    # one-point exact cache avoids repeating that work without changing the
    # numerical path or reusing results at merely-nearby parameter vectors.
    residual_cache_key: bytes | None = None
    residual_cache_value: np.ndarray | None = None

    def data_residual_scaled(x_scaled: np.ndarray) -> np.ndarray:
        nonlocal residual_cache_key, residual_cache_value
        scaled = np.ascontiguousarray(np.asarray(x_scaled, dtype=float).reshape(-1))
        key = scaled.tobytes()
        if residual_cache_key == key and residual_cache_value is not None:
            return residual_cache_value.copy()
        value = np.asarray(residual_fn_native(parameters.to_native(scaled)), dtype=float).reshape(-1)
        residual_cache_key = key
        residual_cache_value = value.copy()
        return value

    prior_count = 0
    prior_indices = np.array([], dtype=int)
    prior_names: list[str] = []
    if (prior_mean_native is None) != (prior_covariance_native is None):
        raise ValueError("prior_mean_native and prior_covariance_native must be provided together.")
    if prior_mean_native is not None and prior_covariance_native is not None:
        prior_mean = np.asarray(prior_mean_native, dtype=float).reshape(-1)
        if prior_parameter_names is None:
            prior_names = list(parameters.names)
        else:
            prior_names = [str(name) for name in prior_parameter_names]
            if not prior_names or len(prior_names) != len(set(prior_names)):
                raise ValueError("prior_parameter_names must be non-empty and unique when provided.")
            unknown = sorted(set(prior_names) - set(parameters.names))
            if unknown:
                raise ValueError(f"Unknown prior parameter names: {unknown}.")
        prior_indices = np.array([parameters.names.index(name) for name in prior_names], dtype=int)
        if prior_mean.size != prior_indices.size:
            raise ValueError("prior_mean_native must match the selected prior parameter dimension.")
        prior_covariance = np.asarray(prior_covariance_native, dtype=float)
        prior_count = int(prior_mean.size)
        prior_factor = prepare_covariance_whitener(
            prior_covariance,
            dimension=prior_count,
            field_name="parameter prior covariance",
        )
    else:
        prior_mean = None
        prior_covariance = None
        prior_factor = None

    def residual_scaled(x_scaled: np.ndarray) -> np.ndarray:
        data = data_residual_scaled(x_scaled)
        if prior_mean is None or prior_covariance is None:
            return data
        assert prior_factor is not None
        prior = whiten_residual_with_factor(
            parameters.to_native(x_scaled)[prior_indices] - prior_mean,
            prior_factor,
            field_name="parameter prior covariance",
        )
        return np.concatenate((data, prior))

    initial_residual = residual_scaled(x0_scaled)
    initial_cost = 0.5 * float(np.dot(initial_residual, initial_residual))
    result = _run_least_squares(
        residual_scaled,
        x0_scaled=x0_scaled,
        lower_scaled=lower_scaled,
        upper_scaled=upper_scaled,
        max_nfev=max_nfev,
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        robust_loss=loss_key,
        robust_f_scale=float(robust_f_scale),
    )
    raw_first = residual_scaled(np.asarray(result.x, dtype=float))
    data_count = int(raw_first.size - prior_count)
    rejected = np.zeros(raw_first.size, dtype=bool)
    if clip_threshold is not None and data_count > 0:
        candidates = np.abs(raw_first[:data_count]) > clip_threshold
        if int(np.count_nonzero(~candidates)) >= int(x0_scaled.size):
            rejected[:data_count] = candidates
    if np.any(rejected):
        first_x = np.asarray(result.x, dtype=float)

        def clipped_residual(x_scaled: np.ndarray) -> np.ndarray:
            values = residual_scaled(x_scaled)
            values[rejected] = 0.0
            return values

        result = _run_least_squares(
            clipped_residual,
            x0_scaled=first_x,
            lower_scaled=lower_scaled,
            upper_scaled=upper_scaled,
            max_nfev=max_nfev,
            xtol=xtol,
            ftol=ftol,
            gtol=gtol,
            robust_loss=loss_key,
            robust_f_scale=float(robust_f_scale),
        )
    x_scaled = np.asarray(result.x, dtype=float)
    raw_residual = residual_scaled(x_scaled)
    residual = raw_residual.copy()
    residual[rejected] = 0.0
    weights = robust_weights(raw_residual, loss=loss_key, f_scale=float(robust_f_scale))
    weights[rejected] = 0.0
    jac_scaled = None if getattr(result, "jac", None) is None else np.asarray(result.jac, dtype=float)
    cov_native = None
    if jac_scaled is not None and residual.size >= x_scaled.size:
        cov_scaled = covariance_from_jacobian(jac_scaled, residual)
        scale = parameters.scales()
        cov_native = cov_scaled * np.outer(scale, scale)
    diagnostics = _batch_diagnostics(
        jacobian=jac_scaled,
        covariance_native=cov_native,
        x_scaled=x_scaled,
        lower_scaled=lower_scaled,
        upper_scaled=upper_scaled,
        parameters=parameters,
        raw_residual=raw_residual,
        rejected=rejected,
        data_count=data_count,
        prior_count=prior_count,
        prior_parameter_indices=prior_indices,
        prior_parameter_names=prior_names,
        robust_loss=loss_key,
        robust_f_scale=float(robust_f_scale),
        solver_success=bool(getattr(result, "success", False)),
    )
    decision_records = tuple(
        {
            "residual_index": int(index),
            "standardized_residual": float(raw_residual[index]),
            "robust_weight": float(weights[index]),
            "accepted": not bool(rejected[index]),
            "reasons": ["sigma_clip_threshold_exceeded"]
            if bool(rejected[index])
            else (["robust_loss_downweight"] if float(weights[index]) < 1.0 - 1.0e-12 else []),
        }
        for index in range(data_count)
    )
    return BatchLeastSquaresResult(
        parameter_names=parameters.names,
        x_native=parameters.to_native(x_scaled),
        x_scaled=x_scaled,
        residual=residual,
        cost=0.5 * float(np.dot(residual, residual)),
        initial_cost=initial_cost,
        success=bool(getattr(result, "success", False)),
        message=str(getattr(result, "message", "")),
        nfev=int(getattr(result, "nfev", 0)),
        jacobian=jac_scaled,
        covariance_native=cov_native,
        raw_residual=raw_residual,
        residual_weights=weights,
        rejected_indices=tuple(int(index) for index in np.flatnonzero(rejected[:data_count])),
        decision_records=decision_records,
        diagnostics=diagnostics,
    )


def _run_least_squares(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    *,
    x0_scaled: np.ndarray,
    lower_scaled: np.ndarray,
    upper_scaled: np.ndarray,
    max_nfev: int,
    xtol: float,
    ftol: float,
    gtol: float,
    robust_loss: str,
    robust_f_scale: float,
) -> Any:
    try:
        from scipy.optimize import least_squares  # type: ignore
    except Exception:

        def fallback_residual(x_scaled: np.ndarray) -> np.ndarray:
            raw = np.asarray(residual_fn(x_scaled), dtype=float).reshape(-1)
            weights = robust_weights(raw, loss=robust_loss, f_scale=robust_f_scale)
            return np.sqrt(weights) * raw

        return _solve_gauss_newton_fallback(
            fallback_residual,
            x0_scaled=x0_scaled,
            lower_scaled=lower_scaled,
            upper_scaled=upper_scaled,
            max_nfev=max_nfev,
            xtol=xtol,
        )
    return least_squares(
        residual_fn,
        x0_scaled,
        bounds=(lower_scaled, upper_scaled),
        max_nfev=max_nfev,
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        x_scale=np.ones_like(x0_scaled),
        loss=robust_loss,
        f_scale=robust_f_scale,
    )


def _batch_diagnostics(
    *,
    jacobian: np.ndarray | None,
    covariance_native: np.ndarray | None,
    x_scaled: np.ndarray,
    lower_scaled: np.ndarray,
    upper_scaled: np.ndarray,
    parameters: ParameterSet,
    raw_residual: np.ndarray,
    rejected: np.ndarray,
    data_count: int,
    prior_count: int,
    prior_parameter_indices: np.ndarray,
    prior_parameter_names: list[str],
    robust_loss: str,
    robust_f_scale: float,
    solver_success: bool,
) -> dict[str, Any]:
    parameter_count = int(x_scaled.size)
    if jacobian is None:
        data_jacobian = np.empty((0, parameter_count), dtype=float)
        augmented_jacobian = data_jacobian
    else:
        augmented_jacobian = np.asarray(jacobian, dtype=float)
        data_jacobian = augmented_jacobian[:data_count]
    data_rank, data_singular_values, data_condition = _svd_diagnostics(data_jacobian)
    augmented_rank, augmented_singular_values, augmented_condition = _svd_diagnostics(augmented_jacobian)
    bound_records = []
    for index, name in enumerate(parameters.names):
        tolerance = 1.0e-7 * max(1.0, abs(float(x_scaled[index])))
        at_lower = abs(float(x_scaled[index] - lower_scaled[index])) <= tolerance
        at_upper = abs(float(upper_scaled[index] - x_scaled[index])) <= tolerance
        if at_lower or at_upper:
            bound_records.append(
                {
                    "parameter": name,
                    "bound": "lower" if at_lower else "upper",
                    "scaled_value": float(x_scaled[index]),
                }
            )
    correlation = None
    if covariance_native is not None:
        try:
            correlation = correlation_from_covariance(covariance_native)
        except ValueError:
            correlation = None
    parameter_diagnostics = []
    prior_index_set = set(prior_parameter_indices.tolist())
    for index, name in enumerate(parameters.names):
        column_norm = (
            float(np.linalg.norm(data_jacobian[:, index]))
            if data_jacobian.ndim == 2 and data_jacobian.shape[0]
            else 0.0
        )
        max_abs_correlation = None
        if correlation is not None and correlation.shape == (parameter_count, parameter_count):
            peers = np.delete(np.abs(correlation[index]), index)
            max_abs_correlation = float(np.max(peers)) if peers.size else 0.0
        identifiable = bool(
            data_rank >= parameter_count
            and column_norm > 1.0e-10
            and (max_abs_correlation is None or max_abs_correlation < 0.9999)
        )
        parameter_diagnostics.append(
            {
                "parameter": name,
                "data_jacobian_column_norm": column_norm,
                "max_abs_correlation": max_abs_correlation,
                "identifiable": identifiable,
                "prior_dominated": bool(not identifiable and index in prior_index_set),
            }
        )
    active_data = np.asarray(raw_residual[:data_count], dtype=float)[~rejected[:data_count]]
    whiteness = _residual_whiteness(active_data)
    rejected_count = int(np.count_nonzero(rejected[:data_count]))
    rejected_fraction = float(rejected_count / data_count) if data_count else 0.0
    if not np.all(np.isfinite(raw_residual)):
        failure_classification = "numerical_failure"
    elif data_count and float(np.max(np.abs(raw_residual[:data_count]))) >= 1.0e8:
        failure_classification = "propagation_failure"
    elif data_rank < parameter_count:
        failure_classification = "non_observability"
    elif not solver_success:
        failure_classification = "solver_failure"
    elif bound_records:
        failure_classification = "bound_saturation"
    elif rejected_count >= 3 and rejected_fraction >= 0.2:
        failure_classification = "bad_data"
    elif rejected_count:
        failure_classification = "data_contamination_handled"
    else:
        failure_classification = "none"
    return {
        "schema_version": 1,
        "failure_classification": failure_classification,
        "parameter_scaling": [
            {"parameter": name, "native_scale": float(scale)}
            for name, scale in zip(parameters.names, parameters.scales(), strict=True)
        ],
        "prior": {
            "enabled": prior_count > 0,
            "residual_count": prior_count,
            "parameter_names": list(prior_parameter_names),
        },
        "robust_processing": {
            "loss": robust_loss,
            "f_scale": robust_f_scale,
            "rejected_residual_count": rejected_count,
            "rejected_residual_fraction": rejected_fraction,
            "accepted_residual_count": int(np.count_nonzero(~rejected[:data_count])),
        },
        "observability": {
            "parameter_count": parameter_count,
            "data_residual_count": data_count,
            "data_rank": data_rank,
            "data_full_rank": data_rank >= parameter_count,
            "data_condition_number": data_condition,
            "data_singular_values": data_singular_values,
            "augmented_rank": augmented_rank,
            "augmented_condition_number": augmented_condition,
            "augmented_singular_values": augmented_singular_values,
            "parameters": parameter_diagnostics,
        },
        "correlation_matrix": None if correlation is None else correlation.tolist(),
        "bound_activity": bound_records,
        "residual_whiteness": whiteness,
    }


def _svd_diagnostics(matrix: np.ndarray) -> tuple[int, list[float], float | None]:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)):
        return 0, [], None
    singular_values = np.linalg.svd(array, compute_uv=False)
    if singular_values.size == 0:
        return 0, [], None
    tolerance = max(array.shape) * np.finfo(float).eps * float(singular_values[0])
    rank = int(np.count_nonzero(singular_values > tolerance))
    smallest = float(singular_values[-1])
    condition = None if smallest <= tolerance else float(singular_values[0] / smallest)
    return rank, [float(value) for value in singular_values], condition


def _residual_whiteness(residual: np.ndarray) -> dict[str, Any]:
    values = np.asarray(residual, dtype=float).reshape(-1)
    if values.size < 2:
        return {
            "sample_count": int(values.size),
            "lag1_autocorrelation": None,
            "durbin_watson": None,
            "status": "insufficient_samples",
        }
    centered = values - float(np.mean(values))
    denominator = float(np.dot(centered, centered))
    lag1 = None if denominator <= 0.0 else float(np.dot(centered[:-1], centered[1:]) / denominator)
    raw_denominator = float(np.dot(values, values))
    durbin_watson = (
        None if raw_denominator <= 0.0 else float(np.dot(np.diff(values), np.diff(values)) / raw_denominator)
    )
    return {
        "sample_count": int(values.size),
        "lag1_autocorrelation": lag1,
        "durbin_watson": durbin_watson,
        "status": "computed",
    }


@dataclass
class _FallbackResult:
    x: np.ndarray
    fun: np.ndarray
    jac: np.ndarray
    success: bool
    message: str
    nfev: int


def _solve_gauss_newton_fallback(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    *,
    x0_scaled: np.ndarray,
    lower_scaled: np.ndarray,
    upper_scaled: np.ndarray,
    max_nfev: int,
    xtol: float,
) -> _FallbackResult:
    x = np.clip(np.asarray(x0_scaled, dtype=float), lower_scaled, upper_scaled)
    r = residual_fn(x)
    nfev = 1
    last_j = np.zeros((r.size, x.size), dtype=float)
    damping = 1.0e-3
    # A fallback iteration forms a finite-difference Jacobian, so it consumes
    # many residual evaluations compared with SciPy's trust-region step. Treat
    # max_nfev as the caller's iteration-scale budget and expand it into an
    # actual residual-evaluation ceiling for the finite-difference path.
    max_residual_evals = max(int(max_nfev), int(max_nfev) * (2 * int(x.size) + 1))
    for _ in range(max(max_nfev - 1, 0)):
        j = _finite_difference_jacobian(residual_fn, x, r)
        last_j = j
        nfev += 2 * x.size
        base_cost = float(np.dot(r, r))
        gradient = j.T @ r
        if float(np.linalg.norm(gradient, ord=np.inf)) <= xtol:
            return _FallbackResult(x=x, fun=r, jac=j, success=True, message="fallback gradient convergence", nfev=nfev)
        accepted = False
        try:
            gn_step, *_ = np.linalg.lstsq(j, -r, rcond=None)
        except np.linalg.LinAlgError:
            gn_step = np.zeros_like(x)
        alpha = 1.0
        for _ls in range(8):
            if float(np.linalg.norm(alpha * gn_step)) <= xtol * (xtol + float(np.linalg.norm(x))):
                return _FallbackResult(x=x, fun=r, jac=j, success=True, message="fallback step convergence", nfev=nfev)
            trial = np.clip(x + alpha * gn_step, lower_scaled, upper_scaled)
            r_trial = residual_fn(trial)
            nfev += 1
            if float(np.dot(r_trial, r_trial)) < base_cost:
                x, r = trial, r_trial
                accepted = True
                damping = max(damping * 0.3, 1.0e-12)
                break
            alpha *= 0.5
        if accepted:
            if nfev >= max_residual_evals:
                break
            continue
        jt_j = j.T @ j
        diag = np.maximum(np.diag(jt_j), 1.0)
        for _ls in range(12):
            lhs = jt_j + damping * np.diag(diag)
            try:
                step = np.linalg.solve(lhs, -gradient)
            except np.linalg.LinAlgError:
                step, *_ = np.linalg.lstsq(lhs, -gradient, rcond=None)
            if float(np.linalg.norm(step)) <= xtol * (xtol + float(np.linalg.norm(x))):
                return _FallbackResult(x=x, fun=r, jac=j, success=True, message="fallback step convergence", nfev=nfev)
            trial = np.clip(x + step, lower_scaled, upper_scaled)
            r_trial = residual_fn(trial)
            nfev += 1
            if float(np.dot(r_trial, r_trial)) < base_cost:
                x, r = trial, r_trial
                accepted = True
                damping = max(damping * 0.3, 1.0e-12)
                break
            damping = min(damping * 10.0, 1.0e12)
        if not accepted:
            return _FallbackResult(x=x, fun=r, jac=j, success=False, message="fallback damped step stalled", nfev=nfev)
        if nfev >= max_residual_evals:
            break
    return _FallbackResult(x=x, fun=r, jac=last_j, success=False, message="fallback max evaluations reached", nfev=nfev)


def _finite_difference_jacobian(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    r0: np.ndarray,
) -> np.ndarray:
    j = np.zeros((r0.size, x.size), dtype=float)
    for idx in range(x.size):
        step = max(1.0e-6, abs(float(x[idx])) * 1.0e-6)
        xp = x.copy()
        xm = x.copy()
        xp[idx] += step
        xm[idx] -= step
        rp = residual_fn(xp)
        rm = residual_fn(xm)
        j[:, idx] = (rp - rm) / (2.0 * step)
    return j
