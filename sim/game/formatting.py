from __future__ import annotations

import math

import numpy as np

DEFAULT_SIG_FIGS = 4


def format_distance_km(value_km: float, *, sig_figs: int = DEFAULT_SIG_FIGS) -> str:
    value = float(value_km)
    if not np.isfinite(value):
        return "--"
    magnitude = abs(value)
    if magnitude >= 1.0:
        return f"{_format_sigfig(value, sig_figs=sig_figs)} km"
    if magnitude >= 1.0e-3:
        return f"{_format_sigfig(value * 1000.0, sig_figs=sig_figs)} m"
    return f"{_format_sigfig(value * 1.0e6, sig_figs=sig_figs)} mm"


def format_speed_km_s(value_km_s: float, *, sig_figs: int = DEFAULT_SIG_FIGS) -> str:
    value = float(value_km_s)
    if not np.isfinite(value):
        return "--"
    return format_speed_m_s(value * 1000.0, sig_figs=sig_figs)


def format_speed_m_s(value_m_s: float, *, sig_figs: int = DEFAULT_SIG_FIGS) -> str:
    value = float(value_m_s)
    if not np.isfinite(value):
        return "--"
    magnitude = abs(value)
    if magnitude >= 1000.0:
        return f"{_format_sigfig(value / 1000.0, sig_figs=sig_figs)} km/s"
    if magnitude >= 1.0:
        return f"{_format_sigfig(value, sig_figs=sig_figs)} m/s"
    return f"{_format_sigfig(value * 1000.0, sig_figs=sig_figs)} mm/s"


def format_scalar(value: float, *, sig_figs: int = DEFAULT_SIG_FIGS) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "--"
    return _format_sigfig(value, sig_figs=sig_figs)


def _format_sigfig(value: float, *, sig_figs: int = DEFAULT_SIG_FIGS) -> str:
    sig_figs = max(int(sig_figs), 1)
    value = float(value)
    if not np.isfinite(value):
        return "--"
    if value == 0.0:
        return "0"
    decimals = max(sig_figs - int(math.floor(math.log10(abs(value)))) - 1, 0)
    return f"{value:.{decimals}f}"
