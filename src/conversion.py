"""Conversão de unidades de tempo e taxa para cálculos consistentes (base interna: segundo)."""

from __future__ import annotations

import math
import copy

# Duração de 1 "unidade nomeada" em segundos
SECONDS_PER = {
    "s": 1,
    "min": 60,
    "h": 3600,
    "dias": 86400,
}

RATE_UNIT_LABEL = {
    "s": "por segundo",
    "min": "por minuto",
    "h": "por hora",
    "dias": "por dia",
}

DURATION_UNIT_LABEL = {
    "s": "segundos",
    "min": "minutos",
    "h": "horas",
    "dias": "dias",
}


def rate_per_second(value: float, rate_period: str) -> float:
    """Taxa 'por minuto' etc. → eventos por segundo."""
    return value / SECONDS_PER[rate_period]


def duration_to_seconds(value: float, duration_unit: str) -> float:
    """Duração em min/h/... → segundos."""
    return value * SECONDS_PER[duration_unit]


def mu_per_second_from_mean_service(es: float, duration_unit: str) -> float:
    """E[S] na unidade dada → taxa μ (1/s)."""
    ds = duration_to_seconds(es, duration_unit)
    if ds <= 0:
        raise ValueError("E[S] deve ser positivo.")
    return 1.0 / ds


def variance_to_seconds_squared(sigma2: float, duration_unit: str) -> float:
    """Variância informada em (unidade)² → variância em s²."""
    s = SECONDS_PER[duration_unit]
    return sigma2 * (s**2)


def rate_in_period(rate_per_sec: float, period: str) -> float:
    """Taxa 1/s → 'por minuto', 'por hora', etc."""
    return rate_per_sec * SECONDS_PER[period]


def scale_result_times_for_display(res: dict, display_duration_unit: str) -> dict:
    """
    Converte W, Wq (de segundos) e λ̄ (de 1/s) para a unidade de duração escolhida
    nos rótulos (ex.: minutos): W_disp = W_s / seg_por_unidade, λ̄_disp = λ̄_s * seg_por_unidade.
    """
    sec = SECONDS_PER[display_duration_unit]
    out = copy.copy(res)

    def _scale_time(v):
        if v is None or v == float("inf") or (isinstance(v, float) and math.isnan(v)):
            return v
        return v / sec

    if "W" in out:
        out["W"] = _scale_time(out["W"])
    if "Wq" in out:
        out["Wq"] = _scale_time(out["Wq"])

    if "lam_eff" in out:
        le = out["lam_eff"]
        if le is not None and le != float("inf") and not (isinstance(le, float) and math.isnan(le)):
            out["lam_eff"] = le * sec

    return out


__all__ = [
    "SECONDS_PER",
    "RATE_UNIT_LABEL",
    "DURATION_UNIT_LABEL",
    "rate_per_second",
    "duration_to_seconds",
    "mu_per_second_from_mean_service",
    "variance_to_seconds_squared",
    "rate_in_period",
    "scale_result_times_for_display",
]
