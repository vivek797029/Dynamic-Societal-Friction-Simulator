"""Scenario -> intervention on the friction tensors -> forecast delta.

Given a `Scenario` and a baseline pipeline context (the trained Stage B
aggregator + forecasting head, plus the E/T/R tensors the pipeline was
computing on), `simulate()` perturbs the inputs at the effective week and
returns per-state, per-cleavage, per-horizon deltas vs the baseline.

The perturbation rule is intentionally simple and documented -- policy
scenarios aren't ground-truth labeled, so this is closer to a calibrated
sensitivity analysis than a learned counterfactual. For each selected state
and each selected cleavage we add an impulse of magnitude `severity` to the
E channel (new events) and a matching impulse to T (media discourse) for a
short window starting at `effective_week`. A decay kernel matching the
aggregator's exponential memory handles spillover -- we don't re-train.

This is a correctness + UX module, not a research claim. The paper-grade
counterfactual evaluation lives in `evaluation/` and uses
difference-in-differences on actual historical interventions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

try:
    import torch                                          # type: ignore
    _HAS_TORCH = True
except ImportError:                                       # pragma: no cover
    torch = None                                          # type: ignore
    _HAS_TORCH = False

from ..data.india_geo import STATES, STATE_TO_IDX
from ..models.cleavage_classifier import CLEAVAGES
from .scenario import Scenario


# --------------------------- data containers --------------------------- #

@dataclass
class PipelineContext:
    """Everything the simulator needs at inference time.

    In production you'll build this once per server start-up from the
    `artifacts/stage_b/stage_b.pt` checkpoint. For tests / dry-runs you can
    pass a minimal context with placeholder tensors -- see
    `PipelineContext.dry_run()`.
    """
    E: np.ndarray                                 # [S, T, K]
    T: np.ndarray                                 # [S, T, K]
    R: np.ndarray | None                          # [S, T, K] or None
    min_week: int                                 # abs iso_week of t=0
    aggregator: Any | None = None                 # FrictionAggregator or None for dry-run
    head: Any | None = None                       # EscalationHead or None for dry-run
    horizons: tuple[int, ...] = (1, 2, 4)

    @property
    def S(self) -> int: return self.E.shape[0]
    @property
    def T_len(self) -> int: return self.E.shape[1]
    @property
    def K(self) -> int: return self.E.shape[-1]

    @classmethod
    def dry_run(cls, S: int = 36, T: int = 520, K: int | None = None,
                min_week: int = 0, seed: int = 0) -> "PipelineContext":
        """Placeholder context for UI / tests with no trained model."""
        rng = np.random.default_rng(seed)
        K = K or len(CLEAVAGES)
        E = rng.gamma(1.5, 1.0, size=(S, T, K)).astype(np.float32)
        Tt = rng.gamma(1.2, 1.0, size=(S, T, K)).astype(np.float32)
        R = rng.gamma(1.0, 1.0, size=(S, T, K)).astype(np.float32)
        return cls(E=E, T=Tt, R=R, min_week=min_week,
                   aggregator=None, head=None)


@dataclass
class ForecastResult:
    """Output of one forecasting pass (baseline or intervention).

    rates :    [S, H, 2] log-lambda / log-mu per state x horizon x target.
               For the dry-run path this is synthesized from the tensor
               statistics directly so UX can be exercised without torch.
    """
    rates: np.ndarray                             # [S, H, 2]


@dataclass
class SimulationResult:
    scenario: Scenario
    effective_week: int
    baseline: ForecastResult
    intervention: ForecastResult
    # Convenience view of (intervention - baseline), [S, H, 2].
    delta: np.ndarray
    # Pre-aggregator friction change per (state, horizon_offset_weeks, cleavage)
    # for the choropleth heatmap. [S, ΔT, K] with ΔT short window.
    friction_delta: np.ndarray

    def state_delta_ranked(self, horizon_idx: int = 0, target_idx: int = 0
                           ) -> list[tuple[str, float]]:
        """Return (state, delta_log_rate) sorted most-positive first."""
        d = self.delta[:, horizon_idx, target_idx]
        order = np.argsort(-d)
        return [(STATES[i], float(d[i])) for i in order]

    def top_cleavages_per_state(self, state: str, k: int = 3
                                ) -> list[tuple[str, float]]:
        i = STATE_TO_IDX[state]
        # sum over the delta window
        per_k = self.friction_delta[i].sum(axis=0)          # [K]
        order = np.argsort(-per_k)[:k]
        return [(CLEAVAGES[j], float(per_k[j])) for j in order]

    def to_dict(self) -> dict:
        out = {
            "scenario": self.scenario.to_dict(),
            "effective_week": int(self.effective_week),
            "baseline_rates_shape": list(self.baseline.rates.shape),
            "intervention_rates_shape": list(self.intervention.rates.shape),
            "delta_summary": {
                "max": float(self.delta.max()),
                "min": float(self.delta.min()),
                "mean": float(self.delta.mean()),
            },
        }
        return out


# ---------------------------- perturbation ----------------------------- #

def _states_to_indices(scenario: Scenario) -> list[int]:
    if not scenario.affected_states:
        return list(range(len(STATES)))                    # nationwide
    return [STATE_TO_IDX[s] for s in scenario.affected_states]


def _cleavages_to_indices(scenario: Scenario) -> list[int]:
    if not scenario.cleavages:
        return list(range(len(CLEAVAGES)))
    return [CLEAVAGES.index(c) for c in scenario.cleavages]


def apply_perturbation(ctx: PipelineContext, scenario: Scenario,
                       impulse_weeks: int = 4,
                       decay: float = 0.7) -> tuple[np.ndarray, np.ndarray,
                                                    np.ndarray | None, int]:
    """Return (E', T', R', t_effective) -- clones of the context tensors
    with an impulse added to the selected (state, cleavage, t) cells."""
    E = ctx.E.copy()
    Tt = ctx.T.copy()
    R = ctx.R.copy() if ctx.R is not None else None

    if scenario.effective_week is None:
        # Default: perturb the last slot available.
        t_eff_rel = ctx.T_len - impulse_weeks - 1
    else:
        t_eff_rel = int(scenario.effective_week) - ctx.min_week
    t_eff_rel = max(0, min(t_eff_rel, ctx.T_len - 1))

    states = _states_to_indices(scenario)
    cleavs = _cleavages_to_indices(scenario)
    severity = float(scenario.severity)

    # Scale impulse by per-state-per-cleavage baseline std so the perturbation
    # is on the same scale as naturally-occurring shocks.
    for offset in range(impulse_weeks):
        t = t_eff_rel + offset
        if t >= ctx.T_len:
            break
        weight = severity * (decay ** offset)
        for s in states:
            for k in cleavs:
                sd_e = max(ctx.E[s, :, k].std(), 1e-3)
                sd_t = max(ctx.T[s, :, k].std(), 1e-3)
                E[s, t, k] += weight * sd_e
                Tt[s, t, k] += weight * sd_t
                if R is not None:
                    sd_r = max(ctx.R[s, :, k].std(), 1e-3)
                    R[s, t, k] += 0.3 * weight * sd_r
    return E, Tt, R, t_eff_rel


# ---------------------------- forecasting ------------------------------ #

def _forecast_trained(ctx: PipelineContext, E: np.ndarray, T: np.ndarray,
                      R: np.ndarray | None, t_end_rel: int) -> ForecastResult:
    """Run the trained Stage B model on the perturbed tensors.

    We look at the horizon ending at `t_end_rel + window_weeks - 1`. Uses
    `build_windows` under the hood for consistency with training.
    """
    from ..models.forecasting_head import build_windows           # lazy
    if not _HAS_TORCH or ctx.aggregator is None or ctx.head is None:
        return _forecast_dry(ctx, E, T, R)
    with torch.no_grad():
        Et = torch.from_numpy(E).float()
        Tt = torch.from_numpy(T).float()
        Rt = torch.from_numpy(R).float() if R is not None else None
        F_k, F_agg = ctx.aggregator(Et, Tt, Rt)
        Xk, Xag, t_ends = build_windows(F_k, F_agg,
                                        window_len=ctx.aggregator.cfg.window_weeks)
        pred = ctx.head(Xk, Xag)                              # [N, H, 2]
        # Take the slot whose window ends at (or just past) the intervention.
        N_t = len(t_ends)
        target_idx = min(range(N_t), key=lambda i: abs(t_ends[i] - t_end_rel))
        # [S, N_t, H, 2]  (we reshape back from build_windows' flattening)
        pred4 = pred.view(ctx.S, N_t, pred.shape[-2], pred.shape[-1])
        rates = pred4[:, target_idx].detach().cpu().numpy()
    return ForecastResult(rates=rates)


def _forecast_dry(ctx: PipelineContext, E: np.ndarray, T: np.ndarray,
                  R: np.ndarray | None) -> ForecastResult:
    """Fallback when no trained model is available: produce a deterministic
    function of the input tensors so UI work can proceed."""
    H = len(ctx.horizons)
    # Use a smoothed last-window summary as a proxy for log-rate.
    w = min(12, E.shape[1])
    recent = E[:, -w:, :].mean(axis=(1, 2))                   # [S]
    recent_T = T[:, -w:, :].mean(axis=(1, 2))                 # [S]
    base = 0.5 * recent + 0.5 * recent_T                      # [S]
    # Shape to [S, H, 2]. Fatalities get a smaller slope than protests.
    protests = np.outer(base, np.linspace(1.0, 1.3, H))        # [S, H]
    fatalities = np.outer(base * 0.25, np.linspace(1.0, 1.2, H))
    rates = np.stack([protests, fatalities], axis=-1).astype(np.float32)
    return ForecastResult(rates=rates)


# ----------------------------- main entry ------------------------------ #

def simulate(ctx: PipelineContext, scenario: Scenario,
             impulse_weeks: int = 4) -> SimulationResult:
    """End-to-end: baseline forecast + perturbed forecast -> delta bundle."""
    # Baseline: run on the unperturbed tensors, ending at the intervention.
    t_rel_eff = (int(scenario.effective_week) - ctx.min_week
                 if scenario.effective_week is not None else ctx.T_len - impulse_weeks - 1)
    t_rel_eff = max(0, min(t_rel_eff, ctx.T_len - 1))
    baseline = _forecast_trained(ctx, ctx.E, ctx.T, ctx.R, t_rel_eff)

    # Intervention: perturb and re-forecast.
    Ei, Ti, Ri, t_eff = apply_perturbation(ctx, scenario, impulse_weeks)
    intervention = _forecast_trained(ctx, Ei, Ti, Ri, t_eff)

    # Pre-aggregator friction delta window [S, impulse_weeks, K].
    end = min(t_eff + impulse_weeks, ctx.T_len)
    friction_delta = (Ei[:, t_eff:end, :] - ctx.E[:, t_eff:end, :]) \
        + (Ti[:, t_eff:end, :] - ctx.T[:, t_eff:end, :])

    return SimulationResult(
        scenario=scenario,
        effective_week=t_eff + ctx.min_week,
        baseline=baseline,
        intervention=intervention,
        delta=intervention.rates - baseline.rates,
        friction_delta=friction_delta,
    )
