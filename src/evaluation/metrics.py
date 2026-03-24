"""
Evaluation Metrics for the Dynamic Society Friction Simulator.

Measures both model quality (generation coherence, factual grounding)
and simulation quality (realistic friction dynamics, political polarization,
emergent behaviors, and cross-domain cascade effects).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for evaluating the fine-tuned LLM quality."""
    perplexity: float = 0.0
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    diversity_score: float = 0.0
    toxicity_score: float = 0.0


@dataclass
class SimulationMetrics:
    """Metrics for evaluating simulation realism and dynamics."""
    # Social friction
    friction_volatility: float = 0.0
    convergence_rate: float = 0.0
    cascade_frequency: float = 0.0
    resolution_rate: float = 0.0
    emergent_behavior_count: int = 0

    # Political polarization
    polarization_index: float = 0.0
    polarization_trend: float = 0.0           # positive = increasing polarization
    ideology_spread: float = 0.0              # std dev of ideology positions
    faction_dominance: float = 0.0            # how uneven faction sizes are
    extreme_agent_ratio: float = 0.0          # % of agents at ideology extremes
    ideology_drift_magnitude: float = 0.0     # avg total drift per agent

    # Cross-domain
    crossover_event_ratio: float = 0.0        # fraction of events that span domains
    political_social_correlation: float = 0.0 # correlation between social/political friction

    # Elections
    election_count: int = 0
    election_competitiveness: float = 0.0     # avg margin (lower = more competitive)
    post_election_friction_spike: float = 0.0


def compute_friction_volatility(friction_history: list[float]) -> float:
    """Measure how much friction scores fluctuate over time."""
    if len(friction_history) < 2:
        return 0.0
    diffs = np.diff(friction_history)
    return float(np.std(diffs))


def compute_polarization_index(group_scores: dict[str, list[float]]) -> float:
    """
    Measure polarization: how far apart group friction scores are.
    Higher = more polarized society.
    """
    if not group_scores:
        return 0.0
    final_scores = [scores[-1] for scores in group_scores.values() if scores]
    if len(final_scores) < 2:
        return 0.0
    return float(np.std(final_scores))


def compute_convergence_rate(friction_history: list[float]) -> float:
    """Measure whether friction is trending toward resolution over time."""
    if len(friction_history) < 10:
        return 0.0
    first_half = np.mean(friction_history[: len(friction_history) // 2])
    second_half = np.mean(friction_history[len(friction_history) // 2 :])
    if first_half == 0:
        return 0.0
    return float((first_half - second_half) / first_half)


def compute_cascade_frequency(events: list[dict]) -> float:
    """Count how often friction events trigger follow-on events."""
    if not events:
        return 0.0
    cascade_count = 0
    for i in range(1, len(events)):
        prev_severity = events[i - 1].get("severity", 0)
        curr_severity = events[i].get("severity", 0)
        if curr_severity > prev_severity * 1.2:
            cascade_count += 1
    return cascade_count / len(events)


def compute_polarization_trend(polarization_history: list[float]) -> float:
    """Measure whether polarization is increasing or decreasing over time."""
    if len(polarization_history) < 10:
        return 0.0
    first_q = np.mean(polarization_history[: len(polarization_history) // 4])
    last_q = np.mean(polarization_history[-len(polarization_history) // 4 :])
    return float(last_q - first_q)


def compute_ideology_metrics(ideology_snapshots: list[dict]) -> dict:
    """Analyze ideology distribution changes over the simulation."""
    if not ideology_snapshots:
        return {"spread": 0.0, "extreme_ratio": 0.0, "avg_drift": 0.0}

    # Final snapshot
    final = ideology_snapshots[-1].get("agents", [])
    positions = [a["ideology"] for a in final]

    if not positions:
        return {"spread": 0.0, "extreme_ratio": 0.0, "avg_drift": 0.0}

    arr = np.array(positions)
    spread = float(np.std(arr))
    extreme_ratio = float(np.mean(np.abs(arr) > 0.7))

    # Compute drift: compare first and last snapshots
    avg_drift = 0.0
    if len(ideology_snapshots) >= 2:
        first = ideology_snapshots[0].get("agents", [])
        first_map = {a["agent_id"]: a["ideology"] for a in first}
        drifts = []
        for a in final:
            if a["agent_id"] in first_map:
                drifts.append(abs(a["ideology"] - first_map[a["agent_id"]]))
        if drifts:
            avg_drift = float(np.mean(drifts))

    return {"spread": spread, "extreme_ratio": extreme_ratio, "avg_drift": avg_drift}


def compute_election_metrics(elections: list[dict]) -> dict:
    """Analyze election competitiveness and friction effects."""
    if not elections:
        return {"count": 0, "avg_margin": 0.0}

    margins = []
    for e in elections:
        margin_str = e.get("margin", "0%")
        try:
            margin_val = float(margin_str.replace("%", "")) / 100
        except (ValueError, AttributeError):
            margin_val = 0.5
        margins.append(margin_val)

    return {
        "count": len(elections),
        "avg_margin": float(np.mean(margins)),
    }


def compute_crossdomain_correlation(
    social_history: list[float], political_history: list[float]
) -> float:
    """Measure correlation between social and political friction trajectories."""
    if len(social_history) < 5 or len(political_history) < 5:
        return 0.0
    min_len = min(len(social_history), len(political_history))
    corr = np.corrcoef(social_history[:min_len], political_history[:min_len])
    return float(corr[0, 1])


def evaluate_simulation_run(results_dir: str) -> SimulationMetrics:
    """Run full evaluation on a completed simulation."""
    results_path = Path(results_dir)
    metrics = SimulationMetrics()

    # Load metrics history
    metrics_file = results_path / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            history = json.load(f)

        # Social friction analysis
        friction_history = [m["global_friction"] for m in history]
        metrics.friction_volatility = compute_friction_volatility(friction_history)
        metrics.convergence_rate = compute_convergence_rate(friction_history)

        group_histories: dict[str, list[float]] = {}
        for m in history:
            for group, score in m.get("group_scores", {}).items():
                group_histories.setdefault(group, []).append(score)
        metrics.polarization_index = compute_polarization_index(group_histories)

        # Political friction analysis
        pol_friction_history = [m.get("political_friction", 0) for m in history]
        polarization_history = [m.get("polarization_index", 0) for m in history]

        if any(p > 0 for p in polarization_history):
            metrics.polarization_trend = compute_polarization_trend(polarization_history)

        # Cross-domain analysis
        crossover_count = sum(1 for m in history if m.get("event_domain") == "crossover")
        metrics.crossover_event_ratio = crossover_count / len(history) if history else 0.0
        metrics.political_social_correlation = compute_crossdomain_correlation(
            friction_history, pol_friction_history
        )

    # Load events
    events_file = results_path / "events.json"
    if events_file.exists():
        with open(events_file) as f:
            events = json.load(f)
        metrics.cascade_frequency = compute_cascade_frequency(events)

    # Load ideology shifts
    ideology_file = results_path / "ideology_shifts.json"
    if ideology_file.exists():
        with open(ideology_file) as f:
            ideology_snapshots = json.load(f)
        ideo_metrics = compute_ideology_metrics(ideology_snapshots)
        metrics.ideology_spread = ideo_metrics["spread"]
        metrics.extreme_agent_ratio = ideo_metrics["extreme_ratio"]
        metrics.ideology_drift_magnitude = ideo_metrics["avg_drift"]

    # Load elections
    elections_file = results_path / "elections.json"
    if elections_file.exists():
        with open(elections_file) as f:
            elections = json.load(f)
        election_m = compute_election_metrics(elections)
        metrics.election_count = election_m["count"]
        metrics.election_competitiveness = 1.0 - election_m["avg_margin"]

    logger.info(f"Simulation evaluation complete: {metrics}")
    return metrics


def generate_report(
    metrics: SimulationMetrics, output_path: str = "outputs/results/eval_report.json"
):
    """Generate a structured evaluation report."""
    report = {
        "social_dynamics": {
            "friction_volatility": metrics.friction_volatility,
            "convergence_rate": metrics.convergence_rate,
            "cascade_frequency": metrics.cascade_frequency,
        },
        "political_dynamics": {
            "polarization_index": metrics.polarization_index,
            "polarization_trend": metrics.polarization_trend,
            "ideology_spread": metrics.ideology_spread,
            "extreme_agent_ratio": metrics.extreme_agent_ratio,
            "ideology_drift_magnitude": metrics.ideology_drift_magnitude,
        },
        "cross_domain": {
            "crossover_event_ratio": metrics.crossover_event_ratio,
            "political_social_correlation": metrics.political_social_correlation,
        },
        "elections": {
            "election_count": metrics.election_count,
            "competitiveness": metrics.election_competitiveness,
        },
        "interpretation": {
            "social_volatility": "high" if metrics.friction_volatility > 0.1 else "stable",
            "social_trend": "resolving" if metrics.convergence_rate > 0 else "escalating",
            "polarization": (
                "severe" if metrics.polarization_index > 0.3
                else "moderate" if metrics.polarization_index > 0.15
                else "mild"
            ),
            "polarization_direction": (
                "increasing" if metrics.polarization_trend > 0.05
                else "decreasing" if metrics.polarization_trend < -0.05
                else "stable"
            ),
            "radicalization_risk": (
                "high" if metrics.extreme_agent_ratio > 0.3
                else "moderate" if metrics.extreme_agent_ratio > 0.15
                else "low"
            ),
            "cross_domain_coupling": (
                "strong" if metrics.political_social_correlation > 0.6
                else "moderate" if metrics.political_social_correlation > 0.3
                else "weak"
            ),
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to {output_path}")
    return report
