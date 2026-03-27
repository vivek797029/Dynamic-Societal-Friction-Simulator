"""Comprehensive tests for evaluation metrics."""

import json
import tempfile
from pathlib import Path

import pytest

from src.evaluation.metrics import (
    compute_cascade_frequency,
    compute_cognitive_dissonance_metrics,
    compute_convergence_rate,
    compute_crossdomain_correlation,
    compute_election_metrics,
    compute_emotional_contagion_metrics,
    compute_friction_volatility,
    compute_ideology_metrics,
    compute_overton_window_metrics,
    compute_polarization_index,
    compute_polarization_trend,
    evaluate_simulation_run,
    generate_report,
)


# ========================================================================
# TESTS: Friction Volatility
# ========================================================================


class TestFrictionVolatility:
    """Tests for friction volatility computation."""

    def test_volatility_empty_list(self):
        """Test volatility with empty history."""
        volatility = compute_friction_volatility([])
        assert volatility == 0.0

    def test_volatility_single_value(self):
        """Test volatility with single value."""
        volatility = compute_friction_volatility([0.5])
        assert volatility == 0.0

    def test_volatility_stable_values(self):
        """Test volatility with stable (constant) values."""
        friction_history = [0.5, 0.5, 0.5, 0.5]
        volatility = compute_friction_volatility(friction_history)
        assert volatility == 0.0, "Constant values should have zero volatility"

    def test_volatility_increasing_trend(self):
        """Test volatility with increasing trend."""
        friction_history = [0.1, 0.2, 0.3, 0.4, 0.5]
        volatility = compute_friction_volatility(friction_history)
        assert volatility > 0.0, "Non-constant trend should have non-zero volatility"

    def test_volatility_oscillating(self):
        """Test volatility with oscillating values."""
        friction_history = [0.1, 0.9, 0.1, 0.9, 0.1]
        volatility = compute_friction_volatility(friction_history)
        assert volatility > 0.0

    def test_volatility_known_input(self):
        """Test volatility with known input."""
        friction_history = [0.0, 0.5, 1.0]
        volatility = compute_friction_volatility(friction_history)
        assert volatility > 0.0


# ========================================================================
# TESTS: Polarization Index
# ========================================================================


class TestPolarizationIndex:
    """Tests for polarization index computation."""

    def test_polarization_empty_groups(self):
        """Test polarization with empty group scores."""
        polarization = compute_polarization_index({})
        assert polarization == 0.0

    def test_polarization_single_group(self):
        """Test polarization with single group."""
        group_scores = {"Group1": [0.1, 0.2, 0.3, 0.4]}
        polarization = compute_polarization_index(group_scores)
        assert polarization >= 0.0

    def test_polarization_identical_scores(self):
        """Test polarization when all groups have same final score."""
        group_scores = {
            "Group1": [0.5],
            "Group2": [0.5],
            "Group3": [0.5],
        }
        polarization = compute_polarization_index(group_scores)
        assert polarization == 0.0, "Identical scores should have zero polarization"

    def test_polarization_disparate_scores(self):
        """Test polarization when groups have different scores."""
        group_scores = {
            "Group1": [0.1],
            "Group2": [0.5],
            "Group3": [0.9],
        }
        polarization = compute_polarization_index(group_scores)
        assert polarization > 0.0

    def test_polarization_with_histories(self):
        """Test polarization uses final score in each group's history."""
        group_scores = {
            "Group1": [0.2, 0.3, 0.1],  # Final: 0.1
            "Group2": [0.8, 0.7, 0.9],  # Final: 0.9
        }
        polarization = compute_polarization_index(group_scores)
        assert polarization > 0.0


# ========================================================================
# TESTS: Convergence Rate
# ========================================================================


class TestConvergenceRate:
    """Tests for convergence rate computation."""

    def test_convergence_too_few_points(self):
        """Test convergence with insufficient data."""
        history = [0.5, 0.5]
        convergence = compute_convergence_rate(history)
        assert convergence == 0.0, "Insufficient data should return 0"

    def test_convergence_resolving(self):
        """Test convergence when friction is decreasing (resolving)."""
        history = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1]
        convergence = compute_convergence_rate(history)
        assert convergence > 0.0, "Decreasing friction should show positive convergence"

    def test_convergence_escalating(self):
        """Test convergence when friction is increasing (escalating)."""
        history = [0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        convergence = compute_convergence_rate(history)
        assert convergence < 0.0, "Increasing friction should show negative convergence"

    def test_convergence_stable(self):
        """Test convergence when friction is stable."""
        history = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        convergence = compute_convergence_rate(history)
        assert convergence == 0.0, "Stable friction should show zero convergence"


# ========================================================================
# TESTS: Cascade Frequency
# ========================================================================


class TestCascadeFrequency:
    """Tests for cascade frequency computation."""

    def test_cascade_empty_events(self):
        """Test cascade frequency with no events."""
        frequency = compute_cascade_frequency([])
        assert frequency == 0.0

    def test_cascade_single_event(self):
        """Test cascade frequency with single event."""
        events = [{"severity": 0.5}]
        frequency = compute_cascade_frequency(events)
        assert frequency == 0.0, "Single event cannot have cascade"

    def test_cascade_no_cascades(self):
        """Test when no cascades occur."""
        events = [
            {"severity": 0.3},
            {"severity": 0.3},
            {"severity": 0.3},
        ]
        frequency = compute_cascade_frequency(events)
        assert frequency == 0.0, "No severity increases should show no cascades"

    def test_cascade_with_cascades(self):
        """Test when cascades occur (severity increase by 20%+)."""
        events = [
            {"severity": 0.5},
            {"severity": 0.7},  # 40% increase - cascade
            {"severity": 0.5},
            {"severity": 0.75},  # 50% increase - cascade
        ]
        frequency = compute_cascade_frequency(events)
        assert frequency > 0.0, "Events with severity jumps should show cascades"


# ========================================================================
# TESTS: Polarization Trend
# ========================================================================


class TestPolarizationTrend:
    """Tests for polarization trend computation."""

    def test_trend_too_few_points(self):
        """Test trend with insufficient data."""
        polarization = [0.1, 0.2]
        trend = compute_polarization_trend(polarization)
        assert trend == 0.0

    def test_trend_increasing(self):
        """Test increasing polarization trend."""
        polarization = [0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        trend = compute_polarization_trend(polarization)
        assert trend > 0.0, "Increasing polarization should show positive trend"

    def test_trend_decreasing(self):
        """Test decreasing polarization trend."""
        polarization = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1]
        trend = compute_polarization_trend(polarization)
        assert trend < 0.0, "Decreasing polarization should show negative trend"

    def test_trend_stable(self):
        """Test stable polarization trend."""
        polarization = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        trend = compute_polarization_trend(polarization)
        assert trend == 0.0


# ========================================================================
# TESTS: Ideology Metrics
# ========================================================================


class TestIdeologyMetrics:
    """Tests for ideology distribution metrics."""

    def test_ideology_empty_snapshots(self):
        """Test ideology metrics with no snapshots."""
        metrics = compute_ideology_metrics([])
        assert metrics["spread"] == 0.0
        assert metrics["extreme_ratio"] == 0.0
        assert metrics["avg_drift"] == 0.0

    def test_ideology_single_snapshot(self):
        """Test ideology metrics with single snapshot."""
        snapshots = [
            {
                "agents": [
                    {"agent_id": "a1", "ideology": -0.5},
                    {"agent_id": "a2", "ideology": 0.5},
                ],
            }
        ]
        metrics = compute_ideology_metrics(snapshots)
        assert metrics["spread"] > 0.0
        assert metrics["extreme_ratio"] >= 0.0

    def test_ideology_extreme_positions(self):
        """Test extreme agent ratio detection."""
        snapshots = [
            {
                "agents": [
                    {"agent_id": "a1", "ideology": -0.9},
                    {"agent_id": "a2", "ideology": 0.9},
                    {"agent_id": "a3", "ideology": 0.0},
                ],
            }
        ]
        metrics = compute_ideology_metrics(snapshots)
        assert metrics["extreme_ratio"] > 0.0

    def test_ideology_drift_calculation(self):
        """Test ideology drift between snapshots."""
        snapshots = [
            {
                "agents": [
                    {"agent_id": "a1", "ideology": 0.0},
                    {"agent_id": "a2", "ideology": 0.0},
                ],
            },
            {
                "agents": [
                    {"agent_id": "a1", "ideology": 0.3},
                    {"agent_id": "a2", "ideology": -0.3},
                ],
            },
        ]
        metrics = compute_ideology_metrics(snapshots)
        assert metrics["avg_drift"] > 0.0


# ========================================================================
# TESTS: Election Metrics
# ========================================================================


class TestElectionMetrics:
    """Tests for election-related metrics."""

    def test_election_no_elections(self):
        """Test election metrics with no elections."""
        metrics = compute_election_metrics([])
        assert metrics["count"] == 0
        assert metrics["avg_margin"] == 0.0

    def test_election_single_election(self):
        """Test metrics for single election."""
        elections = [
            {
                "margin": "55%",
            }
        ]
        metrics = compute_election_metrics(elections)
        assert metrics["count"] == 1
        assert 0.0 <= metrics["avg_margin"] <= 1.0

    def test_election_competitive(self):
        """Test metrics for competitive elections."""
        elections = [
            {"margin": "51%"},
            {"margin": "52%"},
        ]
        metrics = compute_election_metrics(elections)
        assert metrics["avg_margin"] > 0.0
        assert metrics["avg_margin"] < 0.6


# ========================================================================
# TESTS: Cognitive Dissonance Metrics
# ========================================================================


class TestCognitiveDisssonanceMetrics:
    """Tests for cognitive dissonance metrics."""

    def test_dissonance_empty_history(self):
        """Test dissonance with empty history."""
        avg_dissonance = compute_cognitive_dissonance_metrics([])
        assert avg_dissonance == 0.0

    def test_dissonance_single_snapshot(self):
        """Test dissonance with single snapshot."""
        history = [{"avg_dissonance": 0.4}]
        avg_dissonance = compute_cognitive_dissonance_metrics(history)
        assert avg_dissonance == 0.4

    def test_dissonance_multiple_snapshots(self):
        """Test dissonance averaging."""
        history = [
            {"avg_dissonance": 0.3},
            {"avg_dissonance": 0.5},
            {"avg_dissonance": 0.7},
        ]
        avg_dissonance = compute_cognitive_dissonance_metrics(history)
        assert 0.3 < avg_dissonance < 0.7
        assert abs(avg_dissonance - 0.5) < 0.01


# ========================================================================
# TESTS: Overton Window Metrics
# ========================================================================


class TestOwertonWindowMetrics:
    """Tests for Overton Window metrics."""

    def test_overton_empty_history(self):
        """Test Overton Window metrics with empty history."""
        width, shift = compute_overton_window_metrics([])
        assert width == 2.0
        assert shift == 0.0

    def test_overton_single_snapshot(self):
        """Test Overton Window with single snapshot."""
        history = [{"width": 1.5, "center": 0.0}]
        width, shift = compute_overton_window_metrics(history)
        assert width == 1.5
        assert shift == 0.0

    def test_overton_window_narrowing(self):
        """Test detection of narrowing window."""
        history = [
            {"width": 2.0, "center": 0.0},
            {"width": 1.5, "center": 0.1},
        ]
        width, shift = compute_overton_window_metrics(history)
        assert width == 1.5


# ========================================================================
# TESTS: Emotional Contagion Metrics
# ========================================================================


class TestEmotionalContagionMetrics:
    """Tests for emotional contagion metrics."""

    def test_contagion_empty_history(self):
        """Test contagion with empty history."""
        r0, epidemics = compute_emotional_contagion_metrics([])
        assert r0 == 0.0
        assert epidemics == 0

    def test_contagion_single_snapshot(self):
        """Test contagion with single snapshot."""
        history = [
            {
                "r0_by_emotion": {"angry": 1.2, "fearful": 0.8},
                "epidemic_count": 1,
            }
        ]
        r0, epidemics = compute_emotional_contagion_metrics(history)
        assert r0 > 0.0
        assert epidemics >= 0

    def test_contagion_epidemic_tracking(self):
        """Test epidemic count tracking."""
        history = [
            {"r0_by_emotion": {}, "epidemic_count": 2},
            {"r0_by_emotion": {}, "epidemic_count": 3},
            {"r0_by_emotion": {}, "epidemic_count": 1},
        ]
        r0, epidemics = compute_emotional_contagion_metrics(history)
        assert epidemics == 3, "Should return max epidemic count"


# ========================================================================
# TESTS: Cross-Domain Correlation
# ========================================================================


class TestCrossDomainCorrelation:
    """Tests for cross-domain correlation metrics."""

    def test_correlation_insufficient_data(self):
        """Test correlation with insufficient data."""
        correlation = compute_crossdomain_correlation([0.1], [0.2])
        assert correlation == 0.0

    def test_correlation_perfect_positive(self):
        """Test perfect positive correlation."""
        social = [0.1, 0.2, 0.3, 0.4, 0.5]
        political = [0.1, 0.2, 0.3, 0.4, 0.5]
        correlation = compute_crossdomain_correlation(social, political)
        assert abs(correlation - 1.0) < 0.01

    def test_correlation_perfect_negative(self):
        """Test perfect negative correlation."""
        social = [0.1, 0.2, 0.3, 0.4, 0.5]
        political = [0.5, 0.4, 0.3, 0.2, 0.1]
        correlation = compute_crossdomain_correlation(social, political)
        assert correlation < 0.0

    def test_correlation_no_correlation(self):
        """Test zero correlation."""
        social = [0.1, 0.1, 0.1, 0.1, 0.1]
        political = [0.5, 0.5, 0.5, 0.5, 0.5]
        correlation = compute_crossdomain_correlation(social, political)
        # When both are constant, correlation is NaN or 0
        assert correlation == 0.0 or correlation != correlation


# ========================================================================
# TESTS: Full Evaluation
# ========================================================================


class TestSimulationEvaluation:
    """Tests for full simulation evaluation."""

    def test_evaluate_missing_directory(self):
        """Test evaluation with non-existent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = evaluate_simulation_run(tmpdir)
            # Should return default metrics
            assert metrics.friction_volatility >= 0.0

    def test_evaluate_with_metrics_file(self):
        """Test evaluation with metrics file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create mock metrics file
            metrics_data = [
                {
                    "step": 0,
                    "global_friction": 0.3,
                    "political_friction": 0.2,
                    "polarization_index": 0.1,
                    "group_scores": {"Group1": 0.3},
                    "faction_scores": {"Faction1": 0.2},
                    "event_domain": "social",
                },
                {
                    "step": 1,
                    "global_friction": 0.35,
                    "political_friction": 0.25,
                    "polarization_index": 0.15,
                    "group_scores": {"Group1": 0.35},
                    "faction_scores": {"Faction1": 0.25},
                    "event_domain": "political",
                },
            ]

            metrics_file = tmpdir_path / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics_data, f)

            metrics = evaluate_simulation_run(tmpdir)
            assert metrics.friction_volatility >= 0.0


# ========================================================================
# TESTS: Report Generation
# ========================================================================


class TestReportGeneration:
    """Tests for evaluation report generation."""

    def test_generate_report_creates_file(self):
        """Test that report generation creates file."""
        from src.evaluation.metrics import SimulationMetrics

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"

            metrics = SimulationMetrics()
            report = generate_report(metrics, output_path=str(output_path))

            assert output_path.exists()

    def test_report_structure(self):
        """Test report has expected structure."""
        from src.evaluation.metrics import SimulationMetrics

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"

            metrics = SimulationMetrics(
                friction_volatility=0.15,
                polarization_index=0.25,
                convergence_rate=0.1,
            )
            report = generate_report(metrics, output_path=str(output_path))

            assert "social_dynamics" in report
            assert "political_dynamics" in report
            assert "cross_domain" in report
            assert "elections" in report
            assert "cognitive_models" in report
            assert "interpretation" in report

    def test_report_interpretation_fields(self):
        """Test that report includes interpretation."""
        from src.evaluation.metrics import SimulationMetrics

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"

            metrics = SimulationMetrics()
            report = generate_report(metrics, output_path=str(output_path))

            interp = report["interpretation"]
            assert "social_volatility" in interp
            assert "polarization" in interp
            assert "radicalization_risk" in interp
