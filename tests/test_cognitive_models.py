"""Tests for cognitive models."""

import pytest
import networkx as nx

from src.agents.social_agent import PoliticalProfile, SocialAgent
from src.model.cognitive_models import (
    CognitiveDissonanceTracker,
    DissonanceResolutionStrategy,
    OvertonWindowTracker,
    EmotionalContagionModel,
)


# ========================================================================
# TESTS: CognitiveDissonanceTracker
# ========================================================================


class TestCognitiveDissonanceTracker:
    """Tests for the cognitive dissonance tracker."""

    def test_tracker_creation(self):
        """Test tracker initialization."""
        tracker = CognitiveDissonanceTracker()
        assert tracker.dissonance_scores == {}
        assert tracker.resolution_history == {}
        assert tracker.agent_compartmentalization == {}

    def test_compute_dissonance_no_values(self):
        """Test dissonance computation with no agent values."""
        tracker = CognitiveDissonanceTracker()
        agent = SocialAgent(
            agent_id="a1",
            name="Test",
            group="Group1",
            core_values=[],
            openness_to_change=0.5,
            politics=PoliticalProfile(),
        )
        faction_policies = {"policy1": 0.5}

        dissonance = tracker.compute_dissonance(agent, faction_policies)
        assert dissonance == 0.0

    def test_compute_dissonance_aligned_values(self):
        """Test dissonance when values align with policies."""
        tracker = CognitiveDissonanceTracker()
        agent = SocialAgent(
            agent_id="a1",
            name="Test",
            group="Group1",
            core_values=["equality", "freedom"],
            openness_to_change=0.5,
            politics=PoliticalProfile(partisan_strength=0.7),
        )
        # Policies strongly support agent's values
        faction_policies = {"equality": 1.0, "freedom": 1.0}

        dissonance = tracker.compute_dissonance(agent, faction_policies)
        assert 0.0 <= dissonance <= 1.0
        assert dissonance < 0.5, "Aligned values should have low dissonance"

    def test_compute_dissonance_conflicting_values(self):
        """Test dissonance when values conflict with policies."""
        tracker = CognitiveDissonanceTracker()
        agent = SocialAgent(
            agent_id="a1",
            name="Test",
            group="Group1",
            core_values=["equality", "freedom"],
            openness_to_change=0.5,
            politics=PoliticalProfile(partisan_strength=0.8),
        )
        # Policies oppose agent's values
        faction_policies = {"equality": 0.0, "freedom": 0.0}

        dissonance = tracker.compute_dissonance(agent, faction_policies)
        assert 0.0 <= dissonance <= 1.0
        assert dissonance > 0.3, "Conflicting values should have higher dissonance"

    def test_track_dissonance(self):
        """Test tracking dissonance over time."""
        tracker = CognitiveDissonanceTracker()

        tracker.track_dissonance("agent1", 0.3)
        tracker.track_dissonance("agent1", 0.4)
        tracker.track_dissonance("agent1", 0.35)

        assert len(tracker.dissonance_scores["agent1"]) == 3
        assert tracker.dissonance_scores["agent1"] == [0.3, 0.4, 0.35]

    def test_resolve_dissonance_low_score(self):
        """Test resolution with low dissonance."""
        tracker = CognitiveDissonanceTracker()
        agent = SocialAgent(
            agent_id="a1",
            name="Test",
            group="Group1",
            core_values=["value1"],
            openness_to_change=0.5,
            politics=PoliticalProfile(partisan_strength=0.5),
        )

        result = tracker.resolve_dissonance(agent, 0.2)
        assert result.agent_id == "a1"
        assert result.dissonance_score == 0.2

    def test_resolve_dissonance_high_score(self):
        """Test resolution with high dissonance."""
        tracker = CognitiveDissonanceTracker()
        agent = SocialAgent(
            agent_id="a1",
            name="Test",
            group="Group1",
            core_values=["value1"],
            openness_to_change=0.5,
            politics=PoliticalProfile(partisan_strength=0.7),
        )

        factions = [
            {"name": "Faction1"},
            {"name": "Faction2"},
        ]

        result = tracker.resolve_dissonance(agent, 0.7, available_factions=factions)
        assert result.agent_id == "a1"
        assert result.dissonance_score == 0.7
        # High dissonance should trigger resolution attempt
        assert result.resolution_strategy is not None or result.success


# ========================================================================
# TESTS: OvertonWindowTracker
# ========================================================================


class TestOwertonWindowTracker:
    """Tests for the Overton Window tracker."""

    def test_tracker_creation(self):
        """Test tracker initialization."""
        tracker = OvertonWindowTracker()
        assert tracker.window_history == []

    def test_track_window_basic(self):
        """Test basic window tracking."""
        tracker = OvertonWindowTracker()

        agents = [
            SocialAgent(
                agent_id="a1",
                name="A1",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
                politics=PoliticalProfile(ideology_position=-0.6),
            ),
            SocialAgent(
                agent_id="a2",
                name="A2",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
                politics=PoliticalProfile(ideology_position=0.4),
            ),
        ]

        snapshot = tracker.track_window(agents)
        assert snapshot is not None
        assert hasattr(snapshot, "left_edge")
        assert hasattr(snapshot, "right_edge")
        assert hasattr(snapshot, "width")

    def test_track_window_empty_agents(self):
        """Test window tracking with no agents."""
        tracker = OvertonWindowTracker()
        snapshot = tracker.track_window([])
        assert snapshot is not None

    def test_window_width_history(self):
        """Test width history tracking."""
        tracker = OvertonWindowTracker()

        agents1 = [
            SocialAgent(
                agent_id="a1",
                name="A1",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
                politics=PoliticalProfile(ideology_position=-0.5),
            ),
        ]

        agents2 = [
            SocialAgent(
                agent_id="a1",
                name="A1",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
                politics=PoliticalProfile(ideology_position=-0.3),
            ),
        ]

        tracker.track_window(agents1)
        tracker.track_window(agents2)

        widths = tracker.window_width_history()
        assert len(widths) == 2

    def test_detect_polarization_signature(self):
        """Test polarization signature detection."""
        tracker = OvertonWindowTracker()

        agents = [
            SocialAgent(
                agent_id="a1",
                name="A1",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
                politics=PoliticalProfile(ideology_position=-0.8),
            ),
            SocialAgent(
                agent_id="a2",
                name="A2",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
                politics=PoliticalProfile(ideology_position=0.8),
            ),
        ]

        tracker.track_window(agents)
        signature = tracker.detect_polarization_signature()
        assert isinstance(signature, str)


# ========================================================================
# TESTS: EmotionalContagionModel
# ========================================================================


class TestEmotionalContagionModel:
    """Tests for the emotional contagion model."""

    def test_model_creation(self):
        """Test model initialization."""
        G = nx.complete_graph(3)
        model = EmotionalContagionModel(G)
        assert model.network == G
        assert model.contagion_history == []

    def test_model_with_agents(self):
        """Test model setup with agents."""
        G = nx.complete_graph(3)
        model = EmotionalContagionModel(G)

        agents = [
            SocialAgent(
                agent_id="a1",
                name="A1",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
            ),
            SocialAgent(
                agent_id="a2",
                name="A2",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
            ),
            SocialAgent(
                agent_id="a3",
                name="A3",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
            ),
        ]

        agent_dict = {a.agent_id: a for a in agents}
        model.update_agent_susceptibility(agent_dict)

    def test_track_contagion(self):
        """Test basic contagion tracking."""
        G = nx.complete_graph(2)
        model = EmotionalContagionModel(G)

        agents = {
            "a1": SocialAgent(
                agent_id="a1",
                name="A1",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
            ),
            "a2": SocialAgent(
                agent_id="a2",
                name="A2",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
            ),
        }

        metrics = model.track_contagion(agents, step=0)
        assert metrics is not None
        assert hasattr(metrics, "emotional_state_distribution")
        assert hasattr(metrics, "contagion_r0_by_emotion")

    def test_identify_epidemics(self):
        """Test epidemic identification."""
        G = nx.complete_graph(5)
        model = EmotionalContagionModel(G)

        from src.agents.social_agent import EmotionalState

        agents = {}
        for i in range(5):
            agent = SocialAgent(
                agent_id=f"a{i}",
                name=f"A{i}",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
            )
            # Intentionally set same emotional state for cluster
            if i < 3:
                agent.emotional_state = EmotionalState.ANGRY
            agents[agent.agent_id] = agent

        epidemics = model.identify_epidemics(agents, minimum_size=2)
        assert isinstance(epidemics, list)

    def test_average_network_r0(self):
        """Test R0 computation."""
        G = nx.complete_graph(3)
        model = EmotionalContagionModel(G)

        agents = {
            "a1": SocialAgent(
                agent_id="a1",
                name="A1",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
            ),
            "a2": SocialAgent(
                agent_id="a2",
                name="A2",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
            ),
            "a3": SocialAgent(
                agent_id="a3",
                name="A3",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
            ),
        }

        r0 = model.average_network_r0()
        assert isinstance(r0, (int, float))
        assert r0 >= 0.0


# ========================================================================
# TESTS: Integration
# ========================================================================


class TestCognitiveModelsIntegration:
    """Integration tests for cognitive models."""

    def test_dissonance_and_resolution_workflow(self):
        """Test workflow of detecting and resolving dissonance."""
        tracker = CognitiveDissonanceTracker()

        agent = SocialAgent(
            agent_id="a1",
            name="Test",
            group="Group1",
            core_values=["equality", "fairness"],
            openness_to_change=0.6,
            politics=PoliticalProfile(
                faction="Faction1",
                partisan_strength=0.7,
            ),
        )

        # Simulate discovering conflicting policies
        faction_policies = {"equality": 0.2, "fairness": 0.2}

        dissonance = tracker.compute_dissonance(agent, faction_policies)
        tracker.track_dissonance(agent.agent_id, dissonance)

        if dissonance > 0.3:
            result = tracker.resolve_dissonance(agent, dissonance)
            assert result.dissonance_score == dissonance
            assert result.agent_id == agent.agent_id

    def test_overton_window_and_polarization(self):
        """Test Overton Window evolution with polarization."""
        tracker = OvertonWindowTracker()

        # First snapshot: centered distribution
        agents_centered = [
            SocialAgent(
                agent_id=f"a{i}",
                name=f"A{i}",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
                politics=PoliticalProfile(ideology_position=0.0),
            )
            for i in range(5)
        ]

        snapshot1 = tracker.track_window(agents_centered)

        # Second snapshot: polarized distribution
        agents_polarized = [
            SocialAgent(
                agent_id="a1",
                name="A1",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
                politics=PoliticalProfile(ideology_position=-0.9),
            ),
            SocialAgent(
                agent_id="a2",
                name="A2",
                group="G1",
                core_values=["v1"],
                openness_to_change=0.5,
                politics=PoliticalProfile(ideology_position=0.9),
            ),
        ]

        snapshot2 = tracker.track_window(agents_polarized)

        # Window should shift with polarization
        assert snapshot1 is not None
        assert snapshot2 is not None

    def test_emotional_contagion_with_network(self):
        """Test emotional contagion on specific network."""
        # Create a small network with different topologies
        for net_type in ["complete", "path", "star"]:
            if net_type == "complete":
                G = nx.complete_graph(4)
            elif net_type == "path":
                G = nx.path_graph(4)
            else:  # star
                G = nx.star_graph(3)

            model = EmotionalContagionModel(G)

            agents = {
                f"a{i}": SocialAgent(
                    agent_id=f"a{i}",
                    name=f"A{i}",
                    group="G1",
                    core_values=["v1"],
                    openness_to_change=0.5,
                )
                for i in range(G.number_of_nodes())
            }

            model.update_agent_susceptibility(agents)
            metrics = model.track_contagion(agents, step=0)
            assert metrics is not None
