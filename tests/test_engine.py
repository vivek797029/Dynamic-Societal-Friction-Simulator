"""Comprehensive tests for the SimulationEngine."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from src.agents.social_agent import PoliticalProfile, SocialAgent
from src.simulation.engine import FrictionEvent, SimulationEngine, SimulationState


# ========================================================================
# TESTS: Engine Initialization
# ========================================================================


class TestEngineInitialization:
    """Tests for SimulationEngine initialization."""

    def test_engine_creation_with_config(self, sample_config_file):
        """Test engine creation with configuration file."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        assert engine.cfg is not None
        assert engine.agents == {}
        assert engine.network is not None

    def test_engine_state_initialized(self, sample_config_file):
        """Test that engine state is properly initialized."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        assert engine.state.current_step == 0
        assert engine.state.global_friction_score == 0.0
        assert engine.state.political_friction_score == 0.0
        assert len(engine.state.events_log) == 0

    def test_engine_with_agents(self, sample_config_file, sample_agents_list):
        """Test engine initialization with agents."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        assert len(engine.agents) == len(sample_agents_list)
        assert all(a.agent_id in engine.agents for a in sample_agents_list)

    def test_engine_political_flag(self, sample_config_file):
        """Test that political flag is read from config."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        assert engine._political_enabled is True


# ========================================================================
# TESTS: Network Building
# ========================================================================


class TestNetworkBuilding:
    """Tests for network topology creation."""

    def test_small_world_network(self, sample_config_file, sample_agents_list):
        """Test small-world network construction."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        # Override network type
        engine.cfg["network"]["type"] = "small_world"
        engine.initialize(sample_agents_list)

        assert engine.network.number_of_nodes() > 0
        assert engine.network.number_of_edges() > 0

    def test_scale_free_network(self, tmp_path, sample_agents_list):
        """Test scale-free network construction."""
        config = {
            "simulation": {"seed": 42, "num_steps": 5},
            "network": {
                "type": "scale_free",
                "avg_connections": 4,
            },
            "society": {"groups": [{"name": "Group1"}, {"name": "Group2"}]},
            "friction": {
                "cascade_factor": 0.8,
                "resolution_probability": 0.3,
            },
            "political": {"enabled": False},
            "output": {"save_dir": str(tmp_path)},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        engine = SimulationEngine(config_path=str(config_path), llm=None)
        engine.initialize(sample_agents_list)

        assert engine.network.number_of_nodes() > 0

    def test_random_network(self, tmp_path, sample_agents_list):
        """Test random network construction."""
        config = {
            "simulation": {"seed": 42, "num_steps": 5},
            "network": {
                "type": "random",
                "avg_connections": 3,
            },
            "society": {"groups": [{"name": "Group1"}]},
            "friction": {
                "cascade_factor": 0.8,
                "resolution_probability": 0.3,
            },
            "political": {"enabled": False},
            "output": {"save_dir": str(tmp_path)},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        engine = SimulationEngine(config_path=str(config_path), llm=None)
        engine.initialize(sample_agents_list)

        assert engine.network.number_of_nodes() > 0

    def test_network_agent_connections(self, sample_config_file, sample_agents_list):
        """Test that agents get connection lists from network."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        # All agents should have connections set
        for agent in engine.agents.values():
            assert isinstance(agent.connections, list)


# ========================================================================
# TESTS: Event Generation
# ========================================================================


class TestEventGeneration:
    """Tests for friction event generation."""

    def test_generate_event_structure(self, sample_config_file, sample_agents_list):
        """Test that generated events have required structure."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        event = engine.generate_event(step=0)
        assert isinstance(event, FrictionEvent)
        assert event.event_id == "evt_0000"
        assert event.domain in ("social", "political", "crossover")
        assert event.severity > 0

    def test_event_domain_selection(self, sample_config_file, sample_agents_list):
        """Test that event domains are selected appropriately."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        # Generate multiple events and check domain distribution
        domains = []
        for i in range(20):
            event = engine.generate_event(step=i)
            domains.append(event.domain)

        # Should have mix of domains
        assert "social" in domains or "political" in domains or "crossover" in domains

    def test_social_event_has_affected_groups(self, tmp_path, sample_agents_list):
        """Test that social events have affected groups."""
        config = {
            "simulation": {"seed": 42, "num_steps": 5},
            "network": {"type": "small_world", "avg_connections": 4},
            "society": {"groups": [{"name": "Group1"}, {"name": "Group2"}]},
            "friction": {
                "social_event_types": ["cultural_clash"],
                "cascade_factor": 0.8,
                "resolution_probability": 0.3,
                "crossover_probability": 0.0,
            },
            "political": {"enabled": False},
            "output": {"save_dir": str(tmp_path)},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        engine = SimulationEngine(config_path=str(config_path), llm=None)
        engine.initialize(sample_agents_list)

        event = engine.generate_event(step=0)
        assert event.domain == "social"
        assert len(event.affected_groups) > 0

    def test_political_event_has_factions(self, sample_config_file, sample_agents_list):
        """Test that political events have affected factions."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        # Generate events until we get a political one
        for i in range(20):
            event = engine.generate_event(step=i)
            if event.domain == "political":
                assert len(event.affected_factions) > 0
                break


# ========================================================================
# TESTS: Election Mechanics
# ========================================================================


class TestElectionMechanics:
    """Tests for election-related functionality."""

    def test_should_hold_election_basic(self, sample_config_file, sample_agents_list):
        """Test election frequency check."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        # Config has frequency_steps: 5
        engine.state.current_step = 0
        assert engine._should_hold_election() is False

        engine.state.current_step = 5
        assert engine._should_hold_election() is True

    def test_is_campaign_period(self, sample_config_file, sample_agents_list):
        """Test campaign period detection."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        # frequency_steps: 5, campaign_window: 2
        engine.state.current_step = 4
        assert engine._is_campaign_period() is True

        engine.state.current_step = 3
        assert engine._is_campaign_period() is True

        engine.state.current_step = 1
        assert engine._is_campaign_period() is False

    def test_election_voting_mechanics(self, sample_config_file, sample_agents_list):
        """Test that elections properly count votes."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        result = engine._run_election()
        assert "winning_faction" in result
        assert "margin" in result
        assert "vote_counts" in result
        assert result["winning_faction"] in [f["name"] for f in engine.cfg["political"]["factions"]]

    def test_election_result_recorded(self, sample_config_file, sample_agents_list):
        """Test that election results are recorded in state."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        initial_count = len(engine.state.election_log)
        engine._run_election()
        assert len(engine.state.election_log) == initial_count + 1

    def test_post_election_friction_boost(self, sample_config_file, sample_agents_list):
        """Test that losing factions experience friction boost."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        # Set initial faction friction
        for faction_name in engine.state.faction_friction_scores:
            engine.state.faction_friction_scores[faction_name] = 0.1

        engine._run_election()

        # Some factions should have increased friction
        for faction_name, score in engine.state.faction_friction_scores.items():
            assert score >= 0.1


# ========================================================================
# TESTS: Media Influence
# ========================================================================


class TestMediaInfluence:
    """Tests for media consumption and influence."""

    def test_apply_media_influence_basic(self, sample_config_file, sample_agents_list):
        """Test that media influence is applied."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        # Store initial positions
        initial_positions = {
            a.agent_id: a.politics.ideology_position for a in engine.agents.values()
        }

        engine._apply_media_influence()

        # Some positions may have changed
        # (at least the function ran without error)
        assert len(engine.agents) > 0

    def test_media_influence_disabled(self, tmp_path, sample_agents_list):
        """Test that media influence can be disabled."""
        config = {
            "simulation": {"seed": 42, "num_steps": 5},
            "network": {"type": "small_world", "avg_connections": 4},
            "society": {"groups": [{"name": "Group1"}]},
            "friction": {"cascade_factor": 0.8, "resolution_probability": 0.3},
            "political": {"enabled": False},
            "media": {"enabled": False},
            "output": {"save_dir": str(tmp_path)},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        engine = SimulationEngine(config_path=str(config_path), llm=None)
        engine.initialize(sample_agents_list)
        engine._apply_media_influence()  # Should do nothing


# ========================================================================
# TESTS: Polarization Index
# ========================================================================


class TestPolarizationComputation:
    """Tests for polarization index calculation."""

    def test_polarization_with_all_centered(self, sample_config_file):
        """Test polarization when all agents are centered."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)

        # Create agents with centered ideology
        agents = [
            SocialAgent(
                agent_id=f"a{i}",
                name=f"Agent{i}",
                group="Group1",
                core_values=["value"],
                openness_to_change=0.5,
                politics=PoliticalProfile(ideology_position=0.0),
            )
            for i in range(5)
        ]

        engine.initialize(agents)
        polarization = engine._compute_polarization_index()
        assert polarization >= 0.0
        assert polarization <= 1.0

    def test_polarization_with_split_extremes(self, sample_config_file):
        """Test polarization when agents are at extremes."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)

        # Create agents at extremes
        agents = [
            SocialAgent(
                agent_id="a1",
                name="Alice",
                group="Group1",
                core_values=["value"],
                openness_to_change=0.5,
                politics=PoliticalProfile(ideology_position=-1.0),
            ),
            SocialAgent(
                agent_id="a2",
                name="Bob",
                group="Group1",
                core_values=["value"],
                openness_to_change=0.5,
                politics=PoliticalProfile(ideology_position=1.0),
            ),
        ]

        engine.initialize(agents)
        polarization = engine._compute_polarization_index()
        assert polarization > 0.0


# ========================================================================
# TESTS: Ideology Drift
# ========================================================================


class TestIdeologyDrift:
    """Tests for radicalization and moderation mechanics."""

    def test_apply_ideology_drift_radicalization(self, sample_config_file):
        """Test that high friction causes radicalization."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)

        agents = [
            SocialAgent(
                agent_id="a1",
                name="Alice",
                group="Group1",
                core_values=["value"],
                openness_to_change=0.5,
                politics=PoliticalProfile(
                    ideology_position=0.3,
                    faction="Right Bloc",
                ),
            ),
        ]

        engine.initialize(agents)
        # Set high friction for the faction
        engine.state.faction_friction_scores["Right Bloc"] = 0.8

        original_pos = engine.agents["a1"].politics.ideology_position
        engine._apply_ideology_drift()
        new_pos = engine.agents["a1"].politics.ideology_position

        # Should move toward extreme
        assert abs(new_pos) > abs(original_pos)

    def test_apply_ideology_drift_moderation(self, sample_config_file):
        """Test that low friction causes moderation."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)

        agents = [
            SocialAgent(
                agent_id="a1",
                name="Alice",
                group="Group1",
                core_values=["value"],
                openness_to_change=0.5,
                politics=PoliticalProfile(
                    ideology_position=-0.6,
                    faction="Left Coalition",
                ),
            ),
        ]

        engine.initialize(agents)
        # Set low friction
        engine.state.faction_friction_scores["Left Coalition"] = 0.2
        engine.state.political_friction_score = 0.2

        original_pos = engine.agents["a1"].politics.ideology_position
        engine._apply_ideology_drift()
        new_pos = engine.agents["a1"].politics.ideology_position

        # Should move toward center
        assert abs(new_pos) < abs(original_pos)


# ========================================================================
# TESTS: Single Step Execution
# ========================================================================


class TestSingleStep:
    """Tests for single simulation step."""

    def test_step_increments_counter(self, sample_config_file, sample_agents_list):
        """Test that step increments current_step counter."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        initial_step = engine.state.current_step
        engine.step()
        assert engine.state.current_step == initial_step + 1

    def test_step_generates_metrics(self, sample_config_file, sample_agents_list):
        """Test that step returns metrics."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        metrics = engine.step()
        assert "step" in metrics
        assert "global_friction" in metrics
        assert "event_domain" in metrics

    def test_step_records_events(self, sample_config_file, sample_agents_list):
        """Test that step records events."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        initial_events = len(engine.state.events_log)
        engine.step()
        assert len(engine.state.events_log) > initial_events

    def test_step_updates_friction_scores(self, sample_config_file, sample_agents_list):
        """Test that step updates friction scores."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        engine.step()
        # After step, should have computed friction
        assert engine.state.global_friction_score >= 0.0


# ========================================================================
# TESTS: Checkpoint Saving
# ========================================================================


class TestCheckpointing:
    """Tests for checkpoint save/load functionality."""

    def test_save_checkpoint_creates_file(self, sample_config_file, sample_agents_list):
        """Test that checkpoint saves files."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)
        engine.step()

        engine._save_checkpoint()

        save_dir = Path(engine.cfg["output"]["save_dir"])
        assert save_dir.exists()
        checkpoint_file = save_dir / f"checkpoint_step_{engine.state.current_step}.json"
        assert checkpoint_file.exists()

    def test_save_checkpoint_contains_data(self, sample_config_file, sample_agents_list):
        """Test that checkpoint contains expected data."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)
        engine.step()

        engine._save_checkpoint()

        save_dir = Path(engine.cfg["output"]["save_dir"])
        checkpoint_file = save_dir / f"checkpoint_step_{engine.state.current_step}.json"

        with open(checkpoint_file) as f:
            data = json.load(f)

        assert "metrics_history" in data
        assert "ideology_snapshots" in data


# ========================================================================
# TESTS: Full Simulation Run
# ========================================================================


class TestFullSimulation:
    """Tests for full simulation runs."""

    def test_run_short_simulation(self, sample_config_file, sample_agents_list):
        """Test running a short full simulation."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        all_metrics = engine.run(num_steps=3)

        assert len(all_metrics) == 3
        assert all_metrics[0]["step"] == 0
        assert all_metrics[2]["step"] == 2

    def test_simulation_preserves_agent_count(self, sample_config_file, sample_agents_list):
        """Test that agent count doesn't change during simulation."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        initial_count = len(engine.agents)
        engine.run(num_steps=2)
        assert len(engine.agents) == initial_count

    def test_simulation_saves_results(self, sample_config_file, sample_agents_list):
        """Test that simulation saves final results."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)
        engine.initialize(sample_agents_list)

        engine.run(num_steps=2)

        save_dir = Path(engine.cfg["output"]["save_dir"])
        metrics_file = save_dir / "metrics.json"
        assert metrics_file.exists()


# ========================================================================
# TESTS: Edge Cases
# ========================================================================


class TestEngineEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_engine_with_single_agent(self, sample_config_file):
        """Test engine with only one agent."""
        engine = SimulationEngine(config_path=sample_config_file, llm=None)

        agent = SocialAgent(
            agent_id="a1",
            name="Alone",
            group="Group1",
            core_values=["value"],
            openness_to_change=0.5,
        )

        engine.initialize([agent])
        metrics = engine.step()
        assert metrics["step"] == 0

    def test_engine_step_zero_friction(self, tmp_path):
        """Test engine step with low friction potential."""
        config = {
            "simulation": {"seed": 42, "num_steps": 1},
            "network": {"type": "small_world", "avg_connections": 2},
            "society": {"groups": [{"name": "Group1"}]},
            "friction": {
                "cascade_factor": 0.0,
                "resolution_probability": 1.0,
            },
            "political": {"enabled": False},
            "output": {"save_dir": str(tmp_path)},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        engine = SimulationEngine(config_path=str(config_path), llm=None)

        agents = [
            SocialAgent(
                agent_id="a1",
                name="A1",
                group="Group1",
                core_values=["value"],
                openness_to_change=0.5,
            ),
        ]

        engine.initialize(agents)
        metrics = engine.step()
        assert metrics["global_friction"] >= 0.0
