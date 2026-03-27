"""Shared pytest fixtures for all tests."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.agents.social_agent import AgentMemory, PoliticalProfile, SocialAgent
from src.simulation.engine import SimulationEngine


@pytest.fixture
def sample_agent():
    """Create a sample social agent for testing."""
    return SocialAgent(
        agent_id="agent_001",
        name="Alice",
        group="Progressives",
        core_values=["equality", "innovation", "social_justice"],
        openness_to_change=0.7,
        politics=PoliticalProfile(
            ideology_position=-0.5,
            faction="Left Coalition",
            partisan_strength=0.7,
            political_engagement=0.6,
            policy_priorities=["healthcare", "education"],
        ),
    )


@pytest.fixture
def sample_agents_list():
    """Create a list of sample agents with diverse characteristics."""
    agents = []
    agent_configs = [
        {
            "id": "agent_001",
            "name": "Alice",
            "group": "Progressives",
            "values": ["equality", "innovation"],
            "openness": 0.8,
            "ideology": -0.6,
            "faction": "Left Coalition",
            "partisan": 0.8,
        },
        {
            "id": "agent_002",
            "name": "Bob",
            "group": "Traditionalists",
            "values": ["stability", "heritage"],
            "openness": 0.3,
            "ideology": 0.6,
            "faction": "Right Bloc",
            "partisan": 0.75,
        },
        {
            "id": "agent_003",
            "name": "Carol",
            "group": "Pragmatists",
            "values": ["efficiency", "practicality"],
            "openness": 0.6,
            "ideology": 0.1,
            "faction": "Center Alliance",
            "partisan": 0.5,
        },
        {
            "id": "agent_004",
            "name": "David",
            "group": "Youth_Activists",
            "values": ["change", "justice"],
            "openness": 0.9,
            "ideology": -0.8,
            "faction": "Green Alliance",
            "partisan": 0.6,
        },
        {
            "id": "agent_005",
            "name": "Eve",
            "group": "Working_Class",
            "values": ["security", "fairness"],
            "openness": 0.5,
            "ideology": 0.2,
            "faction": "Populist Movement",
            "partisan": 0.65,
        },
    ]

    for cfg in agent_configs:
        agent = SocialAgent(
            agent_id=cfg["id"],
            name=cfg["name"],
            group=cfg["group"],
            core_values=cfg["values"],
            openness_to_change=cfg["openness"],
            politics=PoliticalProfile(
                ideology_position=cfg["ideology"],
                faction=cfg["faction"],
                partisan_strength=cfg["partisan"],
                political_engagement=0.6,
                policy_priorities=["healthcare", "education"],
            ),
        )
        agents.append(agent)

    return agents


@pytest.fixture
def sample_config():
    """Create a sample simulation configuration."""
    config_dict = {
        "simulation": {
            "seed": 42,
            "num_steps": 10,
        },
        "network": {
            "type": "small_world",
            "avg_connections": 4,
            "rewiring_probability": 0.3,
        },
        "society": {
            "groups": [
                {"name": "Progressives"},
                {"name": "Traditionalists"},
                {"name": "Pragmatists"},
            ]
        },
        "friction": {
            "social_event_types": ["cultural_clash", "resource_competition"],
            "political_event_types": ["policy_disagreement", "election_fallout"],
            "crossover_event_types": ["politicized_cultural_issue"],
            "cascade_factor": 0.8,
            "resolution_probability": 0.3,
            "political_amplification": 1.5,
            "crossover_probability": 0.2,
        },
        "political": {
            "enabled": True,
            "factions": [
                {"name": "Left Coalition", "ideology_center": -0.7, "key_policies": {"equality": 1.0, "innovation": 0.9}},
                {"name": "Center Alliance", "ideology_center": 0.0, "key_policies": {"balance": 0.8, "efficiency": 0.8}},
                {"name": "Right Bloc", "ideology_center": 0.7, "key_policies": {"tradition": 1.0, "security": 0.9}},
            ],
            "elections": {
                "enabled": True,
                "frequency_steps": 5,
                "campaign_window": 2,
                "post_election_friction_boost": 0.15,
            },
            "polarization": {
                "radicalization_rate": 0.05,
                "moderation_rate": 0.03,
            },
        },
        "media": {
            "enabled": True,
            "outlets": [
                {"name": "LeftNews", "ideology_bias": -0.8, "credibility": 0.7, "reach": 0.3},
                {"name": "CenterNews", "ideology_bias": 0.0, "credibility": 0.9, "reach": 0.4},
                {"name": "RightNews", "ideology_bias": 0.8, "credibility": 0.7, "reach": 0.3},
            ],
        },
        "output": {
            "save_dir": "outputs/test_sim",
            "save_every_n_steps": 5,
        },
    }
    return config_dict


@pytest.fixture
def sample_config_file(tmp_path, sample_config):
    """Create a temporary YAML config file."""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return str(config_path)


@pytest.fixture
def sample_simulation_engine(sample_config_file, sample_agents_list):
    """Create a simulation engine initialized with sample agents."""
    engine = SimulationEngine(config_path=sample_config_file, llm=None)
    engine.initialize(sample_agents_list)
    return engine


@pytest.fixture
def sample_metrics_history():
    """Create sample metrics history data."""
    return [
        {
            "step": 0,
            "global_friction": 0.3,
            "political_friction": 0.2,
            "polarization_index": 0.15,
            "group_scores": {"Progressives": 0.2, "Traditionalists": 0.3, "Pragmatists": 0.25},
            "faction_scores": {"Left Coalition": 0.2, "Right Bloc": 0.3},
            "event_domain": "social",
            "num_reactions": 3,
            "num_interactions": 2,
        },
        {
            "step": 1,
            "global_friction": 0.35,
            "political_friction": 0.25,
            "polarization_index": 0.18,
            "group_scores": {"Progressives": 0.25, "Traditionalists": 0.35, "Pragmatists": 0.3},
            "faction_scores": {"Left Coalition": 0.25, "Right Bloc": 0.35},
            "event_domain": "political",
            "num_reactions": 4,
            "num_interactions": 3,
        },
        {
            "step": 2,
            "global_friction": 0.32,
            "political_friction": 0.28,
            "polarization_index": 0.2,
            "group_scores": {"Progressives": 0.22, "Traditionalists": 0.32, "Pragmatists": 0.28},
            "faction_scores": {"Left Coalition": 0.28, "Right Bloc": 0.32},
            "event_domain": "crossover",
            "num_reactions": 5,
            "num_interactions": 4,
        },
    ]


@pytest.fixture
def tmp_output_dir():
    """Create a temporary output directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
