"""Tests for social agent behavior."""

from src.agents.social_agent import AgentMemory, EmotionalState, SocialAgent


def test_agent_creation():
    agent = SocialAgent(
        agent_id="test_001",
        name="TestAgent",
        group="Progressives",
        core_values=["equality", "innovation"],
        openness_to_change=0.8,
    )
    assert agent.agent_id == "test_001"
    assert agent.emotional_state == EmotionalState.CALM
    assert agent.friction_tolerance == 0.5


def test_agent_memory():
    memory = AgentMemory(max_size=3)
    memory.add({"summary": "Event 1"})
    memory.add({"summary": "Event 2"})
    memory.add({"summary": "Event 3"})
    memory.add({"summary": "Event 4"})

    assert len(memory.events) == 3
    assert memory.events[0]["summary"] == "Event 2"  # oldest was dropped


def test_agent_memory_summarize():
    memory = AgentMemory()
    memory.add({"summary": "Cultural festival debate"})
    memory.add({"summary": "Resource allocation conflict"})

    summary = memory.summarize()
    assert "Cultural festival debate" in summary
    assert "Resource allocation conflict" in summary


def test_agent_identity_prompt():
    agent = SocialAgent(
        agent_id="test_002",
        name="Maria",
        group="Traditionalists",
        core_values=["heritage", "stability"],
        openness_to_change=0.2,
    )
    prompt = agent.build_identity_prompt()
    assert "Maria" in prompt
    assert "Traditionalists" in prompt
    assert "heritage" in prompt


def test_emotional_state_update():
    agent = SocialAgent(
        agent_id="test_003",
        name="TestAgent",
        group="Pragmatists",
        core_values=["efficiency"],
        openness_to_change=0.5,
    )

    agent.update_emotional_state(0.9)  # high friction
    assert agent.emotional_state in [EmotionalState.ANGRY, EmotionalState.FEARFUL]

    agent.update_emotional_state(0.1)  # low friction
    assert agent.emotional_state in [EmotionalState.HOPEFUL, EmotionalState.EMPATHETIC]
