"""Comprehensive tests for social agent behavior."""

import pytest

from src.agents.social_agent import (
    AgentMemory,
    EmotionalState,
    PoliticalProfile,
    SocialAgent,
)


# ========================================================================
# TESTS: Agent Creation and Basic Properties
# ========================================================================


def test_agent_creation():
    """Test basic agent creation with required fields."""
    agent = SocialAgent(
        agent_id="test_001",
        name="TestAgent",
        group="Progressives",
        core_values=["equality", "innovation"],
        openness_to_change=0.8,
    )
    assert agent.agent_id == "test_001"
    assert agent.name == "TestAgent"
    assert agent.group == "Progressives"
    assert agent.emotional_state == EmotionalState.CALM
    assert agent.friction_tolerance == 0.5
    assert agent.openness_to_change == 0.8


def test_agent_with_custom_friction_tolerance():
    """Test agent with custom friction tolerance."""
    agent = SocialAgent(
        agent_id="test_001",
        name="Alice",
        group="Progressives",
        core_values=["equality"],
        openness_to_change=0.5,
        friction_tolerance=0.8,
    )
    assert agent.friction_tolerance == 0.8


def test_agent_connections_initialized_empty():
    """Test that agent connections list is initialized empty."""
    agent = SocialAgent(
        agent_id="test_001",
        name="Bob",
        group="Traditionalists",
        core_values=["stability"],
        openness_to_change=0.3,
    )
    assert agent.connections == []


# ========================================================================
# TESTS: PoliticalProfile
# ========================================================================


class TestPoliticalProfile:
    """Tests for PoliticalProfile class."""

    def test_political_profile_creation(self):
        """Test basic political profile creation."""
        profile = PoliticalProfile(
            ideology_position=-0.5,
            faction="Left Coalition",
            partisan_strength=0.7,
            political_engagement=0.6,
        )
        assert profile.ideology_position == -0.5
        assert profile.faction == "Left Coalition"
        assert profile.partisan_strength == 0.7
        assert profile.political_engagement == 0.6

    def test_ideology_label_far_left(self):
        """Test ideology label for far-left position."""
        profile = PoliticalProfile(ideology_position=-0.8)
        assert profile.ideology_label == "far-left"

    def test_ideology_label_center_left(self):
        """Test ideology label for center-left position."""
        profile = PoliticalProfile(ideology_position=-0.4)
        assert profile.ideology_label == "center-left"

    def test_ideology_label_centrist(self):
        """Test ideology label for centrist position."""
        profile = PoliticalProfile(ideology_position=0.0)
        assert profile.ideology_label == "centrist"

        profile.ideology_position = 0.15
        assert profile.ideology_label == "centrist"

    def test_ideology_label_center_right(self):
        """Test ideology label for center-right position."""
        profile = PoliticalProfile(ideology_position=0.4)
        assert profile.ideology_label == "center-right"

    def test_ideology_label_far_right(self):
        """Test ideology label for far-right position."""
        profile = PoliticalProfile(ideology_position=0.8)
        assert profile.ideology_label == "far-right"

    def test_is_moderate(self):
        """Test is_moderate property."""
        # Moderate range: -0.3 to 0.3
        moderate_profile = PoliticalProfile(ideology_position=0.2)
        assert moderate_profile.is_moderate is True

        moderate_profile.ideology_position = -0.25
        assert moderate_profile.is_moderate is True

        # Extreme positions
        extreme_profile = PoliticalProfile(ideology_position=0.5)
        assert extreme_profile.is_moderate is False

        extreme_profile.ideology_position = -0.5
        assert extreme_profile.is_moderate is False

    def test_is_extreme(self):
        """Test is_extreme property."""
        # Extreme range: > 0.7 or < -0.7
        extreme_profile = PoliticalProfile(ideology_position=0.8)
        assert extreme_profile.is_extreme is True

        extreme_profile.ideology_position = -0.75
        assert extreme_profile.is_extreme is True

        # Non-extreme positions
        moderate_profile = PoliticalProfile(ideology_position=0.5)
        assert moderate_profile.is_extreme is False

        moderate_profile.ideology_position = 0.0
        assert moderate_profile.is_extreme is False

    def test_drift_toward_positive(self):
        """Test drift_toward moving toward positive ideology."""
        profile = PoliticalProfile(ideology_position=-0.5)
        profile.drift_toward(0.5, rate=0.1)
        assert -0.5 < profile.ideology_position < 0.5, "Should move toward target"
        assert profile.ideology_position > -0.5, "Should move right"

    def test_drift_toward_negative(self):
        """Test drift_toward moving toward negative ideology."""
        profile = PoliticalProfile(ideology_position=0.5)
        profile.drift_toward(-0.5, rate=0.1)
        assert -0.5 < profile.ideology_position < 0.5, "Should move toward target"
        assert profile.ideology_position < 0.5, "Should move left"

    def test_drift_toward_boundary_clamping(self):
        """Test that drift_toward clamps to [-1.0, 1.0] bounds."""
        profile = PoliticalProfile(ideology_position=0.99)
        profile.drift_toward(1.5, rate=0.1)
        assert profile.ideology_position <= 1.0, "Should not exceed right bound"

        profile = PoliticalProfile(ideology_position=-0.99)
        profile.drift_toward(-1.5, rate=0.1)
        assert profile.ideology_position >= -1.0, "Should not exceed left bound"

    def test_drift_toward_records_history(self):
        """Test that drift_toward records ideology history."""
        profile = PoliticalProfile(ideology_position=0.0)
        assert len(profile.ideology_history) == 0

        profile.drift_toward(0.5, rate=0.1)
        assert len(profile.ideology_history) == 1
        assert profile.ideology_history[0] == 0.0

        profile.drift_toward(0.5, rate=0.1)
        assert len(profile.ideology_history) == 2

    def test_radicalize_from_positive(self):
        """Test radicalization moving toward positive extreme."""
        profile = PoliticalProfile(ideology_position=0.5)
        original = profile.ideology_position
        profile.radicalize(rate=0.1)
        assert profile.ideology_position > original, "Should move right"
        assert profile.ideology_position <= 1.0, "Should not exceed bound"

    def test_radicalize_from_negative(self):
        """Test radicalization moving toward negative extreme."""
        profile = PoliticalProfile(ideology_position=-0.5)
        original = profile.ideology_position
        profile.radicalize(rate=0.1)
        assert profile.ideology_position < original, "Should move left"
        assert profile.ideology_position >= -1.0, "Should not exceed bound"

    def test_radicalize_at_zero(self):
        """Test radicalization from centrist position."""
        profile = PoliticalProfile(ideology_position=0.0)
        profile.radicalize(rate=0.1)
        # At zero, direction = 1.0, so should move right
        assert profile.ideology_position > 0.0

    def test_radicalize_extreme_values(self):
        """Test radicalization at extreme boundary values."""
        profile = PoliticalProfile(ideology_position=-1.0)
        profile.radicalize(rate=0.1)
        assert profile.ideology_position == -1.0, "Already at extreme, no further movement"

        profile = PoliticalProfile(ideology_position=1.0)
        profile.radicalize(rate=0.1)
        assert profile.ideology_position == 1.0, "Already at extreme, no further movement"

    def test_moderate_from_positive(self):
        """Test moderation moving toward center from right."""
        profile = PoliticalProfile(ideology_position=0.5)
        original = profile.ideology_position
        profile.moderate(rate=0.1)
        assert profile.ideology_position < original, "Should move left toward center"

    def test_moderate_from_negative(self):
        """Test moderation moving toward center from left."""
        profile = PoliticalProfile(ideology_position=-0.5)
        original = profile.ideology_position
        profile.moderate(rate=0.1)
        assert profile.ideology_position > original, "Should move right toward center"

    def test_moderate_crosses_zero(self):
        """Test moderation that brings position to zero."""
        profile = PoliticalProfile(ideology_position=0.02)
        profile.moderate(rate=0.05)
        # Since abs(0.02) < 0.05, should go to 0.0
        assert profile.ideology_position == 0.0

    def test_moderate_from_extreme_values(self):
        """Test moderation at extreme boundary values."""
        profile = PoliticalProfile(ideology_position=-1.0)
        profile.moderate(rate=0.1)
        assert profile.ideology_position > -1.0, "Should move toward center"
        assert profile.ideology_position >= -1.0, "Should not exceed left bound"

    def test_extreme_ideology_values(self):
        """Test extreme boundary values for ideology position."""
        profile = PoliticalProfile(ideology_position=-1.0)
        assert profile.ideology_position == -1.0
        assert profile.ideology_label == "far-left"
        assert profile.is_extreme is True

        profile = PoliticalProfile(ideology_position=1.0)
        assert profile.ideology_position == 1.0
        assert profile.ideology_label == "far-right"
        assert profile.is_extreme is True


# ========================================================================
# TESTS: AgentMemory
# ========================================================================


class TestAgentMemory:
    """Tests for AgentMemory class."""

    def test_memory_creation_default(self):
        """Test memory creation with default max_size."""
        memory = AgentMemory()
        assert memory.max_size == 25
        assert len(memory.events) == 0

    def test_memory_creation_custom_size(self):
        """Test memory creation with custom max_size."""
        memory = AgentMemory(max_size=10)
        assert memory.max_size == 10

    def test_memory_add_events(self):
        """Test adding events to memory."""
        memory = AgentMemory(max_size=5)
        memory.add({"summary": "Event 1"})
        memory.add({"summary": "Event 2"})
        assert len(memory.events) == 2
        assert memory.events[0]["summary"] == "Event 1"
        assert memory.events[1]["summary"] == "Event 2"

    def test_memory_overflow_drops_oldest(self):
        """Test that memory drops oldest events when exceeding max_size."""
        memory = AgentMemory(max_size=3)
        memory.add({"summary": "Event 1"})
        memory.add({"summary": "Event 2"})
        memory.add({"summary": "Event 3"})
        memory.add({"summary": "Event 4"})

        assert len(memory.events) == 3, "Should have max_size events"
        assert memory.events[0]["summary"] == "Event 2", "Oldest should be dropped"
        assert memory.events[-1]["summary"] == "Event 4", "Newest should be retained"

    def test_memory_boundary_max_size(self):
        """Test memory at boundary conditions for max_size."""
        memory = AgentMemory(max_size=1)
        memory.add({"summary": "Event 1"})
        assert len(memory.events) == 1

        memory.add({"summary": "Event 2"})
        assert len(memory.events) == 1
        assert memory.events[0]["summary"] == "Event 2"

    def test_memory_recent_default(self):
        """Test recent() method with default count."""
        memory = AgentMemory()
        for i in range(10):
            memory.add({"summary": f"Event {i+1}"})

        recent = memory.recent()
        assert len(recent) == 5, "Default recent should return last 5"
        assert recent[0]["summary"] == "Event 6"
        assert recent[-1]["summary"] == "Event 10"

    def test_memory_recent_custom_count(self):
        """Test recent() method with custom count."""
        memory = AgentMemory()
        for i in range(10):
            memory.add({"summary": f"Event {i+1}"})

        recent = memory.recent(n=3)
        assert len(recent) == 3
        assert recent[0]["summary"] == "Event 8"
        assert recent[-1]["summary"] == "Event 10"

    def test_memory_recent_exceeds_available(self):
        """Test recent() when requesting more than available."""
        memory = AgentMemory()
        memory.add({"summary": "Event 1"})
        memory.add({"summary": "Event 2"})

        recent = memory.recent(n=10)
        assert len(recent) == 2, "Should return available events"

    def test_memory_summarize_empty(self):
        """Test summarize() on empty memory."""
        memory = AgentMemory()
        summary = memory.summarize()
        assert summary == "No significant events experienced."

    def test_memory_summarize_basic(self):
        """Test summarize() with events."""
        memory = AgentMemory()
        memory.add({"summary": "Cultural festival debate"})
        memory.add({"summary": "Resource allocation conflict"})

        summary = memory.summarize()
        assert "Cultural festival debate" in summary
        assert "Resource allocation conflict" in summary
        assert "-" in summary, "Should be bulleted format"

    def test_memory_summarize_uses_last_10(self):
        """Test that summarize uses only last 10 events."""
        memory = AgentMemory()
        for i in range(20):
            memory.add({"summary": f"Event {i+1}"})

        summary = memory.summarize()
        # Should contain events 11-20 only
        assert "Event 20" in summary
        assert "Event 1" not in summary, "Should not include first event"

    def test_memory_political_history_filter(self):
        """Test political_history() filter."""
        memory = AgentMemory()
        memory.add({"summary": "Cultural event", "domain": "social"})
        memory.add({"summary": "Election happening", "domain": "political"})
        memory.add({"summary": "Community festival", "domain": "cultural"})
        memory.add({"summary": "Policy change", "domain": "political"})

        political = memory.political_history()
        assert len(political) == 2
        assert all(e["domain"] == "political" for e in political)

    def test_memory_political_history_empty(self):
        """Test political_history() when no political events exist."""
        memory = AgentMemory()
        memory.add({"summary": "Cultural event", "domain": "social"})
        memory.add({"summary": "Community festival", "domain": "cultural"})

        political = memory.political_history()
        assert len(political) == 0

    def test_memory_social_history_filter(self):
        """Test social_history() filter."""
        memory = AgentMemory()
        memory.add({"summary": "Cultural event", "domain": "social"})
        memory.add({"summary": "Election happening", "domain": "political"})
        memory.add({"summary": "Community festival", "domain": "cultural"})
        memory.add({"summary": "Policy change", "domain": "political"})

        social = memory.social_history()
        assert len(social) == 2
        assert all(e["domain"] in ("social", "cultural") for e in social)

    def test_memory_social_history_both_domains(self):
        """Test that social_history includes both 'social' and 'cultural'."""
        memory = AgentMemory()
        memory.add({"summary": "Social event", "domain": "social"})
        memory.add({"summary": "Cultural event", "domain": "cultural"})

        social = memory.social_history()
        assert len(social) == 2


# ========================================================================
# TESTS: SocialAgent with Political Content
# ========================================================================


class TestSocialAgentPolitical:
    """Tests for SocialAgent with political behavior."""

    def test_build_identity_prompt_includes_politics(self):
        """Test that build_identity_prompt includes political information."""
        agent = SocialAgent(
            agent_id="test_001",
            name="Alice",
            group="Progressives",
            core_values=["equality", "innovation"],
            openness_to_change=0.8,
            politics=PoliticalProfile(
                ideology_position=-0.6,
                faction="Left Coalition",
                partisan_strength=0.8,
                political_engagement=0.7,
                policy_priorities=["healthcare", "education"],
            ),
        )

        prompt = agent.build_identity_prompt()
        assert "Alice" in prompt
        assert "Progressives" in prompt
        assert "equality" in prompt
        assert "center-left" in prompt or "far-left" in prompt
        assert "Left Coalition" in prompt
        assert "healthcare" in prompt

    def test_build_identity_prompt_format(self):
        """Test that build_identity_prompt has proper structure."""
        agent = SocialAgent(
            agent_id="test_001",
            name="Bob",
            group="Traditionalists",
            core_values=["stability"],
            openness_to_change=0.3,
        )

        prompt = agent.build_identity_prompt()
        assert "Political Identity" in prompt or "political" in prompt.lower()
        assert "core values" in prompt.lower() or "stability" in prompt
        assert "emotional state" in prompt.lower()

    def test_update_emotional_state_high_friction(self):
        """Test emotional state with high friction."""
        agent = SocialAgent(
            agent_id="test_001",
            name="Alice",
            group="Progressives",
            core_values=["equality"],
            openness_to_change=0.5,
        )

        agent.update_emotional_state(0.85)
        assert agent.emotional_state in [
            EmotionalState.ANGRY,
            EmotionalState.FEARFUL,
            EmotionalState.OUTRAGED,
        ], f"High friction should produce negative state, got {agent.emotional_state}"

    def test_update_emotional_state_high_political_friction(self):
        """Test emotional state with high political friction."""
        agent = SocialAgent(
            agent_id="test_001",
            name="Bob",
            group="Traditionalists",
            core_values=["stability"],
            openness_to_change=0.5,
        )

        agent.update_emotional_state(0.2, political_friction=0.85)
        assert agent.emotional_state in [
            EmotionalState.ANGRY,
            EmotionalState.FEARFUL,
            EmotionalState.OUTRAGED,
        ]

    def test_update_emotional_state_low_friction(self):
        """Test emotional state with low friction."""
        agent = SocialAgent(
            agent_id="test_001",
            name="Carol",
            group="Pragmatists",
            core_values=["efficiency"],
            openness_to_change=0.5,
        )

        agent.update_emotional_state(0.05)
        assert agent.emotional_state in [
            EmotionalState.HOPEFUL,
            EmotionalState.EMPATHETIC,
        ]

    def test_update_emotional_state_medium_friction(self):
        """Test emotional state with medium friction."""
        agent = SocialAgent(
            agent_id="test_001",
            name="David",
            group="Youth_Activists",
            core_values=["justice"],
            openness_to_change=0.8,
        )

        agent.update_emotional_state(0.5)
        # Medium friction could produce multiple states
        assert agent.emotional_state in [
            EmotionalState.ANXIOUS,
            EmotionalState.DISILLUSIONED,
            EmotionalState.CALM,
        ]

    def test_update_emotional_state_combines_friction(self):
        """Test that update_emotional_state uses combined maximum."""
        agent = SocialAgent(
            agent_id="test_001",
            name="Eve",
            group="Working_Class",
            core_values=["security"],
            openness_to_change=0.5,
        )

        # Max should be used: max(0.3, 0.7) = 0.7
        agent.update_emotional_state(0.3, political_friction=0.7)
        assert agent.emotional_state in [
            EmotionalState.ANXIOUS,
            EmotionalState.HOSTILE,
            EmotionalState.GALVANIZED,
        ]

    def test_consume_media_echo_chamber_effect(self):
        """Test that aligned media causes drift in same direction."""
        agent = SocialAgent(
            agent_id="test_001",
            name="Alice",
            group="Progressives",
            core_values=["equality"],
            openness_to_change=0.6,
            politics=PoliticalProfile(ideology_position=-0.5),
        )

        original_position = agent.politics.ideology_position
        # Strongly left-biased media, well aligned with agent
        left_outlet = {
            "name": "LeftNews",
            "ideology_bias": -0.8,
            "credibility": 0.8,
        }
        agent.consume_media(left_outlet)
        # Should drift left (more negative)
        assert agent.politics.ideology_position < original_position

    def test_consume_media_cross_cutting_effect(self):
        """Test that cross-cutting media causes moderation."""
        agent = SocialAgent(
            agent_id="test_001",
            name="Bob",
            group="Traditionalists",
            core_values=["stability"],
            openness_to_change=0.8,  # Open to change
            politics=PoliticalProfile(ideology_position=0.6),
        )

        original_position = agent.politics.ideology_position
        # Left-biased media, poorly aligned
        cross_outlet = {
            "name": "LeftNews",
            "ideology_bias": -0.8,
            "credibility": 0.8,
        }
        agent.consume_media(cross_outlet)
        # Should moderate (move toward center)
        assert abs(agent.politics.ideology_position) < abs(original_position)

    def test_ideology_summary(self):
        """Test ideology_summary output format."""
        agent = SocialAgent(
            agent_id="test_001",
            name="Alice",
            group="Progressives",
            core_values=["equality"],
            openness_to_change=0.8,
            politics=PoliticalProfile(
                ideology_position=-0.6,
                faction="Left Coalition",
                partisan_strength=0.8,
                political_engagement=0.7,
            ),
        )

        summary = agent.ideology_summary()
        assert summary["agent_id"] == "test_001"
        assert summary["group"] == "Progressives"
        assert summary["faction"] == "Left Coalition"
        assert abs(summary["ideology"] - (-0.6)) < 0.01
        assert summary["label"] in [
            "far-left",
            "center-left",
            "centrist",
            "center-right",
            "far-right",
        ]
        assert summary["engagement"] == 0.7
        assert summary["partisan_strength"] == 0.8
        assert "is_extreme" in summary


# ========================================================================
# TESTS: Edge Cases
# ========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_agent_with_empty_values(self):
        """Test agent creation with empty core values."""
        agent = SocialAgent(
            agent_id="test_001",
            name="Empty",
            group="Progressives",
            core_values=[],
            openness_to_change=0.5,
        )
        assert agent.core_values == []

    def test_agent_with_zero_openness(self):
        """Test agent with minimum openness to change."""
        agent = SocialAgent(
            agent_id="test_001",
            name="Rigid",
            group="Traditionalists",
            core_values=["stability"],
            openness_to_change=0.0,
        )
        assert agent.openness_to_change == 0.0

    def test_agent_with_max_openness(self):
        """Test agent with maximum openness to change."""
        agent = SocialAgent(
            agent_id="test_001",
            name="Flexible",
            group="Progressives",
            core_values=["change"],
            openness_to_change=1.0,
        )
        assert agent.openness_to_change == 1.0

    def test_zero_friction_score(self):
        """Test emotional state update with zero friction."""
        agent = SocialAgent(
            agent_id="test_001",
            name="Peaceful",
            group="Pragmatists",
            core_values=["harmony"],
            openness_to_change=0.5,
        )
        agent.update_emotional_state(0.0)
        assert agent.emotional_state in [
            EmotionalState.HOPEFUL,
            EmotionalState.EMPATHETIC,
        ]

    def test_max_friction_score(self):
        """Test emotional state update with maximum friction."""
        agent = SocialAgent(
            agent_id="test_001",
            name="Stressed",
            group="Youth_Activists",
            core_values=["justice"],
            openness_to_change=0.5,
        )
        agent.update_emotional_state(1.0)
        assert agent.emotional_state in [
            EmotionalState.ANGRY,
            EmotionalState.FEARFUL,
            EmotionalState.OUTRAGED,
        ]

    def test_extreme_ideology_drift(self):
        """Test ideology drift at extreme boundaries."""
        profile = PoliticalProfile(ideology_position=-1.0)
        profile.drift_toward(1.0, rate=0.1)
        assert -1.0 <= profile.ideology_position <= 1.0

    def test_political_profile_default_faction(self):
        """Test PoliticalProfile with default faction."""
        profile = PoliticalProfile()
        assert profile.faction == "unaligned"
