"""
Social Agent — An LLM-powered autonomous actor in the friction simulation.

Each agent represents an individual within a social group, with beliefs,
political ideology, memory, emotional state, and the ability to interact
with other agents across both social/cultural and political dimensions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.model.inference import FrictionLLM


class EmotionalState(Enum):
    """Valence-arousal emotional states for agents."""
    CALM = "calm"
    ANXIOUS = "anxious"
    ANGRY = "angry"
    HOPEFUL = "hopeful"
    FEARFUL = "fearful"
    EMPATHETIC = "empathetic"
    HOSTILE = "hostile"
    INDIFFERENT = "indifferent"
    OUTRAGED = "outraged"          # political: strong moral anger
    DISILLUSIONED = "disillusioned"  # political: loss of faith in system
    GALVANIZED = "galvanized"      # political: motivated to act


@dataclass
class AgentMemory:
    """Rolling memory buffer for an agent's experiences."""
    events: list[dict] = field(default_factory=list)
    max_size: int = 25

    def add(self, event: dict):
        self.events.append(event)
        if len(self.events) > self.max_size:
            self.events.pop(0)

    def recent(self, n: int = 5) -> list[dict]:
        return self.events[-n:]

    def summarize(self) -> str:
        if not self.events:
            return "No significant events experienced."
        summaries = [e.get("summary", str(e)) for e in self.events[-10:]]
        return "\n".join(f"- {s}" for s in summaries)

    def political_history(self) -> list[dict]:
        """Filter memory for political events only."""
        return [e for e in self.events if e.get("domain") == "political"]

    def social_history(self) -> list[dict]:
        """Filter memory for social/cultural events only."""
        return [e for e in self.events if e.get("domain") in ("social", "cultural")]


@dataclass
class PoliticalProfile:
    """
    An agent's political identity and dynamics.

    ideology_position: -1.0 (far-left) to +1.0 (far-right)
    faction: which political faction they currently align with
    """
    ideology_position: float = 0.0          # -1.0 to 1.0
    faction: str = "unaligned"
    partisan_strength: float = 0.5          # how strongly they identify with faction
    political_engagement: float = 0.5       # how active they are politically
    policy_priorities: list[str] = field(default_factory=list)
    voted_history: list[dict] = field(default_factory=list)

    # Drift tracking
    ideology_history: list[float] = field(default_factory=list)

    def drift_toward(self, target: float, rate: float = 0.05):
        """Gradually shift ideology toward a target position."""
        self.ideology_history.append(self.ideology_position)
        delta = (target - self.ideology_position) * rate
        self.ideology_position = max(-1.0, min(1.0, self.ideology_position + delta))

    def radicalize(self, rate: float = 0.05):
        """Move further from center (toward whichever extreme is closer)."""
        direction = 1.0 if self.ideology_position >= 0 else -1.0
        self.ideology_position = max(-1.0, min(1.0,
            self.ideology_position + direction * rate
        ))

    def moderate(self, rate: float = 0.03):
        """Move closer to center."""
        direction = -1.0 if self.ideology_position > 0 else 1.0
        if abs(self.ideology_position) < rate:
            self.ideology_position = 0.0
        else:
            self.ideology_position += direction * rate

    @property
    def ideology_label(self) -> str:
        """Human-readable label for current ideology position."""
        pos = self.ideology_position
        if pos < -0.6:
            return "far-left"
        elif pos < -0.2:
            return "center-left"
        elif pos <= 0.2:
            return "centrist"
        elif pos <= 0.6:
            return "center-right"
        else:
            return "far-right"

    @property
    def is_moderate(self) -> bool:
        return abs(self.ideology_position) < 0.3

    @property
    def is_extreme(self) -> bool:
        return abs(self.ideology_position) > 0.7


@dataclass
class SocialAgent:
    """
    An autonomous agent in the society simulation.

    Each agent has a cultural identity, political ideology, personal values,
    memory of past interactions, and an emotional state that influences their
    behavior in both social and political contexts.
    """
    agent_id: str
    name: str
    group: str                                # cultural/social group
    core_values: list[str]
    openness_to_change: float                 # 0.0 (rigid) to 1.0 (very open)

    # Political identity
    politics: PoliticalProfile = field(default_factory=PoliticalProfile)

    # Dynamic state
    emotional_state: EmotionalState = EmotionalState.CALM
    friction_tolerance: float = 0.5
    trust_scores: dict[str, float] = field(default_factory=dict)
    memory: AgentMemory = field(default_factory=AgentMemory)
    connections: list[str] = field(default_factory=list)

    # Internal tracking
    stance_history: list[dict] = field(default_factory=list)

    def build_identity_prompt(self) -> str:
        """Create the agent's full identity context for LLM prompts."""
        return (
            f"You are {self.name}, a member of the {self.group} community.\n"
            f"Your core values: {', '.join(self.core_values)}.\n"
            f"Your openness to change: {self.openness_to_change:.1f}/1.0.\n"
            f"Current emotional state: {self.emotional_state.value}.\n"
            f"Friction tolerance: {self.friction_tolerance:.2f}.\n"
            f"\n--- Political Identity ---\n"
            f"Political ideology: {self.politics.ideology_label} "
            f"(position: {self.politics.ideology_position:+.2f})\n"
            f"Faction: {self.politics.faction}\n"
            f"Partisan strength: {self.politics.partisan_strength:.2f}\n"
            f"Political engagement: {self.politics.political_engagement:.2f}\n"
            f"Policy priorities: {', '.join(self.politics.policy_priorities) if self.politics.policy_priorities else 'none specified'}.\n"
        )

    def react_to_event(self, event: dict, llm: FrictionLLM) -> dict:
        """Generate this agent's reaction to a friction event using the LLM."""
        identity = self.build_identity_prompt()
        memory_context = self.memory.summarize()
        domain = event.get("domain", "social")

        # Tailor instructions based on event domain
        if domain == "political":
            domain_instruction = (
                f"This is a POLITICAL event. React as {self.name} considering your ideology "
                f"({self.politics.ideology_label}), faction loyalty ({self.politics.faction}), "
                f"and policy priorities. Consider how this affects your political engagement "
                f"and whether it pushes you toward more extreme or moderate positions."
            )
        elif domain == "crossover":
            domain_instruction = (
                f"This event spans BOTH social/cultural AND political domains. React as "
                f"{self.name} considering how your cultural identity ({self.group}) and "
                f"political ideology ({self.politics.ideology_label}) interact. "
                f"Does your cultural background conflict with your political faction's stance?"
            )
        else:
            domain_instruction = (
                f"This is a SOCIAL/CULTURAL event. React as {self.name} considering your "
                f"cultural identity, values, and community bonds."
            )

        prompt = (
            f"<|system|>\nYou are simulating a social agent in a multi-domain friction scenario.\n"
            f"<|identity|>\n{identity}\n"
            f"<|memory|>\n{memory_context}\n"
            f"<|event|>\n{event.get('description', '')}\n"
            f"<|domain|>\n{domain_instruction}\n"
            f"<|instruction|>\n"
            f"React to this event as {self.name}. Consider your values, emotional state, "
            f"political ideology, and past experiences. Respond with:\n"
            f"1. Your emotional reaction\n"
            f"2. Your stance (support / oppose / neutral / compromise)\n"
            f"3. What action you would take\n"
            f"4. How this changes your view of other groups and factions\n"
            f"5. Whether this shifts your political position at all\n"
            f"<|response|>\n"
        )

        raw_response = llm.generate(prompt, max_new_tokens=400, temperature=0.7)

        reaction = {
            "agent_id": self.agent_id,
            "event_id": event.get("event_id"),
            "domain": domain,
            "raw_response": raw_response,
            "emotional_state_before": self.emotional_state.value,
            "ideology_before": self.politics.ideology_position,
        }

        # Update memory with domain tagging
        self.memory.add({
            "type": "reaction",
            "domain": domain,
            "event": event.get("description", ""),
            "summary": raw_response[:200],
        })

        return reaction

    def react_to_election(self, election_result: dict, llm: FrictionLLM) -> dict:
        """React to an election result."""
        won = election_result.get("winning_faction") == self.politics.faction
        identity = self.build_identity_prompt()

        prompt = (
            f"<|system|>\nSimulate an agent reacting to an election result.\n"
            f"<|identity|>\n{identity}\n"
            f"<|election_result|>\n"
            f"Winning faction: {election_result.get('winning_faction')}\n"
            f"Vote margin: {election_result.get('margin', 'unknown')}\n"
            f"Your faction {'WON' if won else 'LOST'} this election.\n"
            f"<|instruction|>\n"
            f"React as {self.name}. How do you feel about this result? "
            f"Does it change your engagement level, trust in the system, "
            f"or willingness to compromise? Are you more or less radicalized?\n"
            f"<|response|>\n"
        )

        raw_response = llm.generate(prompt, max_new_tokens=300, temperature=0.7)

        # Record vote
        self.politics.voted_history.append({
            "result": "won" if won else "lost",
            "faction_voted": self.politics.faction,
            "winner": election_result.get("winning_faction"),
        })

        self.memory.add({
            "type": "election",
            "domain": "political",
            "summary": f"Election: {'our faction won' if won else 'our faction lost'}",
        })

        return {
            "agent_id": self.agent_id,
            "faction": self.politics.faction,
            "won": won,
            "raw_response": raw_response,
        }

    def interact_with(self, other: SocialAgent, topic: str, llm: FrictionLLM) -> dict:
        """Simulate a dialogue/interaction between two agents."""
        trust = self.trust_scores.get(other.agent_id, 0.5)
        ideology_gap = abs(self.politics.ideology_position - other.politics.ideology_position)

        prompt = (
            f"<|system|>\nSimulate a conversation between two people with different backgrounds "
            f"and political views.\n"
            f"<|speaker_1|>\n{self.build_identity_prompt()}\n"
            f"<|speaker_2|>\n{other.build_identity_prompt()}\n"
            f"<|relationship|>\n"
            f"Trust level: {trust:.2f}\n"
            f"Ideological gap: {ideology_gap:.2f} "
            f"({self.politics.ideology_label} vs {other.politics.ideology_label})\n"
            f"Same social group: {'yes' if self.group == other.group else 'no'}\n"
            f"Same faction: {'yes' if self.politics.faction == other.politics.faction else 'no'}\n"
            f"<|topic|>\n{topic}\n"
            f"<|instruction|>\n"
            f"Generate a brief exchange (3-4 turns) between {self.name} and {other.name} "
            f"about this topic. Show how their different values AND political views "
            f"create friction or find common ground. Consider whether cultural bonds can "
            f"bridge political divides, or whether politics deepens cultural friction.\n"
            f"<|dialogue|>\n"
        )

        dialogue = llm.generate(prompt, max_new_tokens=500, temperature=0.8)

        # Update trust — ideology gap reduces trust recovery
        trust_modifier = 1.0 - (ideology_gap * 0.5)  # closer ideology = easier trust
        trust_delta = random.uniform(-0.1, 0.1) * trust_modifier
        self.trust_scores[other.agent_id] = max(0.0, min(1.0, trust + trust_delta))

        # Possible ideology influence (small drift toward each other if high trust)
        if trust > 0.6 and random.random() < 0.3:
            midpoint = (self.politics.ideology_position + other.politics.ideology_position) / 2
            self.politics.drift_toward(midpoint, rate=0.02)
            other.politics.drift_toward(midpoint, rate=0.02)

        return {
            "agents": [self.agent_id, other.agent_id],
            "topic": topic,
            "dialogue": dialogue,
            "trust_change": trust_delta,
            "ideology_gap": ideology_gap,
            "factions": [self.politics.faction, other.politics.faction],
        }

    def consume_media(self, outlet: dict):
        """
        Agent consumes media from an outlet, potentially shifting their views.
        Echo chamber effect: biased media reinforces existing position.
        """
        media_bias = outlet.get("ideology_bias", 0.0)
        credibility = outlet.get("credibility", 0.5)

        # If media aligns with agent's ideology, it reinforces
        alignment = 1.0 - abs(self.politics.ideology_position - media_bias) / 2.0
        if alignment > 0.6:
            # Echo chamber: drift toward media's position
            self.politics.drift_toward(media_bias, rate=0.02 * credibility)
        elif alignment < 0.3 and self.openness_to_change > 0.5:
            # Cross-cutting: may moderate slightly
            self.politics.moderate(rate=0.01 * self.openness_to_change)

    def update_emotional_state(self, friction_score: float, political_friction: float = 0.0):
        """Update emotional state based on social AND political friction."""
        combined = max(friction_score, political_friction)

        if combined > 0.8:
            self.emotional_state = random.choice([
                EmotionalState.ANGRY, EmotionalState.OUTRAGED, EmotionalState.FEARFUL
            ])
        elif combined > 0.6:
            self.emotional_state = random.choice([
                EmotionalState.ANXIOUS, EmotionalState.HOSTILE, EmotionalState.GALVANIZED
            ])
        elif combined > 0.4:
            self.emotional_state = random.choice([
                EmotionalState.ANXIOUS, EmotionalState.DISILLUSIONED, EmotionalState.CALM
            ])
        elif combined > 0.2:
            self.emotional_state = EmotionalState.CALM
        else:
            self.emotional_state = random.choice([
                EmotionalState.HOPEFUL, EmotionalState.EMPATHETIC
            ])

    def ideology_summary(self) -> dict:
        """Return a summary of the agent's political state for metrics."""
        return {
            "agent_id": self.agent_id,
            "group": self.group,
            "faction": self.politics.faction,
            "ideology": self.politics.ideology_position,
            "label": self.politics.ideology_label,
            "engagement": self.politics.political_engagement,
            "partisan_strength": self.politics.partisan_strength,
            "is_extreme": self.politics.is_extreme,
        }
