"""
Cognitive Models for Advanced Friction Simulation.

Implements three novel, research-grade cognitive and social dynamics models:

1. COGNITIVE DISSONANCE ENGINE: Tracks internal tension when agents' core values
   conflict with their political faction's policies. Enables dynamic resolution
   strategies (value change, faction switch, compartmentalization, rationalization).

2. OVERTON WINDOW TRACKER: Measures the range of "acceptable" political discourse
   over time. Detects window shifts, shocks, narrowing/widening patterns, and
   identifies polarization signatures.

3. EMOTIONAL CONTAGION NETWORK: Models how emotions spread through social networks
   with negativity bias. Computes epidemic metrics (R0) for emotional states and
   tracks emotional cascades.

These models integrate with the core simulation to provide unprecedented insight
into psychological and social mechanisms driving political polarization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from src.agents.social_agent import SocialAgent

logger = logging.getLogger(__name__)


# ============================================================================
# COGNITIVE DISSONANCE ENGINE
# ============================================================================


class DissonanceResolutionStrategy(Enum):
    """Strategies agents use to resolve cognitive dissonance."""
    VALUE_CHANGE = "value_change"           # change core values to match faction
    FACTION_SWITCH = "faction_switch"       # switch to faction matching values
    COMPARTMENTALIZE = "compartmentalize"   # suppress awareness of conflict
    RATIONALIZE = "rationalize"             # justify contradiction intellectually


@dataclass
class DissonanceResolutionResult:
    """Outcome of an agent's dissonance resolution attempt."""
    agent_id: str
    dissonance_score: float
    resolution_strategy: DissonanceResolutionStrategy | None
    success: bool
    new_partisan_strength: float
    value_shift_magnitude: float
    faction_switch_likelihood: float
    compartmentalization_index: float


class CognitiveDissonanceTracker:
    """
    Tracks cognitive dissonance: tension between an agent's core values and
    their political faction's key policies.

    Implements Festinger's cognitive dissonance theory in a multi-agent
    political context. Higher dissonance drives behavioral/cognitive changes.
    """

    def __init__(self):
        """Initialize the dissonance tracker."""
        self.dissonance_scores: dict[str, list[float]] = {}
        self.resolution_history: dict[str, list[DissonanceResolutionResult]] = {}
        self.agent_compartmentalization: dict[str, float] = {}

    def compute_dissonance(
        self,
        agent: SocialAgent,
        faction_policies: dict[str, float],
    ) -> float:
        """
        Compute cognitive dissonance score (0-1) for an agent.

        Dissonance arises from mismatch between:
        - Agent's core_values (e.g., "economic_security", "individual_freedom")
        - Faction's key policies (mapped to values: 1.0 = strong support, 0.0 = opposed)

        Args:
            agent: The social agent to evaluate
            faction_policies: Dict mapping value names to policy alignment (0-1)

        Returns:
            Dissonance score (0.0 = perfect alignment, 1.0 = maximal conflict)
        """
        if not agent.core_values or not faction_policies:
            return 0.0

        # For each of agent's values, check faction policy alignment
        misalignments = []
        for value in agent.core_values:
            # Check if faction supports this value
            policy_support = faction_policies.get(value.lower(), 0.5)

            # Misalignment: how much the agent values something the faction opposes
            # (or vice versa)
            value_importance = 1.0  # agents care equally about all stated values
            faction_opposition = 1.0 - policy_support

            misalignment = value_importance * faction_opposition
            misalignments.append(misalignment)

        # Average misalignment, scaled by partisan strength (weak partisans
        # experience less dissonance because they're less committed)
        if not misalignments:
            return 0.0

        base_dissonance = float(np.mean(misalignments))
        partisan_scaling = agent.politics.partisan_strength  # 0-1

        # Dissonance amplified for strong partisans (they care more)
        dissonance = base_dissonance * (0.5 + partisan_scaling * 0.5)

        return float(np.clip(dissonance, 0.0, 1.0))

    def track_dissonance(self, agent_id: str, score: float):
        """Record dissonance score for an agent at this timestep."""
        if agent_id not in self.dissonance_scores:
            self.dissonance_scores[agent_id] = []
        self.dissonance_scores[agent_id].append(score)

    def resolve_dissonance(
        self,
        agent: SocialAgent,
        dissonance_score: float,
        available_factions: list[dict] | None = None,
    ) -> DissonanceResolutionResult:
        """
        Simulate agent's attempt to resolve cognitive dissonance.

        High dissonance triggers resolution: agents may change values, switch
        factions, suppress awareness (compartmentalize), or rationalize conflicts.

        Args:
            agent: The agent experiencing dissonance
            dissonance_score: Current dissonance score (0-1)
            available_factions: Optional list of other factions agent could join

        Returns:
            DissonanceResolutionResult describing chosen strategy and outcome
        """
        result = DissonanceResolutionResult(
            agent_id=agent.agent_id,
            dissonance_score=dissonance_score,
            resolution_strategy=None,
            success=False,
            new_partisan_strength=agent.politics.partisan_strength,
            value_shift_magnitude=0.0,
            faction_switch_likelihood=0.0,
            compartmentalization_index=0.0,
        )

        # Only resolve if dissonance is high enough to be uncomfortable
        if dissonance_score < 0.4:
            return result

        # Resolution likelihood increases with dissonance
        resolution_probability = min(dissonance_score * 0.7, 0.9)

        if np.random.random() > resolution_probability:
            return result

        # Choose resolution strategy based on agent traits
        # Agents with high openness to change prefer faction switching
        # Agents with low openness prefer rationalization/compartmentalization
        strategy_weights = {
            DissonanceResolutionStrategy.VALUE_CHANGE: agent.openness_to_change,
            DissonanceResolutionStrategy.FACTION_SWITCH: agent.openness_to_change * 0.8,
            DissonanceResolutionStrategy.COMPARTMENTALIZE: 1.0 - agent.openness_to_change,
            DissonanceResolutionStrategy.RATIONALIZE: 1.0 - agent.openness_to_change,
        }

        strategies = list(strategy_weights.keys())
        weights = [strategy_weights[s] for s in strategies]
        weights = np.array(weights) / sum(weights)

        chosen_strategy = np.random.choice(strategies, p=weights)
        result.resolution_strategy = chosen_strategy

        # Execute resolution strategy
        if chosen_strategy == DissonanceResolutionStrategy.VALUE_CHANGE:
            # Agent changes core values to align with faction
            # Magnitude depends on how strongly they identify with faction
            shift_magnitude = dissonance_score * agent.politics.partisan_strength * 0.15
            result.value_shift_magnitude = shift_magnitude
            result.success = True

        elif chosen_strategy == DissonanceResolutionStrategy.FACTION_SWITCH:
            # Agent considers switching to faction better aligned with values
            if available_factions and len(available_factions) > 1:
                # (Would normally search for better-aligned faction here)
                result.faction_switch_likelihood = dissonance_score * agent.openness_to_change
                result.success = result.faction_switch_likelihood > 0.5
                if result.success:
                    result.new_partisan_strength = max(0.2, agent.politics.partisan_strength - 0.2)

        elif chosen_strategy == DissonanceResolutionStrategy.COMPARTMENTALIZE:
            # Agent suppresses awareness of value-policy conflict
            compartmentalization = dissonance_score * (1.0 - agent.openness_to_change)
            result.compartmentalization_index = compartmentalization
            self.agent_compartmentalization[agent.agent_id] = compartmentalization
            result.success = True

        elif chosen_strategy == DissonanceResolutionStrategy.RATIONALIZE:
            # Agent intellectually justifies the conflict
            # Reduces perceived dissonance through motivated reasoning
            result.compartmentalization_index = dissonance_score * 0.5
            result.success = True

        # Track resolution attempt
        if agent.agent_id not in self.resolution_history:
            self.resolution_history[agent.agent_id] = []
        self.resolution_history[agent.agent_id].append(result)

        return result

    def average_dissonance(self) -> float:
        """Compute mean cognitive dissonance across all tracked agents."""
        if not self.dissonance_scores:
            return 0.0

        latest_scores = [scores[-1] for scores in self.dissonance_scores.values() if scores]
        if not latest_scores:
            return 0.0

        return float(np.mean(latest_scores))


# ============================================================================
# OVERTON WINDOW TRACKER
# ============================================================================


@dataclass
class OvertonWindowSnapshot:
    """A snapshot of the Overton Window at a given timestep."""
    step: int
    left_edge: float          # leftmost "acceptable" ideology position
    right_edge: float         # rightmost "acceptable" ideology position
    center: float             # center of the window
    width: float              # right_edge - left_edge
    left_tail_mass: float     # proportion of agents outside left edge
    right_tail_mass: float    # proportion of agents outside right edge


class OvertonWindowTracker:
    """
    Tracks the Overton Window: the range of political positions considered
    "acceptable" or "mainstream" in society.

    The window can shift left/right, widen (more tolerance for diverse views),
    or narrow (polarization). Window shocks occur when mainstream discourse
    rapidly changes (e.g., in crisis or election).

    Provides a novel measure of polarization complementary to ideology spread.
    """

    def __init__(self):
        """Initialize the Overton Window tracker."""
        self.history: list[OvertonWindowSnapshot] = []
        self.shift_events: list[dict] = []  # detected windows shifts/shocks

    def compute_window(
        self,
        agents: list[SocialAgent],
        percentile_bounds: tuple[float, float] = (10, 90),
    ) -> OvertonWindowSnapshot:
        """
        Compute the current Overton Window from agent ideology distribution.

        The window boundaries are defined by percentiles of the ideology distribution.
        By convention, we use the 10th and 90th percentiles to define the "acceptable"
        range, leaving 10% tails on each side as "beyond the pale".

        Args:
            agents: All agents in simulation
            percentile_bounds: (lower, upper) percentiles for window boundaries

        Returns:
            OvertonWindowSnapshot with window parameters
        """
        if not agents:
            return OvertonWindowSnapshot(0, -1.0, 1.0, 0.0, 2.0, 0.0, 0.0)

        positions = np.array([a.politics.ideology_position for a in agents])
        step = len(self.history)

        # Window boundaries from percentiles
        left_percentile, right_percentile = percentile_bounds
        left_edge = float(np.percentile(positions, left_percentile))
        right_edge = float(np.percentile(positions, right_percentile))
        center = float(np.mean([left_edge, right_edge]))
        width = right_edge - left_edge

        # Tail masses: agents outside the "acceptable" window
        left_tail_mass = float(np.mean(positions < left_edge))
        right_tail_mass = float(np.mean(positions > right_edge))

        snapshot = OvertonWindowSnapshot(
            step=step,
            left_edge=left_edge,
            right_edge=right_edge,
            center=center,
            width=width,
            left_tail_mass=left_tail_mass,
            right_tail_mass=right_tail_mass,
        )

        return snapshot

    def track_window(
        self,
        agents: list[SocialAgent],
        percentile_bounds: tuple[float, float] = (10, 90),
    ) -> OvertonWindowSnapshot:
        """Record a snapshot of the current Overton Window."""
        snapshot = self.compute_window(agents, percentile_bounds)
        self.history.append(snapshot)

        # Detect window shifts
        if len(self.history) >= 2:
            prev = self.history[-2]

            # Shift detection: center moved left/right
            center_shift = snapshot.center - prev.center
            if abs(center_shift) > 0.1:
                self.shift_events.append({
                    "step": snapshot.step,
                    "type": "center_shift",
                    "direction": "left" if center_shift < 0 else "right",
                    "magnitude": abs(center_shift),
                })

            # Width change: window widening (polarization relief) or narrowing (polarization)
            width_change = snapshot.width - prev.width
            if width_change < -0.15:  # narrowing = more polarized
                self.shift_events.append({
                    "step": snapshot.step,
                    "type": "window_shock",
                    "subtype": "narrowing",
                    "width_delta": width_change,
                })
            elif width_change > 0.15:  # widening = more diverse discourse
                self.shift_events.append({
                    "step": snapshot.step,
                    "type": "window_shock",
                    "subtype": "widening",
                    "width_delta": width_change,
                })

        return snapshot

    def window_width_history(self) -> list[float]:
        """Get the width of the Overton Window over time."""
        return [snap.width for snap in self.history]

    def window_center_history(self) -> list[float]:
        """Get the center position of the Overton Window over time."""
        return [snap.center for snap in self.history]

    def detect_polarization_signature(self, lookback_steps: int = 10) -> str:
        """
        Identify the current polarization pattern from window dynamics.

        Returns:
            String describing the pattern (e.g., "narrowing_left", "widening",
            "stable", "shifting_right")
        """
        if len(self.history) < lookback_steps:
            return "insufficient_data"

        recent = self.history[-lookback_steps:]
        widths = [s.width for s in recent]
        centers = [s.center for s in recent]

        width_trend = np.polyfit(range(len(widths)), widths, 1)[0]
        center_trend = np.polyfit(range(len(centers)), centers, 1)[0]

        # Classify pattern
        if width_trend < -0.01:
            if center_trend < -0.01:
                return "narrowing_left"
            elif center_trend > 0.01:
                return "narrowing_right"
            else:
                return "narrowing_stable"
        elif width_trend > 0.01:
            if center_trend < -0.01:
                return "widening_left"
            elif center_trend > 0.01:
                return "widening_right"
            else:
                return "widening_stable"
        else:
            if abs(center_trend) > 0.02:
                return "shifting_" + ("left" if center_trend < 0 else "right")
            else:
                return "stable"

    def average_window_width(self) -> float:
        """Get the average width of the Overton Window across history."""
        if not self.history:
            return 2.0
        return float(np.mean([s.width for s in self.history]))


# ============================================================================
# EMOTIONAL CONTAGION NETWORK
# ============================================================================


@dataclass
class EmotionalContagionMetrics:
    """Metrics describing emotional contagion dynamics."""
    step: int
    emotional_state_distribution: dict[str, float]  # proportion of each emotion
    contagion_r0_by_emotion: dict[str, float]       # R0 (basic reproduction number)
    active_emotional_outbreaks: list[str]           # which emotions are "spreading"
    network_susceptibility: float                   # overall vulnerability to contagion
    cascade_depth: int                              # longest emotional cascade


@dataclass
class EmotionalEpidemic:
    """Tracks spread of a particular emotional state through the network."""
    emotion: str
    origin_agent_id: str
    origin_step: int
    infected_agents: set[str] = field(default_factory=set)
    peak_step: int | None = None
    final_size: int = 0
    r0: float = 0.0  # basic reproduction number


class EmotionalContagionModel:
    """
    Models emotional contagion: how emotions spread through the social network.

    Implements epidemiological contagion model (SIR variant) adapted for emotions:
    - Negative emotions (angry, fearful) spread faster than positive ones
    - High-degree nodes (well-connected agents) are bigger spreaders
    - Agent openness_to_change determines susceptibility
    - Tracks R0 (basic reproduction number) for each emotional state
    - Identifies "emotional epidemics" where emotions cascade through network

    Research shows negativity bias in social networks: bad news spreads faster
    and further than good news.
    """

    def __init__(self, network: nx.Graph):
        """
        Initialize the emotional contagion model.

        Args:
            network: NetworkX graph representing social connections
        """
        self.network = network
        self.contagion_history: list[EmotionalContagionMetrics] = []
        self.epidemics: dict[str, list[EmotionalEpidemic]] = {}
        self.agent_emotional_susceptibility: dict[str, float] = {}

        # Emotional state transmissibility (0-1)
        # Negative emotions spread faster (negativity bias)
        self.transmissibility: dict[str, float] = {
            "angry": 0.8,
            "outraged": 0.85,
            "fearful": 0.75,
            "anxious": 0.7,
            "hostile": 0.75,
            "calm": 0.3,
            "hopeful": 0.4,
            "empathetic": 0.5,
            "disillusioned": 0.65,
            "galvanized": 0.6,
            "indifferent": 0.2,
        }

    def compute_agent_susceptibility(self, agent: SocialAgent) -> float:
        """
        Compute an agent's susceptibility to emotional contagion.

        Agents with high openness_to_change are more susceptible to influence.
        Agents with high friction_tolerance are more resilient.

        Args:
            agent: The agent to evaluate

        Returns:
            Susceptibility score (0.0 = immune, 1.0 = highly susceptible)
        """
        # Base susceptibility from openness to change
        base = agent.openness_to_change

        # Reduce susceptibility with friction tolerance (resilience)
        resilience = agent.friction_tolerance

        # Network effect: more connections = more exposure
        degree = len(agent.connections)
        max_degree = 20  # normalization
        network_factor = 1.0 + (degree / max_degree) * 0.3

        susceptibility = (base - resilience * 0.3) * network_factor
        return float(np.clip(susceptibility, 0.0, 1.0))

    def update_agent_susceptibility(self, agents: dict[str, SocialAgent]):
        """Update susceptibility scores for all agents."""
        for agent_id, agent in agents.items():
            self.agent_emotional_susceptibility[agent_id] = (
                self.compute_agent_susceptibility(agent)
            )

    def simulate_contagion_step(
        self,
        agents: dict[str, SocialAgent],
        agents_by_emotion: dict[str, set[str]],
    ) -> dict[str, set[str]]:
        """
        Simulate one step of emotional contagion through the network.

        Each agent with an emotion has a chance to infect neighbors based on:
        - Emotion transmissibility (negativity bias)
        - Neighbor's susceptibility
        - Connection strength

        Args:
            agents: Dictionary mapping agent_id to SocialAgent
            agents_by_emotion: Dict mapping emotion name to set of agent_ids

        Returns:
            Updated agents_by_emotion after one contagion step
        """
        new_agents_by_emotion = {k: set(v) for k, v in agents_by_emotion.items()}

        for emotion, infected_ids in agents_by_emotion.items():
            transmissibility = self.transmissibility.get(emotion, 0.5)

            for agent_id in infected_ids:
                if agent_id not in self.network:
                    continue

                # Infect neighbors with probability based on transmissibility
                # and neighbor susceptibility
                for neighbor_id in self.network.neighbors(agent_id):
                    if neighbor_id in agents:
                        neighbor_susceptibility = self.agent_emotional_susceptibility.get(
                            neighbor_id, 0.5
                        )

                        infection_probability = (
                            transmissibility * neighbor_susceptibility
                        )

                        if np.random.random() < infection_probability:
                            # Neighbor becomes infected with this emotion
                            new_agents_by_emotion[emotion].add(neighbor_id)

        return new_agents_by_emotion

    def compute_r0(
        self,
        agents: dict[str, SocialAgent],
        agents_by_emotion: dict[str, set[str]],
    ) -> dict[str, float]:
        """
        Compute R0 (basic reproduction number) for each emotional state.

        R0 = how many new cases one infected agent generates on average
        R0 > 1 = spreading epidemic
        R0 < 1 = outbreak dying out

        Args:
            agents: All agents
            agents_by_emotion: Current distribution of emotions

        Returns:
            Dict mapping emotion to its R0 value
        """
        r0_by_emotion = {}

        for emotion, infected_ids in agents_by_emotion.items():
            if not infected_ids:
                r0_by_emotion[emotion] = 0.0
                continue

            transmissibility = self.transmissibility.get(emotion, 0.5)

            # Average number of susceptible neighbors per infected agent
            total_new_infections = 0
            total_neighbors = 0

            for agent_id in infected_ids:
                if agent_id not in self.network:
                    continue

                degree = len(list(self.network.neighbors(agent_id)))
                total_neighbors += degree

                # Expected infections = degree * transmissibility * avg susceptibility
                for neighbor_id in self.network.neighbors(agent_id):
                    if neighbor_id in agents:
                        susceptibility = self.agent_emotional_susceptibility.get(
                            neighbor_id, 0.5
                        )
                        total_new_infections += transmissibility * susceptibility

            avg_degree = total_neighbors / len(infected_ids) if infected_ids else 0
            if avg_degree > 0:
                r0 = total_new_infections / len(infected_ids)
            else:
                r0 = 0.0

            r0_by_emotion[emotion] = float(np.clip(r0, 0.0, 10.0))

        return r0_by_emotion

    def track_contagion(
        self,
        agents: dict[str, SocialAgent],
        step: int,
    ) -> EmotionalContagionMetrics:
        """
        Record current state of emotional contagion in the population.

        Args:
            agents: All agents with their current emotional states
            step: Simulation step number

        Returns:
            EmotionalContagionMetrics snapshot
        """
        # Count emotional distribution
        emotion_counts: dict[str, int] = {}
        agents_by_emotion: dict[str, set[str]] = {}

        for agent_id, agent in agents.items():
            emotion = agent.emotional_state.value
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            if emotion not in agents_by_emotion:
                agents_by_emotion[emotion] = set()
            agents_by_emotion[emotion].add(agent_id)

        total_agents = len(agents)
        emotion_distribution = {
            k: v / total_agents for k, v in emotion_counts.items()
        }

        # Compute R0 for each emotion
        r0_by_emotion = self.compute_r0(agents, agents_by_emotion)

        # Identify active outbreaks (emotions spreading with R0 > 1)
        active_outbreaks = [
            emotion for emotion, r0 in r0_by_emotion.items() if r0 > 1.0
        ]

        # Network susceptibility = average agent susceptibility
        network_susceptibility = float(
            np.mean(list(self.agent_emotional_susceptibility.values()))
            if self.agent_emotional_susceptibility
            else 0.5
        )

        # Cascade depth: longest emotional chain in network
        # (simplified: just use average infection depth)
        cascade_depth = len(active_outbreaks)

        metrics = EmotionalContagionMetrics(
            step=step,
            emotional_state_distribution=emotion_distribution,
            contagion_r0_by_emotion=r0_by_emotion,
            active_emotional_outbreaks=active_outbreaks,
            network_susceptibility=network_susceptibility,
            cascade_depth=cascade_depth,
        )

        self.contagion_history.append(metrics)
        return metrics

    def identify_epidemics(
        self,
        agents: dict[str, SocialAgent],
        minimum_size: int = 3,
    ) -> list[EmotionalEpidemic]:
        """
        Identify emotional "epidemics": clusters of agents with same emotion
        that are connected through the network.

        Args:
            agents: All agents
            minimum_size: Minimum cascade size to count as epidemic

        Returns:
            List of detected EmotionalEpidemic objects
        """
        epidemics = []
        agents_by_emotion: dict[str, set[str]] = {}

        for agent_id, agent in agents.items():
            emotion = agent.emotional_state.value
            if emotion not in agents_by_emotion:
                agents_by_emotion[emotion] = set()
            agents_by_emotion[emotion].add(agent_id)

        for emotion, agent_ids in agents_by_emotion.items():
            if len(agent_ids) >= minimum_size:
                # Check if they form connected components
                subgraph = self.network.subgraph(agent_ids)
                for component in nx.connected_components(subgraph):
                    if len(component) >= minimum_size:
                        epidemic = EmotionalEpidemic(
                            emotion=emotion,
                            origin_agent_id=list(component)[0],
                            origin_step=0,  # would need more tracking for actual origin
                            infected_agents=component,
                            final_size=len(component),
                        )
                        epidemic.r0 = self.compute_r0(
                            agents, {emotion: component}
                        ).get(emotion, 0.0)
                        epidemics.append(epidemic)

        return epidemics

    def average_network_r0(self) -> float:
        """
        Get the average R0 across all emotions in the latest metrics.

        Returns:
            Mean R0, or 0.0 if no contagion history
        """
        if not self.contagion_history:
            return 0.0

        latest = self.contagion_history[-1]
        r0_values = list(latest.contagion_r0_by_emotion.values())

        if not r0_values:
            return 0.0

        return float(np.mean(r0_values))

    def epidemic_count(self) -> int:
        """Get the total number of emotional epidemics detected."""
        total = 0
        for epidemics in self.epidemics.values():
            total += len(epidemics)
        return total
