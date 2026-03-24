"""
Simulation Engine — Orchestrates friction dynamics across all agents.

Manages the social network, generates both social/cultural AND political
friction events, runs elections, tracks polarization, and collects metrics.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import yaml

if TYPE_CHECKING:
    from src.agents.social_agent import SocialAgent
    from src.model.inference import FrictionLLM

logger = logging.getLogger(__name__)


@dataclass
class FrictionEvent:
    """A friction-inducing event in the simulation."""
    event_id: str
    event_type: str
    domain: str                   # "social", "political", or "crossover"
    description: str
    affected_groups: list[str]
    affected_factions: list[str]
    severity: float               # 0.0 to 1.0
    step: int


@dataclass
class SimulationState:
    """Tracks the global state of the simulation."""
    current_step: int = 0
    global_friction_score: float = 0.0
    political_friction_score: float = 0.0

    group_friction_scores: dict[str, float] = field(default_factory=dict)
    faction_friction_scores: dict[str, float] = field(default_factory=dict)
    polarization_index: float = 0.0

    events_log: list[dict] = field(default_factory=list)
    interaction_log: list[dict] = field(default_factory=list)
    election_log: list[dict] = field(default_factory=list)
    metrics_history: list[dict] = field(default_factory=list)
    ideology_snapshots: list[dict] = field(default_factory=list)


class SimulationEngine:
    """
    Core engine that runs the Dynamic Society Friction Simulation.

    Covers both social/cultural conflicts and political polarization,
    including elections, media influence, ideology drift, and cross-domain
    friction cascades.
    """

    def __init__(
        self,
        config_path: str = "configs/simulation_config.yaml",
        llm: FrictionLLM | None = None,
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.llm = llm
        self.agents: dict[str, SocialAgent] = {}
        self.network: nx.Graph = nx.Graph()
        self.state = SimulationState()

        self._political_enabled = self.cfg.get("political", {}).get("enabled", False)
        self._media_enabled = self.cfg.get("media", {}).get("enabled", False)
        self._elections_enabled = (
            self._political_enabled
            and self.cfg.get("political", {}).get("elections", {}).get("enabled", False)
        )

        random.seed(self.cfg["simulation"].get("seed", 42))

    # ----------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------

    def initialize(self, agents: list[SocialAgent]):
        """Set up the simulation with agents and network."""
        for agent in agents:
            self.agents[agent.agent_id] = agent

        self._build_network()
        self._initialize_group_friction()
        if self._political_enabled:
            self._initialize_faction_friction()

        logger.info(
            f"Simulation initialized: {len(self.agents)} agents, "
            f"{self.network.number_of_edges()} connections, "
            f"political={'ON' if self._political_enabled else 'OFF'}"
        )

    def _build_network(self):
        """Build the social network connecting agents."""
        net_cfg = self.cfg["network"]
        n = len(self.agents)
        agent_ids = list(self.agents.keys())

        if net_cfg["type"] == "small_world":
            k = min(net_cfg["avg_connections"], n - 1)
            G = nx.watts_strogatz_graph(n, k, net_cfg["rewiring_probability"])
        elif net_cfg["type"] == "scale_free":
            m = net_cfg["avg_connections"] // 2
            G = nx.barabasi_albert_graph(n, m)
        elif net_cfg["type"] == "random":
            p = net_cfg["avg_connections"] / n
            G = nx.erdos_renyi_graph(n, p)
        else:
            G = nx.grid_2d_graph(int(n**0.5), int(n**0.5))

        mapping = {i: agent_ids[i] for i in range(min(n, G.number_of_nodes()))}
        self.network = nx.relabel_nodes(G, mapping)

        for agent_id in self.agents:
            if agent_id in self.network:
                neighbors = list(self.network.neighbors(agent_id))
                self.agents[agent_id].connections = neighbors

    def _initialize_group_friction(self):
        """Set initial friction scores between social groups."""
        groups = [g["name"] for g in self.cfg["society"]["groups"]]
        for g in groups:
            self.state.group_friction_scores[g] = 0.1

    def _initialize_faction_friction(self):
        """Set initial friction scores between political factions."""
        factions = self.cfg.get("political", {}).get("factions", [])
        for f in factions:
            self.state.faction_friction_scores[f["name"]] = 0.1

    # ----------------------------------------------------------------
    # Event Generation
    # ----------------------------------------------------------------

    def _pick_event_domain(self) -> str:
        """Decide whether this step's event is social, political, or crossover."""
        if not self._political_enabled:
            return "social"

        crossover_prob = self.cfg["friction"].get("crossover_probability", 0.2)
        roll = random.random()
        if roll < crossover_prob:
            return "crossover"
        elif roll < 0.5 + crossover_prob / 2:
            return "political"
        else:
            return "social"

    def generate_event(self, step: int) -> FrictionEvent:
        """Generate a friction event — social, political, or crossover."""
        domain = self._pick_event_domain()
        friction_cfg = self.cfg["friction"]
        groups = [g["name"] for g in self.cfg["society"]["groups"]]
        factions = [f["name"] for f in self.cfg.get("political", {}).get("factions", [])]

        # Pick event type based on domain
        if domain == "social":
            event_type = random.choice(friction_cfg.get("social_event_types", ["cultural_clash"]))
            affected_groups = random.sample(groups, k=min(2, len(groups)))
            affected_factions = []
        elif domain == "political":
            event_type = random.choice(friction_cfg.get("political_event_types", ["policy_disagreement"]))
            affected_factions = random.sample(factions, k=min(2, len(factions))) if factions else []
            affected_groups = []
        else:  # crossover
            event_type = random.choice(friction_cfg.get("crossover_event_types", ["politicized_cultural_issue"]))
            affected_groups = random.sample(groups, k=min(2, len(groups)))
            affected_factions = random.sample(factions, k=min(2, len(factions))) if factions else []

        severity = random.uniform(0.2, 0.9)
        if domain == "political":
            severity *= friction_cfg.get("political_amplification", 1.0)
            severity = min(severity, 1.0)

        # Build description
        affected_names = affected_groups + affected_factions
        description = (
            f"[{domain.upper()}] A {event_type.replace('_', ' ')} event has occurred "
            f"affecting {', '.join(affected_names)}. Severity: {severity:.1f}/1.0."
        )

        if self.llm:
            prompt = (
                f"<|instruction|>\nGenerate a realistic {domain} friction event.\n"
                f"Type: {event_type}\n"
                f"Groups involved: {', '.join(affected_groups) if affected_groups else 'N/A'}\n"
                f"Political factions: {', '.join(affected_factions) if affected_factions else 'N/A'}\n"
                f"Severity: {severity:.1f}\n"
                f"Domain: {domain}\n"
                f"<|event_description|>\n"
            )
            description = self.llm.generate(prompt, max_new_tokens=250, temperature=0.8)

        return FrictionEvent(
            event_id=f"evt_{step:04d}",
            event_type=event_type,
            domain=domain,
            description=description,
            affected_groups=affected_groups,
            affected_factions=affected_factions,
            severity=severity,
            step=step,
        )

    # ----------------------------------------------------------------
    # Elections
    # ----------------------------------------------------------------

    def _should_hold_election(self) -> bool:
        """Check if it's time for an election."""
        if not self._elections_enabled:
            return False
        freq = self.cfg["political"]["elections"]["frequency_steps"]
        return self.state.current_step > 0 and self.state.current_step % freq == 0

    def _is_campaign_period(self) -> bool:
        """Check if we're in the campaign window before an election."""
        if not self._elections_enabled:
            return False
        freq = self.cfg["political"]["elections"]["frequency_steps"]
        window = self.cfg["political"]["elections"]["campaign_window"]
        steps_until_election = freq - (self.state.current_step % freq)
        return steps_until_election <= window

    def _run_election(self) -> dict:
        """Simulate an election among all agents."""
        faction_votes: dict[str, int] = Counter()

        for agent in self.agents.values():
            # Agents vote for their faction (with some randomness)
            if random.random() < agent.politics.partisan_strength:
                faction_votes[agent.politics.faction] += 1
            else:
                # Swing vote — pick faction closest to ideology
                factions = self.cfg["political"]["factions"]
                closest = min(
                    factions,
                    key=lambda f: abs(f["ideology_center"] - agent.politics.ideology_position),
                )
                faction_votes[closest["name"]] += 1

        total = sum(faction_votes.values())
        winner = max(faction_votes, key=faction_votes.get)
        margin = faction_votes[winner] / total if total > 0 else 0

        result = {
            "step": self.state.current_step,
            "winning_faction": winner,
            "margin": f"{margin:.1%}",
            "vote_counts": dict(faction_votes),
            "total_votes": total,
        }

        self.state.election_log.append(result)
        logger.info(f"Election result: {winner} wins with {margin:.1%}")

        # Post-election reactions
        if self.llm:
            for agent in self.agents.values():
                agent.react_to_election(result, self.llm)

        # Post-election friction boost for losing factions
        boost = self.cfg["political"]["elections"].get("post_election_friction_boost", 0.15)
        for faction_name, score in self.state.faction_friction_scores.items():
            if faction_name != winner:
                self.state.faction_friction_scores[faction_name] = min(1.0, score + boost)

        return result

    # ----------------------------------------------------------------
    # Media Influence
    # ----------------------------------------------------------------

    def _apply_media_influence(self):
        """Have agents consume media, creating echo chamber effects."""
        if not self._media_enabled:
            return

        outlets = self.cfg["media"].get("outlets", [])
        if not outlets:
            return

        for agent in self.agents.values():
            # Agent picks an outlet biased toward their ideology
            weights = []
            for outlet in outlets:
                bias = outlet.get("ideology_bias", 0.0)
                alignment = 1.0 - abs(agent.politics.ideology_position - bias) / 2.0
                reach = outlet.get("reach", 0.5)
                weights.append(alignment * reach)

            total_w = sum(weights)
            if total_w == 0:
                continue
            probs = [w / total_w for w in weights]
            chosen_outlet = random.choices(outlets, weights=probs, k=1)[0]
            agent.consume_media(chosen_outlet)

    # ----------------------------------------------------------------
    # Polarization Tracking
    # ----------------------------------------------------------------

    def _compute_polarization_index(self) -> float:
        """
        Compute society-wide polarization: how spread out ideology positions are.
        Uses bimodality coefficient — higher = more polarized.
        """
        positions = [a.politics.ideology_position for a in self.agents.values()]
        if len(positions) < 2:
            return 0.0

        import numpy as np
        arr = np.array(positions)
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0

        # Bimodality: agents clustering at extremes vs center
        left_mass = np.sum(arr < -0.3) / len(arr)
        right_mass = np.sum(arr > 0.3) / len(arr)
        center_mass = np.sum(np.abs(arr) <= 0.3) / len(arr)

        # Polarization = extreme mass minus center mass, normalized
        polarization = (left_mass + right_mass - center_mass + 1) / 2
        return float(np.clip(polarization, 0, 1))

    def _snapshot_ideology(self):
        """Record current ideology distribution for tracking shifts over time."""
        snapshot = {
            "step": self.state.current_step,
            "agents": [agent.ideology_summary() for agent in self.agents.values()],
        }
        self.state.ideology_snapshots.append(snapshot)

    # ----------------------------------------------------------------
    # Radicalization / Moderation Dynamics
    # ----------------------------------------------------------------

    def _apply_ideology_drift(self):
        """Apply radicalization or moderation based on current friction levels."""
        if not self._political_enabled:
            return

        pol_cfg = self.cfg["political"]["polarization"]
        for agent in self.agents.values():
            faction_friction = self.state.faction_friction_scores.get(
                agent.politics.faction, 0.0
            )

            if faction_friction > 0.6:
                # High friction → radicalize
                agent.politics.radicalize(rate=pol_cfg.get("radicalization_rate", 0.05))
            elif faction_friction < 0.3 and self.state.political_friction_score < 0.3:
                # Low friction → moderate
                agent.politics.moderate(rate=pol_cfg.get("moderation_rate", 0.03))

    # ----------------------------------------------------------------
    # Main Simulation Loop
    # ----------------------------------------------------------------

    def step(self) -> dict:
        """Execute one simulation step."""
        step_num = self.state.current_step
        logger.info(f"--- Step {step_num} ---")

        # 0. Check for election
        election_result = None
        if self._should_hold_election():
            election_result = self._run_election()

        # 1. Generate friction event
        event = self.generate_event(step_num)
        self.state.events_log.append({
            "event_id": event.event_id,
            "type": event.event_type,
            "domain": event.domain,
            "description": event.description,
            "severity": event.severity,
            "affected_groups": event.affected_groups,
            "affected_factions": event.affected_factions,
            "step": step_num,
        })

        # 2. Agents react to event
        reactions = []
        for agent_id, agent in self.agents.items():
            should_react = (
                agent.group in event.affected_groups
                or agent.politics.faction in event.affected_factions
            )
            if should_react and self.llm:
                reaction = agent.react_to_event(
                    {
                        "event_id": event.event_id,
                        "description": event.description,
                        "domain": event.domain,
                    },
                    self.llm,
                )
                reactions.append(reaction)

        # 3. Agent-to-agent interactions
        interactions = []
        sampled_edges = random.sample(
            list(self.network.edges()),
            k=min(5, self.network.number_of_edges()),
        )
        for a_id, b_id in sampled_edges:
            if a_id in self.agents and b_id in self.agents:
                if self.llm:
                    interaction = self.agents[a_id].interact_with(
                        self.agents[b_id], event.description, self.llm
                    )
                    interactions.append(interaction)

        self.state.interaction_log.extend(interactions)

        # 4. Media influence
        self._apply_media_influence()

        # 5. Update social friction scores
        friction_cfg = self.cfg["friction"]
        for group in event.affected_groups:
            current = self.state.group_friction_scores.get(group, 0.0)
            delta = event.severity * friction_cfg["cascade_factor"] * 0.1
            if random.random() < friction_cfg["resolution_probability"]:
                delta *= -0.5
            self.state.group_friction_scores[group] = max(0.0, min(1.0, current + delta))

        # 6. Update political friction scores
        if self._political_enabled:
            amplification = friction_cfg.get("political_amplification", 1.5)
            for faction in event.affected_factions:
                current = self.state.faction_friction_scores.get(faction, 0.0)
                delta = event.severity * amplification * 0.1
                if random.random() < friction_cfg["resolution_probability"] * 0.8:
                    delta *= -0.3  # political friction harder to resolve
                self.state.faction_friction_scores[faction] = max(0.0, min(1.0, current + delta))

            # Campaign period heightens political friction
            if self._is_campaign_period():
                for faction in self.state.faction_friction_scores:
                    self.state.faction_friction_scores[faction] = min(
                        1.0, self.state.faction_friction_scores[faction] + 0.02
                    )

        # 7. Ideology drift (radicalization / moderation)
        self._apply_ideology_drift()

        # 8. Update global scores
        social_scores = list(self.state.group_friction_scores.values())
        self.state.global_friction_score = (
            sum(social_scores) / len(social_scores) if social_scores else 0.0
        )

        if self._political_enabled:
            pol_scores = list(self.state.faction_friction_scores.values())
            self.state.political_friction_score = (
                sum(pol_scores) / len(pol_scores) if pol_scores else 0.0
            )
            self.state.polarization_index = self._compute_polarization_index()

        # 9. Update agent emotions
        for agent in self.agents.values():
            group_friction = self.state.group_friction_scores.get(agent.group, 0.0)
            pol_friction = self.state.faction_friction_scores.get(
                agent.politics.faction, 0.0
            ) if self._political_enabled else 0.0
            agent.update_emotional_state(group_friction, pol_friction)

        # 10. Snapshot ideology
        if self._political_enabled:
            self._snapshot_ideology()

        # 11. Collect metrics
        step_metrics = {
            "step": step_num,
            "global_friction": self.state.global_friction_score,
            "political_friction": self.state.political_friction_score,
            "polarization_index": self.state.polarization_index,
            "group_scores": dict(self.state.group_friction_scores),
            "faction_scores": dict(self.state.faction_friction_scores),
            "event_domain": event.domain,
            "num_reactions": len(reactions),
            "num_interactions": len(interactions),
            "election": election_result,
        }
        self.state.metrics_history.append(step_metrics)

        self.state.current_step += 1
        return step_metrics

    def run(self, num_steps: int | None = None) -> list[dict]:
        """Run the full simulation."""
        steps = num_steps or self.cfg["simulation"]["num_steps"]
        logger.info(f"Starting simulation for {steps} steps...")

        all_metrics = []
        for _ in range(steps):
            metrics = self.step()
            all_metrics.append(metrics)

            save_every = self.cfg["output"].get("save_every_n_steps", 10)
            if self.state.current_step % save_every == 0:
                self._save_checkpoint()

        self._save_final_results()
        logger.info("Simulation complete.")
        return all_metrics

    def _save_checkpoint(self):
        save_dir = Path(self.cfg["output"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"checkpoint_step_{self.state.current_step}.json"
        with open(path, "w") as f:
            json.dump(self.state.metrics_history, f, indent=2)

    def _save_final_results(self):
        save_dir = Path(self.cfg["output"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, data in [
            ("metrics.json", self.state.metrics_history),
            ("events.json", self.state.events_log),
            ("interactions.json", self.state.interaction_log),
            ("elections.json", self.state.election_log),
            ("ideology_shifts.json", self.state.ideology_snapshots),
        ]:
            with open(save_dir / name, "w") as f:
                json.dump(data, f, indent=2)

        logger.info(f"Results saved to {save_dir}")
