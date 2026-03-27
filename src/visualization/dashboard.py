"""
Publication-quality visualization module for the Dynamic Society Friction Simulator.

Generates matplotlib-based visualizations from simulation output data with
academic styling, colorblind-friendly palettes, and 300 DPI publication quality.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)

# Colorblind-friendly palette (colorbrewer Set2 + viridis for extended range)
PALETTE = {
    "primary": ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"],
    "accent": ["#a6761d", "#666666"],
    "sequential": plt.cm.viridis,
    "neutral": "#cccccc",
}


def _setup_style():
    """Configure matplotlib for publication-quality output."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
    })


def plot_friction_timeline(
    metrics_history: list[dict],
    save_path: str,
) -> None:
    """
    Line chart showing global_friction and political_friction over time steps.
    Two y-axes. Mark election steps with vertical dashed lines.

    Args:
        metrics_history: List of metric dicts from engine.state.metrics_history
        save_path: Path to save the figure
    """
    _setup_style()

    if not metrics_history:
        logger.warning("No metrics history provided for friction timeline")
        return

    steps = [m["step"] for m in metrics_history]
    global_friction = [m.get("global_friction", 0.0) for m in metrics_history]
    political_friction = [m.get("political_friction", 0.0) for m in metrics_history]

    # Find election steps
    election_steps = [
        m["step"] for m in metrics_history
        if m.get("election") is not None
    ]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Primary y-axis: global friction
    color1 = PALETTE["primary"][0]
    ax1.set_xlabel("Simulation Step", fontsize=11)
    ax1.set_ylabel("Global Friction Score", color=color1, fontsize=11)
    line1 = ax1.plot(
        steps, global_friction,
        color=color1, linewidth=2.0, label="Global Friction",
        marker="o", markersize=3, alpha=0.8
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1.0)

    # Secondary y-axis: political friction
    if any(political_friction):
        ax2 = ax1.twinx()
        color2 = PALETTE["primary"][1]
        ax2.set_ylabel("Political Friction Score", color=color2, fontsize=11)
        line2 = ax2.plot(
            steps, political_friction,
            color=color2, linewidth=2.0, label="Political Friction",
            marker="s", markersize=3, alpha=0.8
        )
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0, 1.0)

    # Mark election steps
    for elec_step in election_steps:
        ax1.axvline(x=elec_step, color="red", linestyle="--", alpha=0.5, linewidth=1.0)

    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(steps[0], steps[-1])

    # Combined legend
    if any(political_friction):
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        election_patch = mpatches.Patch(color="red", label="Election Event", alpha=0.5)
        ax1.legend(lines + [election_patch], labels + ["Election Event"],
                   loc="upper left", fontsize=10)
    else:
        ax1.legend(line1, ["Global Friction"], loc="upper left", fontsize=10)

    plt.title("Friction Dynamics Over Time", fontsize=12, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved friction timeline to {save_path}")


def plot_polarization_heatmap(
    ideology_snapshots: list[dict],
    save_path: str,
    n_bins: int = 20,
) -> None:
    """
    Heatmap showing ideology distribution across groups over time.
    X-axis = simulation steps, Y-axis = ideology bins (-1 to +1), color = agent count.

    Args:
        ideology_snapshots: List of ideology snapshot dicts from engine.state.ideology_snapshots
        save_path: Path to save the figure
        n_bins: Number of ideology bins (default 20)
    """
    _setup_style()

    if not ideology_snapshots:
        logger.warning("No ideology snapshots provided for polarization heatmap")
        return

    steps = [s["step"] for s in ideology_snapshots]
    ideology_bins = np.linspace(-1.0, 1.0, n_bins)
    heatmap_data = np.zeros((n_bins - 1, len(steps)))

    for i, snapshot in enumerate(ideology_snapshots):
        agents = snapshot.get("agents", [])
        positions = [a.get("ideology", 0.0) for a in agents]
        if positions:
            counts, _ = np.histogram(positions, bins=ideology_bins)
            heatmap_data[:, i] = counts

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        origin="lower",
        cmap="YlOrRd",
        extent=[steps[0], steps[-1], -1.0, 1.0],
        interpolation="nearest",
    )

    ax.set_xlabel("Simulation Step", fontsize=11)
    ax.set_ylabel("Ideology Position", fontsize=11)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(["Left", "", "Center", "", "Right"])

    cbar = plt.colorbar(im, ax=ax, label="Agent Count")
    plt.title("Polarization Heatmap: Ideology Distribution Over Time",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved polarization heatmap to {save_path}")


def plot_ideology_drift(
    ideology_snapshots: list[dict],
    save_path: str,
    sample_agents: Optional[int] = None,
) -> None:
    """
    Multi-frame plot showing how agent ideology positions shift over time.
    Color by group/faction.

    Args:
        ideology_snapshots: List of ideology snapshot dicts
        save_path: Path to save the figure
        sample_agents: If specified, only plot this many random agents (for clarity)
    """
    _setup_style()

    if not ideology_snapshots:
        logger.warning("No ideology snapshots provided for ideology drift")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # Collect all agents across snapshots
    agent_data: dict[str, list[dict]] = {}
    for snapshot in ideology_snapshots:
        step = snapshot["step"]
        for agent in snapshot.get("agents", []):
            agent_id = agent.get("agent_id")
            if agent_id not in agent_data:
                agent_data[agent_id] = []
            agent_data[agent_id].append({
                "step": step,
                "ideology": agent.get("ideology", 0.0),
                "group": agent.get("group", "unknown"),
            })

    # Sample agents if needed
    agents_to_plot = list(agent_data.items())
    if sample_agents and len(agents_to_plot) > sample_agents:
        agents_to_plot = agents_to_plot[:sample_agents]

    # Get unique groups and assign colors
    all_groups = set()
    for agent_history in agent_data.values():
        for entry in agent_history:
            all_groups.add(entry["group"])
    groups = sorted(list(all_groups))
    color_map = {g: PALETTE["primary"][i % len(PALETTE["primary"])]
                 for i, g in enumerate(groups)}

    # Plot each agent's trajectory
    for agent_id, history in agents_to_plot:
        history = sorted(history, key=lambda x: x["step"])
        steps = [h["step"] for h in history]
        ideologies = [h["ideology"] for h in history]
        group = history[0]["group"] if history else "unknown"
        color = color_map[group]

        ax.plot(steps, ideologies, color=color, alpha=0.3, linewidth=0.8)

    # Add legend for groups
    group_patches = [mpatches.Patch(color=color_map[g], label=g) for g in groups]
    ax.legend(handles=group_patches, loc="best", fontsize=10)

    ax.set_xlabel("Simulation Step", fontsize=11)
    ax.set_ylabel("Ideology Position", fontsize=11)
    ax.set_ylim(-1.0, 1.0)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)
    plt.title("Agent Ideology Drift Over Time", fontsize=12, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ideology drift to {save_path}")


def plot_election_results(
    elections: list[dict],
    save_path: str,
) -> None:
    """
    Bar chart of election results over time, showing faction vote shares per election.

    Args:
        elections: List of election result dicts from engine.state.election_log
        save_path: Path to save the figure
    """
    _setup_style()

    if not elections:
        logger.warning("No elections provided for election results plot")
        return

    # Collect all factions
    all_factions = set()
    for election in elections:
        all_factions.update(election.get("vote_counts", {}).keys())
    factions = sorted(list(all_factions))
    color_map = {f: PALETTE["primary"][i % len(PALETTE["primary"])]
                 for i, f in enumerate(factions)}

    # Prepare data
    election_nums = [i for i in range(len(elections))]
    faction_data = {f: [] for f in factions}

    for election in elections:
        total_votes = election.get("total_votes", 1)
        for faction in factions:
            votes = election.get("vote_counts", {}).get(faction, 0)
            faction_data[faction].append(100 * votes / total_votes if total_votes > 0 else 0)

    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(elections))

    for faction in factions:
        ax.bar(election_nums, faction_data[faction], label=faction,
               color=color_map[faction], alpha=0.85)

    ax.set_xlabel("Election Number", fontsize=11)
    ax.set_ylabel("Vote Share (%)", fontsize=11)
    ax.set_xticks(election_nums)
    ax.set_xticklabels([f"E{i+1}" for i in election_nums])
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.title("Election Results: Faction Vote Share Over Time",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved election results to {save_path}")


def plot_network_snapshot(
    agents: dict[str, any],
    network: nx.Graph,
    save_path: str,
) -> None:
    """
    NetworkX graph visualization colored by group/faction, node size by influence,
    edge thickness by trust.

    Args:
        agents: Dict of agents (agent_id -> agent object)
        network: NetworkX graph of social connections
        save_path: Path to save the figure
    """
    _setup_style()

    if not network.number_of_nodes():
        logger.warning("Network has no nodes")
        return

    # Get unique groups and assign colors
    groups = set()
    for agent_id in network.nodes():
        if agent_id in agents:
            groups.add(agents[agent_id].group)

    groups = sorted(list(groups))
    color_map = {g: PALETTE["primary"][i % len(PALETTE["primary"])]
                 for i, g in enumerate(groups)}

    # Node colors by group
    node_colors = [
        color_map.get(agents.get(node_id, None).group if node_id in agents else "unknown", PALETTE["neutral"])
        for node_id in network.nodes()
    ]

    # Node sizes by influence (simplified: connectivity)
    node_sizes = [100 + 50 * network.degree(node_id) for node_id in network.nodes()]

    # Edge widths by trust (simplified: use weight if available)
    edge_widths = []
    for u, v in network.edges():
        weight = network[u][v].get("weight", 1.0) if isinstance(network[u][v], dict) else 1.0
        edge_widths.append(weight * 0.5)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Use spring layout for cleaner visualization
    pos = nx.spring_layout(network, k=2, iterations=50, seed=42)

    nx.draw_networkx_nodes(
        network, pos,
        node_color=node_colors,
        node_size=node_sizes,
        ax=ax,
        alpha=0.8,
    )

    nx.draw_networkx_edges(
        network, pos,
        width=edge_widths,
        ax=ax,
        alpha=0.3,
    )

    # Add legend for groups
    group_patches = [mpatches.Patch(color=color_map[g], label=g) for g in groups]
    ax.legend(handles=group_patches, loc="upper left", fontsize=10)

    ax.set_title("Social Network Snapshot: Groups & Connections",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved network snapshot to {save_path}")


def plot_emotional_landscape(
    metrics_history: list[dict],
    save_path: str,
) -> None:
    """
    Stacked area chart showing distribution of emotional states across all agents over time.

    Args:
        metrics_history: List of metric dicts from engine.state.metrics_history
        save_path: Path to save the figure
    """
    _setup_style()

    if not metrics_history:
        logger.warning("No metrics history provided for emotional landscape")
        return

    # Extract emotional contagion data if available
    steps = []
    emotions_data: dict[str, list[float]] = {}

    for m in metrics_history:
        step = m.get("step")
        if step is not None:
            steps.append(step)

        contagion = m.get("emotional_contagion", {})
        if contagion:
            emotion_dist = contagion.get("emotion_distribution", {})
            for emotion, count in emotion_dist.items():
                if emotion not in emotions_data:
                    emotions_data[emotion] = []
                emotions_data[emotion].append(count)

    if not steps or not emotions_data:
        logger.warning("No emotional data available for landscape plot")
        return

    # Pad missing values
    for emotion in emotions_data:
        while len(emotions_data[emotion]) < len(steps):
            emotions_data[emotion].insert(0, 0)

    # Normalize to percentages
    for i in range(len(steps)):
        total = sum(emotions_data[e][i] for e in emotions_data if i < len(emotions_data[e]))
        if total > 0:
            for emotion in emotions_data:
                if i < len(emotions_data[emotion]):
                    emotions_data[emotion][i] = 100 * emotions_data[emotion][i] / total

    # Sort emotions for consistent coloring
    emotions = sorted(emotions_data.keys())
    colors = PALETTE["primary"] + PALETTE["accent"]
    color_map = {e: colors[i % len(colors)] for i, e in enumerate(emotions)}

    fig, ax = plt.subplots(figsize=(14, 7))

    # Stacked area plot
    ax.stackplot(
        steps,
        *[emotions_data[e][:len(steps)] for e in emotions],
        labels=emotions,
        colors=[color_map[e] for e in emotions],
        alpha=0.8,
    )

    ax.set_xlabel("Simulation Step", fontsize=11)
    ax.set_ylabel("Emotional State Distribution (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    plt.title("Emotional Landscape: State Distribution Over Time",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved emotional landscape to {save_path}")


def plot_cognitive_dissonance_map(
    metrics_history: list[dict],
    save_path: str,
) -> None:
    """
    Plot average dissonance per group over time if cognitive dissonance data exists.

    Args:
        metrics_history: List of metric dicts from engine.state.metrics_history
        save_path: Path to save the figure
    """
    _setup_style()

    if not metrics_history:
        logger.warning("No metrics history provided for cognitive dissonance map")
        return

    steps = []
    dissonance_scores = []

    for m in metrics_history:
        step = m.get("step")
        if step is not None:
            steps.append(step)

        dissonance = m.get("cognitive_dissonance", {})
        if dissonance:
            score = dissonance.get("avg_dissonance", 0.0)
            dissonance_scores.append(score)

    if not steps or not dissonance_scores:
        logger.warning("No cognitive dissonance data available")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    color = PALETTE["primary"][2]
    ax.fill_between(steps, dissonance_scores, alpha=0.3, color=color)
    ax.plot(steps, dissonance_scores, color=color, linewidth=2.0,
            marker="o", markersize=4, label="Avg Cognitive Dissonance")

    ax.set_xlabel("Simulation Step", fontsize=11)
    ax.set_ylabel("Cognitive Dissonance Score", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.title("Cognitive Dissonance Over Time",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved cognitive dissonance map to {save_path}")


def plot_overton_window(
    metrics_history: list[dict],
    save_path: str,
) -> None:
    """
    Plot the Overton window boundaries (left_edge, right_edge) as a band chart over time.

    Args:
        metrics_history: List of metric dicts from engine.state.metrics_history
        save_path: Path to save the figure
    """
    _setup_style()

    if not metrics_history:
        logger.warning("No metrics history provided for Overton window plot")
        return

    steps = []
    left_edges = []
    right_edges = []
    centers = []

    for m in metrics_history:
        step = m.get("step")
        if step is not None:
            steps.append(step)

        overton = m.get("overton_window", {})
        if overton:
            left_edges.append(overton.get("left_edge", -1.0))
            right_edges.append(overton.get("right_edge", 1.0))
            centers.append(overton.get("center", 0.0))

    if not steps or not left_edges:
        logger.warning("No Overton window data available")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot the band (acceptable discourse range)
    ax.fill_between(steps, left_edges, right_edges, alpha=0.3,
                    color=PALETTE["primary"][0], label="Overton Window")

    # Plot center
    ax.plot(steps, centers, color=PALETTE["primary"][1], linewidth=2.0,
            marker="o", markersize=4, label="Window Center")

    # Plot edges
    ax.plot(steps, left_edges, color=PALETTE["primary"][0], linewidth=1.0,
            linestyle="--", alpha=0.7, label="Left Edge")
    ax.plot(steps, right_edges, color=PALETTE["primary"][2], linewidth=1.0,
            linestyle="--", alpha=0.7, label="Right Edge")

    ax.axhline(y=-1.0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=0.0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

    ax.set_xlabel("Simulation Step", fontsize=11)
    ax.set_ylabel("Ideology Position", fontsize=11)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(["Left", "", "Center", "", "Right"])
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.title("Overton Window: Discourse Range Over Time",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Overton window plot to {save_path}")


def generate_full_report(
    results_dir: str,
    output_dir: str,
) -> list[str]:
    """
    Generate ALL plots and save to output_dir. Return list of saved file paths.

    Args:
        results_dir: Path to simulation results directory (contains JSON files)
        output_dir: Path to output directory for visualizations

    Returns:
        List of saved file paths
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Load all available data
    metrics_history = []
    if (results_path / "metrics.json").exists():
        with open(results_path / "metrics.json") as f:
            metrics_history = json.load(f)
    logger.info(f"Loaded {len(metrics_history)} metric snapshots")

    ideology_snapshots = []
    if (results_path / "ideology_shifts.json").exists():
        with open(results_path / "ideology_shifts.json") as f:
            ideology_snapshots = json.load(f)
    logger.info(f"Loaded {len(ideology_snapshots)} ideology snapshots")

    elections = []
    if (results_path / "elections.json").exists():
        with open(results_path / "elections.json") as f:
            elections = json.load(f)
    logger.info(f"Loaded {len(elections)} election results")

    # Plot 1: Friction Timeline
    try:
        friction_path = output_path / "01_friction_timeline.png"
        plot_friction_timeline(metrics_history, str(friction_path))
        saved_files.append(str(friction_path))
    except Exception as e:
        logger.error(f"Error generating friction timeline: {e}")

    # Plot 2: Polarization Heatmap
    try:
        heatmap_path = output_path / "02_polarization_heatmap.png"
        plot_polarization_heatmap(ideology_snapshots, str(heatmap_path))
        saved_files.append(str(heatmap_path))
    except Exception as e:
        logger.error(f"Error generating polarization heatmap: {e}")

    # Plot 3: Ideology Drift
    try:
        drift_path = output_path / "03_ideology_drift.png"
        plot_ideology_drift(ideology_snapshots, str(drift_path), sample_agents=50)
        saved_files.append(str(drift_path))
    except Exception as e:
        logger.error(f"Error generating ideology drift: {e}")

    # Plot 4: Election Results
    try:
        if elections:
            election_path = output_path / "04_election_results.png"
            plot_election_results(elections, str(election_path))
            saved_files.append(str(election_path))
    except Exception as e:
        logger.error(f"Error generating election results: {e}")

    # Plot 5: Network Snapshot (requires agents and network)
    # Skipped in generate_full_report as we don't have live agents/network
    # This would need to be called separately if agents are available
    logger.info("Skipping network snapshot (requires live agent data)")

    # Plot 6: Emotional Landscape
    try:
        emotion_path = output_path / "05_emotional_landscape.png"
        plot_emotional_landscape(metrics_history, str(emotion_path))
        saved_files.append(str(emotion_path))
    except Exception as e:
        logger.error(f"Error generating emotional landscape: {e}")

    # Plot 7: Cognitive Dissonance Map
    try:
        dissonance_path = output_path / "06_cognitive_dissonance.png"
        plot_cognitive_dissonance_map(metrics_history, str(dissonance_path))
        saved_files.append(str(dissonance_path))
    except Exception as e:
        logger.error(f"Error generating cognitive dissonance map: {e}")

    # Plot 8: Overton Window
    try:
        overton_path = output_path / "07_overton_window.png"
        plot_overton_window(metrics_history, str(overton_path))
        saved_files.append(str(overton_path))
    except Exception as e:
        logger.error(f"Error generating Overton window plot: {e}")

    logger.info(f"Generated {len(saved_files)} visualizations")
    return saved_files
