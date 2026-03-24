"""
CLI entrypoint for the Dynamic Society Friction Simulator.

Usage:
    dsfs train          — Fine-tune the LLM on friction data
    dsfs generate-data  — Generate synthetic training data
    dsfs simulate       — Run a friction simulation
    dsfs evaluate       — Evaluate simulation results
"""

import logging

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(
    name="dsfs",
    help="Dynamic Society Friction Simulator — LLM-powered social friction modeling",
)
console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@app.command()
def train(
    config: str = typer.Option("configs/model_config.yaml", help="Path to model config"),
    resume: bool = typer.Option(None, "--resume/--no-resume", help="Resume from checkpoint (default: auto-detect)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Fine-tune the base LLM on social friction data.

    By default, auto-detects checkpoints and resumes if found.
    Use --resume to force resume, or --no-resume to start fresh.
    """
    setup_logging(verbose)

    if resume is True:
        console.print("[bold green]Starting training (forced resume from checkpoint)...[/]")
    elif resume is False:
        console.print("[bold green]Starting training (fresh start, ignoring checkpoints)...[/]")
    else:
        console.print("[bold green]Starting training (auto-resume if checkpoints exist)...[/]")

    from src.model.trainer import train as run_training
    run_training(config_path=config, resume=resume)

    console.print("[bold green]Training complete![/]")


@app.command()
def generate_data(
    num_samples: int = typer.Option(1000, help="Number of synthetic samples"),
    output_dir: str = typer.Option("data/processed", help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate synthetic training data for friction scenarios."""
    setup_logging(verbose)
    console.print(f"[bold blue]Generating {num_samples} synthetic samples...[/]")

    from src.model.data_pipeline import generate_dataset
    stats = generate_dataset(num_samples=num_samples, output_dir=output_dir, seed=seed)

    console.print(f"[bold green]Done! {stats['train_samples']} train, {stats['eval_samples']} eval[/]")


@app.command()
def simulate(
    config: str = typer.Option("configs/simulation_config.yaml", help="Simulation config"),
    model_config: str = typer.Option("configs/model_config.yaml", help="Model config"),
    adapter_path: str = typer.Option(None, help="Path to trained LoRA adapter"),
    steps: int = typer.Option(None, help="Override number of steps"),
    no_llm: bool = typer.Option(False, help="Run without LLM (random behaviors)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run a friction simulation."""
    setup_logging(verbose)
    console.print("[bold magenta]Initializing simulation...[/]")

    from src.simulation.engine import SimulationEngine
    from src.agents.social_agent import SocialAgent, AgentMemory

    llm = None
    if not no_llm:
        from src.model.inference import FrictionLLM
        llm = FrictionLLM(config_path=model_config, adapter_path=adapter_path)

    engine = SimulationEngine(config_path=config, llm=llm)

    # Create agents from config
    import yaml
    with open(config) as f:
        sim_cfg = yaml.safe_load(f)

    agents = []
    agent_id = 0
    for group_cfg in sim_cfg["society"]["groups"]:
        count = int(group_cfg["size_ratio"] * sim_cfg["society"]["population_size"])
        for i in range(count):
            agents.append(SocialAgent(
                agent_id=f"agent_{agent_id:03d}",
                name=f"{group_cfg['name']}_{i}",
                group=group_cfg["name"],
                core_values=group_cfg["core_values"],
                openness_to_change=group_cfg["openness_to_change"],
                memory=AgentMemory(max_size=sim_cfg["agents"]["memory_window"]),
            ))
            agent_id += 1

    engine.initialize(agents)
    metrics = engine.run(num_steps=steps)

    console.print(f"[bold green]Simulation complete! {len(metrics)} steps executed.[/]")
    final = metrics[-1] if metrics else {}
    console.print(f"Final global friction: {final.get('global_friction', 'N/A'):.3f}")


@app.command()
def evaluate(
    results_dir: str = typer.Option("outputs/results", help="Path to simulation results"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Evaluate a completed simulation run."""
    setup_logging(verbose)
    console.print("[bold cyan]Evaluating simulation results...[/]")

    from src.evaluation.metrics import evaluate_simulation_run, generate_report
    metrics = evaluate_simulation_run(results_dir)
    report = generate_report(metrics)

    console.print("[bold green]Evaluation complete![/]")
    interp = report["interpretation"]
    console.print(f"Social volatility:    {interp['social_volatility']}")
    console.print(f"Social trend:         {interp['social_trend']}")
    console.print(f"Polarization:         {interp['polarization']} ({interp['polarization_direction']})")
    console.print(f"Radicalization risk:  {interp['radicalization_risk']}")
    console.print(f"Cross-domain coupling:{interp['cross_domain_coupling']}")


if __name__ == "__main__":
    app()
