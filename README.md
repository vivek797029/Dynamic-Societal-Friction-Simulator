# Dynamic Society Friction Simulator (DSFS)

An LLM-powered simulation platform for modeling social/cultural friction, political polarization, and emergent collective behavior. DSFS fine-tunes an open-source language model, then deploys it to power autonomous agents that interact within a simulated society — generating realistic friction events, political debates, elections, media influence effects, cognitive dissonance dynamics, and emotional contagion patterns.

## What Makes This Different

DSFS introduces **three novel computational models** not found in any existing agent-based simulation:

1. **Cognitive Dissonance Engine** — Agents experience measurable psychological tension when their cultural identity conflicts with their political stance. Resolution strategies (value change, faction switch, compartmentalization, rationalization) are personality-driven and trackable over time.

2. **Dynamic Overton Window Tracker** — Monitors how the range of "acceptable" political discourse shifts across the simulation. Detects polarization signatures (narrowing vs. widening), window shocks, and feedback loops between discourse boundaries and political friction.

3. **Emotional Contagion Network** — Emotions propagate through the social network graph with negativity bias (anger spreads faster than hope). Computes epidemic R0 metrics for emotional states and detects emotional cascades — the first application of epidemiological modeling to emotional dynamics in polarization simulations.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                            CLI (typer)                                │
├──────────┬──────────┬─────────────┬────────────┬─────────┬───────────┤
│ Training │ Data Gen │  Simulation │ Elections  │  Eval   │  Visualize│
│ Pipeline │ Pipeline │   Engine    │  System    │ Metrics │ Dashboard │
├──────────┴──────────┼─────────────┴────────────┴─────────┴───────────┤
│    Fine-Tuned LLM   │           Social Agents                        │
│ (QLoRA on Mistral)  │  (Identity + Politics + Memory +               │
│                      │   Emotion + Ideology + Dissonance)             │
├──────────────────────┼──────────────────────────────────────────────┤
│  Cognitive Models    │       Social Network (NetworkX)               │
│  (Dissonance +      │  (Homophily + Echo Chambers +                  │
│   Overton Window +  │   Emotional Contagion +                        │
│   Contagion)        │   Cross-cutting exposure)                      │
├──────────────────────┼──────────────────────────────────────────────┤
│  Media Ecosystem     │       Visualization Dashboard                 │
│  (Outlets + Bias +   │  (Friction timeline + Polarization heatmap + │
│   Echo Chambers)     │   Ideology drift + Election results)          │
└──────────────────────┴──────────────────────────────────────────────┘
```

## What It Simulates

**Social/Cultural Friction** — Cultural clashes, identity conflicts, migration tensions, resource competition, generational divides, and economic inequality between 6 distinct social groups (200 agents).

**Political Polarization** — Ideological divides across a left-right spectrum (-1.0 to +1.0), 5 partisan factions, legislative deadlock, protest movements, disinformation campaigns, and corruption scandals. Agents drift via radicalization or moderation.

**Cross-Domain Events** — Social issues that become politically charged (and vice versa): politicized cultural issues, identity politics clashes, immigration policy debates, education curriculum wars.

**Elections** — Periodic elections where agents vote based on faction loyalty, ideology, and swing dynamics. Includes campaign periods with heightened rhetoric and post-election friction spikes.

**Media Ecosystem** — 6 biased media outlets that create echo chambers. Agents consume ideology-aligned media, reinforcing existing views. Includes social media virality and misinformation dynamics.

**Cognitive Dissonance** — Agents track internal psychological tension between their values and their faction's policies. High dissonance triggers resolution strategies that reshape the political landscape.

**Emotional Contagion** — Emotions spread through the social network with negativity bias. Emotional epidemics can cascade through communities, amplifying friction events.

**Overton Window** — The range of "mainstream" political discourse shifts dynamically. Narrowing windows signal dangerous polarization; widening signals pluralistic discourse.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/vivek797029/Dynamic-Societal-Friction-Simulator.git
cd Dynamic-Societal-Friction-Simulator
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh

# 2. Generate synthetic training data
dsfs generate-data --num-samples 5000

# 3. Fine-tune the model (requires GPU)
dsfs train

# 4. Run a simulation
dsfs simulate --adapter-path outputs/checkpoints/final_adapter

# 5. Evaluate results
dsfs evaluate

# 6. Generate visualizations
dsfs visualize
```

## Training on Google Colab (100 Compute Units)

We provide an optimized Colab notebook with built-in budget management:

```
notebooks/train_on_colab.ipynb
```

The notebook includes a `BudgetTracker` that monitors compute unit consumption in real-time and auto-stops training when 90% of budget is consumed. GPU-specific auto-configuration adjusts batch size, LoRA rank, and epochs based on available hardware.

| GPU | Cost/Hour | Max Training Hours | Recommended Epochs |
|-----|-----------|-------------------|--------------------|
| L4  | ~2.35 units | ~42 hours | 8-10 |
| A100 | ~7.35 units | ~13.5 hours | 4-5 |
| T4  | ~1.67 units | ~59 hours | 10-12 |

## Training Your Own Model

The project uses **QLoRA** (4-bit quantized LoRA) for memory-efficient fine-tuning:

| Setting | Value | Why |
|---|---|---|
| Base model | Mistral 7B Instruct v0.3 | Best quality/cost at 7B scale |
| Quantization | NF4 (4-bit) | Fits in 16 GB VRAM |
| LoRA rank | 128 (cloud) / 64 (local) | Maximum learning capacity |
| Effective batch size | 32 | Via gradient accumulation |
| Learning rate | 1e-4 | Optimized for cloud training |
| Sequence length | 4096 | Full context for complex scenarios |
| Scheduler | Cosine with restarts | Escapes local minima |
| NEFTune alpha | 5.0 | +5-10% generation quality |

Edit `configs/model_config.yaml` to switch base models or adjust hyperparameters.

## Project Structure

```
dynamic-society-friction-simulator/
├── configs/
│   ├── model_config.yaml          # LLM training configuration
│   └── simulation_config.yaml     # Society, politics, media, friction
├── data/
│   ├── raw/                       # Raw collected data
│   ├── processed/                 # Train/eval JSONL files
│   ├── synthetic/                 # Generated synthetic data
│   └── prompts/                   # Prompt templates
├── src/
│   ├── model/
│   │   ├── trainer.py             # QLoRA fine-tuning pipeline
│   │   ├── inference.py           # Model loading and generation
│   │   ├── data_pipeline.py       # Data processing and synthesis
│   │   └── cognitive_models.py    # Dissonance + Overton + Contagion
│   ├── agents/
│   │   └── social_agent.py        # Agent (cultural + political + cognitive)
│   ├── simulation/
│   │   └── engine.py              # Core engine (all domains + elections)
│   ├── evaluation/
│   │   └── metrics.py             # Social, political, cognitive metrics
│   ├── visualization/
│   │   └── dashboard.py           # Publication-quality plots
│   ├── utils/
│   │   └── config.py              # Config loading and validation
│   └── cli.py                     # Command-line interface
├── scripts/
│   ├── setup_env.sh               # Local environment setup
│   ├── setup_gcp.sh               # GCP VM setup
│   └── train.sh                   # Training launch script
├── tests/
│   ├── conftest.py                # Shared pytest fixtures
│   ├── test_agents.py             # Agent behavior tests
│   ├── test_engine.py             # Simulation engine tests
│   ├── test_metrics.py            # Evaluation metrics tests
│   ├── test_data_pipeline.py      # Data generation tests
│   ├── test_cognitive_models.py   # Novel model tests
│   └── test_config.py             # Configuration tests
├── notebooks/
│   └── train_on_colab.ipynb       # Optimized Colab training notebook
├── outputs/
├── pyproject.toml
├── .gitignore
└── README.md
```

## Configuration

All behavior is controlled through YAML configs in `configs/`:

- **`model_config.yaml`** — Base model selection, LoRA settings, training hyperparameters, Google Drive sync, W&B logging
- **`simulation_config.yaml`** — Society groups, political factions, ideology spectrum, media outlets, election rules, friction event types, network topology

Key settings in `simulation_config.yaml`:

| Setting | Default | What it controls |
|---|---|---|
| `echo_chamber_strength` | 0.7 | How much agents reinforce same-ideology views |
| `radicalization_rate` | 0.06 | Speed of ideology drift toward extremes |
| `media_bias_factor` | 0.5 | How much media outlets amplify division |
| `election_frequency` | Every 50 steps | How often elections occur |
| `crossover_probability` | 0.25 | Chance social events become political |
| `population_size` | 200 | Number of autonomous agents |
| `num_steps` | 1000 | ~19 years of simulated weekly events |

## Evaluation Metrics

The evaluation system tracks metrics across five domains:

**Social** — Friction volatility, convergence rate, cascade frequency, resolution rate

**Political** — Polarization index and trend, ideology spread, extreme agent ratio, ideology drift magnitude, election competitiveness

**Cross-Domain** — Crossover event ratio, social-political friction correlation

**Cognitive** — Average cognitive dissonance, dissonance resolution rates, value-policy alignment scores

**Emotional/Discourse** — Emotional contagion R0, epidemic count, Overton window width and shift

## Visualization

Generate publication-quality plots (300 DPI, colorblind-friendly):

```bash
dsfs visualize --results-dir outputs/results --output-dir outputs/visualizations
```

Available plots: friction timeline, polarization heatmap, ideology drift trajectories, election results, network snapshot, emotional landscape, cognitive dissonance map, Overton window dynamics.

## Hardware Requirements

| Tier | GPU | What you can do |
|---|---|---|
| Minimum | RTX 3060 (12 GB) | Train with batch size 1-2, shorter sequences |
| Recommended | RTX 3090/4090 (24 GB) | Full training pipeline as configured |
| Cloud option | 1x A100 (40 GB) | Faster training, larger batch sizes |
| Budget option | Google Colab (100 units) | Full pipeline with budget management |

No GPU? Use `dsfs simulate --no-llm` to run with rule-based agent behaviors for testing.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_agents.py -v
pytest tests/test_cognitive_models.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## License

MIT
