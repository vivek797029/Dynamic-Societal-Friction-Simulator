# Dynamic Society Friction Simulator (DSFS)

An LLM-powered simulation platform for modeling social/cultural friction **and** political polarization dynamics. DSFS fine-tunes an open-source language model, then deploys it to power autonomous agents that interact within a simulated society — generating realistic friction events, political debates, elections, media influence effects, and emergent group behaviors across both domains.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         CLI (typer)                          │
├──────────┬──────────┬─────────────┬────────────┬─────────────┤
│ Training │ Data Gen │  Simulation │ Elections  │  Evaluation │
│ Pipeline │ Pipeline │   Engine    │  System    │   Metrics   │
├──────────┴──────────┼─────────────┴────────────┴─────────────┤
│    Fine-Tuned LLM   │          Social Agents                 │
│ (QLoRA on Mistral)  │  (Identity + Politics + Memory +       │
│                      │   Emotion + Ideology + Network)        │
├──────────────────────┼───────────────────────────────────────┤
│  Media Ecosystem     │       Social Network (NetworkX)        │
│  (Outlets + Bias +   │  (Homophily + Echo Chambers +          │
│   Echo Chambers)     │   Cross-cutting exposure)              │
└──────────────────────┴───────────────────────────────────────┘
```

## What It Simulates

**Social/Cultural Friction** — Cultural clashes, identity conflicts, migration tensions, resource competition, generational divides, and economic inequality between social groups.

**Political Polarization** — Ideological divides across a left-right spectrum, partisan faction dynamics, legislative deadlock, protest movements, disinformation campaigns, and corruption scandals. Agents have ideology positions (-1.0 far-left to +1.0 far-right) that drift over time through radicalization or moderation.

**Cross-Domain Events** — Social issues that become politically charged (and vice versa): politicized cultural issues, identity politics clashes, immigration policy debates, education curriculum wars.

**Elections** — Periodic elections where agents vote based on faction loyalty and ideology. Includes campaign periods with heightened rhetoric and post-election friction spikes for losing factions.

**Media Ecosystem** — Biased media outlets that create echo chambers. Agents consume media aligned with their ideology, reinforcing existing views. Includes social media virality and misinformation dynamics.

## Quick Start

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd dynamic-society-friction-simulator
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
```

## Training Your Own Model

The project uses **QLoRA** (4-bit quantized LoRA) for memory-efficient fine-tuning:

| Setting | Value | Why |
|---|---|---|
| Base model | Mistral 7B Instruct | Best quality/cost at 7B scale |
| Quantization | NF4 (4-bit) | Fits in 16 GB VRAM |
| LoRA rank | 64 | Good balance of capacity vs. efficiency |
| Effective batch size | 32 | Via gradient accumulation (4 x 8) |
| Learning rate | 2e-4 | Standard for QLoRA |
| Sequence length | 4096 | Long enough for complex scenarios |

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
│   │   └── data_pipeline.py       # Data processing and synthesis
│   ├── agents/
│   │   └── social_agent.py        # Agent with cultural + political identity
│   ├── simulation/
│   │   └── engine.py              # Core engine (social + political + elections)
│   ├── evaluation/
│   │   └── metrics.py             # Social, political, cross-domain metrics
│   ├── utils/
│   │   └── config.py              # Config loading and validation
│   └── cli.py                     # Command-line interface
├── scripts/
│   ├── setup_env.sh               # Environment setup
│   └── train.sh                   # Training launch script
├── tests/
├── notebooks/
├── outputs/
├── pyproject.toml
├── .gitignore
└── README.md
```

## Configuration

All behavior is controlled through YAML configs in `configs/`:

- **`model_config.yaml`** — Base model selection, LoRA settings, training hyperparameters
- **`simulation_config.yaml`** — Society groups, political factions, ideology spectrum, media outlets, election rules, friction event types, network topology

Key political settings in `simulation_config.yaml`:

| Setting | Default | What it controls |
|---|---|---|
| `echo_chamber_strength` | 0.6 | How much agents reinforce same-ideology views |
| `radicalization_rate` | 0.05 | Speed of ideology drift toward extremes under stress |
| `media_bias_factor` | 0.4 | How much media outlets amplify division |
| `election_frequency` | Every 25 steps | How often elections occur |
| `crossover_probability` | 0.2 | Chance social events become political (and vice versa) |

## Evaluation Metrics

The evaluation system tracks metrics across three domains:

**Social** — Friction volatility, convergence rate, cascade frequency

**Political** — Polarization index & trend, ideology spread, extreme agent ratio, ideology drift magnitude, election competitiveness

**Cross-Domain** — Crossover event ratio, social-political friction correlation

## Hardware Requirements

| Tier | GPU | What you can do |
|---|---|---|
| Minimum | RTX 3060 (12 GB) | Train with batch size 1-2, shorter sequences |
| Recommended | RTX 3090/4090 (24 GB) | Full training pipeline as configured |
| Cloud option | 1x A100 (40 GB) | Faster training, larger batch sizes |

No GPU? Use `dsfs simulate --no-llm` to run with rule-based agent behaviors for testing.

## License

MIT
