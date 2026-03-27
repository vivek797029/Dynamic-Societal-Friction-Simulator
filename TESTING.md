# Comprehensive Test Suite for Dynamic Society Friction Simulator

## Overview

A professional-grade test suite with **3,090+ lines of test code** covering all major components of the Dynamic Society Friction Simulator. The tests are designed to run without GPU/LLM access and provide comprehensive coverage of:

- **Agent behavior** (political profiles, emotional states, memory)
- **Simulation engine** (network building, event generation, elections)
- **Metrics and evaluation** (friction, polarization, convergence)
- **Cognitive models** (dissonance, Overton window, emotional contagion)
- **Configuration** (YAML loading, validation, schema)
- **Data pipeline** (scenario generation, formatting, JSONL output)

## Test Files and Structure

### 1. `tests/conftest.py` (233 lines)
**Shared pytest fixtures used across all test modules**

```python
# Key fixtures:
- sample_agent: Single SocialAgent for basic tests
- sample_agents_list: 5 diverse agents with different characteristics
- sample_config: Complete simulation configuration dictionary
- sample_config_file: Temporary YAML config for engine tests
- sample_simulation_engine: Initialized engine with agents
- sample_metrics_history: Mock metrics data for evaluation tests
- tmp_output_dir: Temporary directory for file operations
```

**Benefits:**
- Eliminates fixture duplication
- Ensures test data consistency
- Reduces test setup boilerplate
- Easy to extend and maintain

### 2. `tests/test_agents.py` (695 lines)
**Comprehensive tests for social agents and political profiles**

#### PoliticalProfile Tests (25+ tests)
- `ideology_label` property for all positions
- `is_moderate` and `is_extreme` properties
- `drift_toward(target, rate)` with boundary clamping
- `radicalize(rate)` moving toward extremes
- `moderate(rate)` moving toward center
- Edge cases: extreme values (-1.0, +1.0), zero movements
- History tracking through `ideology_history`

#### AgentMemory Tests (13+ tests)
- FIFO buffer behavior with `max_size` boundary
- `add(event)` with overflow handling
- `recent(n)` returning last N events
- `summarize()` generating memory summaries
- `political_history()` filtering events by domain="political"
- `social_history()` filtering events by domain in ("social", "cultural")
- Edge cases: empty memory, exceeding available events

#### SocialAgent Tests (12+ tests)
- `build_identity_prompt()` includes all required political info
- `consume_media()` echo chamber effect (alignment > 0.6 → drift)
- `consume_media()` cross-cutting effect (alignment < 0.3 + open → moderate)
- `update_emotional_state(friction_score, political_friction)`
- `ideology_summary()` output format
- Political identity integration

#### Edge Cases (8+ tests)
- Empty core values
- Zero/max openness to change (0.0, 1.0)
- Zero/max friction scores (0.0, 1.0)
- Extreme ideology boundaries
- Default faction handling

### 3. `tests/test_data_pipeline.py` (331 lines)
**Tests for synthetic scenario generation and data augmentation**

#### Synthetic Scenario Generation (7 tests)
- Required fields: scenario, category, group_a, group_b, severity
- Severity range: 0.0 ≤ severity ≤ 1.0
- Group distinctness: group_a ≠ group_b
- Valid categories: cultural_clash, resource_competition, migration_tension, etc.
- Scenario text quality and length
- Variety across multiple generations

#### Scenario Formatting (6 tests)
- `format_as_instruction()` creates valid instruction/output pairs
- Metadata preservation through formatting
- Text content validation
- Multiple scenario formatting

#### Dataset Generation (8 tests)
- File creation (train.jsonl, eval.jsonl)
- Train/eval split: train_samples + eval_samples = total
- JSONL format validation (one JSON object per line)
- Reproducibility with seed (same seed → same data)
- Different seeds produce different outputs
- Sample quality (instruction length > 20, output present)

#### Data Quality (4 tests)
- Generated samples have substantive content
- Dataset contains diverse scenarios
- Category information in metadata

#### Edge Cases (3 tests)
- Small sample counts (1 sample)
- Large sample counts (100+ samples)
- Empty/non-existent directories

### 4. `tests/test_engine.py` (623 lines)
**Tests for the core SimulationEngine**

#### Initialization Tests (3 tests)
- Engine creation with config
- State initialization
- Agent initialization

#### Network Building Tests (5 tests)
- Small-world network (Watts-Strogatz)
- Scale-free network (Barabási-Albert)
- Random network (Erdős-Rényi)
- Agent connection assignment from network topology

#### Event Generation Tests (4 tests)
- FrictionEvent structure validation
- Domain distribution (social, political, crossover)
- Social events have affected_groups
- Political events have affected_factions

#### Election Tests (5 tests)
- `_should_hold_election()` frequency check
- `_is_campaign_period()` detection
- `_run_election()` voting mechanics
- Election result recording in state
- Post-election friction boost for losing factions

#### Media Influence Tests (2 tests)
- `_apply_media_influence()` applies consumption
- Media disabled flag respected

#### Polarization Tests (2 tests)
- Centered agents: low polarization
- Extreme distribution: high polarization

#### Ideology Drift Tests (2 tests)
- High friction → radicalization
- Low friction → moderation

#### Step Execution Tests (4 tests)
- Current step counter increment
- Metrics generation per step
- Event logging
- Friction score updates

#### Checkpoint Tests (2 tests)
- Checkpoint file creation
- Data preservation (metrics_history, ideology_snapshots)

#### Full Simulation Tests (3 tests)
- Short 5-step runs
- Agent count preservation
- Result file saving

#### Edge Cases (2 tests)
- Single agent simulation
- Zero friction scenarios

### 5. `tests/test_metrics.py` (563 lines)
**Tests for evaluation metrics suite**

#### Friction Volatility (6 tests)
- Empty history → 0.0
- Constant values → 0.0
- Increasing/oscillating → > 0.0
- Known input verification

#### Polarization Index (4 tests)
- Empty groups → 0.0
- Identical scores → 0.0
- Disparate scores → > 0.0
- Uses final score in history

#### Convergence Rate (4 tests)
- Resolving friction → > 0.0
- Escalating friction → < 0.0
- Stable friction → 0.0
- Insufficient data → 0.0

#### Cascade Frequency (4 tests)
- No events → 0.0
- No cascades (no 20% jump) → 0.0
- With cascades → > 0.0

#### Polarization Trend (4 tests)
- Increasing → > 0.0
- Decreasing → < 0.0
- Stable → 0.0

#### Ideology Metrics (3 tests)
- Spread (standard deviation)
- Extreme ratio (agents at |ideology| > 0.7)
- Average drift between snapshots

#### Election Metrics (3 tests)
- Count and average margin
- Competitive elections

#### Cognitive Dissonance Metrics (3 tests)
- Empty history → 0.0
- Single/multiple snapshots
- Averaging across time

#### Overton Window Metrics (3 tests)
- Window width tracking
- Center shift detection
- History accumulation

#### Emotional Contagion Metrics (3 tests)
- R0 (basic reproduction number)
- Epidemic count tracking
- Multiple emotions

#### Cross-Domain Correlation (4 tests)
- Perfect positive/negative correlation
- No correlation (constant values)
- Insufficient data → 0.0

#### Evaluation Integration (2 tests)
- evaluate_simulation_run() with missing/present files
- Full metric computation workflow

#### Report Generation (3 tests)
- File creation
- Report structure (social_dynamics, political_dynamics, etc.)
- Interpretation fields (volatility, trend, risk levels)

### 6. `tests/test_cognitive_models.py` (479 lines)
**Tests for advanced cognitive models**

#### CognitiveDissonanceTracker (6 tests)
- Tracker initialization
- `compute_dissonance()` with aligned/conflicting values
- `track_dissonance()` over time
- `resolve_dissonance()` with low/high scores
- Partisan strength effects on dissonance

#### OvertonWindowTracker (4 tests)
- Basic window tracking
- Empty agents handling
- `window_width_history()` accumulation
- `detect_polarization_signature()` patterns

#### EmotionalContagionModel (5 tests)
- Model initialization with network graph
- Agent susceptibility updates
- `track_contagion()` metrics
- `identify_epidemics()` with minimum size threshold
- `average_network_r0()` computation

#### Integration Tests (3 tests)
- Dissonance detection → resolution workflow
- Overton Window evolution with polarization
- Contagion on different network topologies (complete, path, star)

### 7. `tests/test_config.py` (399 lines)
**Tests for configuration loading and validation**

#### YAML Config Loading (6 tests)
- Basic YAML loading
- String and Path object handling
- Missing file error handling
- Complex nested structures

#### Config Classes (5 tests)
- `BaseModelConfig`: name (required), revision, trust_remote_code
- `QuantizationConfig`: 4-bit settings, compute dtype
- `LoraAdapterConfig`: rank, alpha, dropout, modules
- `TrainingConfig`: epochs, batch size, learning rate, optimizers
- `DataConfig`: file paths, max samples

#### ModelConfiguration Validation (5 tests)
- Complete valid config passes
- Missing base_model raises ValueError
- Missing quantization raises ValueError
- Missing training raises ValueError
- Optional fields (wandb, gdrive) handled

#### Edge Cases (4 tests)
- Extra fields in config (ignored by Pydantic)
- Invalid field types raise ValueError
- YAML boolean strings (yes/no)
- Different learning rate formats

## Test Statistics

```
Total Lines of Test Code:      3,090
Number of Test Files:          7
Number of Test Classes:        40+
Number of Test Functions:      200+

Breakdown by file:
- conftest.py:               233 lines (fixtures)
- test_agents.py:            695 lines (agent/profile tests)
- test_data_pipeline.py:      331 lines (data generation)
- test_engine.py:             623 lines (simulation engine)
- test_metrics.py:            563 lines (evaluation metrics)
- test_cognitive_models.py:   479 lines (cognitive models)
- test_config.py:             399 lines (configuration)
```

## Coverage Areas

✓ **Agent Behavior**
  - Political ideology drift, radicalization, moderation
  - Emotional state updates based on friction
  - Memory management with FIFO buffer and domain filtering
  - Media consumption with echo chamber and cross-cutting effects
  - Identity prompt generation with political context

✓ **Simulation Dynamics**
  - Network topologies (small-world, scale-free, random)
  - Event generation across domains (social, political, crossover)
  - Election mechanics (voting, margins, post-election effects)
  - Media influence on ideology
  - Friction dynamics and escalation

✓ **Advanced Models**
  - Cognitive dissonance (value-policy conflicts)
  - Overton window (discourse range tracking)
  - Emotional contagion (R0, epidemic detection)

✓ **Evaluation & Metrics**
  - Friction volatility and convergence
  - Polarization index and trends
  - Ideology distribution and drift
  - Cascade frequency detection
  - Cross-domain correlations

✓ **Configuration**
  - YAML file loading and parsing
  - Pydantic schema validation
  - Required/optional field handling
  - Type checking and coercion

## Test Execution

### Quick Start

```bash
# Install pytest (requires internet)
pip install pytest

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_agents.py -v

# Run specific test class
pytest tests/test_agents.py::TestPoliticalProfile -v

# Run specific test
pytest tests/test_agents.py::TestPoliticalProfile::test_drift_toward_positive -v
```

### With Coverage Report

```bash
# Install coverage tools
pip install pytest-cov

# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# View report
open htmlcov/index.html
```

### Continuous Integration

```bash
# Run with JUnit XML output for CI/CD
pytest tests/ --junit-xml=test_results.xml

# Run with multiple workers (faster)
pip install pytest-xdist
pytest tests/ -n auto
```

## Key Design Patterns

1. **Fixture Composition**: Common fixtures in conftest.py reduce duplication
2. **Edge Case Coverage**: Every metric/function tested at boundaries
3. **Property-Based Assertions**: Tests verify invariants, not just values
4. **Integration Tests**: Tests combining multiple modules
5. **Temporary Resources**: Temp directories for file I/O tests
6. **Mock Data**: Realistic sample data avoiding need for GPU/LLM
7. **Clear Names**: Test names describe exactly what they test
8. **Descriptive Assertions**: Error messages show expected vs actual

## What Gets Tested

✓ Happy path (normal operation)
✓ Edge cases (empty, single, extreme values)
✓ Boundary conditions (max_size, -1.0/+1.0 values)
✓ Error conditions (missing files, invalid data)
✓ Integration (multiple components together)
✓ State changes (before/after operations)
✓ Data consistency (invariants preserved)

## What Is NOT Tested

✗ LLM inference (mocked/skipped)
✗ GPU operations (designed to skip)
✗ Real file I/O (uses temp directories)
✗ Network requests (not applicable)
✗ Visualization (no display needed)

## Running Specific Test Scenarios

```bash
# Test political profile behavior
pytest tests/test_agents.py::TestPoliticalProfile -v

# Test agent memory operations
pytest tests/test_agents.py::TestAgentMemory -v

# Test simulation engine
pytest tests/test_engine.py -v

# Test metrics computation
pytest tests/test_metrics.py -v

# Test data generation
pytest tests/test_data_pipeline.py -v

# Test cognitive models
pytest tests/test_cognitive_models.py -v

# Test configuration
pytest tests/test_config.py -v
```

## Troubleshooting

**Import errors?**
- Ensure you're in the project root directory
- Check that src/ modules are importable
- Missing dependencies? See setup instructions

**Tests failing?**
- Check fixture definitions in conftest.py
- Verify temporary directory permissions
- Look at assertion messages for details

**Slow tests?**
- Use pytest-xdist for parallel execution
- Skip network/GPU tests in CI
- Use lightweight fixtures

## Future Enhancements

- Property-based testing with hypothesis
- Parametrized tests for multiple scenarios
- Performance benchmarks
- Stress testing with large agent counts
- Mutation testing for test quality
- Coverage analysis with coverate.py

## Files and Paths

```
/tests/
├── conftest.py                 # Shared fixtures
├── test_agents.py              # Agent and profile tests
├── test_data_pipeline.py       # Data generation tests
├── test_engine.py              # Simulation engine tests
├── test_metrics.py             # Metrics and evaluation tests
├── test_cognitive_models.py    # Cognitive model tests
└── test_config.py              # Configuration tests
```

## Summary

This comprehensive test suite provides:
- **3,090+ lines** of professional test code
- **200+ test functions** covering all major components
- **Edge case handling** for robust validation
- **No GPU/LLM dependencies** for fast CI/CD
- **Clear documentation** for maintainability
- **Extensible fixtures** for future tests

The tests validate that the Dynamic Society Friction Simulator correctly models:
- Agent political ideology and emotional dynamics
- Social network effects on friction propagation
- Election mechanics and their effects
- Media influence on polarization
- Advanced cognitive models of political behavior
