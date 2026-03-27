# Quick Reference: Test Suite Commands

## Installation

```bash
pip install pytest pytest-cov
```

## Running Tests

```bash
# All tests
pytest tests/

# Verbose output
pytest tests/ -v

# Stop after first failure
pytest tests/ -x

# Show print statements
pytest tests/ -s

# Specific file
pytest tests/test_agents.py -v

# Specific class
pytest tests/test_agents.py::TestPoliticalProfile -v

# Specific test
pytest tests/test_agents.py::TestPoliticalProfile::test_drift_toward_positive -v

# Pattern matching
pytest tests/ -k "political" -v

# Parallel execution (requires pytest-xdist)
pytest tests/ -n auto
```

## Coverage Reports

```bash
# Generate coverage
pytest tests/ --cov=src --cov-report=html

# View HTML report
open htmlcov/index.html

# Terminal report
pytest tests/ --cov=src --cov-report=term-missing
```

## Test Organization

| Module | Tests | Focus |
|--------|-------|-------|
| test_agents.py | 58 | Agents, profiles, memory |
| test_data_pipeline.py | 28 | Data generation |
| test_engine.py | 40 | Simulation engine |
| test_metrics.py | 45 | Evaluation metrics |
| test_cognitive_models.py | 18 | Cognitive models |
| test_config.py | 20 | Configuration |

## Key Test Classes

### PoliticalProfile (25 tests)
```python
# Tests ideology movement
- test_drift_toward_positive/negative
- test_radicalize_from_positive/negative
- test_moderate_from_positive/negative
- test_ideology_label_*
- test_is_moderate/is_extreme
- test_extreme_ideology_values
```

### AgentMemory (13 tests)
```python
# Tests memory buffer
- test_memory_overflow_drops_oldest
- test_memory_boundary_max_size
- test_memory_political_history_filter
- test_memory_social_history_filter
```

### SocialAgent (12 tests)
```python
# Tests agent behavior
- test_build_identity_prompt_includes_politics
- test_consume_media_echo_chamber_effect
- test_consume_media_cross_cutting_effect
- test_update_emotional_state_*
```

### SimulationEngine (40 tests)
```python
# Tests simulation
- Network building (small_world, scale_free, random)
- Event generation (social, political, crossover)
- Elections (voting, friction, margins)
- Media influence
- Ideology drift (radicalization, moderation)
- Full runs (checkpoints, results)
```

## Common Assertions

```python
# Equality
assert agent.agent_id == "test_001"

# Range
assert 0.0 <= profile.ideology_position <= 1.0

# Container membership
assert agent.emotional_state in [EmotionalState.ANGRY, EmotionalState.FEARFUL]

# Type checking
assert isinstance(metrics, dict)

# Comparison
assert polarization > 0.0
assert convergence_rate == 0.0

# File existence
assert checkpoint_file.exists()

# List length
assert len(memory.events) == 3
```

## Fixtures Available

```python
# From conftest.py
@pytest.fixture
def sample_agent():
    """Single test agent"""

@pytest.fixture
def sample_agents_list():
    """5 diverse agents"""

@pytest.fixture
def sample_config():
    """Simulation configuration dict"""

@pytest.fixture
def sample_config_file(tmp_path):
    """Temporary YAML config"""

@pytest.fixture
def sample_simulation_engine(sample_config_file, sample_agents_list):
    """Initialized engine"""

@pytest.fixture
def sample_metrics_history():
    """Mock metrics data"""

@pytest.fixture
def tmp_output_dir():
    """Temporary output directory"""
```

## Using Fixtures in Tests

```python
def test_agent_creation(sample_agent):
    """Use fixture directly"""
    assert sample_agent.agent_id == "agent_001"

def test_engine_step(sample_simulation_engine):
    """Use initialized engine"""
    metrics = sample_simulation_engine.step()
    assert metrics["step"] == 0

def test_multiple_fixtures(sample_agent, sample_config, tmp_output_dir):
    """Combine multiple fixtures"""
    pass
```

## Common Test Patterns

### Testing Boundaries
```python
def test_extreme_ideology_values():
    profile = PoliticalProfile(ideology_position=-1.0)
    assert profile.ideology_position == -1.0
    assert profile.ideology_label == "far-left"
```

### Testing Effects
```python
def test_consume_media_echo_chamber():
    agent = SocialAgent(...)
    original = agent.politics.ideology_position
    agent.consume_media(left_outlet)
    assert agent.politics.ideology_position < original
```

### Testing Counts
```python
def test_memory_overflow():
    memory = AgentMemory(max_size=3)
    for i in range(5):
        memory.add({"summary": f"Event {i}"})
    assert len(memory.events) == 3
```

### Testing File Operations
```python
def test_save_checkpoint(sample_simulation_engine):
    sample_simulation_engine._save_checkpoint()
    save_dir = Path(sample_simulation_engine.cfg["output"]["save_dir"])
    assert save_dir.exists()
```

## Debugging Failed Tests

```bash
# Show full tracebacks
pytest tests/ --tb=long

# Enter debugger on failure
pytest tests/ --pdb

# Show local variables
pytest tests/ -l

# Verbose output with all details
pytest tests/ -vv

# Capture output even on success
pytest tests/ -s
```

## Test Statistics Commands

```bash
# Count tests
pytest tests/ --collect-only | grep "test session starts" -A 100

# Show test summary
pytest tests/ -v --tb=no

# Generate report
pytest tests/ --html=report.html --self-contained-html
```

## CI/CD Integration

```bash
# JUnit XML for Jenkins
pytest tests/ --junit-xml=test_results.xml

# Coverage for SonarQube
pytest tests/ --cov=src --cov-report=xml

# Exit codes for CI
pytest tests/  # 0 = all pass, 1 = failures
```

## Performance Tips

```bash
# Run tests in parallel
pytest tests/ -n auto

# Stop after N failures
pytest tests/ --maxfail=3

# Run only changed tests (requires pytest-watch)
ptw tests/

# Skip slow tests
pytest tests/ -m "not slow"

# Show slowest 10 tests
pytest tests/ --durations=10
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Ensure in project root, src/ in path |
| Fixture not found | Check conftest.py, ensure pytest installed |
| Permission denied | Check tmp directory permissions |
| Tests too slow | Use -n auto for parallel, skip slow tests |
| Import errors | Missing dependencies (networkx, pydantic) |

## Test File Locations

```
/tests/
├── conftest.py                    # Fixtures (233 lines)
├── test_agents.py                 # Agent tests (695 lines)
├── test_data_pipeline.py          # Data tests (331 lines)
├── test_engine.py                 # Engine tests (623 lines)
├── test_metrics.py                # Metric tests (563 lines)
├── test_cognitive_models.py       # Cognitive tests (479 lines)
└── test_config.py                 # Config tests (399 lines)
```

Total: 3,090 lines of test code across 200+ tests
