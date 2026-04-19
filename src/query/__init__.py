"""Query layer — turns natural-language prompts into forecasts, narratives,
and evidence citations over the friction pipeline.

Public surface:

    from src.query import answer, AnswerBundle
    from src.query import PipelineContext, Scenario
    from src.query import route, Intent
    from src.query import choropleth
    from src.query import get_llm
"""
from .analogues import Analogue, find_analogues  # noqa: F401
from .api import AnswerBundle, answer            # noqa: F401
from .intervention import (PipelineContext,      # noqa: F401
                            SimulationResult, simulate)
from .llm import (AnthropicLLM, LLMProvider,     # noqa: F401
                   OpenAILLM, StubLLM, get_llm)
from .narrative import off_domain_answer         # noqa: F401
from .router import Intent, RouteDecision, route  # noqa: F401
from .scenario import Scenario, extract_scenario  # noqa: F401
from .viz import ColorRamp, choropleth           # noqa: F401
