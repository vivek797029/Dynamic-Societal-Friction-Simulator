"""LLM provider abstraction.

The query layer calls an LLM in a few places: scenario extraction from a
free-form chat prompt, narrative generation, and off-domain fallback. We
hide the vendor behind a single `LLMProvider` protocol so you can plug in
Anthropic, OpenAI, a local Llama, or anything else without touching the
rest of the code.

Three implementations ship:
  * StubLLM         -- deterministic echo, no network, used by tests.
  * AnthropicLLM    -- `anthropic` SDK wrapper (lazy-imported).
  * OpenAILLM       -- `openai` SDK wrapper (lazy-imported).

Pick via env vars:
    DSFS_LLM=stub                 # default, no network
    DSFS_LLM=anthropic            # ANTHROPIC_API_KEY must be set
    DSFS_LLM=openai               # OPENAI_API_KEY must be set
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

log = logging.getLogger(__name__)


# ------------------------------- protocol ------------------------------- #

class LLMProvider(Protocol):
    name: str

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.2) -> str:
        """Return a completion string. Never raises on network issues --
        the implementation is responsible for its own retries; callers may
        choose to fall through on an empty/None return."""
        ...

    def complete_json(self, prompt: str, *, schema_hint: str,
                      system: str | None = None,
                      max_tokens: int = 1024) -> dict | None:
        """Return a parsed JSON dict or None if parsing failed."""
        ...


# -------------------------------- stub ---------------------------------- #

@dataclass
class StubLLM:
    """No network, deterministic. Good for tests and sandbox dev.

    `complete` returns a fixed-but-prompt-aware string. `complete_json`
    returns an empty dict {} so callers can rely on the type but know no
    extraction happened.
    """
    name: str = "stub"

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.2) -> str:
        head = (system or "").splitlines()[0] if system else "stub"
        return f"[{self.name}/{head[:40]}] {prompt[:120]}"

    def complete_json(self, prompt: str, *, schema_hint: str,
                      system: str | None = None,
                      max_tokens: int = 1024) -> dict | None:
        return {}


# ----------------------------- Anthropic -------------------------------- #

class AnthropicLLM:
    """Anthropic Messages API wrapper. Requires `anthropic` SDK + key."""
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6",
                 api_key: str | None = None) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "AnthropicLLM needs `pip install anthropic`"
            ) from e
        self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.2) -> str:
        try:
            resp = self._client.messages.create(
                model=self.model, max_tokens=max_tokens, temperature=temperature,
                system=system or "You are a helpful analyst.",
                messages=[{"role": "user", "content": prompt}],
            )
            # Messages API returns a list of content blocks.
            parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            return "".join(parts).strip()
        except Exception as e:
            log.warning("AnthropicLLM.complete failed: %s", e)
            return ""

    def complete_json(self, prompt: str, *, schema_hint: str,
                      system: str | None = None,
                      max_tokens: int = 1024) -> dict | None:
        sys_msg = (
            (system or "You extract structured data from text.") +
            "\nReply with ONLY valid JSON matching this schema hint:\n" + schema_hint
        )
        raw = self.complete(prompt, system=sys_msg, max_tokens=max_tokens, temperature=0.0)
        return _safe_json(raw)


# ------------------------------- OpenAI --------------------------------- #

class OpenAILLM:
    """OpenAI Chat Completions wrapper."""
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini",
                 api_key: str | None = None) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError("OpenAILLM needs `pip install openai`") from e
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.2) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model, max_tokens=max_tokens, temperature=temperature,
                messages=[
                    {"role": "system", "content": system or "You are a helpful analyst."},
                    {"role": "user", "content": prompt},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            log.warning("OpenAILLM.complete failed: %s", e)
            return ""

    def complete_json(self, prompt: str, *, schema_hint: str,
                      system: str | None = None,
                      max_tokens: int = 1024) -> dict | None:
        sys_msg = (
            (system or "You extract structured data from text.") +
            "\nReply with ONLY valid JSON matching this schema hint:\n" + schema_hint
        )
        raw = self.complete(prompt, system=sys_msg, max_tokens=max_tokens, temperature=0.0)
        return _safe_json(raw)


# ------------------------------- factory -------------------------------- #

def get_llm(name: str | None = None) -> LLMProvider:
    """Look up `DSFS_LLM` (or the `name` arg). Falls back to the stub.

    Never raises -- if the requested provider can't be constructed we log
    a warning and return StubLLM so the query layer still runs.
    """
    provider = (name or os.getenv("DSFS_LLM") or "stub").lower()
    if provider in ("stub", "none", "off", ""):
        return StubLLM()
    try:
        if provider == "anthropic":
            return AnthropicLLM()
        if provider == "openai":
            return OpenAILLM()
    except Exception as e:
        log.warning("LLM provider %r unavailable (%s); falling back to stub", provider, e)
    return StubLLM()


# ------------------------------ helpers --------------------------------- #

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _safe_json(s: str) -> dict | None:
    if not s:
        return None
    # Try direct.
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    # Fallback: extract the largest {...} blob.
    m = _JSON_RE.search(s)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        return None
