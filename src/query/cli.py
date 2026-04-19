"""Command-line entry point for the query layer.

Usage:

    python -m src.query.cli "What if the CAA is extended to Kerala?"
    python -m src.query.cli --ctx artifacts/stage_b/stage_b.pt \\
        "Why is Punjab flagged high this week?"
    python -m src.query.cli --save-heatmap out.svg \\
        --llm anthropic --json results.json \\
        "Effects of a new farm law in Punjab and Haryana"

Exit codes:
    0  success (even for off-domain answers)
    1  argparse / invalid input error
    2  uncaught runtime error

The default `ctx` is `PipelineContext.dry_run()` so this script is
runnable on a fresh checkout without any trained artifacts. When you
pass `--ctx PATH` we try to load a real PipelineContext from disk via
`loader.load_context(PATH)` (see `src/query/loader.py` for the format).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .api import AnswerBundle, answer
from .intervention import PipelineContext
from .llm import get_llm


def _maybe_load_ctx(path: str | None) -> PipelineContext:
    if not path:
        return PipelineContext.dry_run()
    try:
        from .loader import load_context                                # type: ignore
    except ImportError:
        print(f"[cli] loader module not available; falling back to dry-run "
              f"(requested: {path})", file=sys.stderr)
        return PipelineContext.dry_run()
    try:
        return load_context(path)
    except Exception as e:                                              # noqa: BLE001
        print(f"[cli] failed to load context from {path!r}: {e}", file=sys.stderr)
        print("[cli] falling back to dry-run context", file=sys.stderr)
        return PipelineContext.dry_run()


def _render_terminal(bundle: AnswerBundle, wide: bool = False) -> str:
    """A human-readable rendering for stdout."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"prompt   : {bundle.prompt}")
    lines.append(f"intent   : {bundle.intent}  "
                 f"(grounded={bundle.is_model_grounded})")
    if bundle.route:
        lines.append(f"routing  : {bundle.route.get('reason')}")
    if bundle.warnings:
        for w in bundle.warnings:
            lines.append(f"[warning] {w}")
    lines.append("-" * 72)
    lines.append(bundle.narrative)
    lines.append("-" * 72)
    if bundle.state_deltas:
        lines.append("State deltas (top 10):")
        for row in bundle.state_deltas[:10]:
            pieces = [f"{row['state']:<32}"]
            for key, val in row.items():
                if key == "state":
                    continue
                pieces.append(f"{key}={val:+.3f}")
            lines.append("  " + "  ".join(pieces))
    if bundle.top_cleavages:
        lines.append("Top cleavages:")
        for block in bundle.top_cleavages:
            cl = ", ".join(f"{c['name']} ({c['weight']:+.2f})"
                           for c in block["cleavages"])
            lines.append(f"  {block['state']}: {cl}")
    if bundle.analogues:
        lines.append("Historical analogues:")
        for a in bundle.analogues[:5]:
            lines.append(f"  {a['state']} wk={a['iso_week']} "
                          f"sim={a['similarity']:.2f}")
    if bundle.citations:
        lines.append("Citations:")
        for c in bundle.citations[:8]:
            t = c.get("title") or "(untitled)"
            s = c.get("source") or "?"
            lines.append(f"  [{s}] {t}")
    lines.append("=" * 72)
    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dsfs.query",
        description="Natural-language interface to the societal-friction "
                    "forecaster. Accepts what-if, explain-now, or off-topic "
                    "prompts.",
    )
    p.add_argument("prompt", nargs="+",
                   help="The user's question. Quote it if it contains spaces.")
    p.add_argument("--ctx", default=None,
                   help="Path to a PipelineContext checkpoint. Defaults to "
                        "dry-run mode (placeholder tensors, no trained head).")
    p.add_argument("--llm", default=None,
                   help="Override DSFS_LLM for this invocation: "
                        "'stub' | 'anthropic' | 'openai'.")
    p.add_argument("--k-analogues", type=int, default=5,
                   help="Top-k historical analogues to return.")
    p.add_argument("--save-heatmap", default=None,
                   help="If set, write the choropleth SVG to this path.")
    p.add_argument("--no-heatmap", action="store_true",
                   help="Skip SVG rendering entirely (faster).")
    p.add_argument("--json", dest="json_out", default=None,
                   help="If set, write the full AnswerBundle as JSON here.")
    p.add_argument("--quiet", action="store_true",
                   help="Only print the narrative (no table / routing info).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        print("error: empty prompt", file=sys.stderr)
        return 1

    if args.llm:
        os.environ["DSFS_LLM"] = args.llm
    llm = get_llm(args.llm)

    ctx = _maybe_load_ctx(args.ctx)

    try:
        bundle = answer(
            prompt, ctx=ctx, llm=llm,
            k_analogues=args.k_analogues,
            render_heatmap=not args.no_heatmap,
        )
    except Exception as e:                                              # noqa: BLE001
        print(f"[cli] uncaught error: {e}", file=sys.stderr)
        return 2

    if args.quiet:
        print(bundle.narrative)
    else:
        print(_render_terminal(bundle))

    if args.save_heatmap and bundle.heatmap_svg:
        try:
            Path(args.save_heatmap).write_text(bundle.heatmap_svg, encoding="utf-8")
            print(f"[cli] heatmap written to {args.save_heatmap}",
                  file=sys.stderr)
        except OSError as e:
            print(f"[cli] failed to write heatmap: {e}", file=sys.stderr)

    if args.json_out:
        try:
            payload: dict[str, Any] = bundle.to_dict()
            Path(args.json_out).write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
            print(f"[cli] bundle JSON written to {args.json_out}",
                  file=sys.stderr)
        except (OSError, TypeError) as e:
            print(f"[cli] failed to write JSON: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
