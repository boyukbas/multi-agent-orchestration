"""Command-line entrypoint.

    orchestrate "Design a rate limiter for a public API"      # real Claude if ANTHROPIC_API_KEY is set
    orchestrate --offline "..."                                # force the deterministic backend

Without a key it transparently uses the offline backend so the pipeline always runs.
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from .graph import build_orchestrator, initial_state
from .llm import AnthropicLLM, ScriptedLLM, default_llm


def _backend_label(llm) -> str:
    if isinstance(llm, AnthropicLLM):
        return f"AnthropicLLM (model={llm.model}) - real autonomous decomposition"
    return "ScriptedLLM (offline, deterministic) - no API calls"


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Orchestrator-worker multi-agent demo (LangGraph).")
    parser.add_argument("goal", nargs="*", help="The high-level goal to decompose and execute.")
    parser.add_argument("--offline", action="store_true", help="Force the deterministic offline backend.")
    parser.add_argument("--max-review", type=int, default=2, help="Max evaluator-optimizer rounds (default 2).")
    args = parser.parse_args(argv)

    goal = " ".join(args.goal).strip() or "Explain how a multi-agent orchestrator decomposes a task."
    llm = default_llm(offline=args.offline)

    print(f"backend : {_backend_label(llm)}")
    if isinstance(llm, ScriptedLLM) and not args.offline:
        print("note    : no ANTHROPIC_API_KEY found - using the offline backend. Set the key for real decomposition.")
    print(f"goal    : {goal}\n")

    graph = build_orchestrator(llm, max_review=args.max_review)
    final = graph.invoke(initial_state(goal), config={"recursion_limit": 50})

    print("=== run trace " + "=" * 52)
    for line in final["log"]:
        print(line)

    print("\n=== final deliverable " + "=" * 44)
    print(final["final_output"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
