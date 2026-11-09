"""Zero-setup demo: runs the orchestrator end-to-end with the offline backend.

    python examples/demo_offline.py

No API key required — it exercises the full decomposition -> dispatch -> worker ->
synthesize -> critique pipeline deterministically.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from orchestrator.graph import build_orchestrator, initial_state  # noqa: E402
from orchestrator.llm import ScriptedLLM  # noqa: E402

GOAL = "Recommend a caching strategy for a read-heavy product catalog API."


def main() -> None:
    graph = build_orchestrator(ScriptedLLM(), max_review=2)
    final = graph.invoke(initial_state(GOAL), config={"recursion_limit": 50})

    print(f"GOAL: {GOAL}\n")
    print("RUN TRACE")
    for line in final["log"]:
        print(" ", line)
    print("\nFINAL DELIVERABLE\n")
    print(final["final_output"])


if __name__ == "__main__":
    main()
