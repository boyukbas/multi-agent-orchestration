"""The LangGraph state object that flows between nodes.

LangGraph merges each node's returned partial dict into this state. Keys without
a reducer are replaced wholesale; `log` uses `operator.add` so every node can append
trace lines that accumulate across the whole run.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from .models import SubTask


class OrchestrationState(TypedDict):
    """Shared blackboard for the orchestrator graph."""

    goal: str
    subtasks: list[SubTask]            # the live plan; workers mutate status/result
    current_task_id: str | None        # chosen by the dispatcher for the worker node
    final_output: str | None           # produced by the synthesizer
    review_feedback: str | None        # critic feedback fed back into a re-synthesis
    review_iterations: int             # bounds the evaluator-optimizer loop
    approved: bool                     # critic verdict on the latest draft
    log: Annotated[list[str], operator.add]  # append-only human-readable trace
