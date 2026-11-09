"""The orchestrator graph (LangGraph `StateGraph`).

Topology — the orchestrator-worker / supervisor pattern:

    START -> plan -> dispatch --(ready task)--> worker --+
                        |  ^------------------------------+
                        |  (loop until no ready tasks)
                        +--(none left)--> synthesize -> critique --(approved)--> END
                                              ^------(revise)----------+

  * plan        decomposes the goal into a dependency-aware task DAG (autonomous decomposition).
  * dispatch    the supervisor: picks the next subtask whose dependencies are all done.
  * worker      runs that subtask as its assigned specialist, with upstream results as context.
  * synthesize  merges all subtask results into one deliverable.
  * critique    a SEPARATE evaluator; loops back to re-synthesize on rejection (bounded).
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .llm import LLMClient
from .models import SubTask
from .state import OrchestrationState


def initial_state(goal: str) -> OrchestrationState:
    """A fresh state for a run."""
    return {
        "goal": goal,
        "subtasks": [],
        "current_task_id": None,
        "final_output": None,
        "review_feedback": None,
        "review_iterations": 0,
        "approved": False,
        "log": [],
    }


def _ready_task(subtasks: list[SubTask]) -> SubTask | None:
    """Next pending subtask whose dependencies are all done (topological readiness)."""
    done = {t.id for t in subtasks if t.status == "done"}
    for t in subtasks:
        if t.status == "pending" and all(dep in done for dep in t.depends_on):
            return t
    # Deadlock guard (bad/cyclic deps): fall back to any pending task so we never hang.
    return next((t for t in subtasks if t.status == "pending"), None)


def build_orchestrator(llm: LLMClient, max_review: int = 2):
    """Compile the orchestrator graph around a given LLM backend."""

    def plan_node(state: OrchestrationState) -> dict:
        plan = llm.plan(state["goal"])
        subtasks = [SubTask.from_plan(p) for p in plan.subtasks]
        lines = [f"plan: decomposed goal into {len(subtasks)} subtasks"]
        lines += [
            f"  - {t.id} [{t.role.value}] {t.title}"
            + (f" (depends on {', '.join(t.depends_on)})" if t.depends_on else "")
            for t in subtasks
        ]
        return {"subtasks": subtasks, "log": lines}

    def dispatch_node(state: OrchestrationState) -> dict:
        task = _ready_task(state["subtasks"])
        if task is None:
            return {"current_task_id": None, "log": ["dispatch: all subtasks complete"]}
        return {
            "current_task_id": task.id,
            "log": [f"dispatch: selected {task.id} ({task.role.value})"],
        }

    def worker_node(state: OrchestrationState) -> dict:
        subtasks = [t.model_copy(deep=True) for t in state["subtasks"]]
        by_id = {t.id: t for t in subtasks}
        task = by_id[state["current_task_id"]]

        # Context = the results of this task's already-completed dependencies.
        context = "\n\n".join(
            f"[{by_id[d].title}] {by_id[d].result}" for d in task.depends_on if by_id[d].result
        )
        task.result = llm.run_subtask(task.role, task.description, context)
        task.status = "done"
        return {
            "subtasks": subtasks,
            "log": [f"worker: executed {task.id} ({task.role.value}) - {task.title}"],
        }

    def synthesize_node(state: OrchestrationState) -> dict:
        results = "\n\n".join(
            f"### {t.title}\n{t.result}" for t in state["subtasks"] if t.result
        )
        draft = llm.synthesize(state["goal"], results, state["review_feedback"])
        tag = "re-synthesized" if state["review_feedback"] else "synthesized"
        return {"final_output": draft, "log": [f"synthesize: {tag} deliverable"]}

    def critique_node(state: OrchestrationState) -> dict:
        verdict = llm.critique(state["goal"], state["final_output"] or "")
        n = state["review_iterations"] + 1
        approved = verdict.approved or n >= max_review  # bound the loop
        if approved:
            why = "approved" if verdict.approved else f"forced-approve at limit ({max_review})"
            log = [f"critique: {why} after {n} review(s)"]
        else:
            log = [f"critique: rejected (round {n}) - {verdict.feedback}"]
        return {
            "approved": approved,
            "review_feedback": None if approved else verdict.feedback,
            "review_iterations": n,
            "log": log,
        }

    def route_after_dispatch(state: OrchestrationState) -> str:
        return "worker" if state["current_task_id"] else "synthesize"

    def route_after_critique(state: OrchestrationState) -> str:
        return END if state["approved"] else "synthesize"

    g = StateGraph(OrchestrationState)
    g.add_node("plan", plan_node)
    g.add_node("dispatch", dispatch_node)
    g.add_node("worker", worker_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("critique", critique_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "dispatch")
    g.add_conditional_edges("dispatch", route_after_dispatch, {"worker": "worker", "synthesize": "synthesize"})
    g.add_edge("worker", "dispatch")
    g.add_edge("synthesize", "critique")
    g.add_conditional_edges("critique", route_after_critique, {"synthesize": "synthesize", END: END})

    return g.compile()
