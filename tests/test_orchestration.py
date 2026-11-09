"""Offline tests that prove the orchestration mechanics with no API calls.

They run the real LangGraph graph against the deterministic ScriptedLLM, so a green
suite is genuine evidence the decomposition, dependency scheduling, synthesis, and
evaluator-optimizer loop all work end to end.
"""
from orchestrator.graph import build_orchestrator, initial_state
from orchestrator.llm import ScriptedLLM
from orchestrator.models import AgentRole


def _run(goal="Recommend a caching strategy for a catalog API.", max_review=2):
    graph = build_orchestrator(ScriptedLLM(), max_review=max_review)
    return graph.invoke(initial_state(goal), config={"recursion_limit": 50})


def test_planner_decomposes_into_a_dag():
    final = _run()
    subtasks = final["subtasks"]
    assert len(subtasks) >= 3
    # At least one subtask depends on another -> it's a DAG, not a flat list.
    assert any(t.depends_on for t in subtasks)
    # ids referenced in depends_on actually exist.
    ids = {t.id for t in subtasks}
    assert all(dep in ids for t in subtasks for dep in t.depends_on)


def test_all_subtasks_complete_and_specialists_assigned():
    final = _run()
    assert final["subtasks"], "expected a non-empty plan"
    assert all(t.status == "done" and t.result for t in final["subtasks"])
    # Roles came from the AgentRole enum (specialists were actually assigned).
    assert all(isinstance(t.role, AgentRole) for t in final["subtasks"])


def test_dependencies_are_respected_in_execution_order():
    final = _run()
    order = [ln.split()[2] for ln in final["log"] if ln.startswith("worker: executed")]
    pos = {tid: i for i, tid in enumerate(order)}
    for t in final["subtasks"]:
        for dep in t.depends_on:
            assert pos[dep] < pos[t.id], f"{t.id} ran before its dependency {dep}"


def test_evaluator_optimizer_loop_runs_then_approves():
    final = _run(max_review=2)
    # ScriptedLLM rejects once, then approves -> exactly one revision happened.
    assert final["approved"] is True
    assert final["review_iterations"] >= 1
    assert any("re-synthesized" in ln for ln in final["log"])


def test_final_deliverable_is_produced():
    final = _run()
    assert final["final_output"]
    assert "Deliverable" in final["final_output"]


def test_max_review_bounds_the_loop():
    # A critic that never approves must still terminate at the cap.
    class NeverApprove(ScriptedLLM):
        def critique(self, goal, draft):
            from orchestrator.models import CritiqueOutput

            return CritiqueOutput(approved=False, feedback="never good enough")

    graph = build_orchestrator(NeverApprove(), max_review=3)
    final = graph.invoke(initial_state("x"), config={"recursion_limit": 50})
    assert final["approved"] is True  # forced-approved at the limit
    assert final["review_iterations"] == 3
