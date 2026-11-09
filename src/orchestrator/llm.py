"""The pluggable LLM backend.

The orchestration graph depends only on the `LLMClient` protocol, so the same
graph runs against:
  * `AnthropicLLM`  — real Claude; genuine autonomous decomposition (needs ANTHROPIC_API_KEY).
  * `ScriptedLLM`   — deterministic, offline; proves the orchestration mechanics with no API calls.

This separation is the reason the PoC is testable and demoable without secrets:
the *framework* (decomposition, routing, dependency scheduling, synthesis, critique)
is exercised identically either way — only the "thinking" differs.
"""
from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from . import prompts
from .models import AgentRole, CritiqueOutput, PlannedSubTask, PlannerOutput

DEFAULT_MODEL = "claude-opus-4-8"


@runtime_checkable
class LLMClient(Protocol):
    """What every agent node needs from a language model."""

    def plan(self, goal: str) -> PlannerOutput:
        """Decompose a goal into a dependency-aware list of subtasks."""

    def run_subtask(self, role: AgentRole, instruction: str, context: str) -> str:
        """Execute one subtask as the given specialist, given upstream context."""

    def synthesize(self, goal: str, results: str, feedback: str | None = None) -> str:
        """Merge subtask results into a final deliverable (optionally revising on feedback)."""

    def critique(self, goal: str, draft: str) -> CritiqueOutput:
        """Judge whether the draft satisfies the goal (evaluator-optimizer)."""


# --------------------------------------------------------------------------- #
# Real backend: Claude via the official Anthropic SDK
# --------------------------------------------------------------------------- #
class AnthropicLLM:
    """Backs the agents with Claude. Structured calls (plan/critique) use the
    SDK's `messages.parse`, falling back to forced tool-use for resilience across
    SDK/model versions. Free-form calls use `messages.create` with adaptive thinking.
    """

    def __init__(self, model: str | None = None, client=None) -> None:
        if client is None:
            import anthropic  # imported lazily so offline runs need no SDK/key

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self._client = client  # injectable for testing
        self.model = model or os.getenv("ORCHESTRATOR_MODEL", DEFAULT_MODEL)

    # -- helpers ----------------------------------------------------------- #
    def _text(self, system: str, user: str, max_tokens: int = 2048) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            thinking={"type": "adaptive"},
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def _structured(self, system: str, user: str, schema, max_tokens: int = 2048):
        # Preferred path: native structured output (validated against the schema).
        try:
            resp = self._client.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
            if getattr(resp, "parsed_output", None) is not None:
                return resp.parsed_output
        except Exception:
            pass  # fall through to the universally-supported tool-use path

        # Fallback: force a tool call whose schema is the Pydantic model.
        tool = {
            "name": "emit",
            "description": "Return the structured result.",
            "input_schema": schema.model_json_schema(),
        }
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit"},
        )
        block = next(b for b in resp.content if b.type == "tool_use")
        return schema.model_validate(block.input)

    # -- LLMClient --------------------------------------------------------- #
    def plan(self, goal: str) -> PlannerOutput:
        return self._structured(prompts.PLANNER_SYSTEM, f"GOAL: {goal}", PlannerOutput)

    def run_subtask(self, role: AgentRole, instruction: str, context: str) -> str:
        from .agents import system_for

        user = instruction if not context else f"{instruction}\n\nContext from prior steps:\n{context}"
        return self._text(system_for(role), user)

    def synthesize(self, goal: str, results: str, feedback: str | None = None) -> str:
        user = f"GOAL: {goal}\n\nCompleted subtask results:\n{results}"
        if feedback:
            user += f"\n\nThe previous draft was rejected. Address this feedback:\n{feedback}"
        return self._text(prompts.SYNTHESIZER_SYSTEM, user, max_tokens=3072)

    def critique(self, goal: str, draft: str) -> CritiqueOutput:
        user = f"GOAL: {goal}\n\nDRAFT:\n{draft}"
        return self._structured(prompts.CRITIC_SYSTEM, user, CritiqueOutput, max_tokens=1024)


# --------------------------------------------------------------------------- #
# Offline backend: deterministic, no API calls
# --------------------------------------------------------------------------- #
class ScriptedLLM:
    """A deterministic stand-in used for tests and key-less demos.

    The planner emits a fixed 4-node DAG (two independent research tasks -> analysis
    -> writing) so the dependency scheduler has something real to schedule. The critic
    rejects the first draft and approves the second, so the evaluator-optimizer loop is
    visibly exercised. Reasoning is templated, not real — the point is to prove the
    *orchestration*, not to produce genuine content.
    """

    def __init__(self) -> None:
        self._critique_calls = 0

    def plan(self, goal: str) -> PlannerOutput:
        return PlannerOutput(
            subtasks=[
                PlannedSubTask(
                    id="t1",
                    title="Gather background",
                    description=f"Collect the key facts needed to address: {goal}",
                    role=AgentRole.RESEARCHER,
                    depends_on=[],
                ),
                PlannedSubTask(
                    id="t2",
                    title="Gather constraints",
                    description=f"Identify the constraints and trade-offs relevant to: {goal}",
                    role=AgentRole.RESEARCHER,
                    depends_on=[],
                ),
                PlannedSubTask(
                    id="t3",
                    title="Analyze findings",
                    description="Weigh the gathered facts and constraints and reach a recommendation.",
                    role=AgentRole.ANALYST,
                    depends_on=["t1", "t2"],
                ),
                PlannedSubTask(
                    id="t4",
                    title="Write the deliverable",
                    description="Turn the analysis into a clear final answer for the user.",
                    role=AgentRole.WRITER,
                    depends_on=["t3"],
                ),
            ]
        )

    def run_subtask(self, role: AgentRole, instruction: str, context: str) -> str:
        note = f" (using {len(context)} chars of upstream context)" if context else ""
        return f"[{role.value}] {instruction}{note} -> done."

    def synthesize(self, goal: str, results: str, feedback: str | None = None) -> str:
        head = "Summary: revised to address review feedback.\n" if feedback else ""
        return f"{head}Deliverable for '{goal}':\n{results}"

    def critique(self, goal: str, draft: str) -> CritiqueOutput:
        self._critique_calls += 1
        if self._critique_calls < 2:
            return CritiqueOutput(approved=False, feedback="Add an explicit summary line up front.")
        return CritiqueOutput(approved=True, feedback="Meets the goal.")


def default_llm(offline: bool = False) -> LLMClient:
    """Pick a backend: real Claude when a key is present, else the offline stand-in."""
    if offline or not os.getenv("ANTHROPIC_API_KEY"):
        return ScriptedLLM()
    return AnthropicLLM()
