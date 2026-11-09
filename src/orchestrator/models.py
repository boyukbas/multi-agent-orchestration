"""Typed domain models shared across the orchestrator.

These Pydantic models double as the JSON Schema contract the planner / critic LLM
calls are forced to emit, so the structure here *is* the agent interface.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """The specialist roles a subtask can be routed to."""

    RESEARCHER = "researcher"  # gathers / produces raw information
    ANALYST = "analyst"        # reasons over inputs, compares, draws conclusions
    CODER = "coder"            # writes or explains code / pseudocode
    WRITER = "writer"          # turns material into clear prose
    REVIEWER = "reviewer"      # quality-checks (used by the critic node)


class PlannedSubTask(BaseModel):
    """One subtask as proposed by the planner (no execution state yet)."""

    id: str = Field(description="Stable short id, e.g. 't1'. Referenced by depends_on.")
    title: str = Field(description="Short imperative title.")
    description: str = Field(
        description="A self-contained instruction the assigned specialist can act on "
        "without seeing the original goal."
    )
    role: AgentRole = Field(description="Which specialist should execute this subtask.")
    depends_on: list[str] = Field(
        default_factory=list,
        description="ids of subtasks whose results are required before this one runs.",
    )


class PlannerOutput(BaseModel):
    """The structured object the planner agent returns."""

    subtasks: list[PlannedSubTask] = Field(
        description="The goal decomposed into 2-6 ordered, dependency-aware subtasks."
    )


class CritiqueOutput(BaseModel):
    """The structured verdict the critic agent returns (evaluator-optimizer loop)."""

    approved: bool = Field(description="True if the draft adequately satisfies the goal.")
    feedback: str = Field(
        description="If not approved, concrete, actionable revisions. Else a short note."
    )


class SubTask(BaseModel):
    """A subtask plus its runtime execution state (lives in the graph state)."""

    id: str
    title: str
    description: str
    role: AgentRole
    depends_on: list[str] = Field(default_factory=list)
    status: str = "pending"  # pending | done
    result: str | None = None

    @classmethod
    def from_plan(cls, planned: PlannedSubTask) -> "SubTask":
        return cls(
            id=planned.id,
            title=planned.title,
            description=planned.description,
            role=planned.role,
            depends_on=list(planned.depends_on),
        )


class Plan(BaseModel):
    """A goal plus its decomposed subtasks (convenience wrapper for callers/tests)."""

    goal: str
    subtasks: list[SubTask]
