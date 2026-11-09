"""The specialist registry: maps each role to its system prompt.

This is the "specialized agents" surface. A worker is the pairing of a role's
system prompt with the shared LLM backend; the graph treats every specialist
uniformly and routes work to them by role.
"""
from __future__ import annotations

from . import prompts
from .models import AgentRole

ROLE_SYSTEM: dict[AgentRole, str] = {
    AgentRole.RESEARCHER: prompts.RESEARCHER_SYSTEM,
    AgentRole.ANALYST: prompts.ANALYST_SYSTEM,
    AgentRole.CODER: prompts.CODER_SYSTEM,
    AgentRole.WRITER: prompts.WRITER_SYSTEM,
    AgentRole.REVIEWER: prompts.CRITIC_SYSTEM,
}


def system_for(role: AgentRole) -> str:
    """The system prompt that specializes a worker for `role`."""
    return ROLE_SYSTEM[role]
