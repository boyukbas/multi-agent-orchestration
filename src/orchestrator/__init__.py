"""Orchestrator-worker multi-agent system on LangGraph.

A planner agent decomposes a goal into a dependency-aware task DAG; a supervisor
dispatches each ready subtask to the best-suited specialist agent; a synthesizer
merges the results; a critic loop checks quality. The LLM backend is pluggable
(`AnthropicLLM` for real autonomous decomposition, `ScriptedLLM` for offline runs).
"""
from .graph import build_orchestrator
from .llm import AnthropicLLM, LLMClient, ScriptedLLM, default_llm
from .models import AgentRole, Plan, SubTask

__all__ = [
    "build_orchestrator",
    "AnthropicLLM",
    "ScriptedLLM",
    "LLMClient",
    "default_llm",
    "AgentRole",
    "Plan",
    "SubTask",
]
