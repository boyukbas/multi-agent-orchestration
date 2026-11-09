"""Verifies the AnthropicLLM wiring without a network call or API key.

A fake client stands in for `anthropic.Anthropic()`, so we can assert that:
  * structured calls (plan / critique) go through `messages.parse` and are validated,
  * free-form calls (run_subtask / synthesize) extract text from `messages.create`,
  * when `messages.parse` is unavailable, the code falls back to forced tool-use.

This is the one path the offline demo can't exercise; mocking the SDK keeps it covered.
"""
from types import SimpleNamespace

import pytest

from orchestrator.llm import AnthropicLLM
from orchestrator.models import AgentRole, CritiqueOutput, PlannerOutput


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_block(data):
    return SimpleNamespace(type="tool_use", input=data)


class _Messages:
    """Mimics `client.messages` with configurable parse behaviour."""

    def __init__(self, parse_works=True):
        self.parse_works = parse_works
        self.create_calls = []

    def parse(self, *, output_format, **kwargs):
        if not self.parse_works:
            raise RuntimeError("messages.parse unavailable on this SDK version")
        if output_format is PlannerOutput:
            data = {"subtasks": [
                {"id": "t1", "title": "Find facts", "description": "...", "role": "researcher", "depends_on": []},
                {"id": "t2", "title": "Write", "description": "...", "role": "writer", "depends_on": ["t1"]},
            ]}
        else:  # CritiqueOutput
            data = {"approved": True, "feedback": "ok"}
        return SimpleNamespace(parsed_output=output_format.model_validate(data))

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if "tools" in kwargs:  # structured fallback path -> return a tool_use block
            schema = kwargs["tools"][0]["input_schema"]
            if "approved" in schema.get("properties", {}):
                return SimpleNamespace(content=[_tool_block({"approved": False, "feedback": "fix it"})])
            return SimpleNamespace(content=[_tool_block(
                {"subtasks": [{"id": "t1", "title": "x", "description": "y", "role": "analyst", "depends_on": []}]}
            )])
        return SimpleNamespace(content=[_text_block("free-form answer")])


def _llm(parse_works=True):
    return AnthropicLLM(model="claude-opus-4-8", client=SimpleNamespace(messages=_Messages(parse_works)))


def test_plan_uses_structured_output():
    plan = _llm().plan("build something")
    assert isinstance(plan, PlannerOutput)
    assert [t.id for t in plan.subtasks] == ["t1", "t2"]
    assert plan.subtasks[1].depends_on == ["t1"]


def test_run_subtask_and_synthesize_extract_text():
    llm = _llm()
    assert llm.run_subtask(AgentRole.RESEARCHER, "do x", "ctx") == "free-form answer"
    assert llm.synthesize("goal", "results", feedback=None) == "free-form answer"


def test_critique_returns_typed_verdict():
    verdict = _llm().critique("goal", "draft")
    assert isinstance(verdict, CritiqueOutput)
    assert verdict.approved is True


def test_falls_back_to_tool_use_when_parse_unavailable():
    llm = _llm(parse_works=False)
    plan = llm.plan("build something")  # parse raises -> forced tool-use path
    assert isinstance(plan, PlannerOutput)
    assert plan.subtasks[0].role is AgentRole.ANALYST
    # the fallback issued a forced tool call
    assert llm._client.messages.create_calls[-1]["tool_choice"]["name"] == "emit"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
