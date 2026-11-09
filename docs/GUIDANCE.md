# Guidance — Understanding This System

A guided walkthrough so you can read, explain, and extend the PoC. Written to be read top to bottom in ~15 minutes, then used as a map.

---

## 1. The 60-second mental model

Think of it as a **small project team** that assembles itself for one job:

- A **planner** reads the goal and writes the to-do list — *and the dependencies between items* (do A and B before C).
- A **supervisor** hands out the next to-do whose prerequisites are finished.
- **Specialists** (researcher, analyst, coder, writer) each do one item, and each is handed the outputs of the items it depended on.
- A **synthesizer** staples the pieces into one answer.
- A **critic** — a *different* reviewer, not the writer — signs off or sends it back with notes.

The novel part vs. a normal program: **the to-do list isn't written by you, it's written by the planner at runtime** from a one-line goal. That's "autonomous task decomposition."

Everything is wired together as a **graph** (boxes = steps, arrows = "go here next"). That's what LangGraph gives you: a way to describe "which step runs next, based on the current state."

---

## 2. Read the code in this order

You'll understand the whole thing fastest in this sequence. Each file is small.

| # | File | What to notice |
|---|---|---|
| 1 | [`models.py`](../src/orchestrator/models.py) | The **data contracts**. `SubTask` (`id`, `role`, `depends_on`, …) is the unit of work. `PlannerOutput`/`CritiqueOutput` are what the LLM is *forced* to return — these Pydantic classes literally become the JSON schema the model fills in. Start here; everything else moves these objects around. |
| 2 | [`state.py`](../src/orchestrator/state.py) | The **shared blackboard** that flows through the graph. Note `log` is `Annotated[list[str], operator.add]` — the one field that *accumulates* across steps instead of being overwritten (see §3). |
| 3 | [`prompts.py`](../src/orchestrator/prompts.py) + [`agents.py`](../src/orchestrator/agents.py) | The **specialization**. A "specialist" is nothing more than a system prompt bound to the shared model. `agents.py` is just a `role → prompt` lookup. This is why adding an agent is trivial. |
| 4 | [`llm.py`](../src/orchestrator/llm.py) | The **brain**, behind a 4-method interface (`plan`, `run_subtask`, `synthesize`, `critique`). Two implementations: `AnthropicLLM` (real Claude) and `ScriptedLLM` (deterministic, offline). Read `ScriptedLLM` first — it shows the *shape* of each call without API noise. |
| 5 | [`graph.py`](../src/orchestrator/graph.py) | The **orchestration** — the heart. The ASCII diagram at the top is the whole design. Read the five `_node` functions, then the two `route_*` functions, then the `g.add_edge(...)` wiring at the bottom. |
| 6 | [`cli.py`](../src/orchestrator/cli.py) | The thin entrypoint — picks a backend, builds the graph, invokes it, prints the trace. |

Then run `python examples/demo_offline.py` and re-read `graph.py` with the printed trace next to you. The trace lines map 1:1 to the node functions.

---

## 3. The five LangGraph concepts you now know

These transfer to *any* LangGraph project. Each maps to a spot in `graph.py`:

1. **`StateGraph(State)`** — a state machine typed by a state schema. Our state is `OrchestrationState` (a `TypedDict`).
2. **Nodes** are plain functions `state -> partial update`. A node returns *only the keys it changed* (e.g. `worker_node` returns `{"subtasks": ..., "log": [...]}`); LangGraph merges that into the state.
3. **Edges** are the arrows. `add_edge("worker", "dispatch")` = "after worker, always go to dispatch."
4. **Conditional edges** are arrows that *branch*. `add_conditional_edges("dispatch", route_after_dispatch, {...})` calls your function, which returns the name of the next node. This is how the supervisor loops ("more work? → worker; done? → synthesize") and how the critic loops ("rejected? → synthesize; approved? → END").
5. **Reducers** decide how a returned value *merges* into existing state. Default = overwrite. We annotate `log` with `operator.add` so every node can append a trace line and they accumulate instead of clobbering each other. (This also matters the moment you run workers in parallel — shared keys need a reducer or LangGraph errors on the double-write.)

`g.compile()` turns the description into a runnable object; `graph.invoke(initial_state, config={"recursion_limit": 50})` runs it. The `recursion_limit` is a safety fuse against an infinite supervisor↔worker loop — not a sizing knob.

---

## 4. The one big idea: a pluggable LLM

The graph never imports `anthropic`. It only knows the `LLMClient` *protocol* (4 methods). Two things plug in behind it:

- `AnthropicLLM` → real decomposition by Claude.
- `ScriptedLLM` → a deterministic fake that returns a fixed task DAG and a scripted critique.

Why this matters (and why it's worth calling out as an engineering decision):

- **The tests are real.** They run the *actual* graph; only the brain is faked. So "10 passed" genuinely means the orchestration works — not that a mock returned what we told it to.
- **It always runs.** No key? It falls back to offline. Great for demos and CI.
- **It's honest.** Offline proves the *plumbing*; real decomposition needs the key. We never blur that line.

This is the classic "depend on an interface, inject the implementation" move — the same instinct as constructor-injecting an `IEmailer` in C# so you can unit-test without sending mail. Same idea, applied to an LLM.

---

## 5. How to extend it (three recipes)

**A. Add a new specialist (e.g. a "tester").**
1. Add `TESTER = "tester"` to `AgentRole` in `models.py`.
2. Add a `TESTER_SYSTEM = """..."""` prompt in `prompts.py`.
3. Add one line to `ROLE_SYSTEM` in `agents.py`.
That's it — no graph change. The planner can now route subtasks to it.

**B. Turn on real Claude.**
`cp .env.example .env`, paste your `ANTHROPIC_API_KEY`, and drop `--offline`. `default_llm()` switches backends automatically. Watch the plan change from run to run — that's Claude decomposing, not a script.

**C. Give a specialist a real tool (the highest-value next step).**
Right now specialists *reason* but can't *act*. To let the researcher actually search, you'd give `AnthropicLLM.run_subtask` a tool (web search / retrieval) — the cleanest path is **MCP** (Model Context Protocol), the standard for connecting agents to tools. This is item #1 on the TDD roadmap and where the system stops being a demo and starts being useful.

---

## 6. How to talk about it (for the AI initiative)

Three things you can now say with authority:

1. **"There's a ladder, and you climb it only as far as you must."** Single call → workflow → orchestrator-workers → autonomous agent. Each rung adds power but costs latency, money, and predictability. Most problems are solved a rung lower than people reach for. (Source: Anthropic's *Building Effective Agents*.)
2. **"Multi-agent isn't free, and it isn't always right."** It can use ~15× the tokens of a single chat, and it *helps* on breadth-first, read-heavy work but *hurts* on tightly-coupled work like editing one codebase. The skill is matching the pattern to the task shape — see §7.
3. **"Make the structure explicit and the decomposition inspectable."** We chose LangGraph specifically so the plan is a typed artifact you can print and check before any work runs, and so the flow is controllable and resumable — the properties you need to take this to production.

---

## 7. The debate worth knowing

Two credible, *opposing* primary sources — being able to hold both is what makes you sound senior:

- **Anthropic** ran a multi-agent research system that **beat** a single agent by ~90% on their eval, via parallel subagents. Pro multi-agent — *for breadth-first research*.
- **Cognition** ("Don't Build Multi-Agents") argues *against* it for most work: parallel agents with split context make **conflicting decisions**. They favor a single linear agent. Tellingly, Claude Code's subagents never *write code* in parallel — only answer read-only questions.

**The reconciliation:** they're not really disagreeing — they're describing different *task shapes*. Independent + read-heavy → multi-agent wins. Coupled + write-heavy → single agent wins. Both agree the real lever is **context engineering** (giving each agent exactly the context it needs). Our design leans to the safe side: sequential by default, with each worker fed its dependencies' results. (Sources in the TDD references.)

---

## 8. Where to go next

- Re-read [`docs/technical-design-document.md`](technical-design-document.md) §4 (patterns) and §5 (framework choice) — that's the "why" behind every decision in the code.
- Skim **Anthropic — Building Effective Agents** and **Cognition — Don't Build Multi-Agents** (both linked in the TDD). They're short and shape how you'll think about every agent system after this.
- Try the roadmap's #2 (parallel fan-out with LangGraph `Send`) — it's the most instructive next change and forces you to understand reducers properly.

You built (well, commissioned 🙂) a real orchestrator-worker system, grounded it in the current literature, and can defend the framework choice and the trade-offs. That's exactly the "technical contributor beyond daily delivery" bar.
