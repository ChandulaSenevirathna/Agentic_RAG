# Agentic RAG — Ten Working Notebooks with LangGraph

Ten self-contained Jupyter notebooks. The first four are free and build up the core
LangGraph primitives and a working agentic RAG agent from scratch; the remaining six
are a paid, deeper dive into more advanced ways of making a Retrieval-Augmented
Generation (RAG) pipeline "agentic" — able to correct its own mistakes, defer to a
person, split work across specialists, run branches in parallel, or plan ahead.

## Free

| Notebook | What it covers |
|---|---|
| [`1_LangGraph_Starter.ipynb`](https://chandula7.gumroad.com/l/AgenticRAGFundamentals) | The two ideas every other notebook builds on: **state** (the shared object that flows through a graph) and **reducers** (how a node's return value gets combined into state instead of overwriting it) — shown with a single-node graph and a two-parallel-node graph. |
| [`2_LangGraph_Conditional_Routing.ipynb`](https://chandula7.gumroad.com/l/AgenticRAGFundamentals) | Adds the third core primitive: **conditional edges**. An LLM classifies a question, and a router function sends it down one of two paths at runtime — the same mechanic used later to decide "call a tool" vs. "answer directly." |
| [`3_Agentic_RAG.ipynb`](https://chandula7.gumroad.com/l/AgenticRAGFundamentals) | An LLM agent decides *whether* to retrieve at all and *which* of two knowledge bases (LangGraph docs vs. LangChain docs) to search, using tool calling — and now supports **multi-hop** retrieval: after generating an answer, control returns to the agent to decide whether a second lookup is still needed, capped by a hop counter. |
| [`4_ReAct_MultiHop_Agentic_RAG.ipynb`](https://chandula7.gumroad.com/l/AgenticRAGFundamentals) | The same multi-hop, two-knowledge-base problem as notebook 3, solved a different way: LangChain's prebuilt `create_agent` ReAct loop instead of a hand-wired graph, with no separate grading or rewrite nodes — the agent itself decides whether to answer, retry a tool, or call a different one. |

## Paid

| Notebook | Pattern | What makes it agentic |
|---|---|---|
| [`5_Corrective_RAG.ipynb`](https://chandula7.gumroad.com/l/Advanced_RAG_LangGraph_Patterns) | Corrective RAG (CRAG) | Always retrieves first, then grades what it got — and automatically falls back to a live web search if the local documents aren't good enough. |
| [`6_Adaptive_RAG.ipynb`](https://chandula7.gumroad.com/l/Advanced_RAG_LangGraph_Patterns) | Adaptive RAG | Routes each question to a vectorstore or the web *before* retrieving, then grades both the documents *and* the final answer (hallucination + relevance checks) before returning it. |
| [`7_Human_in_the_Loop_RAG.ipynb`](https://chandula7.gumroad.com/l/Advanced_RAG_LangGraph_Patterns) | Human-in-the-Loop RAG | Takes the same retrieve/grade/generate machinery and replaces the automatic loop-decisions with a person: the graph pauses after retrieval and after generation, shows the LLM's grades as advisory suggestions only, and waits for a human to approve, request a revision, or send it back. |
| [`8_Multi_Agent_Supervisor_RAG.ipynb`](https://chandula7.gumroad.com/l/Advanced_RAG_LangGraph_Patterns) | Multi-Agent Supervisor RAG | A supervisor routes each question to one of three specialists (LangGraph docs, LangChain docs, or live web search) — each its own separately compiled subgraph — then critiques the specialist's answer and can escalate to the web specialist if it's not good enough. |
| [`9_Parallel_FanOut_RAG.ipynb`](https://chandula7.gumroad.com/l/Advanced_RAG_LangGraph_Patterns) | Parallel Fan-Out RAG (Map-Reduce) | Breaks a multi-part or comparison question into sub-questions, fans them out to run concurrently with LangGraph's `Send` API, then aggregates and synthesizes the sub-answers into one final answer. |
| [`10_Plan_and_Execute_RAG.ipynb`](https://chandula7.gumroad.com/l/Advanced_RAG_LangGraph_Patterns) | Plan-and-Execute RAG | Plans an ordered list of steps up front, works through them one at a time (retrieve + answer per step), and a replanner checks after each step whether enough has been gathered before synthesizing the final answer. |

Each notebook is fully commented with markdown cells explaining what every step does and
why — you don't need to already know LangGraph to follow along.

> **A numbering note:** this repo has been renumbered a couple of times as notebooks
> were added, and a handful of markdown cells inside the notebooks still refer to
> earlier numbers (e.g. `3_Agentic_RAG.ipynb` was `1_Agentic_RAG.ipynb`,
> `7_Human_in_the_Loop_RAG.ipynb` was `4_...` and then `5_...`, and so on). Same
> notebooks, just stale labels left over from before the rename — worth a pass to
> update those comments so they match the current filenames above.

## Why start with the free four?

- **LangGraph Starter** — state and reducers, with nothing else in the way.
- **Conditional Routing** — adds the one primitive (conditional edges) that turns a
  fixed pipeline into something that can make a decision at runtime.
- **Agentic RAG** — puts state, reducers, and conditional edges together into an actual
  retrieval agent, and adds multi-hop looping so a single question can require more
  than one lookup.
- **ReAct Multi-Hop Agentic RAG** — the same multi-hop retrieval problem, solved with
  LangChain's prebuilt agent loop instead of a hand-wired graph, so you can compare a
  from-scratch `StateGraph` against the prebuilt alternative directly.

From there, the paid notebooks sit on a spectrum of how much a pipeline second-guesses
itself, who gets the final say, and how the work is structured:

- **Corrective RAG** — retrieval always happens, but the *result* is checked and
  corrected with a web-search fallback.
- **Adaptive RAG** — adds routing at the front (vectorstore vs. web) *and* a second
  self-check at the very end, on the generated answer itself.
- **Human-in-the-Loop RAG** — keeps the same grading/self-check machinery as Adaptive
  RAG, but the LLM's verdicts become suggestions: a real person approves, revises, or
  rejects at two checkpoints using LangGraph's `interrupt` / `Command(resume=...)`
  pattern with a `MemorySaver` checkpointer.
- **Multi-Agent Supervisor RAG** — splits the work across three specialist subgraphs
  (LangGraph docs, LangChain docs, web), with a supervisor routing questions and
  critiquing answers.
- **Parallel Fan-Out RAG** — a single question can become several sub-questions
  answered concurrently and merged, using LangGraph's `Send` API for map-reduce style
  branching.
- **Plan-and-Execute RAG** — separates planning from execution entirely: the full list
  of steps is decided up front (ReWOO-style), rather than one LLM call deciding "what
  to do" and "do it" together on every turn.

## Requirements

- [VS Code](https://code.visualstudio.com/) with the Python and Jupyter extensions
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — a fast, single-binary Python package/environment
  manager. It replaces `pip` + `venv` with one tool and one lockfile.
- A [Groq](https://console.groq.com/keys) API key (free tier available) — used by all
  ten notebooks as the LLM.
- A [Tavily](https://app.tavily.com) API key (free tier available) — used for live web
  search by `5_Corrective_RAG.ipynb`, `6_Adaptive_RAG.ipynb`, and
  `8_Multi_Agent_Supervisor_RAG.ipynb`. Not needed for any of the four free notebooks,
  nor for Human-in-the-Loop, Parallel Fan-Out, or Plan-and-Execute — those only ever
  retrieve from a local vector store.

## Setup with uv

### 1. Install uv

Pick whichever matches your setup — you only need one of these.

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Already have Python + pip and prefer not to run an install script:**

```bash
pip install uv
```

**Other options** (Homebrew, `pipx`, `winget`, `cargo`, standalone downloads) are listed
in the [official install docs](https://docs.astral.sh/uv/getting-started/installation/).

Check it worked:

```bash
uv --version
```

### 2. Set up the project

From the project folder:

```bash
# Creates a .venv and installs every dependency from pyproject.toml,
# pinned exactly via uv.lock, in one step.
uv sync

# Register the environment as a Jupyter kernel.
uv run python -m ipykernel install --user --name agentic-rag
```

Copy the env template and add your API keys:

**macOS / Linux:**
```bash
cp .env.example .env
```

**Windows (Command Prompt):**
```
copy .env.example .env
```

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

Open `.env` and fill in `GROQ_API_KEY` and `TAVILY_API_KEY`.

### 3. Open in VS Code

1. Install the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
   and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
   extensions in VS Code, if you don't already have them.
2. Open the project folder in VS Code (`File > Open Folder...`).
3. Open any of the ten `.ipynb` files.
4. In the top-right of the notebook, click **Select Kernel** and choose the
   **agentic-rag** kernel you registered above (it may show as `.venv (Python 3.11)` —
   pick the one whose path points at this project's `.venv`).
5. Run the cells top to bottom with the ▶ buttons, or **Run All**.

That's the whole setup — no manually creating a virtualenv, no separate `pip install -r
requirements.txt` step, and `uv.lock` means everyone who runs `uv sync` gets the exact
same dependency versions you tested with.

> **Kernel is selected per notebook, not per project.** VS Code remembers a separate
> kernel choice for each `.ipynb` file, so it's easy to open a second notebook and have
> it silently fall back to your global Python install instead of this project's
> `.venv` — especially if the `.venv` kernel hasn't been used in that notebook before.
> If one notebook runs fine but another throws import errors for packages this project
> clearly installs (e.g. `chromadb`, `langchain_chroma`), that's the first thing to
> check: open the kernel picker for the failing notebook and confirm the path points
> into this project's `.venv`, not `AppData\Local\Programs\Python\...` or any other
> system/global install.

## What each notebook builds, briefly

### Free

**`1_LangGraph_Starter.ipynb`** builds the two smallest possible graphs to introduce
**state** and **reducers** before any retrieval or tools are involved: one graph with a
single LLM node (the basic node → edge → compile → run loop), and one graph with two
parallel nodes that both write to the same state key — the smallest example that
actually needs a reducer to combine their results instead of one overwriting the other.

**`2_LangGraph_Conditional_Routing.ipynb`** adds the third core primitive: a
`classify` node labels an incoming question (e.g. `"weather"` vs. `"other"`), and a
conditional edge reads that label to route to one of two handler nodes at runtime —
the same mechanic the Agentic RAG notebooks use to decide "call a tool" vs. "answer
directly," isolated here with no retrieval in the way.

**`3_Agentic_RAG.ipynb`** scrapes the LangGraph and LangChain documentation into two
separate Chroma vector stores, wraps each as a retriever tool, and gives both tools to
a tool-calling agent. The agent decides per question whether to call a tool, which one,
and whether the retrieved documents are good enough to answer from or need a rewritten
query. After generating an answer, control returns to the agent instead of ending
immediately, so it can chain a second lookup for questions that span both knowledge
bases — capped by a `MAX_HOPS` counter so it can't loop forever.

**`4_ReAct_MultiHop_Agentic_RAG.ipynb`** solves the same multi-hop, two-knowledge-base
problem as notebook 3, but with LangChain's prebuilt `create_agent` (the current
`langchain.agents` entry point that replaced `langgraph.prebuilt.create_react_agent` in
LangGraph v1) instead of a hand-wired `StateGraph`. There's no separate grading node and
no rewrite node — one ReAct loop reads each tool result and decides, on its own,
whether to answer, retry the same tool with a refined query, or call a different tool,
letting it naturally chain several retrievals for multi-hop questions.

### Paid

**`5_Corrective_RAG.ipynb`** indexes a small set of blog posts on AI agents into one
vector store. Every question always retrieves from it; a grading step then checks each
retrieved chunk for relevance. If nothing relevant comes back, the question is rewritten
for web search and Tavily fills the gap before the answer is generated.

**`6_Adaptive_RAG.ipynb`** builds on the same idea but adds a router at the very start
(should this question go to the vectorstore or straight to the web?) and two more graders
at the very end, checking that the generated answer is actually grounded in the
retrieved documents and actually answers the question — looping back to retry if either
check fails.

**`7_Human_in_the_Loop_RAG.ipynb`** takes the retrieve/grade/generate/self-check
machinery from the Adaptive RAG notebook and puts a person in charge of the two
decisions that used to be automatic:

- After retrieval, each document is graded for relevance as before, but the grade is
  shown to a human as a suggestion only — the human decides whether to proceed to
  generation or reject and have the question rewritten and re-retrieved.
- After generation, the two self-checks (grounded-in-documents, addresses-the-question)
  are shown for reference, but the human has the final call: approve, ask for a
  revision with their own feedback, or send it back to retrieval.

Mechanically, this runs on LangGraph's current recommended HITL pattern: `interrupt(payload)`
called inside a node pauses the graph and surfaces `payload` to whoever is running it;
resuming happens with `graph.stream(Command(resume=...), config)`, and a `MemorySaver`
checkpointer persists the graph's state while it's paused. All branching still goes
through plain `add_conditional_edges` rather than `Command(goto=...)`, so every routing
decision lives in one place. This notebook only needs `GROQ_API_KEY` — it doesn't call
Tavily. It indexes the same Lilian Weng blog posts used by the Corrective, Adaptive,
and Plan-and-Execute notebooks.

**`8_Multi_Agent_Supervisor_RAG.ipynb`** builds a supervisor + specialist subgraphs
pattern. Instead of one agent calling tools directly or one pipeline self-correcting, a
supervisor routes each question to one of three specialists — a LangGraph-docs
specialist, a LangChain-docs specialist, and a web-search specialist (Tavily) — each
its own small, separately compiled `StateGraph` invoked as a subgraph. The supervisor
then critiques the specialist's answer and either accepts it or escalates to the web
specialist for a second attempt. This is the first notebook in the series to compose
subgraphs as nodes rather than using one flat graph.

**`9_Parallel_FanOut_RAG.ipynb`** builds a decompose → fan-out → aggregate pattern for
multi-part or comparison-style questions. A question is broken into a handful of
focused sub-questions; each sub-question is answered by its own parallel branch of the
graph using LangGraph's `Send` API (the mechanism behind map-reduce workflows); an
`operator.add` reducer collects the branches' results back into one list, which is then
synthesized into a single final answer. Unlike notebook 3's two separate knowledge
bases, this one merges the LangGraph and LangChain docs into a single shared vector
store so any sub-question can draw from either topic. `GROQ_API_KEY` only — no Tavily.

**`10_Plan_and_Execute_RAG.ipynb`** builds a plan-first, then-execute pattern
(ReWOO-style). Rather than an agent that reacts one decision at a time, the question is
broken into an ordered list of steps up front; each step is worked through one at a
time — retrieving and answering just that step — and a replanner checks after every
step whether enough has been gathered yet, looping back for more steps or moving on to
synthesize the final answer. It indexes the same three Lilian Weng blog posts (agents,
prompt engineering, adversarial attacks) used in the Corrective, Adaptive, and
Human-in-the-Loop notebooks — three distinct topics, which is what makes a
one-step-per-topic plan a natural fit. `GROQ_API_KEY` only — no Tavily.

## Notes

- Embeddings run locally via `sentence-transformers` (`BAAI/bge-m3`) — no embedding API
  key needed.
- Vector data is written to a local Chroma store at runtime and is not committed to the
  repo (`chroma/` is gitignored).
- These notebooks scrape live documentation pages and blog posts at run time, so results
  will vary slightly as those pages change.

## License

`1_LangGraph_Starter.ipynb`, `2_LangGraph_Conditional_Routing.ipynb`,
`3_Agentic_RAG.ipynb`, and `4_ReAct_MultiHop_Agentic_RAG.ipynb` are free — use them,
share them, redistribute them
([get them here](https://chandula7.gumroad.com/l/AgenticRAGFundamentals)). The other
six notebooks (`5_Corrective_RAG.ipynb`, `6_Adaptive_RAG.ipynb`,
`7_Human_in_the_Loop_RAG.ipynb`, `8_Multi_Agent_Supervisor_RAG.ipynb`,
`9_Parallel_FanOut_RAG.ipynb`, `10_Plan_and_Execute_RAG.ipynb`) are a paid,
personal-use resource
([get them here](https://chandula7.gumroad.com/l/Advanced_RAG_LangGraph_Patterns)) —
see `LICENSE.md`. In short: use them, learn from them, build on them, but don't resell
or redistribute the notebooks themselves.
