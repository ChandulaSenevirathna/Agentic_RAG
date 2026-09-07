# Agentic RAG — Four Working Patterns with LangGraph

Four self-contained Jupyter notebooks, each implementing a different way of making a
Retrieval-Augmented Generation (RAG) pipeline "agentic" — able to decide, check itself,
correct its own mistakes, or defer to a person, instead of blindly retrieving once and
answering.

| Notebook | Pattern | What makes it agentic |
|---|---|---|
| `1_Agentic_RAG.ipynb` **(free demo)** | Agentic RAG | An LLM agent decides *whether* to retrieve at all, and *which* of two knowledge bases to search, using tool calling. |
| `2_Corrective_RAG.ipynb` | Corrective RAG (CRAG) | Always retrieves first, then grades what it got — and automatically falls back to a live web search if the local documents aren't good enough. |
| `3_Adaptive_RAG.ipynb` | Adaptive RAG | Routes each question to a vectorstore or the web *before* retrieving, then grades both the documents *and* the final answer (hallucination + relevance checks) before returning it. |
| `4_Human_in_the_Loop_RAG.ipynb` | Human-in-the-Loop RAG | Takes the same retrieve/grade/generate machinery and replaces the automatic loop-decisions with a person: the graph pauses after retrieval and after generation, shows the LLM's grades as advisory suggestions only, and waits for a human to approve, request a revision, or send it back. |

Each notebook is fully commented with markdown cells explaining what every step does and
why — you don't need to already know LangGraph to follow along.

## Why four patterns instead of one?

They sit on a spectrum of how much a RAG pipeline second-guesses itself — and who gets
the final say:

- **Agentic RAG** — the retrieval decision itself is delegated to the LLM.
- **Corrective RAG** — retrieval always happens, but the *result* is checked and
  corrected with a web-search fallback.
- **Adaptive RAG** — adds routing at the front (vectorstore vs. web) *and* a second
  self-check at the very end, on the generated answer itself.
- **Human-in-the-Loop RAG** — keeps the same grading/self-check machinery as Adaptive
  RAG, but the LLM's verdicts stop being routing decisions and become suggestions: a
  real person approves, revises, or rejects at two checkpoints using LangGraph's
  `interrupt` / `Command(resume=...)` pattern with a `MemorySaver` checkpointer.

Understanding all four, and where each one is worth the extra complexity, is more
useful than knowing just one.

## Requirements

- [VS Code](https://code.visualstudio.com/) with the Python and Jupyter extensions
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — a fast, single-binary Python package/environment
  manager. It replaces `pip` + `venv` with one tool and one lockfile.
- A [Groq](https://console.groq.com/keys) API key (free tier available) — used by all
  four notebooks as the LLM.
- A [Tavily](https://app.tavily.com) API key (free tier available) — used by the
  Corrective RAG and Adaptive RAG notebooks for live web search. Not needed for
  Agentic RAG or Human-in-the-Loop RAG.

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
3. Open any of the four `.ipynb` files.
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

**`1_Agentic_RAG.ipynb`** scrapes the LangGraph and LangChain documentation into two
separate Chroma vector stores, wraps each as a retriever tool, and gives both tools to a
tool-calling agent. The agent decides per question whether to call a tool, which one,
and whether the retrieved documents are good enough to answer from or need a rewritten
query.

**`2_Corrective_RAG.ipynb`** indexes a small set of blog posts on AI agents into one
vector store. Every question always retrieves from it; a grading step then checks each
retrieved chunk for relevance. If nothing relevant comes back, the question is rewritten
for web search and Tavily fills the gap before the answer is generated.

**`3_Adaptive_RAG.ipynb`** builds on the same idea but adds a router at the very start
(should this question go to the vectorstore or straight to the web?) and two more graders
at the very end, checking that the generated answer is actually grounded in the
retrieved documents and actually answers the question — looping back to retry if either
check fails.

**`4_Human_in_the_Loop_RAG.ipynb`** takes the retrieve/grade/generate/self-check
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
Tavily.

## Notes

- Embeddings run locally via `sentence-transformers` (`BAAI/bge-m3`) — no embedding API
  key needed.
- Vector data is written to a local Chroma store at runtime and is not committed to the
  repo (`chroma/` is gitignored).
- These notebooks scrape live documentation pages and blog posts at run time, so results
  will vary slightly as those pages change.

## License

`1_Agentic_RAG.ipynb` is a free demo — use it, share it, redistribute it. The other
three notebooks are a paid, personal-use resource — see `LICENSE.md`. In short: use
them, learn from them, build on them, but don't resell or redistribute the notebooks
themselves.
