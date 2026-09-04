# Agentic RAG — Three Working Patterns with LangGraph

Three self-contained Jupyter notebooks, each implementing a different way of making a
Retrieval-Augmented Generation (RAG) pipeline "agentic" — able to decide, check itself,
and correct its own mistakes instead of blindly retrieving once and answering.

| Notebook | Pattern | What makes it agentic |
|---|---|---|
| `1_Agentic_RAG.ipynb` | Agentic RAG | An LLM agent decides *whether* to retrieve at all, and *which* of two knowledge bases to search, using tool calling. |
| `2_Corrective_RAG.ipynb` | Corrective RAG (CRAG) | Always retrieves first, then grades what it got — and automatically falls back to a live web search if the local documents aren't good enough. |
| `3_Adaptive_RAG.ipynb` | Adaptive RAG | Routes each question to a vectorstore or the web *before* retrieving, then grades both the documents *and* the final answer (hallucination + relevance checks) before returning it. |

Each notebook is fully commented with markdown cells explaining what every step does and
why — you don't need to already know LangGraph to follow along.

## Why three patterns instead of one?

They sit on a spectrum of how much a RAG pipeline second-guesses itself:

- **Agentic RAG** — the retrieval decision itself is delegated to the LLM.
- **Corrective RAG** — retrieval always happens, but the *result* is checked and
  corrected with a web-search fallback.
- **Adaptive RAG** — adds routing at the front (vectorstore vs. web) *and* a second
  self-check at the very end, on the generated answer itself.

Understanding all three, and where each one is worth the extra complexity, is more
useful than knowing just one.

## Requirements

- [VS Code](https://code.visualstudio.com/) with the Python and Jupyter extensions
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — a fast, single-binary Python package/environment
  manager. It replaces `pip` + `venv` with one tool and one lockfile.
- A [Groq](https://console.groq.com/keys) API key (free tier available) — used by all
  three notebooks as the LLM.
- A [Tavily](https://app.tavily.com) API key (free tier available) — used by the
  Corrective RAG and Adaptive RAG notebooks for live web search.

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
3. Open any of the three `.ipynb` files.
4. In the top-right of the notebook, click **Select Kernel** and choose the
   **agentic-rag** kernel you registered above (it may show as `.venv (Python 3.11)` —
   pick the one whose path points at this project's `.venv`).
5. Run the cells top to bottom with the ▶ buttons, or **Run All**.

That's the whole setup — no manually creating a virtualenv, no separate `pip install -r
requirements.txt` step, and `uv.lock` means everyone who runs `uv sync` gets the exact
same dependency versions you tested with.

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

## Notes

- Embeddings run locally via `sentence-transformers` (`BAAI/bge-m3`) — no embedding API
  key needed.
- Vector data is written to a local Chroma store at runtime and is not committed to the
  repo (`chroma/` is gitignored).
- These notebooks scrape live documentation pages and blog posts at run time, so results
  will vary slightly as those pages change.

## License

This is a paid, personal-use resource — see `LICENSE.md`. In short: use it, learn from
it, build on it, but don't resell or redistribute the notebooks themselves.
