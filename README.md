# SmartHire AI

Multi-agent recruitment system built with **LangGraph + LangChain + Streamlit**.
Agents (Supervisor, Resume Screening, Candidate Matching, Interview Scheduling,
HR Assistant) orchestrate the recruitment workflow, and all data is persisted
to **SQLite** so nothing is lost on app restart.

## Quick Start

All Python package management is done with [`uv`](https://docs.astral.sh/uv/) —
do **not** use `pip`/`pip3`/`conda`.

```bash
# 1. Create the virtual environment and install all deps (incl. dev tools)
uv sync

# 2. Create your local model configuration
Copy-Item .env.example .env

# 3. Start Ollama (local Llama 3.2 backend) if you don't already have it running
ollama serve

# 4. Launch the app
uv run streamlit run app.py
```

The SQLite schema (`db/smarthire.db`) is created automatically on first run and
is idempotent (`CREATE TABLE IF NOT EXISTS`), so subsequent runs are no-ops.

### Model provider configuration

The application reads `.env` at startup. The default is fully local:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

For Gemini deployment, set `LLM_PROVIDER=gemini`, `GEMINI_MODEL` (for example
`gemini-2.5-flash`), and `GEMINI_API_KEY`. `.env` is ignored by Git, so the
key remains local or can be injected by your deployment platform.

## Project Layout

```
smarthire-ai/
├── app.py                    # Streamlit UI (recruiter dashboard + candidate chat)
├── graph.py                  # LangGraph StateGraph orchestration + checkpointer wiring
├── supervisor.py             # Intent detection / agent routing
├── agents/                   # The four specialist agents + reflection node
├── tools/                    # resume_parser, jd_analyzer, calendar_tool, ...
├── memory/
│   ├── conversation_memory.py# Conversation memory, backed by SqliteSaver
│   └── state.py              # Shared LangGraph state schema (TypedDict)
├── db/
│   ├── database.py           # sqlite3 connection manager + persistence helpers
│   ├── schema.sql            # DDL for all recruitment tables
│   └── smarthire.db          # generated at runtime (gitignored)
├── utils/                    # Pydantic models, LLM factory
├── prompts/                  # HR knowledge base
└── tests/                    # pytest suite
```

## Adding a Dependency

Use `uv add` (this project is a `uv`-managed project via `pyproject.toml`):

```bash
# Example: add the LangGraph SQLite checkpointer package
uv add langgraph-checkpoint-sqlite
```

This updates `pyproject.toml` and `uv.lock`. Then either run the app
(`uv run ...`) or sync explicitly:

```bash
uv sync
```

> Note: `requirements.txt` is kept in sync for compatibility, but
> `pyproject.toml` + `uv.lock` are the source of truth.

## SQLite Persistence Layer

All recruitment data is persisted to `db/smarthire.db` using Python's built-in
`sqlite3` module, wrapped by the connection-manager class in
`db/database.py`:

- **WAL journal mode** (`PRAGMA journal_mode=WAL;`) so concurrent Streamlit
  reruns can read/write safely.
- **Idempotent schema init** — `Database().init_db()` runs `db/schema.sql`
  (`CREATE TABLE IF NOT EXISTS`).
- **Defensive error handling** — DB write failures are logged and swallowed so
  a storage hiccup never crashes the app.

### Tables

| Table               | Written by                                          | Purpose                              |
|---------------------|-----------------------------------------------------|--------------------------------------|
| `job_descriptions`  | Upload flow / agents                                | Raw JDs (upload / paste / sample)    |
| `candidates`        | Resume Screening Agent                              | One row per screened resume          |
| `screenings`        | Resume Screening Agent                              | Strengths / gaps / summary           |
| `candidate_rankings`| Candidate Matching Agent (+ Reflection)             | Match scores, rank, reflection status|
| `interviews`        | Interview Scheduling Agent                          | Proposed/confirmed/cancelled slots   |
| `hr_answers`        | HR Assistant Agent                                  | Q&A history                          |
| `sessions`          | App / `ConversationMemory`                          | Maps Streamlit session ↔ thread_id   |

Every agent writes its output to the relevant table after producing a result
(rather than only returning it in-memory). Writes are wrapped around the
existing return values, so agent signatures are unchanged.

## LangGraph Memory (SQLite Checkpointer)

Conversation state is persisted per session using LangGraph's built-in SQLite
checkpointer — **`langgraph.checkpoint.sqlite.SqliteSaver`** — pointed at
`db/smarthire.db` and keyed by `thread_id == session id`.

The checkpointer is created in `memory/conversation_memory.py::get_checkpointer()`
and wired into the graph in `graph.py::build_graph()`:

```python
from memory.conversation_memory import get_checkpointer, get_thread_config

checkpointer = get_checkpointer()  # SqliteSaver over db/smarthire.db
compiled = graph.compile(checkpointer=checkpointer)

config = get_thread_config(session_id)      # {"configurable": {"thread_id": session_id}}
result = compiled.invoke(input_state, config=config)
```

Because the graph is invoked with a stable `thread_id`, LangGraph restores the
previous checkpoint automatically and the conversation survives both Streamlit
reruns **and** app restarts. `run_smarthire()` in `graph.py`:

1. Resolves a session id (creating one, or resuming the most recently active
   session from the `sessions` table).
2. Loads the prior checkpoint; on resume it sends only the new user message as
   input (re-sending the restored history would duplicate it via the append
   reducer).
3. Invokes the compiled graph with the thread config.

`memory/conversation_memory.py` also mirrors each session's cached data (history,
JDs, shortlisted candidates, interview prefs) into a dedicated
`conversation_memory` checkpoint namespace, so the app's System Insight view
shows the persisted history after a restart too.

## Testing

```bash
uv run pytest            # full suite
uv run ruff check .      # lint
```

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full multi-agent
design, routing logic, and the reflection validation checklist.
