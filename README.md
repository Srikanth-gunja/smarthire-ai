# SmartHire AI

**Multi-agent recruitment automation system** powered by LangGraph, LangChain, Streamlit, and SQLite.

SmartHire AI orchestrates a team of specialized AI agents to automate the end-to-end hiring workflow — from resume screening and candidate ranking to interview scheduling and HR Q&A. All conversation state and recruitment data persist to SQLite, so nothing is lost between sessions or app restarts.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Layout](#project-layout)
- [How It Works](#how-it-works)
- [Agents](#agents)
- [Tools](#tools)
- [Database Schema](#database-schema)
- [LangGraph Memory](#langgraph-memory)
- [LLM Provider Configuration](#llm-provider-configuration)
- [Frontend](#frontend)
- [Testing](#testing)
- [Development](#development)

---

## Features

- **Multi-agent pipeline** — Supervisor routes user intent to the right specialist agent(s) in the correct order
- **Resume parsing** — Extracts structured data (skills, experience, education, certifications) from PDF, DOCX, and TXT resumes
- **JD analysis** — Parses job descriptions into required/preferred skills, experience, and education requirements
- **Candidate ranking** — Computes composite match scores with deterministic skill/experience overlap scoring, justified per candidate
- **Interview scheduling** — Proposes non-conflicting interview slots with calendar conflict detection and booking
- **HR Assistant** — Answers recruitment FAQs grounded in an approved knowledge base, with escalation for sensitive topics
- **Reflection Node** — Validates all agent outputs against a 4-point checklist before returning the final response
- **Session persistence** — Conversation history, agent state, and recruitment data survive browser refreshes and app restarts via SQLite
- **Dual UI modes** — Recruiter Dashboard (screen, rank, schedule, chat, insight) and Candidate Chat (HR questions)
- **LLM provider flexibility** — Runs fully local with Ollama + Llama 3.2 or remotely with Google Gemini
- **Persistent audit trail** — Every user/assistant/agent turn is logged to `chat_messages` for compliance and debugging
- **Skill normalization** — Handles LLM inconsistencies (category sentences, aliases, abbreviations) to produce accurate skill matching

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Agent orchestration | **LangGraph** StateGraph | Multi-agent routing, conditional edges, state management |
| LLM framework | **LangChain** | Prompt templates, structured output, message types |
| LLM backend | **Ollama** (local) / **Google Gemini** (remote) | Inference for all agents |
| State schema | **TypedDict** + Annotated reducers | Lightweight graph state with append semantics |
| I/O contracts | **Pydantic v2** BaseModel | Input/output validation for all agents |
| Frontend | **Streamlit** | Recruiter dashboard + candidate chat |
| Data handling | **Pandas** | Resume/JD tabular processing |
| Storage | **SQLite** | Recruitment data + LangGraph checkpointer |
| PDF extraction | **pypdf** | Resume PDF text extraction |
| DOCX extraction | **python-docx** | Resume DOCX text extraction |
| Visualization | **matplotlib** | Session insight charts |
| Package management | **uv** | Dependency resolution and virtual environments |
| Linting | **Ruff** | Code quality enforcement |
| Testing | **pytest** + **pytest-mock** | Unit and integration tests |
| Secrets | **python-dotenv** | `.env` file loading |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌───────────┐  │
│  │ Upload &     │ │ Interview    │ │ Chat     │ │ System    │  │
│  │ Screen       │ │ Scheduling   │ │          │ │ Insight   │  │
│  └──────┬───────┘ └──────┬───────┘ └────┬─────┘ └───────────┘  │
└─────────┼────────────────┼──────────────┼──────────────────────┘
          │                │              │
          ▼                ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      graph.py                                   │
│              LangGraph StateGraph Orchestration                 │
│                                                                 │
│  START → Supervisor ──┬──→ Resume Screening ──┐                 │
│                       ├──→ Candidate Matching ─┤                 │
│                       ├──→ Interview Scheduling─┤                 │
│                       └──→ HR Assistant ───────┘                 │
│                              │                                   │
│                              ▼                                   │
│                      Memory Update                               │
│                              │                                   │
│                              ▼                                   │
│                      Reflection Node ──→ END                     │
│                         (retry loop)                             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   db/smarthire.db                               │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ job_desc.  │ │candidates│ │rankings  │ │interviews        │ │
│  │ screenings │ │sessions  │ │hr_answers│ │chat_messages     │ │
│  └────────────┘ └──────────┘ └──────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Routing Logic

The Supervisor classifies user intent and routes to specialist agents:

| Intent | Trigger Examples | Agent(s) Invoked |
|--------|-----------------|-------------------|
| Resume screening | "review this resume", file upload, "parse this CV" | Resume Screening → Candidate Matching (if JD present) |
| Candidate ranking | "rank candidates", "who's the best fit", "compare applicants" | Candidate Matching |
| Interview scheduling | "schedule interview", "find available slots", "book a time" | Interview Scheduling |
| HR question | "what's your policy on", "how does the process work" | HR Assistant |
| Multi-intent | "screen resumes AND schedule interviews" | Multiple agents in sequence |
| Greeting/filler | "hello", "thanks", "okay" | HR Assistant (conversational) |

### Reflection Node — Validation Checklist

Before returning the final response, the Reflection Node validates:

1. **Candidate recommendations match JD** — Cross-checks ranked candidates' skills against JD requirements. Flags fabricated skills (skills claimed in `skills_match` that don't appear in the extracted resume data). Triggers a correction retry from Candidate Matching if issues are found.

2. **Interview schedules conflict-free** — Detects overlapping slots for the same interviewer. Drops conflicting slots and triggers a correction retry from Interview Scheduling.

3. **All questions answered** — Compares the original user query against agent outputs in state. Flags unanswered questions rather than silently dropping them.

4. **Clarity and consistency** — Combines all agent outputs into a single, polished final_response via an LLM polish pass (or deterministic fallback).

When validation fails on the first pass, the graph loops back to the responsible agent with `reflection_feedback` for a single correction attempt. The second pass returns the result to the user with any remaining issues surfaced rather than hidden.

---

## Quick Start

Two ways to set up: **uv** (recommended) or **venv + pip**.

### Option A — uv (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package manager that handles virtual environments and dependency resolution automatically.

```bash
# 1. Clone the repository
git clone <repository-url>
cd smarthire-ai

# 2. Create the virtual environment and install all deps (incl. dev tools)
uv sync

# 3. Create your local model configuration
Copy-Item .env.example .env

# 4. Start Ollama (local Llama 3.2 backend) if you don't already have it running
ollama serve

# 5. Launch the app
uv run streamlit run app.py
```

### Option B — venv + pip

Standard Python tooling — works everywhere Python 3.11+ is installed.

```bash
# 1. Clone the repository
git clone <repository-url>
cd smarthire-ai

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Install dev dependencies (optional, for testing/linting)
pip install pytest pytest-mock ruff

# 6. Create your local model configuration
# Windows (PowerShell):
Copy-Item .env.example .env
# macOS / Linux:
cp .env.example .env

# 7. Start Ollama (local Llama 3.2 backend) if you don't already have it running
ollama serve

# 8. Launch the app
streamlit run app.py
```

### Using Gemini Instead

Edit `.env` and set:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### App Output

The app opens at `http://localhost:8501`. The SQLite schema (`db/smarthire.db`) is created automatically on first run and is idempotent (`CREATE TABLE IF NOT EXISTS`). No manual database setup is required.

---

## Project Layout

```
smarthire-ai/
├── app.py                          # Streamlit frontend (recruiter dashboard + candidate chat)
├── graph.py                        # LangGraph StateGraph orchestration + checkpointer wiring
├── supervisor.py                   # Intent detection / agent routing
│
├── agents/
│   ├── resume_screening_agent.py   # Parses resumes, extracts structured data, initial match scoring
│   ├── candidate_matching_agent.py # Ranks candidates against JD with justified scores
│   ├── interview_scheduler_agent.py# Proposes/manages interview slots, conflict detection
│   ├── hr_assistant_agent.py       # Answers HR FAQs, escalation for sensitive topics
│   └── reflection_node.py          # 4-point validation checklist, correction retry loop
│
├── tools/
│   ├── resume_parser.py            # LLM-based structured resume extraction (PDF/DOCX/TXT)
│   ├── jd_analyzer.py              # LLM-based JD requirement extraction
│   ├── calendar_tool.py            # Interview scheduling, conflict detection, slot booking
│   ├── skill_normalizer.py         # Skill name normalization, alias resolution, text extraction
│   ├── candidate_database.py       # SQLite-based candidate read/query layer
│   └── email_notification.py       # Log-based email stub (interview invites, status updates)
│
├── memory/
│   ├── state.py                    # SmartHireState TypedDict — shared graph state schema
│   ├── conversation_memory.py      # ConversationMemory — session lifecycle + SqliteSaver checkpointer
│   └── chat_audit.py               # ChatAudit — human-readable conversation transcript + session mirror
│
├── db/
│   ├── database.py                 # SQLite connection manager + typed persistence helpers
│   ├── schema.sql                  # DDL for all 8 recruitment tables + indexes
│   └── smarthire.db                # Generated at runtime (gitignored)
│
├── utils/
│   ├── models.py                   # Pydantic v2 input/output models for all agents
│   └── llm_factory.py              # LLM provider factory (Ollama / Gemini)
│
├── prompts/
│   └── hr_knowledge_base.md        # Approved HR policy knowledge base
│
├── data/
│   └── interview_slots.json        # CalendarTool JSON-backed slot store
│
├── tests/                          # pytest suite (14 test files)
├── docs/
│   └── architecture.md             # Full multi-agent design docs
│
├── .streamlit/config.toml          # Streamlit theme (enterprise teal) + server settings
├── .env.example                    # LLM provider template
├── .gitignore                      # SQLite DB, .env, __pycache__, .venv
├── pyproject.toml                  # Project metadata + dependencies (source of truth)
├── uv.lock                         # Locked dependency versions
└── requirements.txt                # pip-compatible copy (kept in sync)
```

---

## How It Works

### Recruiter Workflow

1. **Upload resumes** — Drop PDF, DOCX, or TXT files. The Resume Screening Agent extracts structured data (name, skills, experience, education, certifications) and computes an initial match score against the JD.

2. **Provide a JD** — Upload a file, paste text, or use the bundled sample JD (Senior Full-Stack Developer). The JD Analyzer extracts required skills, preferred skills, experience requirements, and education.

3. **Screen & Rank** — Click "Screen & Rank Candidates" to run the multi-agent pipeline. The Supervisor routes through Resume Screening → Candidate Matching → Memory Update → Reflection Node, with live progress shown per agent.

4. **View results** — See ranked candidates with match score meters, skill strengths/gaps, experience match, and justification. The Reflection Node's validation status is shown in an expandable section.

5. **Schedule interviews** — Click "Schedule Interview" on any ranked candidate to jump to the Interview Scheduling tab. Pick a date, select an interviewer, and confirm available slots. Conflicting slots are grayed out.

6. **Chat** — Ask follow-up questions about candidates, policies, or the process. Messages are labelled with the agent that produced them.

7. **System Insight** — View session metrics (resumes processed, avg match score, interviews proposed), agent routing log, candidate distribution chart, and raw conversation history.

### Candidate Workflow

Candidates can ask HR questions about the recruitment process — application status, interview prep, policies, timelines. The HR Assistant grounds answers in the approved knowledge base and escalates to human HR for sensitive topics (legal, accommodation, discrimination).

---

## Agents

### Supervisor (`supervisor.py`)

- **Responsibility:** Intent detection and routing
- **LLM structured output:** `ExecutionPlan` (intent, agents_to_invoke, reasoning)
- **Fallback:** Keyword-based classification when LLM is unavailable
- **Never processes data** — only routes to specialist agents

### Resume Screening Agent (`agents/resume_screening_agent.py`)

- **Responsibility:** Parse a resume and extract structured data
- **Output:** `ResumeScreeningOutput` — candidate name, skills, experience years, education, summary, match score
- **Grounding:** Skills are extracted by both LLM and text-level vocabulary scan (`skill_normalizer.py`), then deduplicated
- **Persistence:** Writes to `candidates` and `screenings` tables

### Candidate Matching Agent (`agents/candidate_matching_agent.py`)

- **Responsibility:** Rank screened candidates against the JD
- **Output:** `CandidateMatchingOutput` — ranked list with scores, skill matches/gaps, experience match, justification
- **Scoring:** Deterministic composite score (skills 70%, experience 30%), reconciled against LLM output
- **Reflection-aware:** Accepts `reflection_feedback` for correction retries
- **Persistence:** Writes to `candidate_rankings` table

### Interview Scheduling Agent (`agents/interview_scheduler_agent.py`)

- **Responsibility:** Propose non-conflicting interview slots
- **Output:** `InterviewSchedulingOutput` — proposed slots with status (proposed/confirmed/conflict), conflicts list, summary
- **Conflict detection:** Reuses `CalendarTool._has_conflict()` for overlap detection
- **Persistence:** Writes to `interviews` table

### HR Assistant Agent (`agents/hr_assistant_agent.py`)

- **Responsibility:** Answer recruitment FAQs grounded in the approved knowledge base
- **Output:** `HRAssistantOutput` — answer, sources, confidence (0-1), needs_escalation flag
- **Escalation:** Automatically escalates legal, accommodation, discrimination, and contract questions
- **Persistence:** Writes to `hr_answers` table

### Reflection Node (`agents/reflection_node.py`)

- **Responsibility:** Validate all agent outputs before the final response
- **Checks:** (a) candidate skills vs JD, (b) schedule conflicts, (c) all questions answered, (d) clarity/consistency
- **Retry:** One correction retry per turn — loops back to the responsible agent with feedback
- **Persistence:** Stamps reflection results onto `candidate_rankings` rows

---

## Tools

### Resume Parser (`tools/resume_parser.py`)

LLM-based structured extraction from raw resume text. Supports PDF (via pypdf), DOCX (via python-docx), and TXT files. Supplements LLM output with text-level skill extraction to prevent skill deduplication issues with small local models.

### JD Analyzer (`tools/jd_analyzer.py`)

Extracts structured requirements from job description text: job title, required/preferred skills (atomic names, not category sentences), experience range, and education requirements.

### Calendar Tool (`tools/calendar_tool.py`)

JSON-backed interview calendar at `data/interview_slots.json`. Supports availability checking, conflict detection, slot booking, and available slot proposal. Used by both the Interview Scheduling Agent and the Streamlit scheduling UI.

### Skill Normalizer (`tools/skill_normalizer.py`)

Handles LLM skill extraction inconsistencies:
- Expands grouped entries (e.g., "Languages: Python, SQL, Java" → individual skills)
- Resolves aliases (e.g., "sklearn" → "scikit learn")
- Matches skills with light containment (e.g., "javascript" matches "javascript/react")
- Extracts skills from raw text via vocabulary scan as a safety net

### Candidate Database (`tools/candidate_database.py`)

SQLite-based read/query layer for candidate records. Supports search by name, skills, and experience. Used by the HR Assistant for candidate lookups.

### Email Notification (`tools/email_notification.py`)

Log-based stub for interview invitations and status updates. Logs email actions that would be sent in production. Supports real SMTP integration in future phases.

---

## Database Schema

`db/schema.sql` defines 8 tables with foreign key relationships:

```sql
-- Job descriptions (upload/paste/sample)
CREATE TABLE job_descriptions (
    id TEXT PRIMARY KEY,
    title TEXT,
    raw_text TEXT NOT NULL,
    source TEXT,                    -- 'upload' | 'paste' | 'sample'
    created_at TIMESTAMP
);

-- Candidates (one row per uploaded resume)
CREATE TABLE candidates (
    id TEXT PRIMARY KEY,
    name TEXT,
    email TEXT,
    resume_raw_text TEXT NOT NULL,
    resume_filename TEXT,
    extracted_skills TEXT,           -- JSON array
    extracted_experience_years REAL,
    created_at TIMESTAMP
);

-- Screening results (Resume Screening Agent output)
CREATE TABLE screenings (
    id TEXT PRIMARY KEY,
    candidate_id TEXT REFERENCES candidates(id),
    jd_id TEXT REFERENCES job_descriptions(id),
    summary TEXT,
    strengths TEXT,                  -- JSON array
    gaps TEXT,                       -- JSON array
    created_at TIMESTAMP
);

-- Ranking results (Candidate Matching Agent output)
CREATE TABLE candidate_rankings (
    id TEXT PRIMARY KEY,
    candidate_id TEXT REFERENCES candidates(id),
    jd_id TEXT REFERENCES job_descriptions(id),
    match_score REAL NOT NULL,
    rank_position INTEGER,
    reasoning TEXT,
    reflection_validated BOOLEAN DEFAULT 0,
    reflection_notes TEXT,
    created_at TIMESTAMP
);

-- Interview slots (Interview Scheduling Agent output)
CREATE TABLE interviews (
    id TEXT PRIMARY KEY,
    candidate_id TEXT REFERENCES candidates(id),
    jd_id TEXT REFERENCES job_descriptions(id),
    proposed_start TIMESTAMP,
    proposed_end TIMESTAMP,
    status TEXT DEFAULT 'proposed',  -- proposed | confirmed | cancelled
    created_at TIMESTAMP
);

-- Session management
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,             -- = LangGraph thread_id
    mode TEXT,                       -- 'recruiter' | 'candidate'
    started_at TIMESTAMP,
    last_active_at TIMESTAMP
);

-- HR Assistant answers
CREATE TABLE hr_answers (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    query TEXT,
    answer TEXT NOT NULL,
    sources TEXT,                    -- JSON array
    confidence REAL,
    needs_escalation BOOLEAN DEFAULT 0,
    created_at TIMESTAMP
);

-- Conversation transcript / audit log
CREATE TABLE chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    role TEXT NOT NULL,              -- 'user' | 'assistant' | 'agent:<name>'
    content TEXT NOT NULL,
    agent_name TEXT,
    created_at TIMESTAMP
);
```

**Key design decisions:**
- WAL journal mode for concurrent Streamlit reruns
- Idempotent schema init (`CREATE TABLE IF NOT EXISTS`)
- Defensive error handling — DB write failures are logged and swallowed
- Every agent persists output to SQLite after producing a result
- Reflection results are stamped onto `candidate_rankings` rows

---

## LangGraph Memory

### Conversation State

The shared graph state (`SmartHireState` in `memory/state.py`) is a TypedDict with Annotated reducers:

| Field | Type | Reducer | Purpose |
|-------|------|---------|---------|
| `conversation_history` | `list[BaseMessage]` | `operator.add` (append) | Accumulates messages across turns |
| `current_intent` | `str` | overwrite | Supervisor's classified intent |
| `active_agents` | `list[str]` | overwrite | Agent queue for this turn |
| `resumes` | `list[dict]` | overwrite | Screened resume data |
| `resume_inputs` | `list[dict]` | overwrite | Raw resume documents |
| `candidate_rankings` | `list[dict]` | overwrite | Ranked candidates |
| `job_description` | `dict` | overwrite | Parsed JD data |
| `interview_slots` | `list[dict]` | overwrite | Proposed interview slots |
| `hr_answers` | `list[dict]` | overwrite | HR Assistant responses |
| `reflection_notes` | `dict` | overwrite | Validation results and issues |
| `final_response` | `str` | overwrite | Polished response for the user |

### Checkpointer Integration

```python
from memory.conversation_memory import get_checkpointer, get_thread_config

checkpointer = get_checkpointer()  # SqliteSaver over db/smarthire.db
compiled = graph.compile(checkpointer=checkpointer)

config = get_thread_config(session_id)  # {"configurable": {"thread_id": session_id}}
result = compiled.invoke(input_state, config=config)
```

LangGraph restores the previous checkpoint automatically using the stable `thread_id`. On resume, only the new user message is passed as input (the checkpointer restores prior history, avoiding duplication via the append reducer).

### Session Lifecycle

1. **Create** — `ConversationMemory().create_session(mode)` generates a UUID-based session id
2. **Resume** — `resume_last_session()` returns the most recently active session from the `sessions` table
3. **Reset** — `reset_session()` clears working data while retaining the session identity
4. **Clear** — `clear_session()` erases all data (memory + checkpointer + session row)

The `ChatAudit` class maintains a small mirror file (`data/active_session.local`) so browser refreshes can resume the same conversation.

---

## LLM Provider Configuration

The application reads `.env` at startup via `utils/llm_factory.py`:

### Local (Ollama) — Default

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

### Remote (Google Gemini)

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

`.env` is gitignored. For deployment, inject environment variables via your platform.

---

## Frontend

Built with Streamlit using an enterprise teal theme (`.streamlit/config.toml`).

### Recruiter Dashboard (4 tabs)

| Tab | Description |
|-----|-------------|
| **Upload & Screen** | Upload resumes + JD, run multi-agent pipeline, view ranked results with score meters |
| **Interview Scheduling** | Pick candidates, select interviewers, propose/confirm slots with conflict detection |
| **Chat** | Chat with SmartHire AI — messages labelled by agent |
| **System Insight** | Session metrics, agent routing log, candidate distribution chart, raw history |

### Candidate Chat

A separate mode for candidates to ask HR questions about the recruitment process.

### UI Features

- **Agent-aware chat** — Each turn is badge-labelled with the agent that produced it
- **Live progress** — Per-agent stage checklist shows pending/running/done during pipeline execution
- **Score meters** — Color-coded match score bars (teal ≥80%, amber ≥60%, red <60%)
- **Reflection summary** — Expandable section showing validation checks, corrections, and retry status
- **Session persistence** — Past sessions listed in sidebar, switchable with full state restore

---

## Testing

**With uv:**

```bash
uv run pytest              # full suite
uv run pytest -v           # verbose output
uv run pytest tests/test_skill_normalizer.py   # specific file
uv run ruff check .        # lint
```

**With pip (virtual env activated):**

```bash
pytest                     # full suite
pytest -v                  # verbose output
pytest tests/test_skill_normalizer.py   # specific file
ruff check .               # lint
```

### Test Files

| File | Covers |
|------|--------|
| `test_resume_parser.py` | Resume text extraction, structured field extraction |
| `test_resume_screening_agent.py` | Screening pipeline, match scoring |
| `test_jd_analyzer.py` | JD requirement extraction |
| `test_candidate_matching_agent.py` | Candidate ranking, composite scoring |
| `test_interview_scheduler_agent.py` | Slot proposal, conflict detection |
| `test_calendar_tool.py` | Availability checks, booking |
| `test_candidate_database.py` | SQLite candidate queries |
| `test_email_notification.py` | Email stub logging |
| `test_chat_audit.py` | Conversation transcript persistence |
| `test_reflection_node.py` | Validation checklist, retry logic |
| `test_skill_normalizer.py` | Skill normalization, alias resolution |
| `test_memory.py` | Session lifecycle, ConversationMemory |
| `test_graph.py` | LangGraph orchestration, routing |
| `test_hr_assistant_agent.py` | HR Q&A, escalation |

---

## Development

### Adding a Dependency

**With uv:**

```bash
uv add <package-name>
```

This updates `pyproject.toml` and `uv.lock`. Then sync:

```bash
uv sync
```

**With pip:**

```bash
# Activate your virtual environment first, then:
pip install <package-name>
pip freeze > requirements.txt
```

> `requirements.txt` is kept in sync for compatibility, but `pyproject.toml` + `uv.lock` are the source of truth.

### Running Individual Agents

Each agent has a `__main__` block for standalone testing:

**With uv:**

```bash
uv run python agents/resume_screening_agent.py data/sample_resumes/sarah_chen.txt data/sample_jd.txt
uv run python agents/candidate_matching_agent.py data/sample_jd.txt
uv run python tools/resume_parser.py data/sample_resumes/sarah_chen.txt
uv run python tools/jd_analyzer.py data/sample_jd.txt
```

**With pip (virtual env activated):**

```bash
python agents/resume_screening_agent.py data/sample_resumes/sarah_chen.txt data/sample_jd.txt
python agents/candidate_matching_agent.py data/sample_jd.txt
python tools/resume_parser.py data/sample_resumes/sarah_chen.txt
python tools/jd_analyzer.py data/sample_jd.txt
```

### Graph Smoke Test

**With uv:** `uv run python graph.py`

**With pip:** `python graph.py`

Runs a full pipeline with a sample HR question against the real LLM.

### Code Style

- Linting: **Ruff**
  - With uv: `uv run ruff check .`
  - With pip: `ruff check .`
- Type hints: Python 3.11+ syntax (`str | None`, `list[str]`)
- Models: Pydantic v2 with explicit field descriptions
- No comments unless requested

