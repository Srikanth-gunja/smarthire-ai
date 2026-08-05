-- SmartHire AI — SQLite persistence schema.
--
-- Created idempotently by db/database.py::init_db(). LangGraph conversation
-- state is persisted separately by the SqliteSaver checkpointer in the same
-- db/smarthire.db file (checkpoints / writes tables) — see
-- memory/conversation_memory.py::get_checkpointer().

-- Job descriptions
CREATE TABLE IF NOT EXISTS job_descriptions (
    id TEXT PRIMARY KEY,           -- uuid
    title TEXT,
    raw_text TEXT NOT NULL,
    source TEXT,                   -- 'upload' | 'paste' | 'sample'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Candidates (one row per uploaded resume)
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    name TEXT,
    email TEXT,
    resume_raw_text TEXT NOT NULL,
    resume_filename TEXT,
    extracted_skills TEXT,          -- JSON array as text
    extracted_experience_years REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Screening results (Resume Screening Agent output)
CREATE TABLE IF NOT EXISTS screenings (
    id TEXT PRIMARY KEY,
    candidate_id TEXT REFERENCES candidates(id),
    jd_id TEXT REFERENCES job_descriptions(id),
    summary TEXT,
    strengths TEXT,                 -- JSON array
    gaps TEXT,                      -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Matching / ranking results (Candidate Matching Agent output)
CREATE TABLE IF NOT EXISTS candidate_rankings (
    id TEXT PRIMARY KEY,
    candidate_id TEXT REFERENCES candidates(id),
    jd_id TEXT REFERENCES job_descriptions(id),
    match_score REAL NOT NULL,      -- 0-100
    rank_position INTEGER,
    reasoning TEXT,
    reflection_validated BOOLEAN DEFAULT 0,
    reflection_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Interview slots (Interview Scheduling Agent output)
CREATE TABLE IF NOT EXISTS interviews (
    id TEXT PRIMARY KEY,
    candidate_id TEXT REFERENCES candidates(id),
    jd_id TEXT REFERENCES job_descriptions(id),
    session_id TEXT,
    proposed_start TIMESTAMP,
    proposed_end TIMESTAMP,
    interview_type TEXT DEFAULT 'technical',
    interviewer TEXT,
    status TEXT DEFAULT 'proposed',  -- proposed | confirmed | cancelled
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_interviews_session ON interviews(session_id);

-- Sessions (maps a Streamlit session to a LangGraph thread_id)
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,             -- = LangGraph thread_id
    mode TEXT,                       -- 'recruiter' | 'candidate'
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    paused_at_node TEXT,             -- node name where a transient error occurred
    error_message TEXT               -- human-readable error description
);

-- HR Assistant answers (HRAssistantAgent output)
CREATE TABLE IF NOT EXISTS hr_answers (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    query TEXT,
    answer TEXT NOT NULL,
    sources TEXT,                   -- JSON array as text
    confidence REAL,
    needs_escalation BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Human-readable conversation transcript / audit log.
-- Separate from LangGraph's checkpointer tables (checkpoints / writes):
-- SqliteSaver holds agent state for resuming graph execution, while this
-- table records every user / assistant / agent turn for display and audit.
CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,            -- uuid
    session_id TEXT REFERENCES sessions(id),
    role TEXT NOT NULL,             -- 'user' | 'assistant' | 'agent:<agent_name>'
    content TEXT NOT NULL,
    agent_name TEXT,                -- which agent produced this, null for user turns
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Useful lookups for agents and the Streamlit dashboard.
CREATE INDEX IF NOT EXISTS idx_candidates_name ON candidates(name);
CREATE INDEX IF NOT EXISTS idx_candidate_rankings_jd ON candidate_rankings(jd_id);
CREATE INDEX IF NOT EXISTS idx_screenings_jd ON screenings(jd_id);
CREATE INDEX IF NOT EXISTS idx_interviews_jd ON interviews(jd_id);
CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, created_at);
