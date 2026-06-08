import aiosqlite
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

DB_PATH = Path(__file__).parent.parent / "data" / "analysis.db"

SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT 'gpt-4o-mini',
    temperature REAL NOT NULL DEFAULT 0.0,
    n_runs INTEGER NOT NULL DEFAULT 10,
    max_parallel_llm INTEGER NOT NULL DEFAULT 20,
    openai_api_key TEXT,
    github_token TEXT,
    branch TEXT NOT NULL DEFAULT 'main',
    max_commits INTEGER DEFAULT NULL,
    status TEXT NOT NULL DEFAULT 'idle',
    phase1_status TEXT NOT NULL DEFAULT 'idle',
    phase2_status TEXT NOT NULL DEFAULT 'idle',
    phase1_total INTEGER NOT NULL DEFAULT 0,
    phase1_done INTEGER NOT NULL DEFAULT 0,
    phase2_total INTEGER NOT NULL DEFAULT 0,
    phase2_done INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repositories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    owner TEXT NOT NULL,
    name TEXT NOT NULL,
    local_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    current_commit_index INTEGER NOT NULL DEFAULT 0,
    total_commits INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,
    name TEXT NOT NULL,
    template TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS smell_commits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    repo_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    commit_hash TEXT NOT NULL,
    prev_commit_hash TEXT,
    file_path TEXT NOT NULL,
    function_name TEXT,
    smell_type TEXT NOT NULL,
    smell_line INTEGER,
    smell_message TEXT,
    diff_content TEXT,
    commit_message TEXT,
    issue_summary TEXT,
    pr_summary TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    smell_commit_id INTEGER NOT NULL REFERENCES smell_commits(id) ON DELETE CASCADE,
    prompt_pattern_id INTEGER NOT NULL REFERENCES prompt_patterns(id) ON DELETE CASCADE,
    run_number INTEGER NOT NULL,
    primary_activity TEXT,
    sub_activity TEXT,
    reasoning TEXT,
    raw_response TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(smell_commit_id, prompt_pattern_id, run_number)
);

CREATE TABLE IF NOT EXISTS vote_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    smell_commit_id INTEGER NOT NULL REFERENCES smell_commits(id) ON DELETE CASCADE,
    prompt_pattern_id INTEGER NOT NULL REFERENCES prompt_patterns(id) ON DELETE CASCADE,
    primary_activity TEXT,
    vote_count INTEGER DEFAULT 0,
    total_votes INTEGER DEFAULT 0,
    tied INTEGER NOT NULL DEFAULT 0,
    tied_activities TEXT,
    all_votes TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(smell_commit_id, prompt_pattern_id)
);

CREATE TABLE IF NOT EXISTS sub_activity_mapping (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    raw_sub_activity TEXT NOT NULL,
    canonical TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, raw_sub_activity)
);

CREATE INDEX IF NOT EXISTS idx_sam_session ON sub_activity_mapping(session_id);
CREATE INDEX IF NOT EXISTS idx_sc_session ON smell_commits(session_id, status);
CREATE INDEX IF NOT EXISTS idx_sc_repo ON smell_commits(repo_id);
CREATE INDEX IF NOT EXISTS idx_lr_sc ON llm_results(smell_commit_id, prompt_pattern_id);
CREATE INDEX IF NOT EXISTS idx_vr_sc ON vote_results(smell_commit_id);
CREATE INDEX IF NOT EXISTS idx_repo_session ON repositories(session_id);
"""


def now_iso():
    return datetime.now(timezone.utc).isoformat()


async def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        await db.commit()
        # Migrations — ignore if column already exists
        for migration in [
            "ALTER TABLE sessions ADD COLUMN phase3_status TEXT NOT NULL DEFAULT 'idle'",
            "ALTER TABLE smell_commits ADD COLUMN commit_date TEXT",
        ]:
            try:
                await db.execute(migration)
                await db.commit()
            except Exception:
                pass


@asynccontextmanager
async def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("PRAGMA journal_mode = WAL")
        yield db
