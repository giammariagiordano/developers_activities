import asyncio
import csv
import io
import json
import math
import statistics
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import get_db, init_db, now_iso
from .llm_client import PRESET_TEMPLATES
from .runner import (
    is_running, pause_session, reset_session_on_startup,
    start_phase1, start_phase2, start_phase3, subscribe, unsubscribe,
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await reset_session_on_startup()
    yield


app = FastAPI(title="ML Smell Activity Analyzer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────


class SessionCreate(BaseModel):
    name: str
    classifier_models: list[str] = ["llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"]
    aggregator_model: str = "llama-3.3-70b-versatile"
    ollama_base_url: str = "https://api.groq.com/openai"
    llm_api_key: Optional[str] = None
    temperature: float = 0.0
    max_parallel_llm: int = 3
    github_token: Optional[str] = None
    branch: str = "main"
    max_commits: Optional[int] = None


class SessionUpdate(BaseModel):
    name: Optional[str] = None
    classifier_models: Optional[list[str]] = None
    aggregator_model: Optional[str] = None
    ollama_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    temperature: Optional[float] = None
    max_parallel_llm: Optional[int] = None
    github_token: Optional[str] = None
    branch: Optional[str] = None
    max_commits: Optional[int] = None


class RepoAdd(BaseModel):
    owner: str
    name: str


class PatternData(BaseModel):
    position: int = 0
    name: str
    template: str
    enabled: bool = True


# ─── Sessions ─────────────────────────────────────────────────────────────────


@app.get("/api/sessions")
async def list_sessions():
    async with get_db() as db:
        rows = await (await db.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC"
        )).fetchall()
        return [dict(r) for r in rows]


@app.post("/api/sessions", status_code=201)
async def create_session(body: SessionCreate):
    async with get_db() as db:
        import json as _json
        cur = await db.execute(
            """INSERT INTO sessions
               (name, classifier_models, aggregator_model, ollama_base_url,
                llm_api_key, temperature, max_parallel_llm, github_token, branch, max_commits,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (body.name,
             _json.dumps(body.classifier_models),
             body.aggregator_model,
             body.ollama_base_url,
             body.llm_api_key,
             body.temperature, body.max_parallel_llm,
             body.github_token, body.branch, body.max_commits,
             now_iso(), now_iso()),
        )
        session_id = cur.lastrowid

        # Create default zero-shot pattern
        await db.execute(
            "INSERT INTO prompt_patterns (session_id, position, name, template, enabled) VALUES (?,?,?,?,?)",
            (session_id, 0, "Zero-Shot", PRESET_TEMPLATES["Zero-Shot"], 1),
        )
        await db.commit()

        row = await (await db.execute("SELECT * FROM sessions WHERE id=?", (session_id,))).fetchone()
        return dict(row)


@app.get("/api/sessions/{sid}")
async def get_session(sid: int):
    async with get_db() as db:
        row = await (await db.execute("SELECT * FROM sessions WHERE id=?", (sid,))).fetchone()
        if not row:
            raise HTTPException(404, "Not found")
        return dict(row)


@app.put("/api/sessions/{sid}")
async def update_session(sid: int, body: SessionUpdate):
    async with get_db() as db:
        row = await (await db.execute("SELECT * FROM sessions WHERE id=?", (sid,))).fetchone()
        if not row:
            raise HTTPException(404)
        import json as _json
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        if not updates:
            return dict(row)
        # Serialize list fields to JSON string
        if "classifier_models" in updates and isinstance(updates["classifier_models"], list):
            updates["classifier_models"] = _json.dumps(updates["classifier_models"])
        set_clause = ", ".join(f"{k}=?" for k in updates)
        await db.execute(
            f"UPDATE sessions SET {set_clause}, updated_at=? WHERE id=?",
            [*updates.values(), now_iso(), sid],
        )
        await db.commit()
        row = await (await db.execute("SELECT * FROM sessions WHERE id=?", (sid,))).fetchone()
        return dict(row)


@app.delete("/api/sessions/{sid}", status_code=204)
async def delete_session(sid: int):
    async with get_db() as db:
        await db.execute("DELETE FROM sessions WHERE id=?", (sid,))
        await db.commit()


# ─── Repositories ─────────────────────────────────────────────────────────────


@app.get("/api/sessions/{sid}/repos")
async def list_repos(sid: int):
    async with get_db() as db:
        rows = await (await db.execute(
            "SELECT * FROM repositories WHERE session_id=? ORDER BY id", (sid,)
        )).fetchall()
        return [dict(r) for r in rows]


@app.post("/api/sessions/{sid}/repos", status_code=201)
async def add_repo(sid: int, body: RepoAdd):
    async with get_db() as db:
        cur = await db.execute(
            "INSERT INTO repositories (session_id, owner, name, created_at, updated_at) VALUES (?,?,?,?,?)",
            (sid, body.owner, body.name, now_iso(), now_iso()),
        )
        await db.commit()
        row = await (await db.execute("SELECT * FROM repositories WHERE id=?", (cur.lastrowid,))).fetchone()
        return dict(row)


@app.delete("/api/sessions/{sid}/repos/{rid}", status_code=204)
async def delete_repo(sid: int, rid: int):
    async with get_db() as db:
        await db.execute("DELETE FROM repositories WHERE id=? AND session_id=?", (rid, sid))
        await db.commit()


@app.post("/api/sessions/{sid}/repos/import-csv", status_code=201)
async def import_repos_from_csv(sid: int, file: UploadFile = File(...)):
    """Upload CSV with columns: repo (owner/name) OR owner + name separately."""
    content = await file.read()
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

    inserted = 0
    async with get_db() as db:
        for r in rows:
            repo_str = (r.get("repo") or r.get("full_name") or r.get("repository") or "").strip()
            if "/" in repo_str:
                owner, name = repo_str.split("/", 1)
            else:
                owner = (r.get("owner") or "").strip()
                name = (r.get("name") or r.get("repo_name") or "").strip()

            if not name:
                continue

            exists = await (await db.execute(
                "SELECT id FROM repositories WHERE session_id=? AND owner=? AND name=?",
                (sid, owner, name),
            )).fetchone()
            if exists:
                continue

            await db.execute(
                "INSERT INTO repositories (session_id, owner, name, created_at, updated_at) VALUES (?,?,?,?,?)",
                (sid, owner, name, now_iso(), now_iso()),
            )
            inserted += 1
        await db.commit()

    return {"inserted": inserted}


class LocalPathImport(BaseModel):
    base_path: str


@app.post("/api/sessions/{sid}/repos/import-local", status_code=201)
async def import_repos_from_local(sid: int, body: LocalPathImport):
    """Scan a local directory: each subdirectory becomes a repository entry."""
    base = Path(body.base_path)
    if not base.exists() or not base.is_dir():
        raise HTTPException(400, f"Directory not found: {body.base_path}")

    inserted = 0
    async with get_db() as db:
        for subdir in sorted(base.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue

            local_path = str(subdir)
            name = subdir.name
            # Use parent folder name as "owner" for grouping
            owner = base.name

            exists = await (await db.execute(
                "SELECT id FROM repositories WHERE session_id=? AND local_path=?",
                (sid, local_path),
            )).fetchone()
            if exists:
                continue

            # Pre-set local_path so runner skips cloning
            await db.execute(
                """INSERT INTO repositories
                   (session_id, owner, name, local_path, created_at, updated_at)
                   VALUES (?,?,?,?,?,?)""",
                (sid, owner, name, local_path, now_iso(), now_iso()),
            )
            inserted += 1
        await db.commit()

    return {"inserted": inserted, "base_path": str(base)}


# ─── Prompt Patterns ──────────────────────────────────────────────────────────


@app.get("/api/sessions/{sid}/patterns")
async def list_patterns(sid: int):
    async with get_db() as db:
        rows = await (await db.execute(
            "SELECT * FROM prompt_patterns WHERE session_id=? ORDER BY position", (sid,)
        )).fetchall()
        return [dict(r) for r in rows]


@app.post("/api/sessions/{sid}/patterns", status_code=201)
async def add_pattern(sid: int, body: PatternData):
    async with get_db() as db:
        count = (await (await db.execute(
            "SELECT COUNT(*) as c FROM prompt_patterns WHERE session_id=?", (sid,)
        )).fetchone())["c"]
        if count >= 10:
            raise HTTPException(400, "Max 10 prompt patterns")
        cur = await db.execute(
            "INSERT INTO prompt_patterns (session_id, position, name, template, enabled) VALUES (?,?,?,?,?)",
            (sid, body.position, body.name, body.template, 1 if body.enabled else 0),
        )
        await db.commit()
        row = await (await db.execute("SELECT * FROM prompt_patterns WHERE id=?", (cur.lastrowid,))).fetchone()
        return dict(row)


@app.put("/api/sessions/{sid}/patterns/{pid}")
async def update_pattern(sid: int, pid: int, body: PatternData):
    async with get_db() as db:
        await db.execute(
            "UPDATE prompt_patterns SET position=?, name=?, template=?, enabled=? WHERE id=? AND session_id=?",
            (body.position, body.name, body.template, 1 if body.enabled else 0, pid, sid),
        )
        await db.commit()
        row = await (await db.execute("SELECT * FROM prompt_patterns WHERE id=?", (pid,))).fetchone()
        return dict(row)


@app.delete("/api/sessions/{sid}/patterns/{pid}", status_code=204)
async def delete_pattern(sid: int, pid: int):
    async with get_db() as db:
        await db.execute("DELETE FROM prompt_patterns WHERE id=? AND session_id=?", (pid, sid))
        await db.commit()


@app.put("/api/sessions/{sid}/patterns")
async def bulk_update_patterns(sid: int, patterns: list[PatternData]):
    if len(patterns) > 10:
        raise HTTPException(400, "Max 10 patterns")
    async with get_db() as db:
        await db.execute("DELETE FROM prompt_patterns WHERE session_id=?", (sid,))
        for p in patterns:
            await db.execute(
                "INSERT INTO prompt_patterns (session_id, position, name, template, enabled) VALUES (?,?,?,?,?)",
                (sid, p.position, p.name, p.template, 1 if p.enabled else 0),
            )
        await db.commit()
        rows = await (await db.execute(
            "SELECT * FROM prompt_patterns WHERE session_id=? ORDER BY position", (sid,)
        )).fetchall()
        return [dict(r) for r in rows]


@app.get("/api/preset-templates")
async def get_preset_templates():
    return [{"name": k, "template": v} for k, v in PRESET_TEMPLATES.items()]


# ─── Smell Commits / Data ─────────────────────────────────────────────────────


@app.post("/api/sessions/{sid}/smells/upload", status_code=201)
async def upload_smells(sid: int, file: UploadFile = File(...)):
    """Upload pre-computed smell instances as CSV (skips Phase 1 scan)."""
    content = await file.read()
    text = content.decode("utf-8-sig")

    if file.filename and file.filename.endswith(".json"):
        rows = json.loads(text)
    else:
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)

    if not rows:
        raise HTTPException(400, "No data found")

    now = now_iso()
    inserted = 0

    async with get_db() as db:
        for r in rows:
            repo_str = r.get("repo") or r.get("full_repo_name") or ""
            if "/" in repo_str:
                owner, repo_name = repo_str.split("/", 1)
            else:
                owner = r.get("repo_owner") or r.get("owner") or ""
                repo_name = r.get("repo_name") or r.get("name") or ""

            # Ensure repo exists
            repo_row = await (await db.execute(
                "SELECT id FROM repositories WHERE session_id=? AND owner=? AND name=?",
                (sid, owner, repo_name),
            )).fetchone()
            if not repo_row:
                cur = await db.execute(
                    "INSERT INTO repositories (session_id, owner, name, status, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                    (sid, owner, repo_name, "scanned", now, now),
                )
                repo_id = cur.lastrowid
            else:
                repo_id = repo_row["id"]

            commit = r.get("commit_hash") or r.get("commit") or r.get("sha") or ""
            if not commit:
                continue

            await db.execute(
                """INSERT INTO smell_commits
                   (session_id, repo_id, commit_hash, prev_commit_hash, file_path,
                    function_name, smell_type, smell_line, smell_message,
                    diff_content, commit_message, issue_summary, pr_summary,
                    created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sid, repo_id, commit,
                 r.get("prev_commit_hash") or r.get("prev_commit"),
                 r.get("file_path") or r.get("filename") or "",
                 r.get("function_name"),
                 r.get("smell_type") or r.get("name_smell") or r.get("smell") or "Unknown",
                 r.get("smell_line") or r.get("line"),
                 r.get("smell_message") or r.get("message"),
                 r.get("diff_content"),
                 r.get("commit_message") or r.get("message"),
                 r.get("issue_summary"),
                 r.get("pr_summary"),
                 now, now),
            )
            inserted += 1

        total = (await (await db.execute(
            "SELECT COUNT(*) as c FROM smell_commits WHERE session_id=?", (sid,)
        )).fetchone())["c"]
        await db.execute(
            "UPDATE sessions SET phase2_total=?, phase1_status='completed', updated_at=? WHERE id=?",
            (total, now, sid),
        )
        await db.commit()

    return {"inserted": inserted, "total": total}


@app.get("/api/sessions/{sid}/smells")
async def list_smells(sid: int, page: int = 1, per_page: int = 50, status: Optional[str] = None):
    offset = (page - 1) * per_page
    base_query = """
        SELECT sc.*,
               r.owner || '/' || r.name as repo_name,
               vr.primary_activity,
               vr.tied,
               vr.vote_count,
               vr.total_votes,
               (SELECT lr.sub_activity FROM llm_results lr
                WHERE lr.smell_commit_id = sc.id
                GROUP BY lr.sub_activity
                ORDER BY COUNT(*) DESC LIMIT 1) as raw_sub_activity,
               COALESCE(
                   sam.canonical,
                   (SELECT lr.sub_activity FROM llm_results lr
                    WHERE lr.smell_commit_id = sc.id
                    GROUP BY lr.sub_activity
                    ORDER BY COUNT(*) DESC LIMIT 1)
               ) as sub_activity
        FROM smell_commits sc
        JOIN repositories r ON r.id = sc.repo_id
        LEFT JOIN vote_results vr ON vr.smell_commit_id = sc.id
        LEFT JOIN sub_activity_mapping sam
               ON sam.session_id = sc.session_id
              AND sam.raw_sub_activity = (
                   SELECT lr.sub_activity FROM llm_results lr
                   WHERE lr.smell_commit_id = sc.id
                   GROUP BY lr.sub_activity
                   ORDER BY COUNT(*) DESC LIMIT 1)
        WHERE sc.session_id=?
    """
    async with get_db() as db:
        if status:
            rows = await (await db.execute(
                base_query + " AND sc.status=? LIMIT ? OFFSET ?",
                (sid, status, per_page, offset),
            )).fetchall()
            total = (await (await db.execute(
                "SELECT COUNT(*) as c FROM smell_commits WHERE session_id=? AND status=?", (sid, status)
            )).fetchone())["c"]
        else:
            rows = await (await db.execute(
                base_query + " LIMIT ? OFFSET ?",
                (sid, per_page, offset),
            )).fetchall()
            total = (await (await db.execute(
                "SELECT COUNT(*) as c FROM smell_commits WHERE session_id=?", (sid,)
            )).fetchone())["c"]
        return {"smells": [dict(r) for r in rows], "total": total, "page": page, "per_page": per_page}


@app.get("/api/sessions/{sid}/smells/{sc_id}")
async def get_smell(sid: int, sc_id: int):
    async with get_db() as db:
        row = await (await db.execute(
            "SELECT * FROM smell_commits WHERE id=? AND session_id=?", (sc_id, sid)
        )).fetchone()
        if not row:
            raise HTTPException(404)
        item = dict(row)

        results = await (await db.execute(
            """SELECT lr.*, pp.name as pattern_name FROM llm_results lr
               JOIN prompt_patterns pp ON pp.id=lr.prompt_pattern_id
               WHERE lr.smell_commit_id=? ORDER BY lr.prompt_pattern_id, lr.run_number""",
            (sc_id,),
        )).fetchall()

        votes = await (await db.execute(
            """SELECT vr.*, pp.name as pattern_name FROM vote_results vr
               JOIN prompt_patterns pp ON pp.id=vr.prompt_pattern_id
               WHERE vr.smell_commit_id=?""",
            (sc_id,),
        )).fetchall()

        item["results"] = [dict(r) for r in results]
        item["votes"] = [dict(v) for v in votes]
        return item


# ─── Control ──────────────────────────────────────────────────────────────────


@app.post("/api/sessions/{sid}/start-phase1")
async def start_p1(sid: int):
    async with get_db() as db:
        row = await (await db.execute("SELECT status FROM sessions WHERE id=?", (sid,))).fetchone()
        if not row:
            raise HTTPException(404)
        if is_running(sid):
            return {"status": "already_running"}
        # Reset stuck 'scanning' repos to 'pending'
        await db.execute(
            "UPDATE repositories SET status='pending', updated_at=? WHERE session_id=? AND status='scanning'",
            (now_iso(), sid),
        )
        await db.commit()
    await start_phase1(sid)
    return {"status": "started", "phase": 1}


@app.post("/api/sessions/{sid}/start-phase2")
async def start_p2(sid: int):
    async with get_db() as db:
        row = await (await db.execute("SELECT * FROM sessions WHERE id=?", (sid,))).fetchone()
        if not row:
            raise HTTPException(404)
        if is_running(sid):
            return {"status": "already_running"}
        if not row["ollama_base_url"]:
            raise HTTPException(400, "Ollama base URL required")
        # Reset stuck 'running' smell_commits to 'pending'
        await db.execute(
            "UPDATE smell_commits SET status='pending', updated_at=? WHERE session_id=? AND status='running'",
            (now_iso(), sid),
        )
        await db.commit()
    await start_phase2(sid)
    return {"status": "started", "phase": 2}


@app.post("/api/sessions/{sid}/start-phase3")
async def phase3(sid: int):
    await start_phase3(sid)
    return {"status": "started", "phase": 3}


@app.get("/api/sessions/{sid}/phase3/mapping")
async def get_phase3_mapping(sid: int):
    async with get_db() as db:
        rows = await (await db.execute(
            "SELECT raw_sub_activity, canonical FROM sub_activity_mapping WHERE session_id=? ORDER BY canonical, raw_sub_activity",
            (sid,),
        )).fetchall()
    mapping = [dict(r) for r in rows]
    # Group by canonical
    groups: dict[str, list[str]] = {}
    for r in mapping:
        groups.setdefault(r["canonical"], []).append(r["raw_sub_activity"])
    return {"mapping": mapping, "groups": [{"canonical": k, "raws": v} for k, v in sorted(groups.items())]}


@app.post("/api/sessions/{sid}/backfill-commit-dates")
async def backfill_commit_dates(sid: int):
    """Populate commit_date for existing smell_commits from local git repos."""
    import asyncio
    from pathlib import Path
    from .git_client import get_commits

    loop = asyncio.get_event_loop()

    async with get_db() as db:
        repos = await (await db.execute(
            "SELECT * FROM repositories WHERE session_id=? AND local_path IS NOT NULL",
            (sid,),
        )).fetchall()

    total_updated = 0
    for repo in repos:
        local_path = repo["local_path"]
        if not local_path or not Path(local_path).exists():
            continue
        try:
            commits = await loop.run_in_executor(None, get_commits, local_path, "main", None)
        except Exception:
            try:
                commits = await loop.run_in_executor(None, get_commits, local_path, "HEAD", None)
            except Exception:
                continue

        hash_to_date = {c["hash"]: c["date"] for c in commits}
        if not hash_to_date:
            continue

        async with get_db() as db:
            rows = await (await db.execute(
                "SELECT id, commit_hash FROM smell_commits WHERE repo_id=? AND commit_date IS NULL",
                (repo["id"],),
            )).fetchall()
            for row in rows:
                date = hash_to_date.get(row["commit_hash"])
                if date:
                    await db.execute(
                        "UPDATE smell_commits SET commit_date=? WHERE id=?",
                        (date, row["id"]),
                    )
                    total_updated += 1
            await db.commit()

    return {"updated": total_updated}


@app.post("/api/sessions/{sid}/pause")
async def pause(sid: int):
    await pause_session(sid)
    return {"status": "pausing"}


@app.post("/api/sessions/{sid}/reset-failed")
async def reset_failed(sid: int):
    async with get_db() as db:
        await db.execute(
            "UPDATE smell_commits SET status='pending', error_message=NULL, updated_at=? WHERE session_id=? AND status='failed'",
            (now_iso(), sid),
        )
        await db.commit()
    return {"reset": True}


@app.post("/api/sessions/{sid}/clear-smells")
async def clear_smells(sid: int):
    async with get_db() as db:
        await db.execute("DELETE FROM smell_commits WHERE session_id=?", (sid,))
        await db.execute(
            "UPDATE sessions SET phase2_total=0, phase2_done=0, phase2_status='idle', updated_at=? WHERE id=?",
            (now_iso(), sid),
        )
        await db.commit()
    return {"cleared": True}


# ─── Results ──────────────────────────────────────────────────────────────────


@app.get("/api/sessions/{sid}/results/summary")
async def results_summary(sid: int):
    async with get_db() as db:
        patterns = [dict(p) for p in await (await db.execute(
            "SELECT * FROM prompt_patterns WHERE session_id=? ORDER BY position", (sid,)
        )).fetchall()]

        summary = []
        for p in patterns:
            votes = await (await db.execute(
                """SELECT primary_activity, COUNT(*) as count FROM vote_results
                   WHERE prompt_pattern_id=? AND smell_commit_id IN (
                       SELECT id FROM smell_commits WHERE session_id=?)
                   GROUP BY primary_activity""",
                (p["id"], sid),
            )).fetchall()

            tied_count = (await (await db.execute(
                """SELECT COUNT(*) as c FROM vote_results
                   WHERE prompt_pattern_id=? AND tied=1 AND smell_commit_id IN (
                       SELECT id FROM smell_commits WHERE session_id=?)""",
                (p["id"], sid),
            )).fetchone())["c"]

            summary.append({
                "pattern_id": p["id"],
                "pattern_name": p["name"],
                "enabled": bool(p["enabled"]),
                "distribution": {v["primary_activity"] or "Unknown": v["count"] for v in votes},
                "tied_count": tied_count,
                "total": sum(v["count"] for v in votes),
            })

        status_counts = {
            r["status"]: r["count"]
            for r in await (await db.execute(
                "SELECT status, COUNT(*) as count FROM smell_commits WHERE session_id=? GROUP BY status",
                (sid,),
            )).fetchall()
        }

        tokens = dict(await (await db.execute(
            """SELECT SUM(input_tokens) as inp, SUM(output_tokens) as out FROM llm_results
               WHERE smell_commit_id IN (SELECT id FROM smell_commits WHERE session_id=?)""",
            (sid,),
        )).fetchone())

        return {
            "patterns": summary,
            "smell_status": status_counts,
            "tokens": {"input": tokens.get("inp") or 0, "output": tokens.get("out") or 0},
        }


@app.get("/api/sessions/{sid}/results/charts")
async def results_charts(sid: int, repo_id: Optional[int] = None):
    """All chart data in one call. repo_id=None means all repos."""
    async with get_db() as db:
        scope = "AND sc.repo_id=?" if repo_id else ""
        params_base = (sid, repo_id) if repo_id else (sid,)

        # 1. Activity distribution (majority vote, first pattern)
        act_rows = await (await db.execute(f"""
            SELECT vr.primary_activity, COUNT(*) as cnt
            FROM vote_results vr
            JOIN smell_commits sc ON sc.id = vr.smell_commit_id
            WHERE sc.session_id=? {scope}
            GROUP BY vr.primary_activity
        """, params_base)).fetchall()
        activity_dist = {r["primary_activity"] or "Unknown": r["cnt"] for r in act_rows}

        # 2. Smell type distribution
        smell_rows = await (await db.execute(f"""
            SELECT sc.smell_type, COUNT(*) as cnt
            FROM smell_commits sc
            WHERE sc.session_id=? {scope}
            GROUP BY sc.smell_type ORDER BY cnt DESC
        """, params_base)).fetchall()
        smell_dist = [{"smell_type": r["smell_type"], "count": r["cnt"]} for r in smell_rows]

        # 3. Sub-activity distribution (top 20, canonical if available)
        sub_rows = await (await db.execute(f"""
            SELECT COALESCE(sam.canonical,
                (SELECT lr2.sub_activity FROM llm_results lr2
                 WHERE lr2.smell_commit_id=sc.id
                 GROUP BY lr2.sub_activity ORDER BY COUNT(*) DESC LIMIT 1)
            ) as sub_act,
            COUNT(*) as cnt
            FROM smell_commits sc
            LEFT JOIN sub_activity_mapping sam
                ON sam.session_id=sc.session_id
               AND sam.raw_sub_activity=(
                   SELECT lr.sub_activity FROM llm_results lr
                   WHERE lr.smell_commit_id=sc.id
                   GROUP BY lr.sub_activity ORDER BY COUNT(*) DESC LIMIT 1)
            WHERE sc.session_id=? {scope}
              AND sc.status='completed'
            GROUP BY sub_act
            HAVING sub_act IS NOT NULL AND sub_act != ''
            ORDER BY cnt DESC LIMIT 20
        """, params_base)).fetchall()
        sub_dist = [{"sub_activity": r["sub_act"], "count": r["cnt"]} for r in sub_rows]

        # 4. Smell type × activity cross matrix
        matrix_rows = await (await db.execute(f"""
            SELECT sc.smell_type, vr.primary_activity, COUNT(*) as cnt
            FROM vote_results vr
            JOIN smell_commits sc ON sc.id=vr.smell_commit_id
            WHERE sc.session_id=? {scope}
              AND vr.primary_activity IS NOT NULL
            GROUP BY sc.smell_type, vr.primary_activity
        """, params_base)).fetchall()
        cross_matrix = [{"smell_type": r["smell_type"], "activity": r["primary_activity"], "count": r["cnt"]} for r in matrix_rows]

        # 5. Temporal: smell-introducing commits per month, per smell type
        temporal_rows = await (await db.execute(f"""
            SELECT substr(COALESCE(sc.commit_date, sc.created_at), 1, 7) as month,
                   sc.smell_type,
                   COUNT(*) as cnt
            FROM smell_commits sc
            WHERE sc.session_id=? {scope}
            GROUP BY month, sc.smell_type
            ORDER BY month, sc.smell_type
        """, params_base)).fetchall()
        temporal = [{"month": r["month"], "smell_type": r["smell_type"], "count": r["cnt"]} for r in temporal_rows]

        # 6. Per-repo summary — one activity per smell_commit (first pattern)
        repo_rows = await (await db.execute(f"""
            SELECT r.owner, r.name,
                   COALESCE(
                       (SELECT vr2.primary_activity FROM vote_results vr2
                        WHERE vr2.smell_commit_id=sc.id
                        ORDER BY vr2.prompt_pattern_id LIMIT 1),
                       'Unclassified'
                   ) as activity,
                   COUNT(*) as cnt
            FROM smell_commits sc
            JOIN repositories r ON r.id=sc.repo_id
            WHERE sc.session_id=? {scope}
            GROUP BY r.id, activity
        """, params_base)).fetchall()
        per_repo = [{"repo": f"{r['owner']}/{r['name']}",
                     "activity": r["activity"], "count": r["cnt"]} for r in repo_rows]

        # 7. Repos list for selector
        repos_list = await (await db.execute(
            "SELECT id, owner, name FROM repositories WHERE session_id=? ORDER BY owner, name", (sid,)
        )).fetchall()

    return {
        "activity_dist": activity_dist,
        "smell_dist": smell_dist,
        "sub_dist": sub_dist,
        "cross_matrix": cross_matrix,
        "temporal": temporal,
        "per_repo": per_repo,
        "repos": [dict(r) for r in repos_list],
    }


@app.get("/api/sessions/{sid}/results/cross-matrix")
async def cross_matrix(sid: int, pattern_id: Optional[int] = None):
    """Co-occurrence matrix: activity × smell_type"""
    async with get_db() as db:
        query = """
            SELECT vr.primary_activity, sc.smell_type, COUNT(*) as count
            FROM vote_results vr
            JOIN smell_commits sc ON sc.id = vr.smell_commit_id
            WHERE sc.session_id=?
        """
        params = [sid]
        if pattern_id:
            query += " AND vr.prompt_pattern_id=?"
            params.append(pattern_id)
        query += " GROUP BY vr.primary_activity, sc.smell_type"
        rows = await (await db.execute(query, params)).fetchall()
        return [dict(r) for r in rows]


@app.get("/api/sessions/{sid}/export")
async def export(sid: int):
    async with get_db() as db:
        rows = await (await db.execute(
            """SELECT sc.repo_id, r.owner, r.name as repo_name,
                      sc.commit_hash, sc.prev_commit_hash, sc.file_path,
                      sc.function_name, sc.smell_type, sc.smell_line,
                      sc.commit_message,
                      pp.name as pattern_name,
                      vr.primary_activity, vr.vote_count, vr.total_votes,
                      vr.tied, vr.tied_activities, vr.all_votes
               FROM vote_results vr
               JOIN smell_commits sc ON sc.id = vr.smell_commit_id
               JOIN repositories r ON r.id = sc.repo_id
               JOIN prompt_patterns pp ON pp.id = vr.prompt_pattern_id
               WHERE sc.session_id=?
               ORDER BY sc.id, pp.position""",
            (sid,),
        )).fetchall()

    out = io.StringIO()
    if rows:
        fieldnames = list(dict(rows[0]).keys())
        w = csv.DictWriter(out, fieldnames=fieldnames)
        w.writeheader()
        w.writerows([dict(r) for r in rows])

    return StreamingResponse(
        iter([out.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=results_{sid}.csv"},
    )


# ─── SSE ──────────────────────────────────────────────────────────────────────


@app.get("/api/sessions/{sid}/events")
async def session_events(sid: int):
    import asyncio

    q = subscribe(sid)

    async def generate():
        try:
            async with get_db() as db:
                row = await (await db.execute(
                    "SELECT status, phase1_status, phase2_status, phase1_done, phase1_total, phase2_done, phase2_total FROM sessions WHERE id=?",
                    (sid,),
                )).fetchone()
                if row:
                    yield f"data: {json.dumps({'type': 'init', **dict(row)})}\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=25)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            unsubscribe(sid, q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Ollama Models ────────────────────────────────────────────────────────────


@app.get("/api/ollama/models")
async def list_ollama_models(base_url: str = "http://localhost:11434"):
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{base_url.rstrip('/')}/api/tags")
            if r.status_code == 200:
                data = r.json()
                return {"models": [m["name"] for m in data.get("models", [])]}
    except Exception:
        pass
    return {"models": [], "error": "Ollama not reachable"}


# ─── Dataset Stats ────────────────────────────────────────────────────────────

REPOS_DIR = Path(__file__).parent.parent / "data" / "repos"


def _num_stats(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "count": len(values),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "std": round(statistics.stdev(values) if len(values) > 1 else 0.0, 2),
    }


def _count_loc(owner: str, name: str) -> int | None:
    """Count non-blank Python lines in local clone."""
    local = REPOS_DIR / f"{owner}_{name}"
    if not local.exists():
        return None
    total = 0
    for f in local.rglob("*.py"):
        try:
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
            total += sum(1 for l in lines if l.strip())
        except OSError:
            pass
    return total if total > 0 else None


async def _fetch_github_repo(client: httpx.AsyncClient, owner: str, name: str, token: str | None) -> dict | None:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = await client.get(f"https://api.github.com/repos/{owner}/{name}", headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


@app.get("/api/sessions/{sid}/dataset-stats")
async def session_dataset_stats(sid: int):
    async with get_db() as db:
        session_row = await db.execute_fetchone("SELECT github_token FROM sessions WHERE id = ?", (sid,))
        repo_rows = await db.execute_fetchall(
            "SELECT owner, name FROM repositories WHERE session_id = ?", (sid,)
        )
    if not repo_rows:
        return {"total_repos": 0, "stars": {}, "forks": {}, "open_issues": {}, "size_kb": {}, "lines_of_code": {}}

    token = session_row["github_token"] if session_row else None

    semaphore = asyncio.Semaphore(10)

    async def fetch_with_sem(owner, name):
        async with semaphore:
            return owner, name, await _fetch_github_repo(client, owner, name, token)

    async with httpx.AsyncClient() as client:
        tasks = [fetch_with_sem(r["owner"], r["name"]) for r in repo_rows]
        results = await asyncio.gather(*tasks)

    stars, forks, issues, size_kb, loc = [], [], [], [], []
    errors = []

    for owner, name, gh in results:
        if gh is None:
            errors.append(f"{owner}/{name}")
            continue
        stars.append(float(gh.get("stargazers_count", 0)))
        forks.append(float(gh.get("forks_count", 0)))
        issues.append(float(gh.get("open_issues_count", 0)))
        size_kb.append(float(gh.get("size", 0)))
        cloc = _count_loc(owner, name)
        if cloc is not None:
            loc.append(float(cloc))

    return {
        "total_repos": len(repo_rows),
        "fetched": len(repo_rows) - len(errors),
        "errors": errors,
        "stars": _num_stats(stars),
        "forks": _num_stats(forks),
        "open_issues": _num_stats(issues),
        "size_kb": _num_stats(size_kb),
        "lines_of_code": _num_stats(loc),
    }


# ─── Static Files ─────────────────────────────────────────────────────────────


@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/{path:path}")
async def static(path: str):
    fp = FRONTEND_DIR / path
    if fp.exists() and fp.is_file():
        headers = {"Cache-Control": "no-store"} if fp.suffix in (".js", ".css") else {}
        return FileResponse(fp, headers=headers)
    return FileResponse(FRONTEND_DIR / "index.html")
