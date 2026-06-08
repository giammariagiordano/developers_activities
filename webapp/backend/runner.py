"""
Async orchestration for both phases:
  Phase 1 (CodeSmile scan): runs in ThreadPoolExecutor (sync), parallelized across repos
  Phase 2 (LLM classification): pure asyncio, fully parallel with semaphore
"""
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .db import get_db, now_iso
from .git_client import clone_or_update, get_commits, get_file_diff, extract_diff_around_line
from .codesmile_runner import scan_repo_commits
from .llm_client import run_llm_query, build_prompt, compute_majority_vote, run_normalization_batch

# Global: active asyncio tasks per session
_active: dict[int, asyncio.Task] = {}
# SSE subscribers per session
_queues: dict[int, list[asyncio.Queue]] = {}

_executor = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 4))


# ─── SSE Pub/Sub ─────────────────────────────────────────────────────────────


def subscribe(session_id: int) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _queues.setdefault(session_id, []).append(q)
    return q


def unsubscribe(session_id: int, q: asyncio.Queue):
    if session_id in _queues:
        try:
            _queues[session_id].remove(q)
        except ValueError:
            pass


async def _broadcast(session_id: int, event_type: str, data: dict):
    msg = json.dumps({"type": event_type, **data})
    for q in _queues.get(session_id, []):
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass


# ─── Public Control ──────────────────────────────────────────────────────────


def is_running(session_id: int) -> bool:
    t = _active.get(session_id)
    return t is not None and not t.done()


async def start_phase1(session_id: int):
    if is_running(session_id):
        return
    t = asyncio.create_task(_run_phase1(session_id))
    _active[session_id] = t


async def start_phase2(session_id: int):
    if is_running(session_id):
        return
    t = asyncio.create_task(_run_phase2(session_id))
    _active[session_id] = t


async def start_phase3(session_id: int):
    if is_running(session_id):
        return
    t = asyncio.create_task(_run_phase3(session_id))
    _active[session_id] = t


async def _run_phase3(session_id: int):
    try:
        async with get_db() as db:
            sess = dict(await (await db.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            )).fetchone())
            await db.execute(
                "UPDATE sessions SET status='running', phase3_status='running', updated_at=? WHERE id=?",
                (now_iso(), session_id),
            )
            await db.commit()

        await _broadcast(session_id, "status", {"status": "running", "phase": 3})

        # Collect all unique sub_activities for this session from llm_results
        async with get_db() as db:
            rows = await (await db.execute("""
                SELECT DISTINCT lr.sub_activity
                FROM llm_results lr
                JOIN smell_commits sc ON sc.id = lr.smell_commit_id
                WHERE sc.session_id = ?
                  AND lr.sub_activity IS NOT NULL
                  AND trim(lr.sub_activity) != ''
            """, (session_id,))).fetchall()

        all_labels = [r["sub_activity"] for r in rows]

        if not all_labels:
            async with get_db() as db:
                await db.execute(
                    "UPDATE sessions SET status='idle', phase3_status='completed', updated_at=? WHERE id=?",
                    (now_iso(), session_id),
                )
                await db.commit()
            await _broadcast(session_id, "phase3_complete", {"mapped": 0})
            return

        # Pre-dedup: lowercase+strip+basic singularization before sending to LLM
        def _pre_singularize(s: str) -> str:
            s = s.lower().strip()
            # simple English plural → singular for common endings
            if s.endswith("ies") and len(s) > 4:
                s = s[:-3] + "y"
            elif s.endswith("ses") and len(s) > 4:
                s = s[:-2]
            elif s.endswith("xes") and len(s) > 4:
                s = s[:-2]
            elif s.endswith("ches") and len(s) > 5:
                s = s[:-2]
            elif s.endswith("shes") and len(s) > 5:
                s = s[:-2]
            elif s.endswith("ves") and len(s) > 4:
                s = s[:-3] + "f"
            elif s.endswith("s") and not s.endswith("ss") and len(s) > 3:
                s = s[:-1]
            return s

        pre_norm: dict[str, str] = {l: _pre_singularize(l) for l in all_labels}
        unique_for_llm = list(dict.fromkeys(pre_norm.values()))  # preserve order, dedup

        await _broadcast(session_id, "phase3_progress", {"total": len(unique_for_llm), "done": 0})

        # Pass 1: normalize unique lowercased labels → intermediate canonicals
        BATCH_SIZE = 50
        pass1: dict[str, str] = {}
        for i in range(0, len(unique_for_llm), BATCH_SIZE):
            batch = unique_for_llm[i: i + BATCH_SIZE]
            mappings = await run_normalization_batch(batch, sess["model"], sess["openai_api_key"])
            pass1.update(mappings)
            await _broadcast(session_id, "phase3_progress", {
                "total": len(unique_for_llm),
                "done": min(i + BATCH_SIZE, len(unique_for_llm)),
                "pass": 1,
            })

        # Pass 2: normalize the intermediate canonicals themselves
        # (merges near-duplicates like "test case" / "test cases")
        intermediate_canonicals = list(set(pass1.values()))
        pass2: dict[str, str] = {}
        for i in range(0, len(intermediate_canonicals), BATCH_SIZE):
            batch = intermediate_canonicals[i: i + BATCH_SIZE]
            mappings = await run_normalization_batch(batch, sess["model"], sess["openai_api_key"])
            pass2.update(mappings)
            await _broadcast(session_id, "phase3_progress", {
                "total": len(intermediate_canonicals),
                "done": min(i + BATCH_SIZE, len(intermediate_canonicals)),
                "pass": 2,
            })

        # Compose: original_raw → lowercased → pass1 → pass2 → final canonical
        # pre_norm maps original_raw → lowercased
        # pass1 maps lowercased → intermediate canonical
        # pass2 maps intermediate → final canonical
        all_mappings: dict[str, str] = {}
        for original_raw, lowercased in pre_norm.items():
            intermediate = pass1.get(lowercased, lowercased)
            final = pass2.get(intermediate, intermediate)
            all_mappings[original_raw] = final

        # Persist mapping
        now = now_iso()
        async with get_db() as db:
            for raw, canonical in all_mappings.items():
                await db.execute(
                    """INSERT OR REPLACE INTO sub_activity_mapping
                       (session_id, raw_sub_activity, canonical, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (session_id, raw, canonical, now),
                )
            await db.execute(
                "UPDATE sessions SET status='idle', phase3_status='completed', updated_at=? WHERE id=?",
                (now, session_id),
            )
            await db.commit()

        await _broadcast(session_id, "phase3_complete", {"mapped": len(all_mappings)})

    except Exception as e:
        async with get_db() as db:
            await db.execute(
                "UPDATE sessions SET status='error', phase3_status='error', updated_at=? WHERE id=?",
                (now_iso(), session_id),
            )
            await db.commit()
        await _broadcast(session_id, "error", {"message": str(e)})


async def pause_session(session_id: int):
    async with get_db() as db:
        await db.execute(
            "UPDATE sessions SET status='pausing', updated_at=? WHERE id=?",
            (now_iso(), session_id),
        )
        await db.commit()


async def reset_session_on_startup():
    """Called on server start: reset stuck sessions."""
    async with get_db() as db:
        await db.execute(
            "UPDATE sessions SET status='paused', updated_at=? WHERE status IN ('running','pausing')",
            (now_iso(),),
        )
        # Align phase statuses with session status
        await db.execute(
            "UPDATE sessions SET phase1_status='paused', updated_at=? WHERE status='paused' AND phase1_status='running'",
            (now_iso(),),
        )
        await db.execute(
            "UPDATE sessions SET phase2_status='paused', updated_at=? WHERE status='paused' AND phase2_status='running'",
            (now_iso(),),
        )
        await db.execute(
            "UPDATE sessions SET phase3_status='paused', updated_at=? WHERE status='paused' AND phase3_status='running'",
            (now_iso(),),
        )
        await db.execute(
            "UPDATE repositories SET status='pending', updated_at=? WHERE status IN ('cloning','scanning')",
            (now_iso(),),
        )
        await db.execute(
            "UPDATE smell_commits SET status='pending', updated_at=? WHERE status='running'",
            (now_iso(),),
        )
        await db.commit()


# ─── Phase 1: Commit-by-commit CodeSmile scan ────────────────────────────────


async def _run_phase1(session_id: int):
    try:
        async with get_db() as db:
            sess = dict(await (await db.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            )).fetchone())
            repos = [dict(r) for r in await (await db.execute(
                "SELECT * FROM repositories WHERE session_id=? AND status NOT IN ('scanned','error')",
                (session_id,),
            )).fetchall()]

        if not repos:
            async with get_db() as db:
                await db.execute(
                    "UPDATE sessions SET phase1_status='completed', updated_at=? WHERE id=?",
                    (now_iso(), session_id),
                )
                await db.commit()
            await _broadcast(session_id, "phase1_complete", {})
            return

        async with get_db() as db:
            await db.execute(
                "UPDATE sessions SET status='running', phase1_status='running', updated_at=? WHERE id=?",
                (now_iso(), session_id),
            )
            await db.commit()

        await _broadcast(session_id, "status", {"status": "running", "phase": 1})

        # Process repos in parallel (each in its own thread since CodeSmile is sync)
        tasks = [
            asyncio.create_task(_scan_repo(session_id, repo, sess))
            for repo in repos
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Check pause
        async with get_db() as db:
            status = (await (await db.execute(
                "SELECT status FROM sessions WHERE id=?", (session_id,)
            )).fetchone())["status"]

        if status == "pausing":
            async with get_db() as db:
                await db.execute(
                    "UPDATE sessions SET status='paused', phase1_status='paused', updated_at=? WHERE id=?",
                    (now_iso(), session_id),
                )
                await db.commit()
            await _broadcast(session_id, "status", {"status": "paused"})
            return

        # Recount phase2_total from newly found smell_commits
        async with get_db() as db:
            total = (await (await db.execute(
                "SELECT COUNT(*) as c FROM smell_commits WHERE session_id=?", (session_id,)
            )).fetchone())["c"]
            await db.execute(
                "UPDATE sessions SET phase1_status='completed', phase2_total=?, status='idle', updated_at=? WHERE id=?",
                (total, now_iso(), session_id),
            )
            await db.commit()

        await _broadcast(session_id, "phase1_complete", {"smell_count": total})

    except Exception as e:
        async with get_db() as db:
            await db.execute(
                "UPDATE sessions SET status='error', phase1_status='error', updated_at=? WHERE id=?",
                (now_iso(), session_id),
            )
            await db.commit()
        await _broadcast(session_id, "error", {"message": str(e)})


async def _scan_repo(session_id: int, repo: dict, sess: dict):
    repo_id = repo["id"]
    owner = repo["owner"]
    name = repo["name"]
    start_idx = repo["current_commit_index"]

    try:
        loop = asyncio.get_event_loop()
        existing_path = repo.get("local_path")

        # ── Local path already set (no-clone mode) ──
        if existing_path and __import__("os").path.exists(existing_path):
            local_path = existing_path
            is_git = __import__("os").path.exists(__import__("os").path.join(local_path, ".git"))

            if not is_git:
                # No git history → single-state CodeSmile scan
                await _scan_no_git(session_id, repo_id, owner, name, local_path)
                return

            async with get_db() as db:
                await db.execute(
                    "UPDATE repositories SET status='scanning', updated_at=? WHERE id=?",
                    (now_iso(), repo_id),
                )
                await db.commit()
            await _broadcast(session_id, "repo_status", {
                "repo_id": repo_id, "status": "scanning", "repo": f"{owner}/{name}"
            })

        # ── Clone from GitHub ──
        else:
            async with get_db() as db:
                await db.execute(
                    "UPDATE repositories SET status='cloning', updated_at=? WHERE id=?",
                    (now_iso(), repo_id),
                )
                await db.commit()

            await _broadcast(session_id, "repo_status", {
                "repo_id": repo_id, "status": "cloning", "repo": f"{owner}/{name}"
            })

            local_path = await loop.run_in_executor(
                _executor, clone_or_update, owner, name, sess.get("github_token")
            )

            async with get_db() as db:
                await db.execute(
                    "UPDATE repositories SET local_path=?, status='scanning', updated_at=? WHERE id=?",
                    (local_path, now_iso(), repo_id),
                )
                await db.commit()

        # Get commits
        commits = await loop.run_in_executor(
            _executor, get_commits, local_path,
            sess.get("branch", "main"), sess.get("max_commits")
        )

        async with get_db() as db:
            await db.execute(
                "UPDATE repositories SET total_commits=?, updated_at=? WHERE id=?",
                (len(commits), now_iso(), repo_id),
            )
            await db.commit()

        await _broadcast(session_id, "repo_status", {
            "repo_id": repo_id, "status": "scanning",
            "total_commits": len(commits), "repo": f"{owner}/{name}"
        })

        # Scan commits in a thread (blocking operation)
        # progress_cb runs inside ThreadPoolExecutor — use raw sqlite3 for DB writes
        import sqlite3 as _sqlite3
        _db_path_str = str(__import__('pathlib').Path(__file__).parent.parent / "data" / "analysis.db")
        total_commits = len(commits)
        scanned_index = [start_idx]

        def progress_cb(idx, skip=False, error=None):
            scanned_index[0] = idx
            if idx % 5 == 0 or idx == total_commits - 1:
                try:
                    conn = _sqlite3.connect(_db_path_str, timeout=5)
                    conn.execute(
                        "UPDATE repositories SET current_commit_index=?, updated_at=? WHERE id=?",
                        (idx, now_iso(), repo_id),
                    )
                    conn.commit()
                    conn.close()
                except Exception:
                    pass

        smell_instances = await loop.run_in_executor(
            _executor, scan_repo_commits, local_path, commits, start_idx, progress_cb
        )

        # Save smell instances to DB
        now = now_iso()
        async with get_db() as db:
            for s in smell_instances:
                await db.execute(
                    """INSERT INTO smell_commits
                       (session_id, repo_id, commit_hash, prev_commit_hash, file_path,
                        function_name, smell_type, smell_line, smell_message,
                        commit_message, commit_date, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (session_id, repo_id, s["commit_hash"], s.get("prev_commit_hash"),
                     s["file_path"], s.get("function_name"), s["smell_type"],
                     s.get("smell_line"), s.get("smell_message"),
                     s.get("commit_message"), s.get("commit_date"), now, now),
                )

            phase1_done_inc = len(smell_instances)
            await db.execute(
                "UPDATE sessions SET phase1_done=phase1_done+?, updated_at=? WHERE id=?",
                (phase1_done_inc, now, session_id),
            )
            await db.execute(
                "UPDATE repositories SET status='scanned', current_commit_index=?, updated_at=? WHERE id=?",
                (len(commits), now, repo_id),
            )
            await db.commit()

        await _broadcast(session_id, "repo_complete", {
            "repo_id": repo_id, "repo": f"{owner}/{name}",
            "smells_found": len(smell_instances),
        })

    except Exception as e:
        async with get_db() as db:
            await db.execute(
                "UPDATE repositories SET status='error', error_message=?, updated_at=? WHERE id=?",
                (str(e)[:500], now_iso(), repo_id),
            )
            await db.commit()
        await _broadcast(session_id, "repo_error", {
            "repo_id": repo_id, "repo": f"{owner}/{name}", "error": str(e)
        })


async def _scan_no_git(session_id: int, repo_id: int, owner: str, name: str, local_path: str):
    """Single-state CodeSmile scan for local dirs without git history."""
    from .codesmile_runner import run_codesmile_on_path

    await _broadcast(session_id, "repo_status", {
        "repo_id": repo_id, "status": "scanning", "repo": f"{owner}/{name} (no git)"
    })

    loop = asyncio.get_event_loop()
    try:
        snapshot = await loop.run_in_executor(_executor, run_codesmile_on_path, local_path)
        now = now_iso()
        async with get_db() as db:
            for smell_data in snapshot.values():
                await db.execute(
                    """INSERT INTO smell_commits
                       (session_id, repo_id, commit_hash, file_path,
                        function_name, smell_type, smell_line, smell_message,
                        commit_message, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (session_id, repo_id, "local_snapshot",
                     smell_data["file_path"], smell_data.get("function_name"),
                     smell_data["smell_type"], smell_data.get("smell_line"),
                     smell_data.get("smell_message"), "[local snapshot — no commit history]",
                     now, now),
                )
            await db.execute(
                "UPDATE repositories SET status='scanned', total_commits=0, updated_at=? WHERE id=?",
                (now, repo_id),
            )
            await db.execute(
                "UPDATE sessions SET phase1_done=phase1_done+1, updated_at=? WHERE id=?",
                (now, session_id),
            )
            await db.commit()

        await _broadcast(session_id, "repo_complete", {
            "repo_id": repo_id, "repo": f"{owner}/{name}",
            "smells_found": len(snapshot),
        })
    except Exception as e:
        async with get_db() as db:
            await db.execute(
                "UPDATE repositories SET status='error', error_message=?, updated_at=? WHERE id=?",
                (str(e)[:500], now_iso(), repo_id),
            )
            await db.commit()
        await _broadcast(session_id, "repo_error", {
            "repo_id": repo_id, "repo": f"{owner}/{name}", "error": str(e)
        })


# ─── Phase 2: LLM Classification ─────────────────────────────────────────────


async def _run_phase2(session_id: int):
    try:
        async with get_db() as db:
            sess = dict(await (await db.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            )).fetchone())
            await db.execute(
                "UPDATE sessions SET status='running', phase2_status='running', updated_at=? WHERE id=?",
                (now_iso(), session_id),
            )
            await db.commit()

        await _broadcast(session_id, "status", {"status": "running", "phase": 2})

        sem = asyncio.Semaphore(sess.get("max_parallel_llm", 20))

        while True:
            # Pause check
            async with get_db() as db:
                status = (await (await db.execute(
                    "SELECT status FROM sessions WHERE id=?", (session_id,)
                )).fetchone())["status"]

            if status == "pausing":
                async with get_db() as db:
                    await db.execute(
                        "UPDATE sessions SET status='paused', phase2_status='paused', updated_at=? WHERE id=?",
                        (now_iso(), session_id),
                    )
                    await db.commit()
                await _broadcast(session_id, "status", {"status": "paused"})
                return

            # Get batch of pending smell commits
            async with get_db() as db:
                rows = await (await db.execute(
                    "SELECT * FROM smell_commits WHERE session_id=? AND status='pending' LIMIT 30",
                    (session_id,),
                )).fetchall()
                batch = [dict(r) for r in rows]

                if not batch:
                    running = (await (await db.execute(
                        "SELECT COUNT(*) as c FROM smell_commits WHERE session_id=? AND status='running'",
                        (session_id,),
                    )).fetchone())["c"]
                    if running == 0:
                        await db.execute(
                            "UPDATE sessions SET status='idle', phase2_status='completed', updated_at=? WHERE id=?",
                            (now_iso(), session_id),
                        )
                        await db.commit()
                        await _broadcast(session_id, "phase2_complete", {})
                        return
                    await asyncio.sleep(1)
                    continue

                ids = [t["id"] for t in batch]
                await db.execute(
                    f"UPDATE smell_commits SET status='running', updated_at=? WHERE id IN ({','.join('?'*len(ids))})",
                    [now_iso()] + ids,
                )
                await db.commit()

            await asyncio.gather(*[
                _classify_smell(sc, sess, sem) for sc in batch
            ], return_exceptions=True)

    except Exception as e:
        async with get_db() as db:
            await db.execute(
                "UPDATE sessions SET status='error', phase2_status='error', updated_at=? WHERE id=?",
                (now_iso(), session_id),
            )
            await db.commit()
        await _broadcast(session_id, "error", {"message": str(e)})


async def _classify_smell(sc: dict, sess: dict, sem: asyncio.Semaphore):
    sc_id = sc["id"]
    session_id = sc["session_id"]

    try:
        # Fetch diff if not cached
        diff = sc.get("diff_content")
        if not diff and sc.get("prev_commit_hash"):
            async with get_db() as db:
                repo = dict(await (await db.execute(
                    "SELECT local_path FROM repositories WHERE id=?", (sc["repo_id"],)
                )).fetchone())

            loop = asyncio.get_event_loop()
            raw_diff = await loop.run_in_executor(
                _executor, get_file_diff,
                repo["local_path"], sc["prev_commit_hash"], sc["commit_hash"], sc["file_path"]
            )
            # Extract context around smelly line
            diff = extract_diff_around_line(raw_diff, sc.get("smell_line"), context=40)
            async with get_db() as db:
                await db.execute(
                    "UPDATE smell_commits SET diff_content=?, updated_at=? WHERE id=?",
                    (diff, now_iso(), sc_id),
                )
                await db.commit()

        sc["diff_content"] = diff or "[No diff available]"

        # Get enabled patterns
        async with get_db() as db:
            patterns = [dict(p) for p in await (await db.execute(
                "SELECT * FROM prompt_patterns WHERE session_id=? AND enabled=1 ORDER BY position",
                (session_id,),
            )).fetchall()]

        if not patterns:
            raise ValueError("No enabled prompt patterns")

        # Build all (pattern, run) pairs, skip existing
        async with get_db() as db:
            calls = []
            for p in patterns:
                for run in range(sess["n_runs"]):
                    exists = await (await db.execute(
                        "SELECT id FROM llm_results WHERE smell_commit_id=? AND prompt_pattern_id=? AND run_number=?",
                        (sc_id, p["id"], run),
                    )).fetchone()
                    if not exists:
                        calls.append((p, run))

        # Parallel LLM calls (bounded)
        await asyncio.gather(*[
            _single_llm_call(sc, p, run, sess, sem) for p, run in calls
        ], return_exceptions=True)

        # Compute majority vote per pattern
        for p in patterns:
            async with get_db() as db:
                res_rows = await (await db.execute(
                    "SELECT * FROM llm_results WHERE smell_commit_id=? AND prompt_pattern_id=?",
                    (sc_id, p["id"]),
                )).fetchall()
            results = [dict(r) for r in res_rows]
            if results:
                vote = compute_majority_vote(results)
                async with get_db() as db:
                    await db.execute(
                        """INSERT OR REPLACE INTO vote_results
                           (smell_commit_id, prompt_pattern_id, primary_activity,
                            vote_count, total_votes, tied, tied_activities, all_votes, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?)""",
                        (sc_id, p["id"], vote["primary_activity"],
                         vote["vote_count"], vote["total_votes"],
                         1 if vote["tied"] else 0,
                         json.dumps(vote["tied_activities"]),
                         json.dumps(vote["all_votes"]),
                         now_iso()),
                    )
                    await db.commit()

        # Only mark completed if at least one LLM result was stored
        async with get_db() as db:
            has_results = (await (await db.execute(
                "SELECT COUNT(*) as c FROM llm_results WHERE smell_commit_id=?",
                (sc_id,),
            )).fetchone())["c"]

            if has_results:
                await db.execute(
                    "UPDATE smell_commits SET status='completed', updated_at=? WHERE id=?",
                    (now_iso(), sc_id),
                )
                await db.execute(
                    "UPDATE sessions SET phase2_done=phase2_done+1, updated_at=? WHERE id=?",
                    (now_iso(), session_id),
                )
            else:
                await db.execute(
                    "UPDATE smell_commits SET status='pending', updated_at=? WHERE id=?",
                    (now_iso(), sc_id),
                )
            await db.commit()

        async with get_db() as db:
            prog = dict(await (await db.execute(
                "SELECT phase2_done, phase2_total FROM sessions WHERE id=?", (session_id,)
            )).fetchone())

        await _broadcast(session_id, "progress", {
            "done": prog["phase2_done"], "total": prog["phase2_total"], "sc_id": sc_id
        })

    except Exception as e:
        async with get_db() as db:
            await db.execute(
                "UPDATE smell_commits SET status='failed', error_message=?, updated_at=? WHERE id=?",
                (str(e)[:500], now_iso(), sc_id),
            )
            await db.commit()
        await _broadcast(session_id, "task_failed", {"sc_id": sc_id, "error": str(e)})


async def _single_llm_call(sc: dict, pattern: dict, run_num: int, sess: dict, sem: asyncio.Semaphore):
    async with sem:
        prompt = build_prompt(pattern["template"], sc)
        result = await run_llm_query(
            prompt=prompt,
            model=sess["model"],
            temperature=sess["temperature"],
            api_key=sess["openai_api_key"],
        )
        async with get_db() as db:
            await db.execute(
                """INSERT OR IGNORE INTO llm_results
                   (smell_commit_id, prompt_pattern_id, run_number, primary_activity,
                    sub_activity, reasoning, raw_response, input_tokens, output_tokens, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (sc["id"], pattern["id"], run_num,
                 result["primary_activity"], result["sub_activity"], result["reasoning"],
                 result["raw_response"], result["input_tokens"], result["output_tokens"],
                 now_iso()),
            )
            await db.commit()
