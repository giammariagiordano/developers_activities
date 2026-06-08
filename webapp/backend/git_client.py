import os
import re
import shutil
from pathlib import Path
from typing import Optional
import git

REPOS_DIR = Path(__file__).parent.parent / "data" / "repos"
MERGE_KEYWORDS = {"merge branch", "merge pull request", "merged in", "merge remote"}


def get_repo_local_path(owner: str, name: str) -> str:
    return str(REPOS_DIR / f"{owner}_{name}")


def clone_or_update(owner: str, name: str, github_token: Optional[str] = None) -> str:
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = get_repo_local_path(owner, name)

    if github_token:
        url = f"https://{github_token}@github.com/{owner}/{name}.git"
    else:
        url = f"https://github.com/{owner}/{name}.git"

    if os.path.exists(local_path):
        try:
            repo = git.Repo(local_path)
            repo.git.fetch("--all")
            return local_path
        except Exception:
            shutil.rmtree(local_path, ignore_errors=True)

    git.Repo.clone_from(url, local_path)
    return local_path


def get_commits(local_path: str, branch: str = "main", max_commits: Optional[int] = None) -> list[dict]:
    repo = git.Repo(local_path)

    # Try branch, then HEAD
    try:
        commits = list(repo.iter_commits(branch, reverse=True))
    except Exception:
        try:
            commits = list(repo.iter_commits("HEAD", reverse=True))
        except Exception:
            commits = []

    if max_commits:
        commits = commits[-max_commits:]

    return [
        {
            "hash": c.hexsha,
            "short_hash": c.hexsha[:8],
            "message": c.message.strip(),
            "author": str(c.author),
            "date": c.committed_datetime.isoformat(),
            "parents": [p.hexsha for p in c.parents],
        }
        for c in commits
    ]


def is_merge_commit(commit_data: dict) -> bool:
    msg = commit_data.get("message", "").lower()
    return any(kw in msg for kw in MERGE_KEYWORDS) or len(commit_data.get("parents", [])) > 1


def checkout_commit(local_path: str, commit_hash: str):
    repo = git.Repo(local_path)
    repo.git.checkout(commit_hash, force=True)


def get_file_diff(local_path: str, prev_hash: str, current_hash: str, file_path: str) -> str:
    repo = git.Repo(local_path)
    try:
        diff = repo.git.diff(prev_hash, current_hash, "--", file_path, unified=5)
        return diff if diff else f"[No diff available for {file_path}]"
    except Exception as e:
        return f"[Error getting diff: {e}]"


def get_file_content_at_commit(local_path: str, commit_hash: str, file_path: str) -> str:
    repo = git.Repo(local_path)
    try:
        return repo.git.show(f"{commit_hash}:{file_path}")
    except Exception:
        return ""


def extract_diff_around_line(diff: str, target_line: int, context: int = 30) -> str:
    """Extract the diff hunk closest to the target line."""
    if not diff or target_line is None:
        return diff

    lines = diff.split("\n")
    # Find hunk headers like @@ -a,b +c,d @@
    hunk_pattern = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

    best_hunk_start = 0
    best_distance = float("inf")
    current_line = 0

    for i, line in enumerate(lines):
        m = hunk_pattern.match(line)
        if m:
            hunk_line = int(m.group(1))
            dist = abs(hunk_line - target_line)
            if dist < best_distance:
                best_distance = dist
                best_hunk_start = i
            current_line = hunk_line

    if best_hunk_start == 0 and len(lines) <= context * 2:
        return diff

    # Extract from best hunk start + context lines
    start = best_hunk_start
    end = min(len(lines), start + context * 2)
    return "\n".join(lines[start:end])


def get_changed_python_files(local_path: str, prev_hash: str, current_hash: str) -> list[tuple[str, str]] | None:
    """Returns list of (status, filepath) for changed .py files between two commits.
    status: A=added, M=modified, D=deleted, R=renamed (old path added as D).
    Returns None on error (caller should fall back to full scan).
    """
    repo = git.Repo(local_path)
    try:
        output = repo.git.diff(prev_hash, current_hash, "--name-status")
        changed: list[tuple[str, str]] = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            status = parts[0][0]  # A, M, D, R, C
            new_path = parts[-1]
            if new_path.endswith(".py"):
                changed.append((status, new_path))
            # For renames: also mark old path as deleted
            if status == "R" and len(parts) >= 3:
                old_path = parts[1]
                if old_path.endswith(".py") and old_path != new_path:
                    changed.append(("D", old_path))
        return changed
    except Exception:
        return None


def restore_to_head(local_path: str):
    repo = git.Repo(local_path)
    try:
        repo.git.checkout("HEAD", force=True)
    except Exception:
        pass


def delete_repo(local_path: str):
    if os.path.exists(local_path):
        shutil.rmtree(local_path, ignore_errors=True)
