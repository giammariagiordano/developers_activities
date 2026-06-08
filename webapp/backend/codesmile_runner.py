import sys
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

SMELL_AI_DIR = Path(__file__).parent.parent.parent / "smell_ai"


def _load_codesmile():
    smell_ai_str = str(SMELL_AI_DIR)
    if smell_ai_str not in sys.path:
        sys.path.insert(0, smell_ai_str)
    from components.inspector import Inspector
    from utils.file_utils import FileUtils
    return Inspector, FileUtils


_DICT_PATHS = {
    "dataframe": str(SMELL_AI_DIR / "obj_dictionaries/dataframes.csv"),
    "model":     str(SMELL_AI_DIR / "obj_dictionaries/models.csv"),
    "tensor":    str(SMELL_AI_DIR / "obj_dictionaries/tensors.csv"),
}

# Thread-local Inspector — created once per thread, reused across all files.
# Inspector.inspect() is stateless (no disk writes), so this is safe.
_thread_local = threading.local()


def _get_thread_inspector():
    if not hasattr(_thread_local, "inspector"):
        Inspector, _ = _load_codesmile()
        _thread_local.inspector = Inspector(
            "/tmp",
            dataframe_dict_path=_DICT_PATHS["dataframe"],
            model_dict_path=_DICT_PATHS["model"],
            tensor_dict_path=_DICT_PATHS["tensor"],
        )
    return _thread_local.inspector


# Pre-filter: files with no ML imports cannot have ML-specific code smells.
_ML_IMPORT_RE = re.compile(
    r"^\s*(import|from)\s+(torch|tensorflow|tf|keras|sklearn|pandas|numpy|"
    r"scipy|transformers|xgboost|lightgbm|catboost|cv2|PIL|gym|mlflow|"
    r"wandb|optuna|jax|flax|mxnet|caffe|theano|lasagne|fastai)",
    re.MULTILINE,
)


def _has_ml_imports(filename: str) -> bool:
    try:
        with open(filename, "r", errors="ignore") as f:
            content = f.read(16384)  # 16 KB covers all imports
        return bool(_ML_IMPORT_RE.search(content))
    except Exception:
        return True  # if unreadable, let CodeSmile decide


def _inspect_file(filename: str, project_path: str) -> dict:
    """Thread-safe: reuses thread-local Inspector (no CSV reload per file)."""
    if not _has_ml_imports(filename):
        return {}
    inspector = _get_thread_inspector()
    results = {}
    try:
        df = inspector.inspect(filename)
        if df is not None and not df.empty:
            rel_path = os.path.relpath(filename, project_path)
            for _, row in df.iterrows():
                key = (rel_path, str(row.get("function_name", "")), str(row.get("smell_name", row.get("name_smell", ""))))
                if key not in results:
                    results[key] = {
                        "file_path": rel_path,
                        "function_name": str(row.get("function_name", "")),
                        "smell_type": str(row.get("smell_name", row.get("name_smell", ""))),
                        "smell_line": int(row.get("line", 0)) if row.get("line") else None,
                        "smell_message": str(row.get("additional_info", row.get("message", ""))),
                    }
    except Exception:
        pass
    return results


def run_codesmile_on_path(project_path: str) -> dict[str, list[dict]]:
    """
    Run CodeSmile on all .py files in project_path.
    Parallel across files (thread-local Inspector per worker thread).
    """
    _, FileUtils = _load_codesmile()
    filenames = FileUtils.get_python_files(project_path)

    if not filenames:
        return {}

    results = {}
    workers = min(len(filenames), max(os.cpu_count() or 4, 8))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_inspect_file, f, project_path) for f in filenames]
        for future in futures:
            try:
                results.update(future.result())
            except Exception:
                continue

    return results


def _apply_incremental(local_path: str, prev_snapshot: dict, changed_files: list[tuple[str, str]]) -> dict:
    """
    Build updated snapshot by re-scanning only the changed .py files.
    Unchanged files retain their previous smell entries.
    """
    current_snapshot = dict(prev_snapshot)

    for status, filepath in changed_files:
        # Drop all previous entries for this file
        for k in [k for k in current_snapshot if k[0] == filepath]:
            del current_snapshot[k]

        if status != "D":
            abs_path = os.path.join(local_path, filepath)
            if os.path.exists(abs_path):
                partial = _inspect_file(abs_path, local_path)
                current_snapshot.update(partial)

    return current_snapshot


def diff_smell_snapshots(prev: dict, current: dict) -> list[dict]:
    """Find smells newly introduced (in current but not in prev)."""
    return [smell_data for key, smell_data in current.items() if key not in prev]


def scan_repo_commits(
    local_path: str,
    commits: list[dict],
    start_index: int = 0,
    progress_callback=None,
) -> list[dict]:
    """
    Scan a repo commit-by-commit for smell-introducing commits.
    Uses incremental scanning: only re-runs CodeSmile on files changed per commit.
    Falls back to full scan if git diff fails.
    """
    from .git_client import checkout_commit, is_merge_commit, get_changed_python_files

    smell_instances = []
    prev_snapshot: dict = {}
    actual_prev_hash: str | None = None  # last actually-processed commit hash

    for i, commit in enumerate(commits):
        if i < start_index:
            continue

        if is_merge_commit(commit):
            if progress_callback:
                progress_callback(i, skip=True)
            continue

        current_hash = commit["hash"]

        try:
            checkout_commit(local_path, current_hash)

            if actual_prev_hash is None:
                # First processed commit: full scan to build baseline snapshot
                current_snapshot = run_codesmile_on_path(local_path)
            else:
                changed = get_changed_python_files(local_path, actual_prev_hash, current_hash)
                if changed is None:
                    # git diff failed: fall back to full scan
                    current_snapshot = run_codesmile_on_path(local_path)
                elif not changed:
                    # No .py files touched: snapshot unchanged
                    current_snapshot = prev_snapshot
                else:
                    current_snapshot = _apply_incremental(local_path, prev_snapshot, changed)

            if actual_prev_hash is not None:
                new_smells = diff_smell_snapshots(prev_snapshot, current_snapshot)
                for smell in new_smells:
                    smell_instances.append({
                        "commit_hash": current_hash,
                        "prev_commit_hash": actual_prev_hash,
                        "commit_message": commit["message"],
                        "commit_date": commit.get("date"),
                        **smell,
                    })

            prev_snapshot = current_snapshot
            actual_prev_hash = current_hash

            if progress_callback:
                progress_callback(i, skip=False)

        except Exception as e:
            if progress_callback:
                progress_callback(i, skip=False, error=str(e))
            prev_snapshot = {}
            actual_prev_hash = None

    return smell_instances
