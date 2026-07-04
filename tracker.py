#!/usr/bin/env python3
"""tracker.py - thread-local (per-session) file-creation tracker.

When the main agent creates a file (Write/Edit/NotebookEdit, or a Bash command
that creates files via touch/mkdir/redirect/cp/mv/tee), the PostToolUse hook
records the absolute path. Later, when the permission hook sees a delete/edit
targeting that file, it fast-path allows — the user's rule:
"if the agent created the file itself, it can delete/modify it."

State is stored per session_id in a JSON file so it survives hook restarts.

Concurrency: every hook invocation is a separate process, and the user runs
multiple agent threads in parallel, so concurrent writers are expected. Writes
are therefore serialized with an fcntl.flock on a sidecar lock file, and the
state file is replaced atomically (write tmp -> fsync -> os.replace). Readers
(was_created_by_agent / get_tracked_files) take no lock: os.replace is atomic,
so a reader sees either the fully-old or fully-new file, never a torn write.
"""
import contextlib
import fcntl
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SCRIPT_DIR, log  # noqa: E402


def _state_path(config: dict) -> Path:
    state_file = config.get("file_tracker", {}).get("state_file", "runtime/file_tracker.json")
    p = Path(state_file)
    if not p.is_absolute():
        p = SCRIPT_DIR / state_file
    return p


def _lock_path(config: dict) -> Path:
    """Sidecar lock file path (state file + '.lock')."""
    p = _state_path(config)
    return p.with_suffix(p.suffix + ".lock")


@contextlib.contextmanager
def _tracker_lock(config: dict):
    """Exclusive lock serializing tracker read-modify-write across processes."""
    p = _lock_path(config)
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "a+")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def _load_state(config: dict) -> dict:
    """Load the full tracker state dict. Returns {} on any error."""
    p = _state_path(config)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_state(config: dict, state: dict):
    """Persist tracker state atomically. Silently ignores IO errors."""
    p = _state_path(config)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)  # atomic on POSIX
    except (IOError, OSError):
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def normalize_path(file_path: str, cwd: str) -> str:
    """Resolve a tool-input path to an absolute, normalized path.

    Empty input -> empty string. Relative paths are joined against cwd.
    """
    if not file_path:
        return ""
    if os.path.isabs(file_path):
        return os.path.abspath(file_path)
    return os.path.abspath(os.path.join(cwd, file_path))


def _prune_old_sessions(state: dict, config: dict, session_id: str):
    """Drop oldest sessions until we are under max_sessions (keep current)."""
    max_sessions = config.get("file_tracker", {}).get("max_sessions", 50)
    sessions = list(state.keys())
    if len(sessions) < max_sessions:
        return
    # oldest entries first (insertion order); never drop the current session
    for old_sid in sessions[: len(sessions) - max_sessions + 1]:
        if old_sid != session_id:
            del state[old_sid]


def record_files_created(config: dict, session_id: str,
                         file_paths: Iterable[str], cwd: str):
    """Record that the agent created/modified the given files in this session.

    Called from the PostToolUse hook after a successful Write/Edit/NotebookEdit
    or a file-creating Bash command. Batched + locked so concurrent writers
    never lose records or corrupt the JSON.
    """
    if not session_id:
        return
    abs_paths = [normalize_path(p, cwd) for p in file_paths if p]
    abs_paths = [p for p in abs_paths if p]
    if not abs_paths:
        return

    with _tracker_lock(config):
        state = _load_state(config)
        _prune_old_sessions(state, config, session_id)

        ss = state.get(session_id, [])
        changed = False
        for ap in abs_paths:
            if ap not in ss:
                ss.append(ap)
                changed = True
        if changed:
            max_files = config.get("file_tracker", {}).get("max_files_per_session", 200)
            if len(ss) > max_files:
                ss = ss[-max_files:]
            state[session_id] = ss
            _save_state(config, state)
            log(config, "TRACK", f"recorded {len(abs_paths)} path(s) for {session_id}")


def record_file_created(config: dict, session_id: str, file_path: str, cwd: str):
    """Record a single created file (convenience wrapper around the batch API)."""
    record_files_created(config, session_id, [file_path], cwd)


def was_created_by_agent(config: dict, session_id: str, file_path: str, cwd: str) -> bool:
    """Check if file_path was previously recorded for this session."""
    if not file_path or not session_id:
        return False
    abs_path = normalize_path(file_path, cwd)
    if not abs_path:
        return False

    state = _load_state(config)
    ss = state.get(session_id, [])
    return abs_path in ss


def get_tracked_files(config: dict, session_id: str) -> Set[str]:
    """Return the set of absolute paths tracked for a session."""
    state = _load_state(config)
    return set(state.get(session_id, []))


def prune_session(config: dict, session_id: str):
    """Remove tracking data for a session (cleanup on session end)."""
    if not session_id:
        return
    with _tracker_lock(config):
        state = _load_state(config)
        if session_id in state:
            del state[session_id]
            _save_state(config, state)
