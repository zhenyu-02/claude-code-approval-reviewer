#!/usr/bin/env python3
"""common.py - shared utilities: config loading, logging, circuit breaker."""
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent


def load_config() -> dict:
    """Load config. Priority: REVIEWER_CONFIG env var > config.json next to script."""
    config_path = os.environ.get("REVIEWER_CONFIG")
    if not config_path:
        config_path = str(SCRIPT_DIR / "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def log(config: dict, level: str, message: str):
    """Append a log line."""
    log_file = config.get("log_file")
    if not log_file:
        return
    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = SCRIPT_DIR / log_file
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [{level}] {message}\n")
    except Exception:
        pass


def _resolve_script_path(path_value: str) -> Path:
    p = Path(path_value)
    if not p.is_absolute():
        p = SCRIPT_DIR / path_value
    return p


def _redact_text(text: str, config: dict) -> str:
    redacted = text
    for marker in config.get("approval_records", {}).get("redact_markers", []):
        if marker:
            redacted = redacted.replace(marker, "[REDACTED]")
    return redacted


def _safe_json_value(value, config: dict):
    max_chars = config.get("approval_records", {}).get("max_tool_input_chars", 4000)
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        text = json.dumps(str(value), ensure_ascii=False)
    text = _redact_text(text, config)
    if len(text) > max_chars:
        return {
            "truncated": True,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "preview": text[:max_chars],
        }
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def record_approval(config: dict, record: dict):
    """Append one structured approval decision as JSONL.

    This is deliberately best-effort: approval hooks must not fail only because
    audit logging is unavailable. Concurrent writers are serialized with flock.
    """
    rec_cfg = config.get("approval_records", {})
    if not rec_cfg.get("enabled", False):
        return
    record_file = rec_cfg.get("file", "runtime/approval_records.jsonl")
    record_path = _resolve_script_path(record_file)
    lock_path = record_path.with_suffix(record_path.suffix + ".lock")
    try:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "schema_version": 1,
            **record,
        }
        if "tool_input" in payload:
            payload["tool_input"] = _safe_json_value(payload["tool_input"], config)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with open(lock_path, "a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with open(record_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            fcntl.flock(lock, fcntl.LOCK_UN)
    except Exception as e:
        log(config, "ERROR", f"approval record write failed: {e}")


def read_hook_input() -> dict:
    """Read hook input JSON from stdin."""
    data = sys.stdin.read()
    return json.loads(data) if data.strip() else {}


def resolve_session_id(hook_input: dict) -> str:
    """Pick a stable session id for per-session tracker state.

    Claude Code supplies `session_id` on hook input; that is the correct key and
    is used as-is when present. The fallback matters because each hook call is a
    fresh process, so the id MUST be stable across all calls in one session —
    using the process pid would be wrong (it changes every call and would split
    one session into N single-call sessions, defeating tracking entirely).

    Fallback chain:
      1. transcript_path — one transcript per session, stable across calls;
      2. cwd — last resort, scopes tracking to the working directory.
    """
    sid = (hook_input.get("session_id") or "").strip()
    if sid:
        return sid
    tp = (hook_input.get("transcript_path") or "").strip()
    if tp:
        return "tx:" + tp
    return "cwd:" + (hook_input.get("cwd") or os.getcwd())


def _cb_state_path(config: dict) -> Path:
    cb_cfg = config.get("circuit_breaker", {})
    state_file = cb_cfg.get("state_file", "runtime/circuit_breaker.json")
    p = Path(state_file)
    if not p.is_absolute():
        p = SCRIPT_DIR / state_file
    return p


def circuit_breaker_check(config: dict, session_id: str) -> Optional[str]:
    """Return warning message if breaker tripped for this session, else None."""
    cb_cfg = config.get("circuit_breaker", {})
    if not cb_cfg.get("enabled", False):
        return None
    p = _cb_state_path(config)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
    ss = state.get(session_id, {})
    if ss.get("interrupted"):
        return (f"circuit breaker tripped "
                f"(consecutive={ss.get('consecutive_denials', 0)}, "
                f"window_denials={sum(ss.get('recent', []))})")
    return None


def circuit_breaker_record(config: dict, session_id: str, denied: bool):
    """Record one review outcome; trip breaker if thresholds exceeded."""
    cb_cfg = config.get("circuit_breaker", {})
    if not cb_cfg.get("enabled", False):
        return
    p = _cb_state_path(config)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(p, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (json.JSONDecodeError, IOError):
        state = {}

    ss = state.get(session_id, {"consecutive_denials": 0, "recent": [], "interrupted": False})
    ss["consecutive_denials"] = 0 if not denied else ss.get("consecutive_denials", 0) + 1
    ss["recent"] = ss.get("recent", []) + [denied]
    window = cb_cfg.get("window_size", 50)
    if len(ss["recent"]) > window:
        ss["recent"] = ss["recent"][-window:]

    max_consec = cb_cfg.get("max_consecutive_denials", 3)
    max_in_window = cb_cfg.get("max_denials_in_window", 10)
    denials_in_window = sum(ss["recent"])
    if not ss.get("interrupted") and (
        ss["consecutive_denials"] >= max_consec or denials_in_window >= max_in_window
    ):
        ss["interrupted"] = True

    state[session_id] = ss
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except IOError:
        pass
