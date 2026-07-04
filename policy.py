#!/usr/bin/env python3
"""policy.py - hard red-line fast paths (deterministic, no LLM cost).

Two fast paths that skip the LLM reviewer:
  - check_hard_deny:  matches -> deny immediately (rm -rf /, .env, curl|sh, ...)
  - check_fast_path_allow: matches -> allow (read-only tools, git status, pytest, ...)

Everything else falls through to the LLM reviewer.
"""
import os
import re
import shlex
from typing import Optional


# Split a compound bash command into sub-commands on shell separators.
# (Best effort: does not track cwd changes from `cd`, see README limitations.)
_SUBCMD_SPLIT_RE = re.compile(r"(?:&&|\|\||;|\|)")
# Shell chars to drop before tokenizing a single sub-command so shlex doesn't
# choke on redirection / pipes. `<` and `>` are stripped; redirect targets
# survive as ordinary operands (e.g. `echo hi > foo.txt` -> `foo.txt`).
_OP_STRIP_RE = re.compile(r"[<>]")


def _normalize_path(file_path: str, cwd: str) -> str:
    """Resolve a path to an absolute, normalized path (self-contained, no
    tracker import — keeps the Bash check working under lazy import)."""
    if not file_path:
        return ""
    if os.path.isabs(file_path):
        return os.path.abspath(file_path)
    return os.path.abspath(os.path.join(cwd, file_path))


def _tokenize_command(command: str) -> list:
    """Tokenize a single bash sub-command, dropping redirection chars."""
    cleaned = _OP_STRIP_RE.sub(" ", command)
    try:
        return shlex.split(cleaned)
    except ValueError:
        # Unbalanced quotes etc. — fall back to whitespace split.
        return cleaned.split()


def _bash_targets_tracked(command: str, cwd: str, tracked: set) -> bool:
    """True if a bash command only writes/deletes files the agent created.

    Tokenizes each sub-command, normalizes every non-flag operand to an absolute
    path, and exact-matches against `tracked` (no substring matching — that
    caused both false negatives (`rm ./foo` vs tracked `/abs/foo`) and false
    positives (`rm /abs/foo.bak` matching tracked `/abs/foo`)).

    Conservative for verbs whose written files are identifiable, so a command
    that also writes/deletes an UNTRACKED file is NOT fast-pathed (falls through
    to the LLM reviewer):
      - `rm`/`rmdir`/`unlink`/`tee` : write/delete EVERY named path -> all must
        be tracked (prevents `rm important.conf my.txt` being allowed just
        because `my.txt` is tracked, and `tee important.conf` overwriting it).
      - `cp`/`mv`/`install` : overwrite the DESTINATION (last operand) -> dst
        must be tracked (prevents `cp src important.conf`).
    Other verbs (echo/cat/sed/perl/... including redirection) only need one
    tracked operand; their non-path arguments make "all tracked" impractical,
    and the common redirect case (`echo hi > own.txt`) is safe because an
    untracked redirect target has no tracked operand to ride on. See README for
    residual gaps (cd-subdir, unexpanded vars/globs, in-place editors).
    """
    if not tracked:
        return False
    subs = [s for s in _SUBCMD_SPLIT_RE.split(command) if s and s.strip()]
    if not subs:
        return False
    touched_tracked = False
    for sub in subs:
        tokens = _tokenize_command(sub)
        if not tokens:
            continue
        verb = tokens[0]
        operands = [t for t in tokens[1:] if not t.startswith("-")]
        norm = {_normalize_path(t, cwd) for t in operands if t}
        if not norm:
            continue
        if verb in ("rm", "rmdir", "unlink", "tee"):
            # Writes/deletes every named path: require ALL tracked.
            if not norm.issubset(tracked):
                return False
        elif verb in ("cp", "mv", "install"):
            # Overwrites the destination (last operand): require it tracked.
            dst = _normalize_path(operands[-1], cwd) if operands else ""
            if not dst or dst not in tracked:
                return False
        if norm & tracked:
            touched_tracked = True
    return touched_tracked


def check_hard_deny(tool_name: str, tool_input: dict, config: dict) -> Optional[str]:
    """Return a deny reason if a hard red-line is hit, else None."""
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        for rule in config.get("hard_deny_patterns", []):
            if re.search(rule["pattern"], command):
                return rule["reason"]
        for forbidden in config.get("hard_deny_paths", []):
            if forbidden in command:
                return f"command touches sensitive path: {forbidden}"

    if tool_name in ("Read", "Edit", "Write", "NotebookEdit"):
        file_path = tool_input.get("file_path") or tool_input.get("path") or ""
        for forbidden in config.get("hard_deny_paths", []):
            if forbidden in file_path:
                return f"access to sensitive path: {forbidden}"
    return None


def check_agent_created_file(tool_name: str, tool_input: dict, cwd: str,
                              config: dict, session_id: str) -> bool:
    """True if the tool targets a file the agent created in this session.

    Implements user rule: "删 agent 自己刚创建的文件=放行".
    Covers Write/Edit/NotebookEdit targeting a tracked file, and Bash commands
    whose path operands exact-match tracked files (tokenized + normalized).
    """
    if not config.get("file_tracker", {}).get("enabled", True):
        return False
    # Lazy import: keeps hard_deny working even if tracker/common fail to
    # import, and avoids loading tracker on the many fast-path-only calls that
    # never reach this check. (There is no circular import — policy does not
    # import common at module top; tracker -> common is a linear chain.) The
    # Bash branch uses policy-local _normalize_path, not tracker's, so it does
    # not depend on these imports at all.
    from tracker import was_created_by_agent, get_tracked_files  # noqa: E402

    if tool_name in ("Write", "Edit"):
        # `file_path` is the documented field; `path` is kept as a fallback for
        # Claude Code versions / tools that use it (kept symmetric with the
        # hard-deny check so tracking and hard-deny never disagree on a path).
        file_path = tool_input.get("file_path") or tool_input.get("path") or ""
        if file_path and was_created_by_agent(config, session_id, file_path, cwd):
            return True
        return False
    elif tool_name == "NotebookEdit":
        file_path = tool_input.get("notebook_path") or tool_input.get("path") or ""
        if file_path and was_created_by_agent(config, session_id, file_path, cwd):
            return True
        return False
    elif tool_name == "Bash":
        command = tool_input.get("command", "")
        if not command:
            return False
        tracked = get_tracked_files(config, session_id)
        if not tracked:
            return False
        return _bash_targets_tracked(command, cwd, tracked)
    return False


def check_fast_path_allow(tool_name: str, tool_input: dict, config: dict) -> bool:
    """True if the call is read-only/safe and can skip the LLM reviewer."""
    if tool_name in config.get("fast_path_allow_tools", []):
        return True
    if tool_name == "Bash":
        command = (tool_input.get("command") or "").strip()
        for pattern in config.get("fast_path_allow_bash", []):
            if re.match(pattern, command):
                return True
    return False


def is_in_cwd(file_path: str, cwd: str) -> bool:
    if not file_path:
        return False
    abs_path = os.path.abspath(os.path.join(cwd, file_path)) \
        if not os.path.isabs(file_path) else os.path.abspath(file_path)
    return abs_path.startswith(os.path.abspath(cwd))
