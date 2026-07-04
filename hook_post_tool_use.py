#!/usr/bin/env python3
"""hook_post_tool_use.py - PostToolUse hook entry point.

Fires after every tool execution. Records files created/modified by the agent
into the per-session tracker, so that later permission checks can fast-path
allow operations on files the agent created itself
("删 agent 自己刚创建的文件=放行").

Tracked tools:
  - Write / Edit / NotebookEdit : the target file path.
  - Bash : file-creating commands are parsed best-effort and every created
    path is recorded. Covers `touch`, `mkdir`, redirection (`>`, `>>`),
    `tee`, and `cp`/`mv`/`install` destinations. Other bash commands are a
    no-op (no path extracted).

This hook never blocks — it only records. It passes through (exit 0, no output)
for every tool call. If the tracker write fails, it logs and exits 0 silently;
worst case the LLM reviewer makes the call instead (tracking is an
optimization, not a security boundary).
"""
import json
import os
import re
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, log, read_hook_input, resolve_session_id  # noqa: E402
from tracker import normalize_path, record_files_created  # noqa: E402

# Tools whose file_path / notebook_path we track directly.
WRITE_TOOLS = {"Write", "Edit", "NotebookEdit"}

# Bash verbs that create file(s); handled with verb-specific operand parsing.
_MULTI_OPERAND_CREATORS = {"touch", "mkdir", "tee"}
# cp / mv / install : only the LAST operand (destination) is the created path.
_DEST_CREATORS = {"cp", "mv", "install"}
# Shell separators for splitting compound commands before parsing.
_SUBCMD_SPLIT_RE = re.compile(r"(?:&&|\|\||;|\|)")
# Redirection target: `> file`, `>> file`, `>file` (skip `>&N` / `2>&1`).
# Run against a quote-stripped copy so `>` inside a quoted string is ignored.
_REDIRECT_RE = re.compile(r"(?:>>|>)\s*([^\s;|&<>]+)")
_QUOTE_RE = re.compile(r'"[^"]*"|\'[^\']*\'')


def _extract_path(tool_name: str, tool_input: dict) -> str:
    """Extract the target file path from Write/Edit/NotebookEdit input.

    `file_path` is the documented field; `path` is kept as a fallback so
    tracking matches whatever field shape a given Claude Code version uses
    (symmetric with the hard-deny / agent-created checks in policy.py).
    """
    if tool_name in ("Write", "Edit"):
        return tool_input.get("file_path") or tool_input.get("path") or ""
    if tool_name == "NotebookEdit":
        return tool_input.get("notebook_path") or tool_input.get("path") or ""
    return ""


def _extract_created_paths_from_bash(command: str, cwd: str) -> list:
    """Best-effort extraction of paths a bash command creates.

    Returns absolute, normalized paths. False negatives are acceptable (an
    untracked file just falls through to the LLM reviewer); false positives are
    benign (a wrongly-tracked path can only let the agent touch a file it
    itself referenced). The hard-deny layer is the real security backstop.
    """
    paths = []

    # 1) Redirection targets (`> foo`, `>> foo`, `>foo`) on a quote-stripped
    #    copy so `>` inside a string literal is not mistaken for a redirect.
    stripped = _QUOTE_RE.sub('""', command)
    for m in _REDIRECT_RE.finditer(stripped):
        tgt = m.group(1)
        if tgt and not tgt.startswith("&"):
            paths.append(normalize_path(tgt, cwd))

    # 2) Verb-based creators, split per sub-command so flags stay scoped.
    for sub in _SUBCMD_SPLIT_RE.split(command):
        sub = sub.strip()
        if not sub:
            continue
        try:
            tokens = shlex.split(_QUOTE_RE.sub('""', sub))
        except ValueError:
            tokens = sub.split()
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in _DEST_CREATORS and i + 1 < len(tokens):
                # Skip flags, collect operands; destination is the last one.
                ops = [t for t in tokens[i + 1:] if not t.startswith("-")]
                if ops:
                    paths.append(normalize_path(ops[-1], cwd))
                break  # rest of this sub-command belongs to this verb
            if tok in _MULTI_OPERAND_CREATORS and i + 1 < len(tokens):
                j = i + 1
                while j < len(tokens) and tokens[j].startswith("-"):
                    j += 1
                while j < len(tokens) and not tokens[j].startswith("-"):
                    paths.append(normalize_path(tokens[j], cwd))
                    j += 1
                break
            i += 1

    return [p for p in paths if p]


def main():
    try:
        config = load_config()
    except Exception:
        sys.exit(0)  # never block

    if not config.get("file_tracker", {}).get("enabled", True):
        sys.exit(0)

    try:
        hook_input = read_hook_input()
    except Exception:
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")
    session_id = resolve_session_id(hook_input)
    cwd = hook_input.get("cwd", os.getcwd())

    if tool_name not in WRITE_TOOLS and tool_name != "Bash":
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {}) or {}

    if tool_name == "Bash":
        command = tool_input.get("command", "") or ""
        if not command:
            sys.exit(0)
        paths = _extract_created_paths_from_bash(command, cwd)
    else:
        p = _extract_path(tool_name, tool_input)
        paths = [p] if p else []

    if not paths:
        sys.exit(0)

    try:
        record_files_created(config, session_id, paths, cwd)
        log(config, "INFO", f"PostToolUse tracked: {tool_name} -> {paths}")
    except Exception as e:
        log(config, "ERROR", f"PostToolUse track failed: {e}")

    sys.exit(0)  # always passthrough


if __name__ == "__main__":
    main()
