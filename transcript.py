#!/usr/bin/env python3
"""transcript.py - parse Claude Code transcript JSONL into reviewer context.

Extracts what the user wants the reviewer to see (≈ what the human user sees):
  - recent N turns of user prompts (real human input, NOT tool_result passbacks)
  - recent N turns of assistant pure text (text blocks, NOT thinking/reasoning)
  - recent tool-call names (no inputs, no results - kept thin on purpose)

Claude Code transcript JSONL: each line is one JSON object with fields like
  - type: "user" | "assistant" | "system" | ...
  - message: {role, content, id, ...}
      content may be a string OR an array of blocks
      assistant blocks: {type: "text"|"thinking"|"tool_use", ...}  -> we keep only "text"
      user blocks: may contain {type: "tool_result", ...} passbacks -> we drop those rows

Run `python3 transcript.py <path> [N]` in debug mode to inspect parsing.
"""
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class Turn:
    user_prompt: str = ""
    assistant_text: str = ""
    tool_uses: list = field(default_factory=list)      # [{id, name, input}]
    tool_results: dict = field(default_factory=dict)   # {tool_use_id: result_text}


def _is_real_user_message(content) -> bool:
    """Distinguish real human input from tool_result passbacks wrapped as user role."""
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        has_tool_result = any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )
        has_text = any(
            isinstance(b, dict) and b.get("type") == "text" for b in content
        )
        return has_text and not has_tool_result
    return False


def _extract_user_prompt(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(t for t in texts if t).strip()
    return ""


def _extract_assistant_text(content) -> Tuple[str, list]:
    """Return (pure_text, tool_uses). Excludes thinking blocks.
    tool_uses: list of {id, name, input}."""
    if isinstance(content, str):
        return content.strip(), []
    if isinstance(content, list):
        texts, tool_uses = [], []
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                texts.append(b.get("text", ""))
            elif t == "tool_use":
                tool_uses.append({
                    "id": b.get("id", ""),
                    "name": b.get("name", ""),
                    "input": b.get("input", {}),
                })
            # "thinking" blocks are intentionally skipped
        return "\n".join(s for s in texts if s).strip(), tool_uses
    return "", []


def _extract_tool_results(content) -> dict:
    """Return {tool_use_id: result_text} from a user message's tool_result blocks."""
    out = {}
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict) or b.get("type") != "tool_result":
                continue
            tid = b.get("tool_use_id", "")
            raw = b.get("content", "")
            if isinstance(raw, list):
                # content may be a list of text blocks
                raw = "\n".join(
                    x.get("text", "") for x in raw
                    if isinstance(x, dict) and x.get("type") == "text")
            if not isinstance(raw, str):
                raw = json.dumps(raw, ensure_ascii=False) if raw is not None else ""
            if tid:
                out[tid] = raw
    return out


def parse_transcript(path: str, tail_bytes: int = 256 * 1024) -> list:
    """Read transcript JSONL tail (default 256KB) and parse each line."""
    p = Path(path)
    if not p.exists():
        return []
    entries = []
    size = p.stat().st_size
    offset = max(0, size - tail_bytes)
    with p.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        if offset > 0:
            f.readline()  # drop possibly-truncated first line
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def build_turns(entries: list) -> List[Turn]:
    """Group transcript entries into turns (one user prompt + following assistant reply).

    Captures assistant tool_use blocks (name + input + id) and the matching
    tool_result passbacks (paired by tool_use_id). tool_results live in user-role
    messages that _is_real_user_message filters out as "not real human input";
    we still mine them for evidence before discarding the message as a prompt.
    """
    turns: List[Turn] = []
    current: Turn = None
    for entry in entries:
        etype = entry.get("type")
        message = entry.get("message", {}) or {}
        role = message.get("role", etype)
        content = message.get("content")

        if role == "user" or etype == "user":
            # tool_result passback: capture evidence, do NOT start a new turn
            results = _extract_tool_results(content)
            if results and current is not None:
                current.tool_results.update(results)
            if _is_real_user_message(content):
                if current is not None:
                    turns.append(current)
                current = Turn(user_prompt=_extract_user_prompt(content))
        elif role == "assistant" or etype == "assistant":
            if current is None:
                current = Turn()
            text, tool_uses = _extract_assistant_text(content)
            if text:
                current.assistant_text = (current.assistant_text + "\n" + text).strip() \
                    if current.assistant_text else text
            current.tool_uses.extend(tool_uses)
    if current is not None:
        turns.append(current)
    return turns


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "...[truncated]"


def _spill_result_to_tempfile(text: str) -> str:
    fd, path = tempfile.mkstemp(prefix="reviewer_ctx_result_", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as f:
            f.write(text)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return path


def extract_context(transcript_path: str, turns_to_read: int = 2,
                    max_user_chars: int = 500,
                    max_assistant_chars: int = 1000,
                    tool_result_spill_threshold: int = 4000,
                    tool_result_preview_chars: int = 2000):
    """Build the reviewer context dict.

    Returns (context_dict, temp_files_to_cleanup).

    recent_tool_calls now keeps BOTH the call (name + input) AND the matching
    tool_result, so the reviewer has the evidence the agent gathered (Codex-style).
    Long tool_results are spilled to a temp file and only a small preview +
    sha256 + size + temp path is kept inline; the reviewer can Read/Grep the temp
    file. Caller must delete temp_files when done.
    """
    empty = {"recent_user_prompts": [], "recent_assistant_texts": [],
             "recent_tool_calls": []}
    if not transcript_path:
        return empty, []
    entries = parse_transcript(transcript_path)
    turns = build_turns(entries)
    recent = turns[-turns_to_read:] if len(turns) > turns_to_read else turns

    user_prompts, assistant_texts, tool_interactions = [], [], []
    temp_files = []
    for t in recent:
        if t.user_prompt:
            user_prompts.append(_truncate(t.user_prompt, max_user_chars))
        if t.assistant_text:
            assistant_texts.append(_truncate(t.assistant_text, max_assistant_chars))
        for tu in t.tool_uses:
            name = tu.get("name", "")
            tid = tu.get("id", "")
            raw_result = t.tool_results.get(tid, "") if tid else ""
            result_field = None
            if raw_result:
                if len(raw_result) > tool_result_spill_threshold:
                    tmp = _spill_result_to_tempfile(raw_result)
                    temp_files.append(tmp)
                    preview = raw_result[:tool_result_preview_chars]
                    if len(raw_result) > tool_result_preview_chars:
                        preview += "...[truncated]"
                    result_field = {
                        "_truncated": True,
                        "preview": preview,
                        "sha256": hashlib.sha256(raw_result.encode("utf-8")).hexdigest(),
                        "size_bytes": len(raw_result.encode("utf-8")),
                        "full_content_path": tmp,
                    }
                else:
                    result_field = raw_result
            tool_interactions.append({
                "name": name,
                "input": tu.get("input", {}),
                "result": result_field,
            })

    return {
        "recent_user_prompts": user_prompts,
        "recent_assistant_texts": assistant_texts,
        "recent_tool_calls": tool_interactions,
    }, temp_files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 transcript.py <transcript_path> [turns]", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    ctx, temps = extract_context(path, turns_to_read=n)
    try:
        print(json.dumps(ctx, indent=2, ensure_ascii=False))
    finally:
        for tf in temps:
            try:
                os.remove(tf)
            except OSError:
                pass
