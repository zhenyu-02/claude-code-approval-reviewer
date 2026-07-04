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
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class Turn:
    user_prompt: str = ""
    assistant_text: str = ""
    tool_calls: List[str] = field(default_factory=list)


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


def _extract_assistant_text(content) -> Tuple[str, List[str]]:
    """Return (pure_text, tool_call_names). Excludes thinking blocks."""
    if isinstance(content, str):
        return content.strip(), []
    if isinstance(content, list):
        texts, tool_names = [], []
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                texts.append(b.get("text", ""))
            elif t == "tool_use":
                tool_names.append(b.get("name", ""))
            # "thinking" blocks are intentionally skipped
        return "\n".join(s for s in texts if s).strip(), tool_names
    return "", []


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
    """Group transcript entries into turns (one user prompt + following assistant reply)."""
    turns: List[Turn] = []
    current: Turn = None
    for entry in entries:
        etype = entry.get("type")
        message = entry.get("message", {}) or {}
        role = message.get("role", etype)
        content = message.get("content")

        if role == "user" or etype == "user":
            if _is_real_user_message(content):
                if current is not None:
                    turns.append(current)
                current = Turn(user_prompt=_extract_user_prompt(content))
        elif role == "assistant" or etype == "assistant":
            if current is None:
                current = Turn()
            text, tool_names = _extract_assistant_text(content)
            if text:
                current.assistant_text = (current.assistant_text + "\n" + text).strip() \
                    if current.assistant_text else text
            current.tool_calls.extend(tool_names)
    if current is not None:
        turns.append(current)
    return turns


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "...[truncated]"


def extract_context(transcript_path: str, turns_to_read: int = 2,
                    max_user_chars: int = 500,
                    max_assistant_chars: int = 1000) -> dict:
    """Build the reviewer context dict.

    Returns:
      {
        "recent_user_prompts":   [str, ...],   # last N turns' human prompts
        "recent_assistant_texts":[str, ...],   # last N turns' assistant pure text
        "recent_tool_calls":     [str, ...],   # recent tool names (deduped, max 10)
      }
    """
    if not transcript_path:
        return {"recent_user_prompts": [], "recent_assistant_texts": [], "recent_tool_calls": []}
    entries = parse_transcript(transcript_path)
    turns = build_turns(entries)
    recent = turns[-turns_to_read:] if len(turns) > turns_to_read else turns

    user_prompts, assistant_texts, all_tools = [], [], []
    for t in recent:
        if t.user_prompt:
            user_prompts.append(_truncate(t.user_prompt, max_user_chars))
        if t.assistant_text:
            assistant_texts.append(_truncate(t.assistant_text, max_assistant_chars))
        all_tools.extend(t.tool_calls)

    seen, recent_tools = set(), []
    for name in reversed(all_tools):
        if name and name not in seen:
            seen.add(name)
            recent_tools.append(name)
        if len(recent_tools) >= 10:
            break
    recent_tools.reverse()

    return {
        "recent_user_prompts": user_prompts,
        "recent_assistant_texts": assistant_texts,
        "recent_tool_calls": recent_tools,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 transcript.py <transcript_path> [turns]", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    ctx = extract_context(path, turns_to_read=n)
    print(json.dumps(ctx, indent=2, ensure_ascii=False))
