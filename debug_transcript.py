#!/usr/bin/env python3
"""debug_transcript.py - inspect raw transcript JSONL and the parsed reviewer context.

Usage:
  python3 debug_transcript.py <transcript_path> [turns]

Prints:
  1) the first 3 raw JSONL lines (so you can verify the field layout matches
     what transcript.py expects)
  2) the extracted reviewer context (recent user prompts + assistant pure text)

If parsing looks wrong, compare the raw lines with the field assumptions in
transcript.py (_is_real_user_message / _extract_assistant_text) and adjust.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) if (os := __import__("os")) else None
from transcript import parse_transcript, build_turns, extract_context  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 debug_transcript.py <transcript_path> [turns]", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    p = Path(path)
    if not p.exists():
        print(f"file not found: {path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("RAW LINES (last 5 non-empty):")
    print("=" * 60)
    lines = [l for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    for line in lines[-5:]:
        try:
            obj = json.loads(line)
            # compact view of the key fields
            t = obj.get("type")
            msg = obj.get("message", {}) or {}
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, list):
                block_types = [b.get("type") for b in content if isinstance(b, dict)]
                print(f"  type={t} role={role} content=array[{block_types}]")
            else:
                preview = str(content)[:120] if content else "(empty)"
                print(f"  type={t} role={role} content={preview}")
        except json.JSONDecodeError:
            print(f"  (unparseable) {line[:120]}")

    print()
    print("=" * 60)
    print(f"PARSED TURNS (total={len(build_turns(parse_transcript(path)))}, showing last {n}):")
    print("=" * 60)
    turns = build_turns(parse_transcript(path))
    for i, t in enumerate(turns[-n:]):
        idx = len(turns) - n + i
        print(f"\n--- Turn {idx} ---")
        print(f"  USER: {t.user_prompt[:200]}{'...' if len(t.user_prompt) > 200 else ''}")
        print(f"  ASSISTANT: {t.assistant_text[:200]}{'...' if len(t.assistant_text) > 200 else ''}")
        print(f"  TOOLS: {t.tool_calls}")

    print()
    print("=" * 60)
    print("REVIEWER CONTEXT:")
    print("=" * 60)
    ctx = extract_context(path, turns_to_read=n)
    print(json.dumps(ctx, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
