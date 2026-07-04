#!/usr/bin/env python3
"""hook_permission_request.py - PermissionRequest hook entry point.

Fires when Claude Code is about to show a permission dialog to the human
(in default/plan/acceptEdits modes, or when auto-mode classifier falls back to
manual, or when an `ask` rule matches). The reviewer agent replaces the human
Yes/No.

Output:
  allow -> {"hookSpecificOutput":{"hookEventName":"PermissionRequest","decision":{"behavior":"allow"}}}
  deny  -> {"hookSpecificOutput":{"hookEventName":"PermissionRequest","decision":{"behavior":"deny","message":"..."}}}
  passthrough (let human decide) -> print nothing, exit 0
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    load_config, log, read_hook_input, resolve_session_id,
    circuit_breaker_check, circuit_breaker_record, record_approval,
)
from policy import (  # noqa: E402
    check_hard_deny, check_agent_created_file, check_fast_path_allow,
)
from reviewer import review  # noqa: E402
from transcript import extract_context  # noqa: E402


def _emit_allow():
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PermissionRequest",
        "decision": {"behavior": "allow"}}}))


def _emit_deny(message: str):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PermissionRequest",
        "decision": {"behavior": "deny", "message": message}}}))


def _record(config: dict, hook_input: dict, decision: str, source: str, reason: str = ""):
    record_approval(config, {
        "hook_event": "PermissionRequest",
        "decision": decision,
        "source": source,
        "reason": reason,
        "session_id": resolve_session_id(hook_input),
        "permission_mode": hook_input.get("permission_mode", "default"),
        "cwd": hook_input.get("cwd", os.getcwd()),
        "transcript_path": hook_input.get("transcript_path", ""),
        "tool_name": hook_input.get("tool_name", ""),
        "tool_input": hook_input.get("tool_input", {}) or {},
    })


def main():
    try:
        config = load_config()
    except Exception as e:
        print(f"[reviewer] config load failed: {e}", file=sys.stderr)
        sys.exit(0)  # fail-open on config error: let human decide

    try:
        hook_input = read_hook_input()
    except Exception as e:
        log(config, "ERROR", f"hook input parse failed: {e}")
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {}) or {}
    cwd = hook_input.get("cwd", os.getcwd())
    transcript_path = hook_input.get("transcript_path", "")
    session_id = resolve_session_id(hook_input)
    permission_mode = hook_input.get("permission_mode", "default")

    log(config, "INFO",
        f"PermissionRequest tool={tool_name} mode={permission_mode}")

    # 1) hard red-lines -> deny
    deny_reason = check_hard_deny(tool_name, tool_input, config)
    if deny_reason:
        circuit_breaker_record(config, session_id, denied=True)
        log(config, "DENY", f"hard: {tool_name} - {deny_reason}")
        _record(config, hook_input, "deny", "hard_deny", deny_reason)
        _emit_deny(f"[reviewer] {deny_reason}")
        return

    # 2) circuit breaker tripped -> let human take over
    cb_warn = circuit_breaker_check(config, session_id)
    if cb_warn:
        log(config, "WARN", f"circuit breaker: {cb_warn}")
        _record(config, hook_input, "passthrough", "circuit_breaker", cb_warn)
        sys.exit(0)

    # 3) agent-created file -> allow (the agent can modify its own files)
    if check_agent_created_file(tool_name, tool_input, cwd, config, session_id):
        circuit_breaker_record(config, session_id, denied=False)
        log(config, "ALLOW", f"agent-created: {tool_name}")
        _record(config, hook_input, "allow", "agent_created", "agent-created file")
        _emit_allow()
        return

    # 4) fast-path allow (read-only) -> proactively allow to avoid the dialog
    if check_fast_path_allow(tool_name, tool_input, config):
        circuit_breaker_record(config, session_id, denied=False)
        log(config, "ALLOW", f"fast: {tool_name}")
        _record(config, hook_input, "allow", "fast_path", "fast-path allow")
        _emit_allow()
        return

    # 5) LLM review
    try:
        ctx = extract_context(
            transcript_path,
            turns_to_read=config.get("turns_to_read", 2),
            max_user_chars=config.get("max_user_prompt_chars", 500),
            max_assistant_chars=config.get("max_assistant_text_chars", 1000),
        )
    except Exception as e:
        log(config, "ERROR", f"transcript extract failed: {e}")
        ctx = {}

    try:
        result = review(tool_name, tool_input, cwd, ctx, config, permission_mode)
    except Exception as e:
        if config.get("fail_closed", True):
            circuit_breaker_record(config, session_id, denied=True)
            _record(config, hook_input, "deny", "review_error", str(e))
            _emit_deny(f"[reviewer] review failed (fail-closed): {e}")
        else:
            _record(config, hook_input, "passthrough", "review_error", str(e))
            sys.exit(0)
        return

    decision = result["decision"]
    reason = result.get("reason", "")
    circuit_breaker_record(config, session_id, denied=(decision == "deny"))

    if decision == "allow":
        log(config, "ALLOW", f"llm: {tool_name} - {reason}")
        _record(config, hook_input, "allow", "llm", reason)
        _emit_allow()
    elif decision == "deny":
        log(config, "DENY", f"llm: {tool_name} - {reason}")
        _record(config, hook_input, "deny", "llm", reason)
        _emit_deny(f"[reviewer] {reason}")
    else:  # ask -> passthrough to human
        log(config, "ASK", f"llm: {tool_name} - {reason}")
        _record(config, hook_input, "passthrough", "llm_ask", reason)
        sys.exit(0)


if __name__ == "__main__":
    main()
