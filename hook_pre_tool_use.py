#!/usr/bin/env python3
"""hook_pre_tool_use.py - PreToolUse hook entry point (bypass-mode safety net).

Only activates the reviewer when permission_mode is in pre_tool_use_active_modes
(default: ["bypassPermissions"]). In every other mode (default/acceptEdits/plan/
auto/dontAsk) it passes through so Claude Code's native flow (auto-mode classifier
+ the PermissionRequest hook) handles approval. This realises the user's design:
"only use PreToolUse when bypass is on; trust Claude Code's own judgement otherwise".

Output:
  deny  -> {"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"..."}}
  allow -> {"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","permissionDecisionReason":"..."}}
  ask   -> {"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"..."}}
  passthrough -> print nothing, exit 0
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


def _emit(decision: str, reason: str):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
        "permissionDecisionReason": reason}}))


def _record(config: dict, hook_input: dict, decision: str, source: str, reason: str = ""):
    record_approval(config, {
        "hook_event": "PreToolUse",
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
        sys.exit(0)

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

    # Only activate in configured modes (default: bypassPermissions only).
    active_modes = config.get("pre_tool_use_active_modes", ["bypassPermissions"])
    if permission_mode not in active_modes:
        _record(config, hook_input, "passthrough", "inactive_mode", "native flow handles this mode")
        sys.exit(0)  # let native flow handle it

    log(config, "INFO",
        f"PreToolUse(bypass net) tool={tool_name} mode={permission_mode}")

    # 1) hard red-lines -> deny (works even under bypassPermissions)
    deny_reason = check_hard_deny(tool_name, tool_input, config)
    if deny_reason:
        circuit_breaker_record(config, session_id, denied=True)
        log(config, "DENY", f"hard: {tool_name} - {deny_reason}")
        _record(config, hook_input, "deny", "hard_deny", deny_reason)
        _emit("deny", f"[reviewer] {deny_reason}")
        return

    # 2) circuit breaker
    cb_warn = circuit_breaker_check(config, session_id)
    if cb_warn:
        log(config, "WARN", f"circuit breaker: {cb_warn}")
        _record(config, hook_input, "ask", "circuit_breaker", cb_warn)
        _emit("ask", f"[reviewer] {cb_warn}")
        return

    # 3) agent-created file -> under bypass it would pass anyway; stay silent
    if check_agent_created_file(tool_name, tool_input, cwd, config, session_id):
        circuit_breaker_record(config, session_id, denied=False)
        log(config, "ALLOW", f"agent-created: {tool_name}")
        _record(config, hook_input, "passthrough", "agent_created", "bypass mode already permits tracked agent-created file")
        sys.exit(0)

    # 4) fast-path allow -> under bypass it would pass anyway; stay silent
    if check_fast_path_allow(tool_name, tool_input, config):
        circuit_breaker_record(config, session_id, denied=False)
        log(config, "ALLOW", f"fast: {tool_name}")
        _record(config, hook_input, "passthrough", "fast_path", "bypass mode already permits fast-path command")
        sys.exit(0)

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
            _emit("deny", f"[reviewer] review failed (fail-closed): {e}")
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
        _emit("allow", reason)
    elif decision == "deny":
        log(config, "DENY", f"llm: {tool_name} - {reason}")
        _record(config, hook_input, "deny", "llm", reason)
        _emit("deny", f"[reviewer] {reason}")
    else:  # ask -> force a human dialog even under bypass
        log(config, "ASK", f"llm: {tool_name} - {reason}")
        _record(config, hook_input, "ask", "llm_ask", reason)
        _emit("ask", f"[reviewer] needs human: {reason}")


if __name__ == "__main__":
    main()
