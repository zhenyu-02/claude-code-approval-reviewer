# Claude Code Approval Reviewer

A Claude Code hook that reviews permission requests with an LLM before allowing
or denying tool calls. It is designed for Claude Code's `PermissionRequest`,
`PreToolUse`, and `PostToolUse` hook events.

The reviewer can make deterministic decisions for hard-deny and fast-path cases,
then use a small Anthropic Messages API compatible reviewer agent for the rest.
It also writes structured JSONL approval records for later audit and drift
analysis.

## What It Does

- Denies hard red-line commands such as `rm -rf /`, `curl | sh`, force-push to
  protected branches, and writes to obvious secret paths.
- Allows low-risk read-only commands and known safe commands without an LLM call.
- Tracks files created by the agent in the current session, so later edits or
  deletes of agent-created files can be fast-pathed.
- Calls an LLM reviewer for ambiguous requests. The reviewer can inspect local
  state with read-only `Read`, `Grep`, and `Glob` tools.
- Persists audit records to `runtime/approval_records.jsonl`. Runtime files are
  ignored by Git and must not be committed.

## Requirements

- Claude Code with hook support.
- Python 3.9+; no third-party Python packages are required.
- `rg` is recommended for the reviewer's grep tool, with `grep` as fallback.
- An Anthropic Messages API compatible endpoint.

Important: direct DeepSeek OpenAI-compatible API calls are not implemented yet.
To use DeepSeek, run it behind a Claude/Anthropic-compatible proxy and set
`ANTHROPIC_BASE_URL` to that proxy.

## MacBook Install

Clone into Claude's global hook directory:

```bash
git clone <repo-url> ~/.claude/hooks/reviewer
cd ~/.claude/hooks/reviewer
cp .env.example .env
```

Edit `~/.claude/hooks/reviewer/.env` and set your endpoint variables. Then load
them in your shell startup file, for example `~/.zshrc` on macOS:

```bash
source ~/.claude/hooks/reviewer/.env
```

Merge `settings.global.example.json` into `~/.claude/settings.json`. If you
already have hooks configured, merge the `hooks` object instead of replacing the
whole file.

For project-local installation, clone into `.claude/hooks/reviewer` inside the
project and merge `settings.example.json` into that project's `.claude/settings.json`.

## Configuration

Main configuration lives in `config.json`. Useful keys:

| key | meaning |
| --- | --- |
| `llm.api_key_env` | environment variable for the API key, default `ANTHROPIC_API_KEY` |
| `llm.base_url_env` | environment variable for a compatible proxy URL, default `ANTHROPIC_BASE_URL` |
| `llm.model` | default reviewer model when `ANTHROPIC_MODEL` is unset |
| `fast_path_allow_bash` | regex allowlist for low-risk shell commands |
| `hard_deny_patterns` | regex denylist for high-risk shell commands |
| `approval_records.file` | JSONL audit stream path |
| `file_tracker.state_file` | per-session agent-created file tracker |

The default runtime paths are under `runtime/`, which is ignored by Git.

## Dry Run

```bash
echo '{"tool_name":"Bash","tool_input":{"command":"git status --short"},"cwd":"'"$PWD"'","permission_mode":"default","session_id":"dryrun"}' \
  | python3 hook_permission_request.py
```

You should see an allow decision. The audit record will appear in
`runtime/approval_records.jsonl`.

## Security Notes

- Do not commit `.env`, `runtime/`, approval records, logs, transcript files, or
  Claude Code history.
- Approval records may include command arguments and local paths. Treat them as
  private audit data.
- The reviewer reads recent transcript text and pending tool input as untrusted
  evidence, not instructions.
- The LLM reviewer fails closed by default when calls fail or output cannot be
  parsed.

## License

Apache-2.0.
