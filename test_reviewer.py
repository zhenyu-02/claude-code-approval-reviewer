#!/usr/bin/env python3
"""test_reviewer.py - regression tests for the Claude Code approval reviewer.

Guard against the class of bug where a module alias gets shadowed by a
same-named function. The original incident:

    import glob as _glob
    def _glob(pattern): ...
        matches = _glob.glob(...)   # -> 'function' object has no attribute 'glob'

That made execute_tool("Glob", ...) raise and force fail-closed denies.

Run:
    python3 -m unittest test_reviewer -v
    python3 test_reviewer.py
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reviewer  # noqa: E402
import transcript  # noqa: E402


class ExecuteToolTests(unittest.TestCase):
    """Exercise every read-only investigation tool the reviewer LLM can call.

    If any of these raises, the hook fails closed in production. So each tool
    must return a string (possibly an error *string*) and never raise.
    """

    def setUp(self):
        self._orig_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix="reviewer_test_")
        with open(os.path.join(self.tmp, "alpha.py"), "w") as f:
            f.write("print('hello world')\n")
        with open(os.path.join(self.tmp, "beta.md"), "w") as f:
            f.write("# notes\nSECRET_TOKEN=abc\n")
        os.makedirs(os.path.join(self.tmp, "sub"), exist_ok=True)
        with open(os.path.join(self.tmp, "sub", "gamma.py"), "w") as f:
            f.write("x = 42\n")

    def tearDown(self):
        os.chdir(self._orig_cwd)

    def test_glob_non_recursive(self):
        out = reviewer.execute_tool("Glob", {"pattern": "*.py"}, self.tmp)
        self.assertIsInstance(out, str)
        self.assertIn("alpha.py", out)
        self.assertNotIn("[no matches]", out)

    def test_glob_recursive(self):
        out = reviewer.execute_tool("Glob", {"pattern": "**/*.py"}, self.tmp)
        self.assertIn("alpha.py", out)
        self.assertIn("gamma.py", out)

    def test_glob_no_matches(self):
        out = reviewer.execute_tool("Glob", {"pattern": "nope_*.xyz"}, self.tmp)
        self.assertIn("[no matches]", out)

    def test_read_file(self):
        out = reviewer.execute_tool(
            "Read", {"path": os.path.join(self.tmp, "alpha.py")}, self.tmp)
        self.assertIn("hello world", out)

    def test_grep_finds_match(self):
        out = reviewer.execute_tool(
            "Grep", {"pattern": "SECRET_TOKEN", "path": self.tmp}, self.tmp)
        self.assertIn("SECRET_TOKEN", out)

    def test_unknown_tool(self):
        out = reviewer.execute_tool("Frobnicate", {}, self.tmp)
        self.assertIn("[unknown tool", out)


class NoShadowingTests(unittest.TestCase):
    """Directly guard against module-alias / function-name shadowing regressions."""

    def test_glob_module_alias_intact(self):
        mod = getattr(reviewer, "_glob_module", None)
        self.assertIsNotNone(
            mod,
            "glob module alias must exist under a non-shadowed name "
            "('import glob as _glob' + 'def _glob' is the known footgun)")
        self.assertTrue(hasattr(mod, "glob") and callable(mod.glob),
                        "glob module alias must still expose glob()")

    def test_glob_function_callable(self):
        self.assertTrue(callable(reviewer._glob),
                        "_glob must remain the tool function, not be overwritten by the module")

    def test_execute_glob_no_attribute_error(self):
        """The original bug surfaced here as AttributeError."""
        try:
            out = reviewer.execute_tool(
                "Glob", {"pattern": "*.py"},
                os.path.dirname(os.path.abspath(__file__)))
        except AttributeError as e:
            self.fail(f"execute_tool(Glob) raised AttributeError "
                      f"(shadowing regression?): {e}")
        self.assertIsInstance(out, str)


class PermissionRequestDeferOnErrorTests(unittest.TestCase):
    """When review() raises (network error / reviewer crash / bug), the
    PermissionRequest hook must defer to the human via native flow and must
    NOT emit a deny. Regression for the policy change after the glob incident,
    where upstream DeepSeek proxy blips and the glob bug both caused false
    fail-closed denies of legitimate tool calls."""

    def _run_hook(self, hook_input, side_effect):
        import importlib, io
        import hook_permission_request as hpr
        importlib.reload(hpr)  # reset patched names from any prior test
        # neutralize all I/O / state side effects
        hpr.log = lambda *a, **k: None
        hpr.record_approval = lambda *a, **k: None
        hpr.circuit_breaker_record = lambda *a, **k: None
        hpr.circuit_breaker_check = lambda *a, **k: None
        hpr.check_hard_deny = lambda *a, **k: None
        hpr.check_agent_created_file = lambda *a, **k: False
        hpr.check_fast_path_allow = lambda *a, **k: False
        hpr.extract_context = lambda *a, **k: {}
        hpr.load_config = lambda: {"fail_closed": True}
        hpr.resolve_session_id = lambda x: "test-sid"
        def _raise(*a, **k):
            raise side_effect
        hpr.review = _raise

        old_in, old_out = sys.stdin, sys.stdout
        sys.stdin = io.StringIO(json.dumps(hook_input))
        buf = io.StringIO()
        sys.stdout = buf
        raised = None
        try:
            hpr.main()
        except SystemExit as e:
            raised = e
        finally:
            sys.stdin, sys.stdout = old_in, old_out
        return buf.getvalue(), raised

    def _base_input(self, tool_name, tool_input):
        return {"tool_name": tool_name, "tool_input": tool_input,
                "cwd": "/tmp", "permission_mode": "default",
                "session_id": "s1", "transcript_path": ""}

    def test_network_error_defers_not_denies(self):
        out, raised = self._run_hook(
            self._base_input("Bash", {"command": "ls"}),
            ConnectionError("Connection reset by peer"))
        self.assertNotIn("deny", out)
        self.assertNotIn("behavior", out)  # no decision emitted -> native flow
        self.assertEqual(raised.code if raised else 0, 0)

    def test_reviewer_crash_defers_not_denies(self):
        out, raised = self._run_hook(
            self._base_input("Write", {"file_path": "/tmp/x", "content": "y"}),
            AttributeError("'function' object has no attribute 'glob'"))
        self.assertNotIn("deny", out)
        self.assertEqual(raised.code if raised else 0, 0)


class ToolInputTrimTests(unittest.TestCase):
    """Large tool_input fields are spilled to a temp file for the reviewer LLM;
    short high-signal fields stay inline; temp files are cleaned up after the
    review() call returns (even on failure). Audit keeps a much fuller copy."""

    def tearDown(self):
        # safety net: remove any reviewer_ctx_ temp files left by failed asserts
        import glob, tempfile
        for p in glob.glob(os.path.join(tempfile.gettempdir(), "reviewer_ctx_*.txt")):
            try:
                os.remove(p)
            except OSError:
                pass

    def test_small_input_not_spilled(self):
        ti = {"file_path": "/tmp/x.py", "content": "print('hi')\n"}
        trimmed, temps = reviewer._trim_tool_input_for_llm(ti, {})
        self.assertEqual(temps, [])
        self.assertEqual(trimmed, ti)

    def test_large_content_spilled_with_preview_sha_size_path(self):
        big = "A" * 10000
        ti = {"file_path": "/tmp/big.py", "content": big}
        trimmed, temps = reviewer._trim_tool_input_for_llm(ti, {})
        try:
            self.assertEqual(len(temps), 1)
            self.assertTrue(os.path.exists(temps[0]))
            # file_path stays inline (short, high-signal)
            self.assertEqual(trimmed["file_path"], "/tmp/big.py")
            spilled = trimmed["content"]
            self.assertTrue(spilled["_truncated"])
            self.assertEqual(spilled["size_bytes"], 10000)
            self.assertEqual(spilled["sha256"],
                             __import__("hashlib").sha256(big.encode()).hexdigest())
            self.assertTrue(spilled["preview"].startswith("A"))
            self.assertTrue(spilled["preview"].endswith("...[truncated]"))
            self.assertLessEqual(len(spilled["preview"]), 2000 + len("...[truncated]"))
            self.assertEqual(spilled["full_content_path"], temps[0])
            # temp file holds the full content
            with open(temps[0]) as f:
                self.assertEqual(f.read(), big)
        finally:
            for t in temps:
                os.remove(t)

    def test_short_high_signal_fields_stay_inline(self):
        # command and file_path are short -> never spilled, even alongside a big field
        big = "Z" * 5000
        ti = {"command": "ls -la", "file_path": "/tmp/x", "content": big}
        trimmed, temps = reviewer._trim_tool_input_for_llm(ti, {})
        try:
            self.assertEqual(trimmed["command"], "ls -la")
            self.assertEqual(trimmed["file_path"], "/tmp/x")
            self.assertTrue(trimmed["content"]["_truncated"])
            self.assertEqual(len(temps), 1)
        finally:
            for t in temps:
                os.remove(t)

    def test_review_cleans_up_temp_files_even_on_llm_failure(self):
        # review() must delete spilled temp files in its finally block, even when
        # the LLM call ultimately fails. After the retry policy change, a
        # non-transient LLM error propagates out of review() (the hook defers to
        # the human). Use a non-transient error so no retry/sleep happens.
        big = "B" * 6000
        ti = {"file_path": "/tmp/big2.py", "content": big}
        original_call = reviewer._call_llm
        def _boom(*a, **k):
            raise ValueError("non-transient: bad API key / 4xx")
        reviewer._call_llm = _boom
        raised = None
        try:
            reviewer.review("Write", ti, "/tmp", {}, {"fail_closed": True}, "default")
        except Exception as e:
            raised = e
        finally:
            reviewer._call_llm = original_call
        self.assertIsNotNone(raised, "review() should raise on LLM failure, not return deny")
        import glob, tempfile
        leftover = glob.glob(os.path.join(tempfile.gettempdir(), "reviewer_ctx_*.txt"))
        self.assertEqual(leftover, [], f"temp files not cleaned up: {leftover}")

    def test_transient_llm_error_retried_3x_then_raises(self):
        # ConnectionError is transient -> 3 attempts then raise. Patch sleep to
        # no-op so the test is fast.
        attempts = {"n": 0}
        orig_call = reviewer._call_llm
        orig_sleep = reviewer.time.sleep
        def _flaky(*a, **k):
            attempts["n"] += 1
            raise ConnectionError("simulated proxy outage")
        reviewer._call_llm = _flaky
        reviewer.time.sleep = lambda _s: None
        raised = None
        try:
            reviewer.review("Read", {"path": "/tmp/x"}, "/tmp", {}, {}, "default")
        except ConnectionError as e:
            raised = e
        finally:
            reviewer._call_llm = orig_call
            reviewer.time.sleep = orig_sleep
        self.assertEqual(attempts["n"], 3, f"expected 3 attempts, got {attempts['n']}")
        self.assertIsNotNone(raised)

    def test_non_transient_llm_error_raises_immediately(self):
        # A 4xx-style error (HTTPError 401) must NOT be retried.
        import urllib.error
        attempts = {"n": 0}
        orig_call = reviewer._call_llm
        def _bad(*a, **k):
            attempts["n"] += 1
            raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
        reviewer._call_llm = _bad
        raised = None
        try:
            reviewer.review("Read", {"path": "/tmp/x"}, "/tmp", {}, {}, "default")
        except urllib.error.HTTPError as e:
            raised = e
        finally:
            reviewer._call_llm = orig_call
        self.assertEqual(attempts["n"], 1, f"401 must not retry, got {attempts['n']}")
        self.assertIsNotNone(raised)


class TranscriptContextTests(unittest.TestCase):
    """recent_tool_calls now keeps both the call (name+input) and the matching
    tool_result (Codex-style evidence). Long results spill to a temp file."""

    def _write_transcript(self, lines):
        fd, path = tempfile.mkstemp(prefix="reviewer_tx_", suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(json.dumps(l) for l in lines) + "\n")
        return path

    def test_tool_call_and_result_captured_and_paired(self):
        big_result = "RESULT_LINE\n" * 3000  # ~33KB, above spill threshold
        tx = self._write_transcript([
            {"type": "user", "message": {"role": "user",
              "content": [{"type": "text", "text": "please read the file"}]}},
            {"type": "assistant", "message": {"role": "assistant",
              "content": [{"type": "tool_use", "id": "tu1", "name": "Read",
                           "input": {"path": "/tmp/secret.txt"}}]}},
            {"type": "user", "message": {"role": "user",
              "content": [{"type": "tool_result", "tool_use_id": "tu1",
                           "content": big_result}]}},
        ])
        ctx, temps = transcript.extract_context(tx, turns_to_read=2)
        try:
            self.assertEqual(len(ctx["recent_tool_calls"]), 1)
            ti = ctx["recent_tool_calls"][0]
            self.assertEqual(ti["name"], "Read")
            self.assertEqual(ti["input"], {"path": "/tmp/secret.txt"})
            self.assertTrue(ti["result"]["_truncated"])
            self.assertEqual(len(temps), 1)
            self.assertTrue(os.path.exists(temps[0]))
            with open(temps[0]) as f:
                self.assertEqual(f.read(), big_result)
            self.assertTrue(ti["result"]["preview"].startswith("RESULT_LINE"))
            self.assertTrue(ti["result"]["preview"].endswith("...[truncated]"))
        finally:
            for t in temps:
                os.remove(t)
            os.remove(tx)

    def test_short_result_kept_inline_not_spilled(self):
        tx = self._write_transcript([
            {"type": "user", "message": {"role": "user",
              "content": [{"type": "text", "text": "ls"}]}},
            {"type": "assistant", "message": {"role": "assistant",
              "content": [{"type": "tool_use", "id": "tu9", "name": "Bash",
                           "input": {"command": "ls"}}]}},
            {"type": "user", "message": {"role": "user",
              "content": [{"type": "tool_result", "tool_use_id": "tu9",
                           "content": "file1.txt\nfile2.txt"}]}},
        ])
        ctx, temps = transcript.extract_context(tx, turns_to_read=2)
        try:
            ti = ctx["recent_tool_calls"][0]
            self.assertEqual(ti["name"], "Bash")
            self.assertEqual(ti["result"], "file1.txt\nfile2.txt")  # inline string, not dict
            self.assertEqual(temps, [])
        finally:
            for t in temps:
                os.remove(t)
            os.remove(tx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
