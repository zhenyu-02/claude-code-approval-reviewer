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


if __name__ == "__main__":
    unittest.main(verbosity=2)
