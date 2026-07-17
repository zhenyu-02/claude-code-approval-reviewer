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


if __name__ == "__main__":
    unittest.main(verbosity=2)
