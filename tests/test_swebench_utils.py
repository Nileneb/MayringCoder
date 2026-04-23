import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.swebench_utils import patched_files, determine_match


SAMPLE_PATCH = """\
diff --git a/django/views/generic/base.py b/django/views/generic/base.py
--- a/django/views/generic/base.py
+++ b/django/views/generic/base.py
@@ -1,3 +1,4 @@
 import logging
+import warnings
 from functools import wraps
diff --git a/django/utils/decorators.py b/django/utils/decorators.py
--- a/django/utils/decorators.py
+++ b/django/utils/decorators.py
@@ -10,2 +10,3 @@
 pass
"""


def test_patched_files_extracts_correct_paths():
    result = patched_files(SAMPLE_PATCH)
    assert result == {"django/views/generic/base.py", "django/utils/decorators.py"}


def test_patched_files_empty_patch():
    assert patched_files("") == set()


def test_patched_files_no_minus_a_lines():
    patch = "diff --git a/foo.py b/foo.py\n+++ b/foo.py\n"
    assert patched_files(patch) == set()


def test_determine_match_tp():
    mc = {"django/views/generic/base.py", "some/other.py"}
    gt = {"django/views/generic/base.py"}
    assert determine_match(mc, gt) == "TP"


def test_determine_match_fn():
    mc = {"completely/different.py"}
    gt = {"django/views/generic/base.py"}
    assert determine_match(mc, gt) == "FN"


def test_determine_match_empty_mc():
    assert determine_match(set(), {"django/views/generic/base.py"}) == "FN"


def test_determine_match_empty_gt():
    assert determine_match({"some/file.py"}, set()) == "FN"
