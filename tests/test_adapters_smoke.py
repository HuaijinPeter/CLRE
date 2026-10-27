#!/usr/bin/env python
"""Import-level smoke checks for the UniAD / DriveTransformer adapters.

The adapters' full offline suites (24 fixture groups each, plus AST parity
checks against sha256-pinned upstream files) are NOT part of this release:
they require pinned checkouts of upstream repositories that cannot be
redistributed here. See docs/LICENSE_AUDIT.md and integration/.

What is checked here needs no upstream, no checkpoint and no CARLA: that the
adapters import standalone, that their frozen contract constants survived the
release edits, and that no machine-local checkpoint path is baked in.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from clre.adapters import uniad_adapter as ua            # noqa: E402
from clre.adapters import drivetransformer_adapter as dta  # noqa: E402


def test_adapters_import_without_upstream():
    assert ua.ADAPTER_VERSION.startswith("uniad_adapter/")
    assert dta.ADAPTER_VERSION.startswith("drivetransformer_adapter/")


def test_upstream_pins_present_and_well_formed():
    for mod in (ua, dta):
        pins = mod.UPSTREAM_SHA256
        assert pins, mod.__name__
        for rel, digest in pins.items():
            assert len(digest) == 64 and all(c in "0123456789abcdef"
                                             for c in digest), (rel, digest)
            assert not os.path.isabs(rel), rel


def test_checkpoint_paths_are_relative_and_overridable():
    """No machine-local checkpoint path, and the env override is live."""
    for path in (ua.UNIAD_CKPT_PATH, dta.DT_CKPT_PATH):
        assert not os.path.isabs(path), path
    import importlib
    os.environ["CLRE_UNIAD_CKPT"] = "/tmp/clre-test-ckpt/uniad.pth"
    try:
        importlib.reload(ua)
        assert ua.UNIAD_CKPT_PATH == "/tmp/clre-test-ckpt/uniad.pth"
    finally:
        del os.environ["CLRE_UNIAD_CKPT"]
        importlib.reload(ua)
    assert not os.path.isabs(ua.UNIAD_CKPT_PATH)


def test_adapters_never_import_the_planner():
    """Contract: adapters convert data only; they never duplicate objective
    or OBB mathematics, and never import the refinement layer."""
    import ast
    for mod in (ua, dta):
        src = open(mod.__file__, encoding="utf-8").read()
        for node in ast.walk(ast.parse(src)):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                assert "refinement" not in m and "planner" not in m, (mod, m)


if __name__ == "__main__":
    test_adapters_import_without_upstream()
    test_upstream_pins_present_and_well_formed()
    test_checkpoint_paths_are_relative_and_overridable()
    test_adapters_never_import_the_planner()
    print("adapter smoke OK")
