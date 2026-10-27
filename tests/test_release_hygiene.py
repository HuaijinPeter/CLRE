#!/usr/bin/env python
"""Release hygiene self-check -- runs in CI, needs nothing but the repo.

Fails if a future commit reintroduces anything that must never be published:
machine-local absolute paths, usernames, credential-shaped strings, large
binaries, caches, or run outputs. This is the automated half of the
sanitisation audit in docs/SANITIZATION_REPORT.md.
"""
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TEXT_EXT = {".py", ".md", ".txt", ".sh", ".cfg", ".toml", ".yaml", ".yml",
            ".json", ".in"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv",
             "venv", "node_modules"}

# This file necessarily contains the patterns it searches for, so it is the
# one file exempt from the content scans (its own patterns are the test).
SELF = os.path.basename(__file__)

MAX_BYTES = 2 * 1024 * 1024          # 2 MB: nothing in a source release needs more
BANNED_BINARY_EXT = {".pth", ".pt", ".ckpt", ".npz", ".npy", ".pkl", ".pickle",
                     ".bin", ".h5", ".hdf5", ".so", ".zip", ".tar", ".tgz",
                     ".mp4", ".log", ".rec", ".avi", ".png.aux"}

# Machine-local path shapes. Generic patterns, not one developer's name.
PATH_PATTERNS = [
    re.compile(r"/home/[A-Za-z0-9_.-]+/"),
    re.compile(r"/Users/[A-Za-z0-9_.-]+/"),
    re.compile(r"/data/[A-Za-z0-9_.-]+/projects/"),
    re.compile(r"[A-Za-z]:\\\\Users\\\\"),
]
# Credential shapes. Deliberately narrow so prose like "token budget" passes.
SECRET_PATTERNS = [
    re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|"
               r"auth[_-]?token|client[_-]?secret|passwd|password)\b\s*[:=]\s*"
               r"['\"][^'\"]{6,}['\"]"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),        # GitHub tokens
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),               # OpenAI-style keys
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                  # AWS access key id
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


def _walk():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            yield os.path.join(dirpath, name)


def _rel(p):
    return os.path.relpath(p, ROOT)


def test_no_machine_local_absolute_paths():
    bad = []
    for path in _walk():
        if os.path.basename(path) == SELF:
            continue
        if os.path.splitext(path)[1] not in TEXT_EXT:
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        for pat in PATH_PATTERNS:
            for m in pat.finditer(text):
                line = text[:m.start()].count("\n") + 1
                bad.append(f"{_rel(path)}:{line}: {m.group(0)}")
    assert not bad, "machine-local paths found:\n" + "\n".join(bad)


def test_no_credential_shaped_strings():
    bad = []
    for path in _walk():
        if os.path.basename(path) == SELF:
            continue
        if os.path.splitext(path)[1] not in TEXT_EXT:
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        for pat in SECRET_PATTERNS:
            for m in pat.finditer(text):
                line = text[:m.start()].count("\n") + 1
                bad.append(f"{_rel(path)}:{line}: {m.group(0)[:40]}")
    assert not bad, "credential-shaped strings found:\n" + "\n".join(bad)


def test_no_large_or_binary_artifacts():
    bad = []
    for path in _walk():
        ext = os.path.splitext(path)[1].lower()
        size = os.path.getsize(path)
        if ext in BANNED_BINARY_EXT:
            bad.append(f"{_rel(path)}: banned extension {ext}")
        elif size > MAX_BYTES:
            bad.append(f"{_rel(path)}: {size / 1e6:.1f} MB > 2 MB")
    assert not bad, "unpublishable artifacts found:\n" + "\n".join(bad)


def test_no_upstream_source_vendored():
    """No upstream driving-stack source may be redistributed here.

    Importing upstream names is fine and expected -- the integration guides do
    exactly that. What must never appear is upstream SOURCE: a module that
    DEFINES the upstream classes rather than referencing them.
    """
    import ast
    banned_defs = {"PIDController", "PID", "VadAgent", "UniadAgent",
                   "DriveTransformerAgent", "AutonomousAgent"}
    bad = []
    for path in _walk():
        if os.path.splitext(path)[1] != ".py":
            continue
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in banned_defs:
                bad.append(f"{_rel(path)}:{node.lineno}: defines {node.name}")
    assert not bad, "vendored upstream source found:\n" + "\n".join(bad)


def test_every_python_file_compiles():
    """Syntax check without emitting bytecode anywhere."""
    bad = []
    for path in _walk():
        if os.path.splitext(path)[1] != ".py":
            continue
        src = open(path, encoding="utf-8", errors="replace").read()
        try:
            compile(src, path, "exec")
        except SyntaxError as exc:
            bad.append(f"{_rel(path)}:{exc.lineno}: {exc.msg}")
    assert not bad, "syntax errors:\n" + "\n".join(bad)


if __name__ == "__main__":
    test_no_machine_local_absolute_paths()
    test_no_credential_shaped_strings()
    test_no_large_or_binary_artifacts()
    test_no_upstream_source_vendored()
    test_every_python_file_compiles()
    print("release hygiene OK")
