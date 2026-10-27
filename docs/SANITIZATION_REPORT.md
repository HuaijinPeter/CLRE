# Sanitisation report

Recursive scan of this release directory, performed at release time. Thirteen
checks; everything found is listed, including the findings that were
deliberately accepted rather than changed.

`tests/test_release_hygiene.py` automates the five checks that can regress, so
a future commit cannot quietly reintroduce them.

---

## Summary

| # | Check | Result |
| --- | --- | --- |
| 1 | Machine-local absolute paths | **2 found → fixed** |
| 2 | Usernames / personal identifiers | 5 occurrences, all intended authorship |
| 3 | API keys, tokens, passwords, private keys | none |
| 4 | Private hostnames, IP addresses, internal URLs | none |
| 5 | Large files | none; largest 52 KB, total 357 KB over 34 files |
| 6 | Binary files | none; every file is text |
| 7 | Caches, temp files, editor noise | **3 found → removed** |
| 8 | Debug / temporary / stale code | none |
| 9 | Dead code | **1 found → removed** |
| 10 | Dangling references to the internal research tree | 21 occurrences, **accepted** (see below) |
| 11 | Accidental third-party source | none |
| 12 | Vendored upstream class/function definitions | none |
| 13 | Checkpoints, datasets, run outputs, CARLA recordings | none |

---

## Findings fixed

### 1. Machine-local absolute paths — 2 occurrences

The two adapters hard-coded a developer checkpoint path:

| File | Was | Now |
| --- | --- | --- |
| `clre/adapters/uniad_adapter.py` | `<home>/checkpoints/bench2drive/uniad_base_b2d.pth` | `os.environ.get("CLRE_UNIAD_CKPT", "ckpts/uniad_base_b2d.pth")` |
| `clre/adapters/drivetransformer_adapter.py` | `<home>/checkpoints/bench2drive/drivetransformer_large.pth` | `os.environ.get("CLRE_DT_CKPT", "ckpts/drivetransformer_large.pth")` |

A third occurrence was then introduced *by this audit*: the "before" lines
quoted in `docs/PROVENANCE.md` reproduced the literal home directory. The
automated check caught it, and the path is now shown as `<redacted-home>`.

Worth recording because it is the failure mode these reports usually miss —
the document describing a leak recreating it.

### 7. Caches and build artifacts — 3 directories

`.pytest_cache/` and `__pycache__/` appeared from running the test suite
locally. Removed, and both are in `.gitignore`.

### 9. Dead code — 1 block

A loop in `tests/test_execution_layer.py::test_e_braking_monotone_in_severity`
computed a value and immediately `del`-ed it, left over from drafting the
closed-form ramp assertion. Removed; the surviving assertion covers the
thresholds properly.

---

## Findings accepted, with reasons

### 2. Usernames and personal identifiers — 5 occurrences

| File | Content |
| --- | --- |
| `LICENSE` | copyright line, 4 author names |
| `NOTICE` | copyright line, 4 author names |
| `README.md` | BibTeX `author` field |

These are **intended attribution**, not leakage: the same names appear on the
paper. No email addresses, affiliations, account names or machine identifiers
are included anywhere in the release.

### 10. Dangling references to the internal research tree — 21 occurrences

Docstrings and comments in the byte-identical files point at documents and
scripts that are **not** part of this release:

| File | Referenced path(s) |
| --- | --- |
| `clre/planning/refinement.py` | `docs/phase4b1_candidate_a_formulation.md`, `docs/phase4b2a_candidate_a_mvp_implementation.md`, `docs/phase4b2a_candidate_a_v2_revision.md`, `docs/final_dev_stage_master_plan.md`, `docs/e39_instrumentation_schema.md`, `tools/diagnosis/phase4b1_candidate_a_offline.py`, `tools/diagnosis/phase4b2a_v2_repair_analysis.py`, `research/candidate_a/mvp_planner.py` |
| `clre/planning/reference_prototype.py` | `docs/phase4b1_candidate_a_formulation.md` |
| `clre/adapters/uniad_adapter.py` | `docs/experiments/uniad_newborn_integration_design.md` |
| `clre/adapters/drivetransformer_adapter.py` | `docs/experiments/drivetransformer_newborn_integration_design.md`, `artifacts/drivetransformer_ckpt_sha256.txt` |
| `clre/tracing/trace_writer.py` | `docs/native_vad_36route_trace_schema.md` |

**Accepted deliberately.** These files are published byte-identical to the
frozen research source, and their SHA-256 values are listed in
[PROVENANCE.md](PROVENANCE.md) §1 so a reader can verify that the published
code is the evaluated code. Rewriting the references would edit the evaluated
source, destroy that hash equality, and replace a verifiable artifact with a
plausible-looking one.

They are relative paths, contain no secrets and no absolute paths, and leak
nothing beyond internal document titles. [PROVENANCE.md](PROVENANCE.md) §1
tells readers to treat them as pointers into the research record.

One is resolvable within the release: `research/candidate_a/mvp_planner.py` is
published here as `clre/planning/reference_prototype.py`.

### Experiment labels in comments

The same files carry internal experiment identifiers — `E39`, `E56`, `E58`,
`A-v2`/`A-v3`, `R1`/`R2`/`R3`, `Phase 4B-2A`, `DevVal-16` — and some comments
record superseded designs and negative results (for example, why the
`infeasible_override` branch was deleted, and why the first radar sign
convention was wrong). Accepted for the same byte-identity reason, and in this
case they also carry real information: the discarded alternatives are *why*
the constants are what they are. The `NEWBORN_*` environment-variable names
are kept for the same reason and are documented in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) §4.

---

## Checks that came back empty

**3. Credentials.** No API keys, tokens, passwords, bearer strings, SSH keys
or PEM blocks. Patterns searched: `api[_-]?key`, `secret`, `token`, `passwd`,
`password`, `bearer`, `ssh-rsa`, `BEGIN … PRIVATE KEY`, `gh[pousr]_…`,
`AKIA…`, `sk-…`.

**4. Hosts and networks.** No IP addresses, no internal hostnames, no private
URLs. The only URLs are public project pages for upstream dependencies
(github.com, apache.org) and licence references.

**5 / 6. Size and binaries.** 34 files, 357 KB total. Largest:
`clre/planning/refinement.py` at 52 KB. Every file is text; nothing approaches
GitHub's limits, and the automated check fails anything above 2 MB.

**8. Debug code.** No `TODO`, `FIXME`, `XXX`, `HACK`, `breakpoint()`,
`pdb.set_trace()`, debug prints or commented-out experiments.

**11 / 12. Third-party source.** No upstream file is included, in whole or in
part. An AST walk confirms no release file *defines* `PIDController`, `PID`,
`VadAgent`, `UniadAgent`, `DriveTransformerAgent`, `AutonomousAgent`,
`SparseAgent` or `OrionAgent`. Exactly one upstream import exists anywhere in
the release —
`tests/test_refinement.py::test_native_desired_speed_matches_real_pid`, inside
a `try/except ImportError` that skips with a printed reason when
Bench2DriveZoo is absent. Referencing a class by name is not copying it.

The adapters record SHA-256 **hashes** of pinned upstream files and, in the
DriveTransformer case, dynamically path-load one upstream function at runtime.
Neither embeds upstream source.

**13. Checkpoints, data, run outputs.** No `.pth`, `.pt`, `.ckpt`, `.npz`,
`.npy`, `.pkl`, `.so`, `.zip`, `.log`, `.rec` or `.mp4`. No evaluator results,
per-tick traces, CARLA recordings, figures or dataset files. `.gitignore`
blocks all of these, and the automated check rejects them by extension.

For scale: the frozen research tree these files came from carries about 570 GB
of experiment output (`experiments/` 537 GB, `carla_runs/` 31 GB, `artifacts/`
868 MB). None of it is here.

---

## Re-verified after the pre-push documentation pass

The README example, citation, licence wording, OBB terminology and
`requirements.txt` were revised on 2026-10-05 (README, NOTICE,
`requirements.txt`, `docs/PARAMETERS.md`, `docs/LICENSE_AUDIT.md`, and this
file). No released method code was touched. All thirteen checks were re-run
after those edits and the results in the table above still hold: 35/35 tests
pass, 17/17 Python files compile, 41/41 relative links resolve, and the scans
for machine-local paths, credentials, binaries, caches and vendored upstream
source all come back empty. Caches created by running the suite locally were
removed again.

## Re-running this audit

```bash
python -m pytest tests/test_release_hygiene.py -v
```

Five automated checks: machine-local paths, credential-shaped strings, large
or banned-extension files, vendored upstream definitions, and syntax. The
remaining eight checks in the table above were manual and are reproduced by
the `grep`/AST commands quoted in each section.
