# Provenance

Where every released file came from, and every change made on the way out.
Nothing in this repository was written from scratch to look like the evaluated
method: the method files are byte-identical copies of the frozen research
source, and the three exceptions are listed exhaustively below.

Frozen source: commit `b1f509df68c427fc128d8b1e7c86520a304c7afb`, branch
`phase4b3/e59-pebv2`, of the private research workspace. That commit is the
`PERMANENT_FROZEN` method identity used for every closed-loop campaign in the
paper.

---

## 1. Byte-identical files

Copied without a single character changed. Their SHA-256 values match the
frozen tree's own dependency manifest, so a reader can verify the published
code is the evaluated code:

| Released path | Frozen path | SHA-256 |
| --- | --- | --- |
| `clre/planning/refinement.py` | `newborn_vad/planning/candidate_a_planner.py` | `ff1df4f1f3055be09edd17c0776d9f02d2122beec09304fc20505a28b4c10236` |
| `clre/planning/reference_prototype.py` | `research/candidate_a/mvp_planner.py` | `2bec1b153e209b07b32d6cf4a774607e773bc63de80400d3ca18a0ec825ba911` |
| `clre/control/radar_safety.py` | `newborn_vad/control/radar_safety.py` | `f15cd190d4db27da89144cbdef216930ed7855dc51b2a888d2b2288445d8f5fc` |
| `clre/control/route_centerline.py` | `newborn_vad/control/route_centerline.py` | `ea372514e9514965f45a17c956fa8af1878f7ab12488423e51fa8afc1dee843b` |
| `clre/tracing/trace_writer.py` | `newborn_vad/tracing/trace_writer.py` | `723a315665a545ec0c1671046769c4101595a5f50f8fc42cef398d81c6e5a33d` |

```bash
sha256sum clre/planning/refinement.py clre/control/radar_safety.py \
          clre/control/route_centerline.py clre/tracing/trace_writer.py \
          clre/planning/reference_prototype.py
```

### A note on names you will see inside these files

Because the bodies are untouched, their comments and docstrings still refer to
the private research tree: `docs/phase4b1_candidate_a_formulation.md`,
`tools/diagnosis/phase4b2a_v2_repair_analysis.py`, experiment labels like
`E39`, `E56`, `A-v3`, `R1/R2/R3`, and the package name `newborn_vad`. Those
references are **retained deliberately**. Rewriting them would have meant
editing the evaluated source, which would destroy the hash equality above and
replace a verifiable artifact with a plausible-looking one. Read them as
pointers into the research record, not as missing files.

The one consequence worth knowing: `clre/planning/refinement.py` documents
two replicated quirks of the frozen prototype (a negated curvature profile in
the prior seed, and the `interp_pred` origin knot) that are preserved on
purpose, because fixing either would change the formulation and invalidate the
frozen results. They are not bugs introduced by this release.

## 2. Files with listed edits

Three lines of code changed in total, plus their explanatory comments.

The `-` lines below show the frozen source. The developer home directory in
them is shown as `<redacted-home>`: the point is the *shape* of what was
removed, and publishing the literal path would reintroduce exactly the leak
the edit exists to close (`tests/test_release_hygiene.py` enforces this).

### `clre/adapters/uniad_adapter.py` ← `newborn_vad/planning/uniad_adapter.py`

One edit: a machine-local checkpoint path became an environment variable.

```diff
-UNIAD_CKPT_PATH = "<redacted-home>/checkpoints/bench2drive/uniad_base_b2d.pth"
+UNIAD_CKPT_PATH = os.environ.get(
+    "CLRE_UNIAD_CKPT", os.path.join("ckpts", "uniad_base_b2d.pth"))
```

The adapter's `UNIAD_CKPT_SHA256` pin is unchanged, so the checkpoint identity
check still works against a user-supplied file.

### `clre/adapters/drivetransformer_adapter.py` ← `newborn_vad/planning/drivetransformer_adapter.py`

Two edits: the same checkpoint-path change, plus the only internal import the
package rename affected.

```diff
-from newborn_vad.planning import uniad_adapter as ua
+from clre.adapters import uniad_adapter as ua  # RELEASE EDIT: package renamed

-DT_CKPT_PATH = "<redacted-home>/checkpoints/bench2drive/drivetransformer_large.pth"
+DT_CKPT_PATH = os.environ.get(
+    "CLRE_DT_CKPT", os.path.join("ckpts", "drivetransformer_large.pth"))
```

### `tests/test_refinement.py` ← `tests/test_candidate_a_planner.py`

Two import rewrites, a skip guard, and docstring path updates. No assertion,
tolerance or fixture was touched.

```diff
-from newborn_vad.planning import candidate_a_planner as cap
-from research.candidate_a import mvp_planner as frozen
+from clre.planning import refinement as cap
+from clre.planning import reference_prototype as frozen
```

`test_native_desired_speed_matches_real_pid` compares CLRE's reference-speed
formula against the real upstream `PIDController`. That class is not
redistributed here, so the test now skips with a printed reason when
Bench2DriveZoo is absent from `PYTHONPATH`, instead of failing or silently
passing. Put your Bench2DriveZoo checkout on the path to run it.

## 3. `clre/control/peb_longitudinal.py` — extracted, then verified

The frozen `newborn_vad/control/peb_v2_controller.py` subclasses
Bench2DriveZoo's `PIDController` and reproduces its aim-point selection and
all four steering-angle computations verbatim, so that steering stays
bit-identical to the baseline and only the longitudinal block differs. That
makes the file Adapted Material under CC BY-NC-ND 4.0, so it cannot be
published ([LICENSE_AUDIT.md](LICENSE_AUDIT.md)).

The split is clean, because the two halves do not interact: the braking law is
entirely the authors' work and the steering block is entirely upstream's.
`clre/control/peb_longitudinal.py` contains the former as pure functions with
no upstream import — same branch order, same formulas, same thresholds, same
telemetry keys, and the same retired-branch hole at code `1`.

**This was verified, not assumed.** At release time the extracted law was
replayed against the frozen subclass over the full cross product of 7 ego
speeds × 6 waypoint target speeds × 5 radar caps × 2 emergency states × 3
fallback-override states:

```
PEB longitudinal equivalence: 1260/1260 cases exact
  (throttle, brake, branch, branch code, e_v, override flag,
   final target speed — all matching to < 1e-12)
frozen PEB/execution telemetry keys: 12 | missing from release: none
```

What the extraction does **not** carry, by design: the steering path, and
`speed_controller.step()` itself. The second matters for correctness, not
licensing — the upstream speed PID must still be stepped exactly once per
tick, unconditionally, on every branch, or its integral window diverges from
the baseline and the comparison stops being matched. `window_delta()` exists
to make that integration point explicit; see
[`../integration/peb_v2_controller_patch.md`](../integration/peb_v2_controller_patch.md).

## 4. Files written for this release

`clre/__init__.py` and the four sub-package `__init__.py` files (the frozen
tree used implicit namespace packages); `tests/test_execution_layer.py`,
`tests/test_adapters_smoke.py`, `tests/test_release_hygiene.py`; everything
under `integration/`, `docs/` and `scripts/`; `README.md`, `NOTICE`,
`LICENSE`, `requirements.txt`, `.gitignore`.

`tests/test_execution_layer.py` deserves a note. The frozen execution-layer
suite drives the unpublishable controller directly, so it could not be copied
wholesale. Its radar-supervisor and route-centerline checks — scenarios
(d), (g), (h), (i), (k), (l), (n) — are carried over unchanged, because those
two modules are published verbatim. The braking scenarios — (a), (b), (c),
(e), (m) — were re-expressed against `peb_longitudinal`, and scenario (f), the
structural check that no planner state can reach the execution layer, now
walks the released modules' ASTs. Scenario (m) is the one whose statement
changed: the frozen version simulated 200 ticks of a coasting vehicle through
the real PID to show the latched target produces a restoring force, which
needs the upstream PID; the released version asserts the same contract as a
property of the demand itself — a latched target stays constant while the ego
coasts and feeds the PID a growing error, whereas re-reading the current speed
each tick makes the target ratchet down and the error drain to zero.

## 5. Not carried over

The frozen tree also contains ~50 diagnosis tools, ~36 internal design and
audit documents, the HiP-AD and ORION-Lite integrations, Tier-0 prototypes,
ablation and latency-audit agent subclasses, and about 570 GB of experiment
output. [LICENSE_AUDIT.md](LICENSE_AUDIT.md) §2 lists each category and the
reason it was excluded.

## 6. Stale draft sources in the research tree — check before publishing

This release is titled for the paper **"Closed-Loop Refinement and Execution
for Learned Driving Planners"**, which is exactly the title carried by the
current LaTeX source (`paper/final_paper.tex`). Nothing needs changing for
that — but the research tree also still contains an **older, superseded**
draft (`paper/root.tex` plus `paper/sections/*.tex`) which differs from the
current one in ways that matter if anyone reads it as authoritative:

| | current `final_paper.tex` | superseded `root.tex` / `sections/` |
| --- | --- | --- |
| title | matches this release | *"Progress-Driven Constrained Trajectory Refinement for Closed-Loop End-to-End Autonomous Driving"* |
| third author | Zhongyu Mo | Zoey Mo |
| main protocol | all 220 official validation routes | first 36 routes, with an open `TODO` for 220 |
| ablations | three arms (w/o progress, w/o feasibility test, w/o execution layer) | execution-layer arm only |
| interaction weights | `w_out` / `w_in` | `w_risk` / `w_bar` |

This release's documentation was written against `final_paper.tex`. Earlier
drafts also named the method `CLoRE-VAD` / `CLRE-VAD`. Before making the
repository public, confirm that the repository name, the `README.md` heading
and the BibTeX entry match whatever is actually submitted, and consider
deleting or clearly marking the superseded draft so it cannot be mistaken for
the paper.
