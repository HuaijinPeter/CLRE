# License and provenance audit

Audit performed while preparing this release, against the upstream checkouts
actually used for the paper (the commits in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) §2). Every licence below was read
from the `LICENSE` file in the corresponding checkout, not inferred from a
README, a paper, or a habit.

**This is an engineering audit, not legal advice.** Where it says *needs
manual review*, it means a lawyer or the upstream authors should be asked
before redistributing the file in question, and the conservative choice —
not copying — was taken here.

---

## 1. Upstream licences as found

| Project | Licence file | Licence | Redistribute verbatim? | Redistribute **modified**? |
| --- | --- | --- | --- | --- |
| CARLA 0.9.15 | `CARLA_0.9.15/LICENSE` | MIT | yes | yes |
| CARLA leaderboard (in Bench2Drive) | `Bench2Drive/leaderboard/LICENSE` | MIT (© 2019 CARLA) | yes | yes |
| CARLA scenario_runner (in Bench2Drive) | `Bench2Drive/scenario_runner/LICENSE` | MIT (© 2018 CARLA) | yes | yes |
| Bench2Drive | `Bench2Drive/LICENSE` | **CC BY-NC-ND 4.0** | with attribution, non-commercial | **NO** |
| Bench2DriveZoo (VAD, UniAD, BEVFormer, TCP, merged mmcv) | `Bench2DriveZoo/LICENSE` | **CC BY-NC-ND 4.0** | with attribution, non-commercial | **NO** |
| DriveTransformer | **none found** | **unlicensed / all rights reserved** | **NO** | **NO** |

Two findings drove every decision in this release.

**Finding 1 — Bench2Drive and Bench2DriveZoo are NoDerivatives.**
CC BY-NC-ND 4.0 grants the right to reproduce and share the *licensed
material*, but §2(a) does **not** grant the right to share *Adapted Material*.
A file that copies a substantial block of upstream source and modifies it is
Adapted Material. Such files therefore cannot be published here, regardless of
attribution. Separately, the NonCommercial term means that use of Bench2Drive
or Bench2DriveZoo remains subject to those NonCommercial terms, independently
of CLRE's MIT license.

**Finding 2 — DriveTransformer ships no licence at all.**
The checkout at `19702015eb700e804bafb53e63b04571c7b8604a` contains no
`LICENSE` or `NOTICE` file, and its README makes no licence statement. With no
grant, the default is all rights reserved: nothing derived from its source may
be redistributed. Our DriveTransformer *adapter* is unaffected — it is original
CLRE code that contains no DriveTransformer source — but no upstream file or
modified copy of one appears here.

### Secondary observation, flagged rather than resolved

Bench2DriveZoo redistributes code originating from VAD (hustvl), UniAD
(OpenDriveLab) and BEVFormer (fundamentalvision), all of which are
Apache-2.0 upstream, under a single root CC BY-NC-ND 4.0 `LICENSE` with no
per-file headers and no preserved Apache `NOTICE`. Whether the ND term is
enforceable over those specific sub-trees is a question about Bench2DriveZoo's
own licensing, not about CLRE.

**Status: needs manual review. Not relied upon.** This release assumes the
*stricter* reading — ND applies — and copies nothing from that tree. If the
authors later confirm that the VAD/UniAD sub-trees remain Apache-2.0, the
agent wrappers in `integration/` could be published directly; until then they
are documented only.

## 2. Per-file disposition

### PUBLIC — original CLRE code, verbatim from the frozen commit

Measured verbatim-line overlap against the plausible upstream donor files is
reported below; the residual one or two lines are `import numpy as np` and
similar boilerplate.

| Released path | Frozen path | Upstream overlap | Role |
| --- | --- | --- | --- |
| `clre/planning/refinement.py` | `newborn_vad/planning/candidate_a_planner.py` | 1 / 762 lines (0.1 %) | receding-horizon refinement, objective terms, candidate seeds, warm start, OBB gate, selection, solver |
| `clre/planning/reference_prototype.py` | `research/candidate_a/mvp_planner.py` | 0 % | offline reference implementation of the same objective |
| `clre/control/radar_safety.py` | `newborn_vad/control/radar_safety.py` | 1 / 263 (0.4 %) | forward-radar speed bound |
| `clre/control/route_centerline.py` | `newborn_vad/control/route_centerline.py` | 1 / 158 (0.6 %) | route-centerline backup |
| `clre/tracing/trace_writer.py` | `newborn_vad/tracing/trace_writer.py` | 0 % | per-tick npz trace writer |
| `clre/adapters/uniad_adapter.py` | `newborn_vad/planning/uniad_adapter.py` | 2 / 552 (0.4 %) | UniAD → CLRE pure data adapter |
| `clre/adapters/drivetransformer_adapter.py` | `newborn_vad/planning/drivetransformer_adapter.py` | 1 / 792 (0.1 %) | DriveTransformer → CLRE pure data adapter |
| `tests/test_refinement.py` | `tests/test_candidate_a_planner.py` | 0 % | solver-vs-reference equivalence, gate behaviour, bounds, fallback, warm start |

Both adapters record `sha256` **hashes** of pinned upstream files and, in the
DriveTransformer case, dynamically path-load one upstream function at runtime.
Neither embeds upstream source. Publishing a hash of a file is not publishing
the file.

### PUBLIC — written for this release

| Released path | Role | Why it exists |
| --- | --- | --- |
| `clre/control/peb_longitudinal.py` | saturated proportional-error braking (PEB-v2) as upstream-free pure functions | extraction of the only original part of a controller that cannot be shipped (below) |
| `tests/test_execution_layer.py` | radar / centerline / braking behaviour | the frozen suite drives the unpublishable controller; radar and centerline checks carried over unchanged |
| `tests/test_adapters_smoke.py` | adapter import + contract checks | the frozen adapter suites assert against unpublished upstream files |
| `tests/test_release_hygiene.py` | repo self-check for paths, secrets, binaries, vendored upstream | keeps the sanitisation audit true for future commits |
| `integration/*.md`, `docs/*.md`, `README.md`, `NOTICE`, `LICENSE`, `requirements.txt`, `.gitignore`, `scripts/example_commands.sh` | packaging and documentation | — |

### DOCUMENTED-BUT-NOT-COPIED — embeds upstream code

Each of these is *mostly* original CLRE work, but each contains verbatim
upstream lines, which makes the file Adapted Material under CC BY-NC-ND 4.0
(or, for DriveTransformer, a derivative of unlicensed code).

| Frozen path | Upstream overlap | Donor | Where it is documented |
| --- | --- | --- | --- |
| `newborn_vad/control/peb_v2_controller.py` | 38 / 164 lines (23.2 %) | `Bench2DriveZoo/team_code/pid_controller.py` — the aim-point and all four steering angle computations, copied verbatim so steering stays bit-identical to the baseline | `integration/peb_v2_controller_patch.md` |
| `newborn_vad/agents/candidate_a_agent.py` | 56 / 548 (10.2 %) | `Bench2DriveZoo/team_code/vad_b2d_agent.py` — input construction and the native control path | `integration/vad_integration.md` |
| `newborn_vad/agents/vad_trace_native_agent.py` | 58 / 289 (20.1 %) | same | `integration/vad_integration.md` |
| `newborn_vad/agents/candidate_a_uniad_agent.py` | 107 / 666 (16.1 %) | `Bench2DriveZoo/team_code/uniad_b2d_agent.py` | `integration/uniad_integration.md` |
| `newborn_vad/agents/candidate_a_drivetransformer_agent.py` | 155 / 787 (19.7 %) | `DriveTransformer/team_code/drivetransformer_b2d_agent.py` (**no licence**) | `integration/drivetransformer_integration.md` |
| `tools/closedloop/b2d_route_runner_lib.py` | — | no upstream code, but it is machine-specific campaign orchestration carrying absolute paths | `scripts/example_commands.sh` |

The integration guides give the exact insertion points, the data contract and
the invariants that must hold. They do not reproduce upstream lines: you
already have those in your own checkout.

For `peb_v2_controller.py` specifically, the split is clean. The longitudinal
branch law is 100 % CLRE work and is published as
`clre/control/peb_longitudinal.py`; the steering block is 100 % upstream and
is not. At release time the extracted law was verified to reproduce the frozen
subclass's longitudinal outputs **exactly on 1260 / 1260 synthetic ticks**,
with all 12 execution-layer telemetry keys present — see
[PROVENANCE.md](PROVENANCE.md) §3.

### EXCLUDED — out of scope for the paper

| Frozen path(s) | Reason |
| --- | --- |
| `newborn_vad/planning/hipad_adapter.py`, `newborn_vad/agents/candidate_a_hipad_agent.py`, `configs/hipad/*`, `tests/test_hipad_adapter.py` | HiP-AD is not a backbone in this paper. (Its upstream licence is Apache-2.0, so this is a scope decision, not a licence one.) |
| `newborn_vad/planning/orion_lite_adapter.py`, `newborn_vad/agents/candidate_a_orion_lite_agent.py`, `tests/test_orion_lite_adapter.py`, `tools/probes/orion_lite_stochastic_probe.py` | ORION-Lite is not a backbone in this paper, **and** its upstream repository publishes no licence file. |
| `newborn_vad/planning/bicycle.py`, `newborn_vad/planning/ellipse_risk.py`, `newborn_vad/agents/vad_planning_probe_agent.py`, `tests/test_planning_modules.py` | Tier-0 feasibility prototypes that preceded the method; superseded by `refinement.py` and not part of the published pipeline. |
| `newborn_vad/agents/ablation_*.py`, `newborn_vad/agents/latency_audit_*.py`, `newborn_vad/tracing/latency_probe.py`, `newborn_vad/replay/vad_replay.py` | Ablation and instrumentation subclasses of the unpublishable agents; meaningless without them. |
| `tools/diagnosis/*` (≈ 50 files), `tools/paper/*`, `tools/closedloop/run_*.py`, `tools/visualization/*`, `tools/audit_reconciliation/*` | Campaign orchestration and one-off analysis scripts. Machine-specific absolute paths, hard-coded campaign manifests, and dependence on unpublished experiment trees. |
| `experiments/` (537 GB), `carla_runs/` (31 GB), `artifacts/` (868 MB), `logs/` | Experiment outputs: evaluator results, per-tick traces, CARLA recordings, intermediate CSV/JSON. |
| `docs/` of the frozen tree (≈ 36 design and audit documents), `CLAUDE.md` | Internal research records, negative results and working notes. Valuable, but not release material, and they reference internal paths throughout. |
| All `__pycache__/`, `.pytest_cache/` | Caches. |

## 3. Conclusion

* **May legally be redistributed here:** every file listed PUBLIC above. They
  are the authors' own work, contain no third-party source, and are released
  under MIT.
* **May not:** every file listed DOCUMENTED-BUT-NOT-COPIED, because of
  CC BY-NC-ND 4.0's NoDerivatives term (Bench2Drive, Bench2DriveZoo) or the
  absence of any licence grant (DriveTransformer).
* **Needs manual review before any change of position:** the Apache-2.0
  provenance of the VAD/UniAD sub-trees inside Bench2DriveZoo (§1). Until
  resolved, the stricter reading is applied.
* **Users must obtain for themselves:** CARLA, Bench2Drive, Bench2DriveZoo,
  DriveTransformer, and all checkpoints.

Downstream users should note that obtaining and using Bench2Drive or
Bench2DriveZoo remains governed by those projects' own NonCommercial terms.
That is a constraint on using those dependencies, not a change to the MIT
license under which CLRE's own code is released.
