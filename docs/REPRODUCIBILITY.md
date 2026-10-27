# Reproducibility manifest

Everything a third party needs in order to tell whether they are running the
same method, on the same benchmark, against the same weights as the paper.

This document records **identity and protocol**, not results. It deliberately
contains no Driving Score numbers: those belong to the paper, and quoting them
here would create a second, drift-prone copy.

---

## 1. Frozen source identity

| Item | Value |
| --- | --- |
| CLRE frozen research commit | `b1f509df68c427fc128d8b1e7c86520a304c7afb` |
| Frozen research branch | `phase4b3/e59-pebv2` |
| Method freeze status | `PERMANENT_FROZEN`, 2026-08-27 |
| This public release | derived from that commit; see [PROVENANCE.md](PROVENANCE.md) for the per-file mapping and the complete list of edits |

The frozen tree carried its own `sha256` manifest over the eight files whose
contents define the method. Four of those files are published here verbatim
and their hashes are therefore directly checkable:

| Frozen path | SHA-256 | In this release |
| --- | --- | --- |
| `newborn_vad/planning/candidate_a_planner.py` | `ff1df4f1f3055be09edd17c0776d9f02d2122beec09304fc20505a28b4c10236` | `clre/planning/refinement.py`, byte-identical |
| `newborn_vad/control/radar_safety.py` | `f15cd190d4db27da89144cbdef216930ed7855dc51b2a888d2b2288445d8f5fc` | `clre/control/radar_safety.py`, byte-identical |
| `newborn_vad/control/route_centerline.py` | `ea372514e9514965f45a17c956fa8af1878f7ab12488423e51fa8afc1dee843b` | `clre/control/route_centerline.py`, byte-identical |
| `newborn_vad/control/peb_v2_controller.py` | `53dfa6cfbec27a040025e1d62dabad1198c88255efef4b36e58b27ec8fb2c7f7` | **not published** (embeds upstream code); its longitudinal law is re-released upstream-free as `clre/control/peb_longitudinal.py` |
| `newborn_vad/agents/candidate_a_agent.py` | `80ed3c32248adff0867ff7e76f459d2afeab5cd1787451f9dd23eede5e549239` | **not published**; reconstruct from `integration/vad_integration.md` |
| `newborn_vad/agents/vad_trace_native_agent.py` | `5604f79ad26c21185ccb7efe91b0564ee52ff95856c24b38255287382a4e78c6` | **not published**; same guide |
| `external/v004/Bench2DriveZoo/team_code/pid_controller.py` | `397881ba9dee58cee8f2daa30cb73483ce8c25e778261bf2f20559ab6d9ae891` | upstream; install it yourself |
| `external/v004/Bench2DriveZoo/team_code/vad_b2d_agent.py` | `2dcf7e1523e1bdda2cd2a6972612d2074e1c9d0ca866d07464468e5ade55277c` | upstream; install it yourself |

Verify a published file with:

```bash
sha256sum clre/planning/refinement.py
# ff1df4f1f3055be09edd17c0776d9f02d2122beec09304fc20505a28b4c10236
```

## 2. Upstream dependency versions

| Project | Commit | Tag / release | Source |
| --- | --- | --- | --- |
| CARLA | — | **0.9.15** (packaged release) | https://github.com/carla-simulator/carla/releases/tag/0.9.15 |
| Bench2Drive | `7ec25d1c9f7522d923ce5f3420986cef1cb2d956` | "Release Bench2Drive 0.0.4" | https://github.com/Thinklab-SJTU/Bench2Drive |
| Bench2DriveZoo | `d9caa0af805ec3c435533aa268e2723d80c20017` | "Release Bench2DriveZoo 0.0.4 (uniad/vad)" | https://github.com/Thinklab-SJTU/Bench2DriveZoo |
| DriveTransformer | `19702015eb700e804bafb53e63b04571c7b8604a` | — | https://github.com/Thinklab-SJTU/DriveTransformer |

The CARLA `leaderboard` and `scenario_runner` used are the copies vendored
inside the Bench2Drive tree at the commit above, not separate clones.

**Python environment.** The paper runs used the Bench2DriveZoo environment:
Python 3.8, PyTorch 1.13 + CUDA 11.7, with Bench2DriveZoo's merged
`mmcv`/`mmdet`/`mmdet3d` fork and its compiled CUDA ops. CLRE adds only
`numpy` and `torch`, both already present there.

## 3. Checkpoints — expected names and hashes, not the weights

No checkpoint is distributed in this repository, and none can be
reconstructed from it. Obtain each from its original provider, then confirm
you have the same weights the paper used:

| Planner | Expected file name | SHA-256 | Provider |
| --- | --- | --- | --- |
| VAD (main backbone) | `vad_b2d_base.pth` | `0f2d48b5c6cdb06d0dffb21612791fe740f3f5c7273b3b5e78e1f19e4a527eab` | Bench2DriveZoo |
| UniAD | `uniad_base_b2d.pth` | `8c6fb898033ef62e0f20a45ff68d554c131c0b58ca405d57bb5bfa4af1f8dcb0` | Bench2DriveZoo |
| DriveTransformer-Large | `drivetransformer_large.pth` | `982f2024cb17f59541d0333a1b48ad74267ee0ba91eca3abe880afc60ed8c2f5` | DriveTransformer |

The DriveTransformer project publishes no official checkpoint hash; the value
above is the one observed locally when the release weights were downloaded, so
treat it as "the file we used", not as an upstream-sanctioned pin.

Point the adapters at your copies:

```bash
export CLRE_UNIAD_CKPT=/path/to/uniad_base_b2d.pth
export CLRE_DT_CKPT=/path/to/drivetransformer_large.pth
```

**Planner configs** (from the Bench2DriveZoo checkout):

* VAD: `adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py`
* UniAD: `adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py`

## 4. Frozen CLRE parameters

The complete table, cross-referenced to the paper's symbols and to the
constant that carries each value in the released source, is in
[PARAMETERS.md](PARAMETERS.md). The summary:

**Refinement and solver.** Horizon `N = 12` steps at `Δt = 0.25 s` (3.0 s);
acceleration bound `[-6.0, 2.5] m/s²`; curvature bound `|κ| ≤ 0.2 m⁻¹`;
speed `[0, 15] m/s`; Adam step size `0.06`; `70` iterations cold, `25` warm;
8 behaviour candidates plus the warm-started solution and its unrefined
time-shifted copy (10 maximum); ego box `4.90 × 1.86 m`; ego disk radius
`1.3 m`, three disks offset `±1.633 m` along heading; OBB clearance threshold
`ε = 0.05 m`.

**Objective weights.** `w_pt = 0.05`, `w_ct = 1.2`, `w_prog = 1.0`,
`w_v = 0.6`, `w_risk (outer ellipse) = 1.6`, `w_bar (inner ellipse) = 25.0`,
`w_bnd = 6.0`, `w_cor = 0.5`, `w_sm = 0.8`.

**Cost-term shape.** Huber `δ = 1.0 m`; commitment gate
`σ(2 (s_nat − 1.5 m))`; outer-ellipse margins `(0.6, 0.4) m` with softplus
sharpness `4`; inner-ellipse margin `0.05 m` with sharpness `12`; boundary
margin `0.6 m`, sharpness `8`; corridor half-width `4.0 m`; smoothness
`λ_κ = 30`, `λ_0 = 3`; reference-speed clamp `[0.5, 8.0] m/s`.

**Scene selection.** At most 8 predicted agents within 30 m with detection
confidence ≥ 0.30, ordered by distance, top-probability motion mode. Up to 4
additional actors with confidence in `[0.15, 0.30)` within 20 m whose
predicted 3 s displacement exceeds 2 m. Map elements restricted to
solid-marking boundary polylines with score > 0.3.

**Execution layer.** `E_db = 0.5 m/s`, `E_full = 2.0 m/s`; radar
`a_brake = 4.0 m/s²`, `t_reaction = 0.4 s`, `d_standoff = 2.0 m`; TTC
emergency `1.2 s` sustained for `3` consecutive frames; corridor `±1.2 m`;
accepted height band `[0.30, 2.20] m`; range gate `[0.5, 60] m`.

**Sensing.** The upstream planner's own suite (6 RGB cameras, IMU, GPS,
speedometer) plus **one** forward radar at `x = 2.0 m`, `z = 1.0 m`, FOV
`30° × 10°`, within the benchmark's allowance of four. The leaderboard
overrides radar `range` to 100 m and `points_per_second` to 1500 regardless of
what the agent requests, so only the FOV and pose are actually the agent's.

> **Arms without the execution layer run with one sensor fewer.** Comparisons
> involving that layer are therefore not sensor-matched. The paper states this
> wherever it applies, and so should any reuse.

### Mandatory runtime configuration

Every CLRE arm in the paper ran with exactly this environment. The names keep
their frozen research spelling so released code stays byte-comparable to the
evaluated code.

```bash
NEWBORN_CANDA_MODE=exec
NEWBORN_CANDA_DEVICE=cpu
NEWBORN_CANDA_DTYPE=float64
NEWBORN_CANDA_ITERS_COLD=70
NEWBORN_CANDA_ITERS_WARM=25
NEWBORN_CANDA_BUDGET_MS=1000.0
NEWBORN_CANDA_FIXES=r1,r2,r3,geom,feas
NEWBORN_PEBV2=1
```

`NEWBORN_CANDA_FIXES` is load-bearing and defaults to `none`:

| Flag | Effect if present |
| --- | --- |
| `r1` | additive inclusion of near, moving, low-confidence actors |
| `r2` | anchors each agent's interpolated prediction at its observed position |
| `r3` | extends the prior path along its terminal tangent, so a stalled prior carries no along-track stop authority |
| `geom` | evaluates the ellipse field at three ego disk centres instead of one (completes the longitudinal geometry) |
| `feas` | **enables the OBB hard feasibility gate** |

Running without `feas` disables the paper's hard safety gate entirely. An
unset `NEWBORN_CANDA_FIXES` does not reproduce the paper's method.

## 5. Evaluation protocol

### Main protocol — official 220 routes

| Item | Value |
| --- | --- |
| Route file | `Bench2Drive/leaderboard/data/bench2drive_0.0.4_val.xml` |
| SHA-256 | `885debbc3d87f14351b521d8da0dc9d25fcee1f0d926ed1c92cc9ff62504ca45` |
| Routes | 220 (the whole file) |
| Executions | one per route per arm |
| Control rate | 20 Hz; refinement invoked every tick |
| Arms | CLRE vs the matched upstream planner alone |

There are two different 220-route files in circulation in this ecosystem.
`bench2drive220.xml` is **not** the official validation set and shares no
routes with it. The official set is `bench2drive_0.0.4_val.xml`; check the
hash above before trusting a comparison.

### Ablation protocol — first 36 routes

The ablations use the **first 36 routes of that same file, in original XML
order** — a cohort fixed before any run in the campaign and not selected on
any outcome.

| Ablation arm | One factor changed |
| --- | --- |
| w/o progress | the forward-progress term removed from the objective |
| w/o feasibility test | `NEWBORN_CANDA_FIXES=r1,r2,r3,geom` — the OBB gate omitted |
| w/o execution layer | `NEWBORN_PEBV2=0` — CLRE refinement with the upstream PID |

Each arm changes exactly one factor against the full method, on the same
routes, paired by route.

> This section describes the protocol of the **current** paper source
> (`paper/final_paper.tex`: 220 routes for the main evaluation, first 36 for
> the ablation table). A superseded draft in the research tree reports the
> 36-route cohort as the main protocol and only the execution-layer ablation
> — see [PROVENANCE.md](PROVENANCE.md) §6 before reconciling anything against
> it.

### Metrics and statistics

Official Bench2Drive metrics: Driving Score (DS, the primary endpoint), Route
Completion, Success, and per-category infraction counts. Mechanism read-outs
(braking statistics, speed, fallback activation, solver latency) come from the
per-tick traces written by `clre/tracing/trace_writer.py`; they describe *how*
an arm behaves, not how well it scores.

Closed-loop runs in this stack are **not deterministic**: same-logic re-runs
were measured to differ by up to 28 DS on individual routes. Consequently:

* all comparisons are **paired by route**;
* no single-route difference is interpreted on its own;
* paired differences use the two-sided exact sign test with `|ΔDS| ≤ 0.5`
  treated as a tie, plus a bootstrap 95 % CI for the mean difference;
* binary outcomes use McNemar's exact test;
* decision rules were registered before the corresponding runs, and anything
  reported outside that registration is labelled post-hoc.

If you re-run the benchmark, expect per-route disagreement with the paper and
compare distributions over the cohort, not individual routes.

## 6. What requires CARLA / GPU execution

| Component | CARLA | GPU | Checkpoint |
| --- | --- | --- | --- |
| `clre/planning/refinement.py` | no | no (CPU float64 in the paper) | no |
| `clre/control/*` | no | no | no |
| `clre/adapters/*` (conversion + assertions) | no | no | no |
| `clre/adapters/*` (`verify_upstream` sha256 gate) | no | no | needs the pinned upstream files on disk |
| `tests/` (all of it) | no | no | no |
| Upstream planner forward pass | no | **yes** | **yes** |
| Any closed-loop route, any DS/RC number, any trace | **yes** | **yes** | **yes** |

The refinement solver ran on **CPU in float64** for every paper run. That is
the configuration the frozen results were produced with; moving it to GPU or
float32 changes round-off and has not been validated against the frozen
numbers.

## 7. Known gaps in reproducibility

Stated plainly rather than implied:

1. **The agent wrappers are not in this repository.** They embed upstream
   code under licences that forbid redistributing modified copies
   ([LICENSE_AUDIT.md](LICENSE_AUDIT.md)). You must reconstruct them in your
   own checkout from `integration/`. A faithful reconstruction is achievable —
   the guides give the insertion points, data contract and invariants — but it
   is reconstruction, not a byte-identical artifact.
2. **Closed-loop results are not bit-reproducible by anyone**, including the
   authors; see §5.
3. **The UniAD and DriveTransformer adapters' full offline suites are not
   released.** They assert against `sha256`-pinned upstream files that are not
   redistributed here. `tests/test_adapters_smoke.py` covers only what is
   checkable standalone.
4. **The DriveTransformer arm depends on a project with no published
   licence.** Its adapter here is original CLRE code and safe to publish, but
   obtaining and using the upstream model is between you and its authors.
