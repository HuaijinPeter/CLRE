# CLRE — Closed-Loop Refinement and Execution for Learned Driving Planners

Reference implementation of **CLRE**, a training-free layer that sits between a
frozen learned end-to-end driving planner and the vehicle. CLRE never retrains
or edits the upstream planner: at every control step it takes that planner's
nominal trajectory and its predicted agents, re-solves a short
receding-horizon refinement problem around them, screens the candidates with a
prediction-conditioned oriented-bounding-box (OBB) feasibility gate, and hands
the survivor to a safety-aware execution layer.

Closed-loop driving performance is the target, and the paper's main
performance results are evaluated in CARLA under the Bench2Drive protocol. The
paper additionally reports offline solver and sensitivity analyses, which do
not involve closed-loop execution.

> **Paper.** *Closed-Loop Refinement and Execution for Learned Driving
> Planners.* — [arXiv:2610.00992](https://arxiv.org/abs/2610.00992).
> See [Citation](#citation).
>
> This repository is the **method code only**. It deliberately does not
> contain the benchmark, the simulator, the upstream planners, their
> checkpoints, or any experiment output. See
> [Installation](#installation) and [docs/LICENSE_AUDIT.md](docs/LICENSE_AUDIT.md).

---

## 1. Pipeline

```
          ┌──────────────────────────┐
          │  frozen learned planner  │   VAD / UniAD / DriveTransformer
          │  (never retrained)       │   — not included, see Installation
          └────────────┬─────────────┘
                       │  nominal trajectory  +  predicted agents  +  map
                       ▼
          ┌──────────────────────────┐
          │  receding-horizon        │   clre/planning/refinement.py
          │  refinement              │   9-term objective, kinematic bicycle
          │                          │   rollout, 8 behaviour candidates
          │                          │   + warm start, Adam
          └────────────┬─────────────┘
                       │  candidate trajectories
                       ▼
          ┌──────────────────────────┐
          │  prediction-conditioned  │   clre/planning/refinement.py
          │  OBB feasibility gate    │   exact signed OBB clearance, ego box vs
          │                          │   each predicted agent box
          └────────────┬─────────────┘
              feasible │                      │ none feasible
              (min J)  │                      ▼
                       │          ┌──────────────────────────┐
                       │          │ route-centerline backup  │  clre/control/
                       │          │ (hold latched speed)     │  route_centerline.py
                       │          └────────────┬─────────────┘
                       ▼                       ▼
          ┌───────────────────────────────────────────────────┐
          │  execution layer                                  │
          │   forward-radar speed bound  clre/control/radar_safety.py
          │   saturated proportional     clre/control/peb_longitudinal.py
          │   error braking (PEB-v2)                          │
          └────────────────────────┬──────────────────────────┘
                                   ▼
                        throttle / brake / steer
```

Three properties are structural rather than tuned, and they are the reason the
layer behaves the way the paper reports:

* **Progress cannot buy predicted contact.** The OBB gate is a hard filter
  applied *after* optimisation. An infeasible candidate can never win on the
  scalar objective, however much progress it promises.
* **The radar supervisor is independent of planner predictions.** It consumes
  only measured range and closing rate, and does not depend on predicted agent
  motion, so it cannot inherit a prediction defect; it also covers static
  props that are absent from the planner's agent set entirely. The rest of the
  execution layer does track the selected reference trajectory — the
  independence claim is about the radar supervisor specifically, not the whole
  layer.
* **"No feasible candidate" is not treated as "there is an obstacle".** It
  means the optimiser has nothing to say this tick, so the vehicle follows the
  navigation centerline at its latched entry speed instead of braking. Braking
  on that signal was measured to convert a prediction defect into a stall.

## 2. What is in this repository

| Path | Role |
| --- | --- |
| `clre/planning/refinement.py` | The CLRE core: objective terms, candidate seeds, warm-start buffer, kinematic rollout, OBB feasibility gate, candidate selection, solver, trace fields. |
| `clre/planning/reference_prototype.py` | Single-candidate-loop offline reference implementation of the same objective. Kept so the batched solver can be checked against it term by term. |
| `clre/control/radar_safety.py` | Forward-radar speed bound from the constant-deceleration stopping-distance law, with corridor/height gating, column-layout self-check and TTC emergency persistence. |
| `clre/control/peb_longitudinal.py` | Saturated proportional-error braking (PEB-v2), as upstream-free pure functions. |
| `clre/control/route_centerline.py` | Route-centerline backup: dense navigation route → ego-frame waypoints. |
| `clre/adapters/uniad_adapter.py` | UniAD → CLRE pure data adapter. |
| `clre/adapters/drivetransformer_adapter.py` | DriveTransformer → CLRE pure data adapter. |
| `clre/tracing/trace_writer.py` | Buffered per-tick `npz` trace writer used by the analyses. |
| `integration/` | How to attach CLRE to each upstream planner, including the parts that cannot be redistributed. |
| `docs/` | Reproducibility manifest, parameters, provenance, license audit, sanitisation report. |
| `tests/` | Offline tests. No CARLA, no checkpoint, no GPU. |

**Not in this repository, by design:** the benchmark, the simulator, the
upstream planners, any checkpoint, any dataset, and every experiment output
(evaluator results, per-tick traces, CARLA recordings, figures, logs). The
agent wrappers that embed upstream code are described in `integration/` rather
than shipped — see §6 and [docs/LICENSE_AUDIT.md](docs/LICENSE_AUDIT.md).

## 3. Supported / tested upstream planners

| Planner | Role in the paper | Status |
| --- | --- | --- |
| **VAD** (`Bench2DriveZoo`) | **Main evaluation backbone.** All headline closed-loop results and all ablations. | Tested end to end. |
| **UniAD** (`Bench2DriveZoo`) | Additional demonstration that the layer is not VAD-specific. | Adapter released; evaluated on a small route subset only. |
| **DriveTransformer-Large** | Additional demonstration. | Adapter released; evaluated on a small route subset only. |

The UniAD and DriveTransformer arms are *demonstrations of portability on a
small number of routes*, not full benchmark comparisons. Do not read them as
equivalent evidence to the VAD results.

## 4. Environment used for the paper

| Component | Version / commit |
| --- | --- |
| CARLA | 0.9.15 |
| Bench2Drive | `v0.0.4` — `7ec25d1c9f7522d923ce5f3420986cef1cb2d956` |
| Bench2DriveZoo (VAD, UniAD) | `v0.0.4` — `d9caa0af805ec3c435533aa268e2723d80c20017` |
| DriveTransformer | `19702015eb700e804bafb53e63b04571c7b8604a` |
| CLRE frozen research commit | `b1f509df68c427fc128d8b1e7c86520a304c7afb` |
| Python / PyTorch | 3.8 / 1.13 + cu117 (the Bench2DriveZoo environment) |

Full hashes, checkpoint names, route files and the exact frozen runtime
configuration are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## 5. Installation

CLRE itself has almost no dependencies; the heavy stack belongs to the
upstream projects, and this repository intentionally points you at them
instead of vendoring them.

**Step 1 — install the upstream stack from its own official sources.**

```bash
# CARLA 0.9.15 — https://github.com/carla-simulator/carla/releases/tag/0.9.15
# Bench2Drive  — https://github.com/Thinklab-SJTU/Bench2Drive
git clone https://github.com/Thinklab-SJTU/Bench2Drive.git
git -C Bench2Drive checkout 7ec25d1c9f7522d923ce5f3420986cef1cb2d956

# Bench2DriveZoo (VAD / UniAD) — https://github.com/Thinklab-SJTU/Bench2DriveZoo
git clone https://github.com/Thinklab-SJTU/Bench2DriveZoo.git
git -C Bench2DriveZoo checkout d9caa0af805ec3c435533aa268e2723d80c20017

# optional, only for the DriveTransformer demonstration
# https://github.com/Thinklab-SJTU/DriveTransformer
```

Follow each project's own installation instructions (they build a merged
`mmcv`/`mmdet3d` fork and CUDA ops). CLRE adds nothing to that process.

**Step 2 — install CLRE.**

```bash
git clone <this-repository> CLRE
cd CLRE
pip install -r requirements.txt     # numpy, torch only
```

**Step 3 — verify the offline tests pass.** These need no checkpoint, no
simulator and no GPU:

```bash
python -m pytest tests/ -q
```

## 6. Where CLRE files go, and how the adapters connect

CLRE is an **overlay**: keep the upstream repositories unmodified where you
can, put `CLRE/` on `PYTHONPATH`, and add one agent file of your own that
subclasses the upstream agent.

```
<workspace>/
├── Bench2Drive/            # upstream, unmodified
├── Bench2DriveZoo/         # upstream, unmodified
├── CLRE/                   # this repository
│   └── clre/...
└── my_agents/
    └── clre_vad_agent.py   # YOU write this, following integration/vad_integration.md
```

```bash
export PYTHONPATH=/path/to/CLRE:/path/to/Bench2DriveZoo:/path/to/Bench2Drive/leaderboard:$PYTHONPATH
```

The agent file is yours to write because the published version of it embeds
upstream code that the upstream licences do not let us redistribute. Each
guide therefore gives you the exact insertion points, the data contract, and
the invariants that must hold — everything except the upstream lines you
already have in your own checkout:

* [`integration/vad_integration.md`](integration/vad_integration.md) — the main backbone.
* [`integration/uniad_integration.md`](integration/uniad_integration.md)
* [`integration/drivetransformer_integration.md`](integration/drivetransformer_integration.md)
* [`integration/peb_v2_controller_patch.md`](integration/peb_v2_controller_patch.md) — wiring the braking law into the upstream PID controller without touching its steering path.

A one-screen sketch of the per-tick call sequence (full version in the VAD
guide). Note that **three distinct outcomes execute three different
trajectories** — `res.ok` alone does not decide what is driven:

```python
from clre.planning import refinement as clre
from clre.control.radar_safety import RadarSafetyOverride
from clre.control.route_centerline import RouteCenterline

# once, in setup()
cfg  = clre.PlannerConfig(device="cpu", dtype="float64")
clre.apply_fixes(cfg, "r1,r2,r3,geom,feas")   # REQUIRED: see docs/REPRODUCIBILITY.md
warm = clre.WarmBuffer(tick_dt=0.05)
radar        = RadarSafetyOverride()              # measured returns only
centerline   = RouteCenterline.from_agent(self)   # once, from the dense GPS route
hold_speed   = None                               # latched at fallback entry

# ---- every tick, after the upstream forward pass ----
warm.advance()                                 # EVERY tick, before solving
centerline.note_position(pos)                  # EVERY tick, keeps the hint fresh

window = clre.make_window(speed, nominal_plan, route_target_xy,
                          clre.agents_from_arrays(..., fix_r1=cfg.fix_r1),
                          clre.boundaries_from_map(...),
                          v_ref_in=max(clre.native_desired_speed(nominal_plan), 2.5))
res = clre.solve(window, cfg, warm=warm.seed())

exec_plan, exec_source = nominal_plan, "native"      # the default is Native

# ---- stage 1: may the refined plan be adopted at all? ----
if not res.ok:
    # solver failure: non-finite objective/trajectory, or hard-bound
    # violation (res.reason says which). Execute Native, and DROP the warm
    # start -- rejected controls must not seed the next cycle.
    warm.reset()

elif res.solve_ms > cfg.runtime_budget_ms:
    # (3) RUNTIME-BUDGET FAILURE -> Native trajectory.
    #     The solution is numerically valid, it merely arrived late, so it is
    #     DISCARDED for execution but KEPT as the warm start.
    warm.update(res.a, res.k)

else:
    warm.update(res.a, res.k)
    exec_plan, exec_source = res.wp6, "clre"         # (6, 2) waypoints @ 0.5 s

# ---- stage 2: was the adopted plan actually feasible? ----
# res.ok == True does NOT mean a feasible candidate was found: when the OBB
# gate rejects every candidate, solve() still returns ok=True and res.wp6
# holds the max-min-clearance candidate. That candidate must NOT be executed.
want_fallback = (exec_source == "clre"
                 and res.sel_mode == "all_infeasible_safest")

if want_fallback and centerline is not None:
    # (2) COMPLETED SOLVE, NO FEASIBLE CANDIDATE -> centerline backup.
    #     Follow the navigation centerline at the ego speed LATCHED at
    #     fallback entry. Never re-read the current speed per tick.
    if hold_speed is None:
        hold_speed = speed
    exec_plan, exec_source = (
        centerline.waypoints_ego(pos, compass_rad, hold_speed), "clre_fallback")
else:
    # (1) FEASIBLE COMPLETED SOLVE -> the refined trajectory already adopted
    #     above, with res.sel_mode == "feasible_best" ("unconstrained" when
    #     the OBB gate is disabled, which is not the paper's configuration).
    #     Also reached on the Native-fallback ticks, where nothing is latched.
    hold_speed = None                    # episode over, or never began

# The radar supervisor runs in ALL of the above cases, on measured returns only
speed_cap, radar_tel = radar.evaluate(radar_points, speed)
```

The `centerline is not None` guard is load-bearing: if
`RouteCenterline.from_agent` returned `None` (the dense GPS route was
unavailable), case (2) cannot be served, and the frozen agent then counts a
`centerline_unavailable` event and executes the `all_infeasible_safest`
candidate rather than inventing geometry. Count that event, do not hide it.

## 7. Example commands

**Do not expect these to run unchanged.** They record the shape of the paper's
evaluation so it can be reconstructed; paths, ports and the agent file are
yours. Each one needs CARLA, a GPU and a checkpoint.

```bash
# --- the frozen configuration used for every CLRE arm in the paper ---
export NEWBORN_CANDA_MODE=exec          # execute the refined plan
export NEWBORN_CANDA_DEVICE=cpu         # solver device (paper: CPU, float64)
export NEWBORN_CANDA_DTYPE=float64
export NEWBORN_CANDA_ITERS_COLD=70      # Adam iterations, cold start
export NEWBORN_CANDA_ITERS_WARM=25      # Adam iterations, warm start
export NEWBORN_CANDA_BUDGET_MS=1000.0   # per-tick budget; overrun => use Native plan
export NEWBORN_CANDA_FIXES=r1,r2,r3,geom,feas   # MANDATORY, see below
export NEWBORN_PEBV2=1                  # enable the execution layer

# --- main protocol: the official 220 validation routes ---
python Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py \
    --agent=my_agents/clre_vad_agent.py \
    --agent-config=Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py \
    --routes=Bench2Drive/leaderboard/data/bench2drive_0.0.4_val.xml \
    --checkpoint=results/clre_vad_official220.json \
    --track=SENSORS --port=2000 --traffic-manager-port=8000

# --- matched baseline: the same routes, upstream planner alone ---
#     (use the stock Bench2DriveZoo agent, unmodified)

# --- ablation protocol: the first 36 routes of the same file ---
#     (same command, restricted to routes 1-36; see docs/REPRODUCIBILITY.md)
```

`NEWBORN_CANDA_FIXES=r1,r2,r3,geom,feas` is **not optional**. With it unset
the refinement layer runs an earlier formulation: no three-disk ego geometry
and, critically, **no OBB feasibility gate** (`feas`). Every number in the
paper was produced with all five flags on.

No module in `clre/` reads these variables — they are consumed by the agent
file you write, which passes the values into `PlannerConfig` and
`apply_fixes()` (see `integration/vad_integration.md` §3). They are listed
here with their frozen research spelling so your runs stay directly comparable
to the published configuration; keeping that spelling is what lets the
released modules stay byte-identical to the evaluated ones, rather than being
a requirement of the library itself.

## 8. Checkpoints

**No checkpoint is distributed here, and none can be reconstructed from this
repository.** Obtain each one from its original provider:

| Planner | Checkpoint | Source |
| --- | --- | --- |
| VAD | `vad_b2d_base.pth` | Bench2DriveZoo |
| UniAD | `uniad_base_b2d.pth` | Bench2DriveZoo |
| DriveTransformer | `drivetransformer_large.pth` | DriveTransformer |

Point the adapters at your own copies with `CLRE_UNIAD_CKPT` and
`CLRE_DT_CKPT`. The SHA-256 values the paper runs verified are recorded in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) so you can confirm you have
the same weights — they are not download links.

## 9. Reproducibility scope

Be aware of what this repository can and cannot give you.

* **Fully reproducible here:** the refinement objective, the candidate
  initialisation and warm start, the OBB gate, candidate selection, the
  route-centerline backup, the radar bound, the braking law, and the UniAD /
  DriveTransformer data adapters — together with offline tests that check the
  batched solver against the reference implementation and pin the execution
  layer's behaviour.
* **Reproducible only with upstream components you install yourself:** the
  agent wrappers (`integration/`), and therefore any closed-loop run.
* **Not reproducible bit-for-bit by anyone, including us:** closed-loop CARLA
  runs are not deterministic. Repeated executions of identical logic in this
  stack were measured to differ by up to 28 Driving Score on individual
  routes. That is why the paper pairs every comparison by route and interprets
  no single-route difference on its own; see
  [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

We therefore do **not** claim full end-to-end reproducibility from this
repository alone. Two upstream dependencies carry licence terms that forbid
redistributing modified copies, and one publishes no licence at all, so the
glue code that binds CLRE to them must be reconstructed from the integration
guides in your own checkout.

## 10. Citation

```bibtex
@misc{hu2026clre,
  title         = {Closed-Loop Refinement and Execution for Learned Driving Planners},
  author        = {Hu, Huaijin and Wang, Shanting and Mo, Zhongyu and Malikopoulos, Andreas A.},
  year          = {2026},
  eprint        = {2610.00992},
  archivePrefix = {arXiv}
}
```

Please also cite the upstream work you build on — at minimum CARLA,
Bench2Drive, and the planner you evaluate (VAD, UniAD or DriveTransformer).

## 11. License and third-party attribution

The code in this repository is released under the [MIT License](LICENSE) and
contains no third-party source.

It is built to run against third-party projects that are **not** included
here. Their terms are not ours to waive, and two of them constrain what you
may do with the combination:

| Project | License | Consequence |
| --- | --- | --- |
| CARLA 0.9.15 | MIT | Permissive. |
| CARLA leaderboard / scenario_runner | MIT | Permissive. |
| Bench2Drive | CC BY-NC-ND 4.0 | **NonCommercial**, and **no redistribution of modified copies**. |
| Bench2DriveZoo (VAD / UniAD) | CC BY-NC-ND 4.0 | Same. |
| DriveTransformer | **no license file published** | Treat as all-rights-reserved; ask the authors before redistributing anything derived from it. |

Use of Bench2Drive or Bench2DriveZoo remains subject to their NonCommercial
license terms, independently of CLRE's MIT license. See [NOTICE](NOTICE) and
[docs/LICENSE_AUDIT.md](docs/LICENSE_AUDIT.md) for the full audit, including
which files were deliberately not published and why.
