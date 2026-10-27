# Frozen parameters, mapped to the source

Every value below was read back from the released modules at release time, not
transcribed from a design document. Reproduce the table yourself with:

```bash
python - <<'EOF'
from clre.planning import refinement as r
from clre.control import radar_safety as rs, peb_longitudinal as pl
print(r.WEIGHTS, r.FEAS_EPS, r.DT, r.N_STEPS)
print(rs.RadarSafetyOverride().params(), pl.E_DB_DEFAULT, pl.E_FULL_DEFAULT)
EOF
```

Weights compensate for the heterogeneous units and typical magnitudes of the
cost terms. They do **not** encode a priority ordering, and the measured
sensitivity ordering was found to be uncorrelated with weight magnitude.

---

## Refinement and solver

| Paper symbol | Value | Source constant |
| --- | --- | --- |
| `N`, `Δt` | 12, 0.25 s (3.0 s horizon) | `refinement.N_STEPS`, `refinement.DT` |
| acceleration bounds | `[-6.0, 2.5]` m/s² | `refinement.A_MIN`, `A_MAX` |
| curvature bound `|κ|` | 0.2 m⁻¹ (R ≥ 5 m) | `refinement.K_MAX` |
| speed bounds | `[0, 15]` m/s, no reverse | `refinement.V_MAX` |
| Adam step size | 0.06 | `PlannerConfig.lr` |
| iterations | 70 cold / 25 warm | `PlannerConfig.iters_cold`, `iters_warm` |
| candidates `N_c` | 8 behaviour seeds (+ `warm`, `warm_raw`) | `refinement.SEED_NAMES` |
| ego box | 4.90 × 1.86 m | `2 × EGO_HALF_LEN`, `2 × EGO_HALF_W` |
| ego disk radius `R` | 1.3 m | `refinement.R_EGO` |
| ego disk offsets | 0, ±1.633 m along heading | `refinement.EGO_DISC_OFF` |
| OBB clearance `ε` | 0.05 m | `refinement.FEAS_EPS` |
| per-tick budget | 1000 ms (overrun ⇒ execute the Native plan; the solution is still kept as the warm start) | `PlannerConfig.runtime_budget_ms` |
| solver precision | CPU, float64 | `PlannerConfig.device`, `dtype` |

The 8 behaviour candidates are, in order: `native` (reconstruction of the
upstream prior), `brake`, `commit` (accelerate), `swerve_pos_x`,
`brake_swerve_pos_x`, `swerve_neg_x`, `brake_swerve_neg_x`, `creep_release`
(burst-and-coast). The warm start adds the previous cycle's solution,
time-shifted onto the current control grid, as a 9th candidate, and its
**unrefined** copy as a 10th — that second entry is what makes a warm cycle
unable to end worse than the time-shifted previous solution under the reduced
warm iteration budget. All 8 basins keep competing every cycle; the brake
basin is never dropped.

## Objective weights

| Paper symbol | Value | Key in `refinement.WEIGHTS` |
| --- | --- | --- |
| `w_pt` (point-wise prior tracking) | 0.05 | `pt` |
| `w_ct` (geometric path commitment) | 1.2 | `ct` |
| `w_prog` (forward progress) | 1.0 | `prog` |
| `w_v` (speed regularisation) | 0.6 | `v` |
| `w_out` / `w_risk` (outer ellipse) | 1.6 | `risk` |
| `w_in` / `w_bar` (inner ellipse) | 25.0 | `bar` |
| `w_bnd` (road-boundary proximity) | 6.0 | `bnd` |
| `w_cor` (lateral corridor) | 0.5 | `cor` |
| `w_sm` (control regularisation) | 0.8 | `sm` |

> **Notation warning.** The paper's method section and sensitivity analysis
> write the two interaction weights as `w_out` / `w_in`; the code and every
> exported CSV call them `risk` / `bar`. They are the same two numbers. An
> earlier draft's parameter table used `w_risk` / `w_bar` instead, matching the
> code — so both spellings appear in the LaTeX sources. Check which one the
> submitted version uses and make it consistent.

`sum(weighted.values()) == J` holds **by construction** in
`refinement.objective`, not merely to a tolerance: the returned decomposition
is built from the same per-term tensors the objective sums. This is what makes
the per-term trace read-outs exact rather than a parallel re-computation.

## Cost-term shape

| Paper symbol | Value | Where |
| --- | --- | --- |
| Huber transition `δ` | 1.0 m | `objective`, `j_pt` |
| commitment gate `β_γ`, `s_0` | 2, 1.5 m | `make_window`: `γ = σ(2 (s_nat − 1.5))` |
| outer-ellipse margins `(m_u, m_v)`, `β_r` | (0.6, 0.4) m, 4 | `_WindowTensors.ax/bx`, `objective` softplus |
| inner-ellipse margin `m_p`, `β_b` | 0.05 m, 12 | `_WindowTensors.axh/bxh`, `objective` softplus |
| boundary margin `δ_m`, `β_m` | 0.6 m, 8 | `objective`, `j_bnd` |
| corridor half-width `w_c` | 4.0 m | `objective`, `j_cor` |
| smoothness `λ_κ`, `λ_0` | 30, 3 | `objective`, `j_sm` |
| reference speed `v_ref` range | `[0.5, 8.0]` m/s | `make_window` |

Ellipse semi-axes are per-agent and extent-aware: the agent's own half-extents
plus `R_EGO` plus the margin above, evaluated in the agent's frame. The
reference-speed clamp is what bounds the progress reward — progress is linear
in terminal route advance but saturates at `v_ref`, so "faster than sensible"
is never rewarded. The commitment gate is the other side of that trade: a
stalled prior carries no stop authority, so imitation cannot re-create a
no-commit stall.

## Scene selection

| Quantity | Value | Source constant |
| --- | --- | --- |
| detection confidence floor | 0.30 | `AGENT_SCORE_MIN` |
| distance gate | 30.0 m | `AGENT_DIST_MAX` |
| max agents | 8 | `AGENT_MAX` |
| min half-extent | 0.4 m | `EXT_MIN` |
| low-confidence band (additive) | `[0.15, 0.30)` | `R1_SCORE_MIN` |
| low-confidence distance gate | 20.0 m | `R1_DIST_MAX` |
| low-confidence motion gate | 2.0 m over 3 s | `R1_MOVE_MIN` |
| max extra low-confidence actors | 4 | `R1_EXTRA_MAX` |
| boundary score floor | > 0.3 | `BOUNDARY_SCORE_MIN` |
| boundary labels | Solid, SolidSolid = `(1, 2)` | `BOUNDARY_LABELS` |
| prior-path densification | 0.4 m | `densify_path`, `R3_EXT_STEP` |
| prior-path tangent extension | 45 m (≥ `V_MAX × 3 s`) | `R3_EXT_LEN` |

The low-confidence pass is **strictly additive**: the top-8 selection is never
reordered or reduced. It exists so that low-confidence *moving* traffic is not
discarded together with low-confidence static clutter. Only the
top-probability motion mode of each retained agent is used, weighted by its
confidence.

## Execution layer

| Paper symbol | Value | Source |
| --- | --- | --- |
| `Δv_min` / `E_db` (brake onset) | 0.5 m/s | `peb_longitudinal.E_DB_DEFAULT` |
| `Δv_max` / `E_full` (full brake) | 2.0 m/s | `peb_longitudinal.E_FULL_DEFAULT` |
| `a_br` (usable deceleration) | 4.0 m/s² | `RadarSafetyOverride.a_brake` |
| `T_r` (reaction horizon) | 0.4 s | `RadarSafetyOverride.t_reaction` |
| `d_0` (standoff) | 2.0 m | `RadarSafetyOverride.d_standoff` |
| TTC emergency threshold | 1.2 s | `RadarSafetyOverride.ttc_emergency` |
| TTC emergency persistence | 3 consecutive frames (0.15 s at 20 Hz) | `radar_safety.EMERGENCY_PERSIST_N` |
| radar corridor half-width | ±1.2 m | `RadarSafetyOverride.corridor_half_w` |
| accepted height band | `[0.30, 2.20]` m above road | `h_ground_clear`, `h_overhead` |
| radar range gate | `[0.5, 60]` m | `min_range`, `max_range` |
| radar mounting | `x = 2.0`, `z = 1.0` m, FOV 30° × 10° | agent `sensors()`; see `integration/vad_integration.md` |
| radar velocity-column sign | `-1.0` (range rate, negative while closing) | `radar_safety.RADAR_V_SIGN` |

`a_brake = 4.0` is deliberately below the planner's own `-6.0` bound: the
supervisor reserves authority rather than assuming the full envelope.

Two things in this layer are measured facts about the simulator rather than
tuned values, and both are load-bearing:

* **`RADAR_V_SIGN = -1.0`.** CARLA's documentation describes the velocity
  column as the object's speed *towards* the sensor, which would make positive
  mean closing. It was measured to be the opposite in this build — a range
  rate, negative while closing, matching ego speed on static clutter to 0.7 %.
  The first draft shipped `+1.0` and the layer was structurally inert for a
  whole run. `radar_v_static_med` is logged every tick for exactly this
  reason, and must track `+ego_speed`; it is the first number to read on any
  new run, before any behavioural statistic.
* **Radar column order** is `[depth, altitude, azimuth, velocity]` as
  delivered to the agent, derived from CARLA's `RadarDetection` wire struct
  plus Bench2Drive's `np.flip`. Two nearby upstream comments state a different
  order and are wrong. `assert_layout_plausible()` cross-checks it at runtime
  (azimuth must spread wider than altitude, since horizontal FOV is 3 ×
  vertical) and goes **fail-transparent** on mismatch — publishing no ceiling
  for the rest of the route rather than gating on garbage columns.

The leaderboard overrides radar `range` to 100 m and `points_per_second` to
1500 regardless of the agent's request, giving roughly 75 returns per frame
across the whole FOV. This layer is therefore a coarse supervisor, not a
tracker: it takes the closest in-corridor return and never associates across
frames.

### Documented geometric limits

The height gate is computed in the sensor frame and is exact only on a flat
road at zero ego pitch. A rising grade of ≳ 3 %, or a nose-down pitch of
2–3° under hard braking, can admit road-surface returns into the corridor
(phantom braking on empty road); a ≈ 5 % downgrade can reject most of a lead
vehicle's returns at 20–30 m, shrinking effective range exactly where stopping
distance grows. Correcting this needs IMU-fed pitch compensation, which is
outside the scope of what the paper evaluates. Watch
`radar_n_in_corridor` on open road: it should be ≈ 0, and persistently large
values on grades indicate the first failure mode.
