# Integrating CLRE with VAD (main backbone)

Target: a leaderboard agent that behaves exactly like Bench2DriveZoo's
`VadAgent`, except that the trajectory handed to the controller is CLRE's
refined plan, and the controller's longitudinal block is PEB-v2.

Prerequisites: a Bench2DriveZoo checkout at
`d9caa0af805ec3c435533aa268e2723d80c20017`, the `vad_b2d_base.pth` checkpoint,
CARLA 0.9.15, and this repository on `PYTHONPATH`.

---

## 1. Shape of the file

```python
# my_agents/clre_vad_agent.py
from Bench2DriveZoo.team_code.vad_b2d_agent import VadAgent
from Bench2DriveZoo.team_code.pid_controller import PIDController

from clre.planning import refinement as clre
from clre.control.radar_safety import RadarSafetyOverride
from clre.control.route_centerline import RouteCenterline
from clre.control.peb_longitudinal import peb_v2_longitudinal, window_delta


def get_entry_point():
    return "CLREVadAgent"


class CLREVadAgent(VadAgent):
    ...
```

Subclassing is the whole design: every line you do not override stays the
upstream line, which is what makes the comparison matched.

## 2. `sensors()` — add one forward radar

```python
RADAR_ID = "RADAR_FRONT"

def sensors(self):
    return super().sensors() + [{
        'type': 'sensor.other.radar',
        'x': 2.0, 'y': 0.0, 'z': 1.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'horizontal_fov': 30.0, 'vertical_fov': 10.0,
        'id': RADAR_ID,
    }]
```

Notes that matter:

* VAD ships no radar and no LiDAR, so the execution layer genuinely requires
  adding a sensor. The leaderboard permits four; this adds one.
* The wrapper **overrides** `range` to 100 m and `points_per_second` to 1500
  whatever you request. Only the FOV and pose are yours. A narrow 30° × 10°
  cone concentrates the forced ~75 returns/frame on the forward corridor.
* The sensor set must track the layer's enable flag exactly: `sensors()` and
  `setup()` are both called during agent construction, and an arm with the
  execution layer disabled has to be *sensor-identical* to a plain VAD arm or
  it is not a matched baseline. Read the flag from the environment in both
  places rather than from instance state.
* **Arms with and without this radar are not sensor-matched.** Say so wherever
  you report such a comparison.

## 3. `setup()` — construct the layers

```python
def setup(self, path_to_conf_file):
    super().setup(path_to_conf_file)

    # refinement
    self.planner_cfg = clre.PlannerConfig(
        device=os.environ.get("NEWBORN_CANDA_DEVICE", "cpu"),
        dtype=os.environ.get("NEWBORN_CANDA_DTYPE", "float64"),
        iters_cold=int(os.environ.get("NEWBORN_CANDA_ITERS_COLD", "70")),
        iters_warm=int(os.environ.get("NEWBORN_CANDA_ITERS_WARM", "25")),
        runtime_budget_ms=float(os.environ.get("NEWBORN_CANDA_BUDGET_MS", "1000")),
    )
    clre.apply_fixes(self.planner_cfg,
                     os.environ.get("NEWBORN_CANDA_FIXES", "none"))
    self.warm = clre.WarmBuffer(tick_dt=0.05)
    self.canda_mode = os.environ.get("NEWBORN_CANDA_MODE", "exec")

    # execution layer
    self.pebv2 = os.environ.get("NEWBORN_PEBV2", "1") == "1"
    self._radar = RadarSafetyOverride() if self.pebv2 else None
    self._centerline = None
    self._centerline_failed = False
    self._fallback_hold = None          # latched at fallback entry
    if self.pebv2:
        self.pidcontroller = PEBv2Controller()   # your subclass; see the patch guide

    # head-output hook on the EXISTING forward pass
    self._head_outs = {}
    self.model.pts_bbox_head.register_forward_hook(self._capture_head)
```

`NEWBORN_CANDA_FIXES` defaults to `none`, which is **not** the paper's method
— it disables the OBB gate among other things. Pass
`r1,r2,r3,geom,feas`. Keep the frozen environment-variable spellings so your
runs stay comparable with the published configuration
([../docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md) §4).

Solver device: CPU/float64 in the paper. The tiny-tensor solve is
kernel-launch bound, so CPU float64 was measured *faster* than CUDA (warm
≈ 50 ms vs ≈ 104 ms) while reproducing the reference numerics exactly and
leaving the GPU to the VAD forward pass.

## 4. The forward hook

```python
HEAD_KEYS = ("all_cls_scores", "all_bbox_preds", "all_traj_preds",
             "all_traj_cls_scores", "ego_fut_preds")

def _capture_head(self, module, inputs, output):
    if isinstance(output, dict):
        for k in HEAD_KEYS:
            v = output.get(k)
            self._head_outs[k] = v.detach() if torch.is_tensor(v) else v
```

Clear `self._head_outs` immediately before each forward call so a stale tick
can never leak into the next one.

## 5. `run_step()` — the single insertion point

Copy your upstream `VadAgent.run_step` and change **only** the trajectory that
reaches `control_pid`. In outline:

```
  tick / input construction ..................... UNCHANGED
  single model forward .......................... UNCHANGED
  nominal plan + route target extraction ........ UNCHANGED
  agent extraction from the head outputs ........ moved earlier (CLRE needs it)
  ─────────────────────────────────────────────── CLRE BLOCK (new)
  controller call ............................... same call, new plan + 3 kwargs
  brake/throttle post-processing ................ UNCHANGED
```

Moving the agent extraction above the control block is a reordering of
existing code, not a change to it: the same arrays feed the trace row.

### 5a. Build the window and solve

```python
nominal = out_truck                       # (6, 2) selected-branch cumsum plan
nat_desired = clre.native_desired_speed(nominal)
exec_plan, exec_source, fallback_reason, res = nominal, "native", "", None

if self.canda_mode != "native":
    self.warm.advance()                   # EVERY tick, before solving
    try:
        agents = clre.agents_from_arrays(
            ag_valid, ag_score, ag_box, ag_traj, ag_prob, ag_label,
            fix_r1=self.planner_cfg.fix_r1)
        bnds = clre.boundaries_from_map(map_pts_3d, map_scores_3d, map_labels_3d)
        win = clre.make_window(speed, nominal, local_command_xy, agents, bnds,
                               v_ref_in=max(nat_desired, 2.5))
        res = clre.solve(win, self.planner_cfg, warm=self.warm.seed())
    except Exception as e:                # never kill the route
        res, fallback_reason = None, f"planner_exception:{type(e).__name__}"
        self.warm.reset()

    if res is not None:
        if not res.ok:
            fallback_reason = res.reason          # non-finite / bound violation
            self.warm.reset()
        elif res.solve_ms > self.planner_cfg.runtime_budget_ms:
            fallback_reason = "runtime_budget_exceeded"
            self.warm.update(res.a, res.k)        # solution valid, keep warm
        else:
            self.warm.update(res.a, res.k)
            if self.canda_mode == "exec":
                exec_plan, exec_source = res.wp6, "clre"
```

**Input contract.**

| Argument | Meaning | Convention |
| --- | --- | --- |
| `speed` | measured ego speed, m/s | — |
| `nominal` | the planner's selected-branch plan | `(6, 2)`, 0.5 s spacing, **cumulative** positions, plan frame |
| `local_command_xy` | next route node in the ego frame | raw, unnormalised; the window applies the > 1 m degeneracy guard itself |
| `ag_box` | agent boxes | columns `[cx, cy, w, l, h, yaw, cz]` |
| `ag_traj` | motion modes | `(K, n_modes, 6, 2)`, **per-step displacements**; the adapter cumsums them |
| `ag_prob` | mode probabilities | `(K, n_modes)`; only the top-1 mode is used |
| `v_ref_in` | reference speed | **must** be `max(native_desired_speed, 2.5)` — the floor is applied at the call site, and it is inside both gate branches of the frozen results |

Plan frame throughout: **x right, y forward**.

`res.wp6` is `(6, 2)` at 0.5 s spacing — deliberately the same shape and
convention as the upstream plan, so the controller cannot tell which source
produced it.

**Three things that are easy to get wrong:**

* `warm.advance()` must be called every tick, including ticks where you do not
  solve. It time-shifts the buffer onto the current control grid.
* Reset the warm buffer on failure (`not res.ok`), but **keep** it on a budget
  overrun — that solution was valid, it just arrived late.
* Mode trajectories are displacements, not positions.

### 5b. Fallback and radar

```python
# centerline, built once from the agent's own dense GPS plan
if self.pebv2 and self._centerline is None and not self._centerline_failed:
    self._centerline = RouteCenterline.from_agent(self)
    self._centerline_failed = self._centerline is None
if self._centerline is not None:
    self._centerline.note_position(pos)      # EVERY tick, keeps the hint fresh

hold_speed, fallback_active = None, False
if self.pebv2 and self._centerline is not None:
    want = (exec_source == "clre" and res is not None
            and res.sel_mode == "all_infeasible_safest")
    if want:
        if self._fallback_hold is None:
            self._fallback_hold = speed      # LATCH at episode entry
        hold_speed = self._fallback_hold
        exec_plan = self._centerline.waypoints_ego(pos, raw_theta, hold_speed)
        exec_source, fallback_active = "clre_fallback", True
    else:
        self._fallback_hold = None           # episode over

# radar: reads ONLY measured returns, in both normal and fallback mode
speed_cap, radar_emergency = float("inf"), False
if self.pebv2:
    item = input_data.get(RADAR_ID)
    speed_cap, radar_tel = self._radar.evaluate(
        item[1] if item is not None else None, speed)
    radar_emergency = bool(radar_tel.get("safety_emergency"))
```

`RouteCenterline.from_agent` reads `self._plan_gps_HACK` (the leaderboard's
full 1 m-resolution GPS route; `self._global_plan` is downsampled to ≤ 50 m
and far too sparse to steer on) and converts it with the **agent's own**
`gps_to_location`. Use the agent's transform, not the leaderboard's true
lat/lon reference: `VadAgent._init` does not read the map georeference, it
numerically solves for `lat_ref`/`lon_ref`, and ego position comes from that
same solved transform. Converting the route with it puts centerline and ego in
the *same* frame even when the solved reference carries a constant offset —
which it does, and which cancels in `centerline − ego`. Using the true
reference here introduces a systematic centerline bias instead.

Watch `centerline_offset_m` (published every tick as
`RouteCenterline.last_offset_m`): a vehicle driving its route stays within
about a lane half-width of the centerline, so a persistently large value means
your frame is wrong.

### 5c. Controller call

```python
if self.pebv2:
    steer, throttle, brake, meta = self.pidcontroller.control_pid(
        exec_plan, tick_data['speed'], local_command_xy,
        speed_cap=speed_cap, emergency=radar_emergency,
        desired_speed_override=hold_speed)
else:
    steer, throttle, brake, meta = self.pidcontroller.control_pid(
        exec_plan, tick_data['speed'], local_command_xy)
```

Everything after this — the `brake < 0.05` and `throttle > brake` clamps, the
`carla.VehicleControl` construction, the throttle clip to 0.75 — stays exactly
as upstream has it.

## 6. Optional: a shadow native controller for logging

To record what the upstream planner *would* have commanded this tick without
contaminating the executed controller's PID windows, keep a second, stock
`PIDController` instance and call it on the nominal plan. Separate state,
logging only. Do not reuse the executed controller for this.

## 7. Modes worth implementing

| `NEWBORN_CANDA_MODE` | Behaviour |
| --- | --- |
| `exec` | execute the refined plan (the paper's arm) |
| `shadow` | solve and log every tick, execute the nominal plan |
| `native` | skip the refinement entirely — a passthrough regression check |

`native` is worth the few lines: it should be byte-identical to the stock agent
through the same code path, which makes it the cheapest possible proof that
your wiring adds nothing when it is switched off.

## 8. Acceptance checks before any campaign

1. `NEWBORN_CANDA_MODE=native` reproduces stock VAD control output exactly.
2. `tests/` passes against your CLRE copy.
3. On a smoke route, `radar_v_static_med` tracks `+ego_speed` (sign
   convention) and `radar_n_in_corridor ≈ 0` on open road (height gating).
4. `radar.layout_warning` is empty — if not, the radar layer has disabled
   itself and the run is a plain refinement arm.
5. `centerline_offset_m` stays within a lane half-width.
6. Fallback episodes show a *constant* `final_target_speed`, not a decaying
   one — that is the latch working.
