# Integrating CLRE with UniAD

Read [`vad_integration.md`](vad_integration.md) first. UniAD differs from VAD
only in how the refinement layer's inputs are obtained; the refinement call,
the execution layer and the five invariants are identical.

**Status.** Additional demonstration that the layer is not VAD-specific.
Evaluated on a small route subset, not the full benchmark. Do not report it as
equivalent evidence to the VAD results.

Prerequisites: a Bench2DriveZoo checkout at
`d9caa0af805ec3c435533aa268e2723d80c20017`, the `uniad_base_b2d.pth`
checkpoint, and this repository on `PYTHONPATH`.

---

## 1. What the adapter does

[`clre/adapters/uniad_adapter.py`](../clre/adapters/uniad_adapter.py) is a
**pure data adapter**. By contract it:

* converts UniAD inference outputs into the frozen `ag_*` array schema and the
  refinement layer's `native_plan` / `boundaries` input forms;
* never imports the refinement layer, never duplicates objective or OBB
  mathematics, never holds model state, never performs I/O.

Every structural violation raises `UniADAdapterError` and **fails loud**:
adapter corruption must never masquerade as legitimate driving behaviour — in
particular, never as an all-infeasible tick, which would silently turn a
plumbing bug into a fallback episode.

## 2. Agent skeleton

```python
import Bench2DriveZoo.team_code.uniad_b2d_agent as _stock
from Bench2DriveZoo.team_code.uniad_b2d_agent import UniadAgent

from clre.planning import refinement as clre
from clre.adapters import uniad_adapter as ua
from clre.control.radar_safety import RadarSafetyOverride
from clre.control.route_centerline import RouteCenterline


def get_entry_point():
    return "CLREUniadAgent"


class CLREUniadAgent(UniadAgent):
    ...
```

## 3. Verify the upstream pins before running

The adapter assumes the layout of nine files in the pinned Bench2DriveZoo
release and carries their SHA-256 values. Gate on them at setup:

```python
ua.require_upstream(zoo_parent_root)   # hard stop on any mismatch
```

`verify_upstream(root)` returns the per-file report if you want to log it.
`root` is the directory the recorded relative paths resolve against — they are
recorded as `external/v004/Bench2DriveZoo/...`, so either mirror that layout
or pass the parent accordingly.

Running against unpinned upstream files is not a configuration choice: the
adapter's column orders and index laws are derived from those exact files.

## 4. Per-tick conversion

```python
out0 = out[0]                              # UniAD's own forward output

prior  = ua.extract_prior(out0)            # (6, 2) nominal plan, validated
agents = ua.extract_agents(out0, ...)      # frozen ag_* schema
bnds   = ua.extract_boundaries(lane_raster, ...)   # list of (P, 2) polylines
speed  = ua.validate_ego_state(speed)
```

Then call the refinement layer exactly as in the VAD guide, with
`v_ref_in = max(clre.native_desired_speed(prior), 2.5)`.

### Validation the adapter performs for you

`validate_prior` enforces hard implausibility bounds on the nominal plan:
`PRIOR_MAX_STEP_M = 8.0` (0.5 s × 16 m/s), `PRIOR_FIRST_STEP_MAX_M = 5.0`,
`PRIOR_STEP_DELTA_MAX_M = 6.0`. A violation is a loud failure, not a clamp —
a prior outside those bounds means something upstream is wrong, and refining
it would hide that.

### Three traps specific to UniAD

These were found the hard way; the adapter handles them, but you need to know
they exist if you touch its inputs.

* **Lane-raster channel semantics.** Boundaries come from the segmentation
  raster, not from a vector map head. Only channels `(1, 2)` — Solid and
  SolidSolid — are boundaries, matching the refinement layer's own
  `BOUNDARY_LABELS`. The grid is 200 × 200 at `102.4 / 200 = 0.512 m` per
  cell. Row/column ↔ `(x, y)` orientation and the half-cell offset follow the
  occupancy-grid convention of the pinned planning head; `GridSpec` encodes
  it. Getting this wrong mirrors the map and the vehicle drifts toward
  boundaries instead of away.
* **The SDC row.** UniAD's track output includes the ego ("SDC") as a row.
  It must not enter the agent set — the vehicle would be avoiding itself.
  The adapter identifies it by cross-key identity and an origin-distance
  bound (`SDC_CROSSKEY_TOL`, `SDC_ORIGIN_MAX_M = 2.0`), and exposes the
  ego-ghost checks (`EGO_GHOST_DIST_M`, `EGO_GHOST_EXT_RTOL`) as **diagnostic
  audit fields only** — deliberately never a fail-loud condition, because a
  genuine nearby vehicle can legitimately look like a ghost.
* **Motion-mode truncation.** UniAD emits 12 steps at 0.5 s; the refinement
  layer uses the first 6 (3.0 s). `MOTION_KEEP_STEPS = 6` is truncation only,
  never resampling.

## 5. Everything downstream is unchanged

The OBB gate, candidate selection, route-centerline backup, radar bound and
PEB-v2 are backbone-agnostic: they consume the refinement layer's output and
measured sensor data, never planner internals. Add the forward radar in
`sensors()` and wire the controller exactly as in the VAD guide and
[`peb_v2_controller_patch.md`](peb_v2_controller_patch.md).

## 6. Not released: the adapter's full offline suite

The frozen tree carries a 24-group offline suite plus an AST parity check for
this adapter. It asserts against the `sha256`-pinned upstream files, which are
not redistributed here, so it is not part of this release.
`tests/test_adapters_smoke.py` covers what is checkable standalone: the module
imports, the pins are well-formed and relative, no machine-local checkpoint
path is baked in, and the adapter never imports the refinement layer.

Two conventions in the adapter were validated against real forward passes
rather than synthetic fixtures, and are documented in the module docstring as
such: instance-level box column order and yaw convention, and the segmentation
raster's orientation. If you port this to a different UniAD release, re-check
both — do **not** weaken the assertions to make synthetic tests pass.
