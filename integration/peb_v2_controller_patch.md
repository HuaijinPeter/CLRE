# Wiring PEB-v2 into the upstream PID controller

Goal: a `PEBv2Controller` whose **steering path is bit-identical** to
Bench2DriveZoo's `PIDController` and whose longitudinal block is CLRE's
saturated proportional-error braking.

You build this file in your own checkout. Bench2DriveZoo is CC BY-NC-ND 4.0,
so a published copy of the subclass — which necessarily reproduces upstream's
steering block — cannot be redistributed here
([../docs/LICENSE_AUDIT.md](../docs/LICENSE_AUDIT.md)). The braking law itself
is CLRE's own work and *is* published, as
[`clre/control/peb_longitudinal.py`](../clre/control/peb_longitudinal.py).

---

## Why you cannot just call `super()`

The obvious mixin — call `super().control_pid(...)`, keep its steering, redo
the longitudinal part — is **wrong**, and silently so.

Upstream's `control_pid` calls `self.speed_controller.step(delta)` with a
`delta` derived from the *uncapped* target. Calling it and then recomputing
with the capped target steps the speed PID **twice per tick**. Its integral
window and derivative term then diverge from the baseline's, and the arm stops
being a matched comparison while still producing plausible-looking driving.

So the longitudinal block has to be replaced in place, inside one copy of
`control_pid`.

## The patch

Take your upstream `PIDController.control_pid` and keep everything down to and
including `steer = np.clip(...)` **exactly as it is**: `num_pairs`, the
aim-point loop, `aim_last`, all four angle computations, `use_target_to_aim`,
and the turn-PID step. That block is upstream's and must not be reformatted,
let alone changed.

Then replace the longitudinal block with this:

```python
import numpy as np
from Bench2DriveZoo.team_code.pid_controller import PIDController
from clre.control.peb_longitudinal import (
    peb_v2_longitudinal, window_delta, E_DB_DEFAULT, E_FULL_DEFAULT)


class PEBv2Controller(PIDController):
    """Drop-in replacement. Steering is inherited UNCHANGED (never overridden);
    only the longitudinal block differs."""

    def __init__(self, e_db=E_DB_DEFAULT, e_full=E_FULL_DEFAULT, **kwargs):
        super().__init__(**kwargs)
        assert 0.0 < e_db < e_full, (e_db, e_full)
        self.e_db, self.e_full = float(e_db), float(e_full)

    def control_pid(self, waypoints, speed, target,
                    speed_cap=float("inf"), emergency=False,
                    desired_speed_override=None):

        # ================= UPSTREAM STEERING BLOCK =================
        # Copy verbatim from PIDController.control_pid, unchanged:
        #   num_pairs / desired_speed accumulation / aim selection
        #   aim_last, angle, angle_last, angle_target, angle_final
        #   steer = np.clip(self.turn_controller.step(angle_final), -1, 1)
        # It also produces `desired_speed`, the waypoint-implied target.
        # ===========================================================

        # ==================== PEB-v2 LONGITUDINAL ==================
        desired_speed_wp = float(desired_speed)
        demand = (desired_speed_wp if desired_speed_override is None
                  else float(desired_speed_override))
        final_target_speed = min(demand, float(speed_cap))

        # the speed PID runs EVERY tick, unconditionally, on the CAPPED
        # target -- exactly once, exactly as the baseline does
        delta = window_delta(final_target_speed, speed, self.clip_delta)
        pid_throttle = float(np.clip(self.speed_controller.step(delta),
                                     0.0, self.max_throttle))

        throttle, brake, peb_tel = peb_v2_longitudinal(
            speed, desired_speed_wp, pid_throttle, self.brake_speed,
            speed_cap=speed_cap, emergency=emergency,
            desired_speed_override=desired_speed_override,
            e_db=self.e_db, e_full=self.e_full)
        # ===========================================================

        metadata = {
            # ... upstream's own keys: speed, steer, throttle, brake,
            #     wp_1..wp_4, aim, target, angle, angle_last, angle_target,
            #     angle_final, delta
            #
            # HISTORICAL SEMANTICS: every consumer of trace `desired_speed`
            # reads it as "the waypoint-implied PID target of the executed
            # plan". Keep exactly that meaning; the demand actually tracked
            # is `final_target_speed`, inside peb_tel.
            'desired_speed': desired_speed_wp,
            **peb_tel,
        }
        return steer, throttle, brake, metadata
```

## Branch order

```
(E) emergency      measured TTC persistently low   -> brake 1.0
(S) stop           v_cmd < brake_speed             -> brake 1.0
(F) full_error     dv >= E_full                    -> brake 1.0
(P) proportional   E_db < dv < E_full              -> ramp
(T) throttle_path  otherwise                       -> upstream PID throttle
```

`(E)` precedes `(S)` only so telemetry attributes the tick to the physical
cause; both actuate identically, so the order cannot change vehicle behaviour.
`(S)` still fires on a planner-demanded stop, so this does not weaken any
legitimate stop the refinement layer asks for.

## The retired branch

An earlier revision had an `(O) infeasible_override` branch that forced a full
brake whenever the refinement layer reported `all_infeasible`:

```python
elif infeasible_override:        # DELETED -- do not reintroduce
    brake, throttle = 1.0, 0.0
```

It was removed because `all_infeasible` is a statement about the optimiser's
search, not about the world: a large majority of such verdicts were measured
to be phantom (an oracle re-check found the space physically clear), and
braking on them converted a prediction defect into a stall. Held-out
evaluation showed the net effect was negative.

**In PEB-v2 no planner state reaches the controller at all** — the argument is
gone from the signature, not merely defaulted off. Branch code `1` is left as
a permanent hole so any v1/v2 trace mix-up is immediately visible.
`tests/test_execution_layer.py::test_f_no_planner_state_reaches_execution_layer`
enforces this by AST inspection.

## Caller contract — the latch

`desired_speed_override` is used **only** by the centerline fallback, and the
caller must **latch it once per fallback episode**:

```python
if self._fallback_hold is None:
    self._fallback_hold = speed      # entry only
hold = self._fallback_hold
```

Re-reading the current speed every tick makes the tracking error identically
zero. The PID's `clip(target - speed, 0, clip_delta)` window drains, throttle
decays to zero, and the target ratchets down with the coasting speed — a
silent coast-to-stop wearing a "hold speed" label, which below `brake_speed`
becomes the very full brake the deleted `(O)` branch used to command.

Supplying the longitudinal target explicitly also decouples it from the
fallback's waypoint spacing. Those waypoints carry a minimum spacing so the
steering aim-point stays well conditioned near standstill (otherwise all six
collapse onto the ego and `arctan2(0, 0)` feeds the turn PID garbage) — and
without the explicit target that floor would leak into the
waypoint-differencing speed estimate and make the fallback *accelerate from
rest*.

## Verifying your copy

The release ships `peb_longitudinal` already verified exact against the frozen
subclass (1260/1260 synthetic ticks,
[../docs/PROVENANCE.md](../docs/PROVENANCE.md) §3). What *your* patch adds is
the steering path and the single PID step, so check those two:

```python
# 1. steering identical to the stock controller on random waypoint sets
for _ in range(25):
    wp  = np.cumsum(rng.normal([0.0, 2.0], 0.6, (6, 2)), 0)
    tgt = rng.normal([0.0, 25.0], 5.0, 2)
    s_new, *_ = PEBv2Controller().control_pid(wp, np.float64(6.0), tgt)
    s_old, *_ = PIDController().control_pid(wp, np.float64(6.0), tgt)
    assert s_new == s_old

# 2. with no cap and no emergency, the longitudinal decision is the stock one
s2, t2, b2, m2 = PEBv2Controller().control_pid(wp, np.float64(6.0), tgt)
s1, t1, b1, m1 = PIDController().control_pid(wp, np.float64(6.0), tgt)
assert (t2, b2) == (t1, b1) or m2['peb_branch'] != 'throttle_path'
```

Check (2) holds exactly on the `throttle_path` branch. On the braking branches
the two controllers are *meant* to differ — that is the contribution — so
compare branch labels there, not actuation.
