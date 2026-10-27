#!/usr/bin/env python
"""Saturated proportional-error braking (PEB-v2) -- longitudinal law only.

WHY THIS FILE EXISTS IN THE PUBLIC RELEASE
==========================================
In the research tree this law lives inside `PEBv2Controller`, a subclass of
Bench2DriveZoo's `team_code.pid_controller.PIDController` whose `control_pid`
reproduces the upstream steering block verbatim so that steering stays
bit-identical to the baseline. Bench2DriveZoo is licensed CC BY-NC-ND 4.0,
whose NoDerivatives term does not grant the right to redistribute adapted
material, so that subclass CANNOT be shipped here (see docs/LICENSE_AUDIT.md).

This module therefore contains ONLY the part that is original CLRE work --
the longitudinal branch law, the radar speed cap, and the telemetry -- as
pure functions with no upstream import and no upstream code. Reconnect it to
your own checkout of the upstream controller following
integration/peb_v2_controller_patch.md; the steering path is left completely
untouched there, which is the whole point of the split.

THE LAW (paper Sec. "Safety-Aware Execution Layer")
===================================================
    v_cmd  = min(v_ref, v_safe)                  radar cap, never raises
    dv     = v - v_cmd                           tracking error
    b      = 0                                   dv <= E_db
           = (dv - E_db) / (E_full - E_db)       E_db < dv < E_full
           = 1                                   dv >= E_full

Branch order and why (telemetry attribution only -- (E) and (S) produce
identical actuation, so the order cannot change vehicle behaviour):

    (E) emergency      measured TTC persistently low   -> brake 1.0
    (S) stop           v_cmd < brake_speed             -> brake 1.0
    (F) full_error     dv >= E_full                    -> brake 1.0
    (P) proportional   E_db < dv < E_full              -> ramp
    (T) throttle_path  otherwise                       -> upstream PID throttle

NOTE on the retired branch: an earlier revision (E48 PEB) also had an
`(O) infeasible_override` branch that forced a full brake whenever the
refinement layer reported `all_infeasible`. It was removed because
`all_infeasible` is a statement about the optimizer's search, not about the
world: a large majority of those verdicts were measured to be phantom, and
braking on them converted a prediction defect into a stall. No planner state
reaches this module at all -- by construction, not by a default argument.
Branch code 1 is left as a permanent hole so that any v1/v2 trace mix-up is
immediately visible in analysis.
"""

E_DB_DEFAULT = 0.5     # m/s -- brake-onset threshold (deadband edge)
E_FULL_DEFAULT = 2.0   # m/s -- full-brake threshold

# Branch codes for telemetry. 1 == the retired (O) branch; never reused.
BRANCH_CODES = {"emergency": 5, "stop": 0, "full_error": 2,
                "proportional": 3, "throttle_path": 4}


def window_delta(final_target_speed, speed, clip_delta):
    """The error the upstream speed PID must be stepped with THIS tick.

    The upstream controller computes `clip(target - speed, 0, clip_delta)` and
    feeds it to `speed_controller.step()`. CLRE changes only the target that
    goes in (the radar-capped one), never the expression, and the call must
    still happen EXACTLY ONCE PER TICK, unconditionally, on every branch --
    otherwise the PID's integral window diverges from the baseline and the
    comparison stops being matched. Exposed as a function so the integration
    point is explicit rather than folklore.
    """
    d = float(final_target_speed) - float(speed)
    if d < 0.0:
        return 0.0
    cd = float(clip_delta)
    return cd if d > cd else d


def peb_v2_longitudinal(speed, desired_speed_wp, pid_throttle, brake_speed,
                        speed_cap=float("inf"), emergency=False,
                        desired_speed_override=None,
                        e_db=E_DB_DEFAULT, e_full=E_FULL_DEFAULT):
    """One tick of the PEB-v2 longitudinal law.

    Parameters
    ----------
    speed : float
        Measured ego speed [m/s].
    desired_speed_wp : float
        The raw waypoint-implied target of the executed plan, i.e. upstream's
        own `desired_speed` quantity. Kept separately from the demand actually
        tracked so that post-hoc analysis can separate fallback effects from
        radar-override effects.
    pid_throttle : float
        The upstream speed PID's throttle for this tick, ALREADY clipped to
        [0, max_throttle] by the caller, computed from `window_delta(...)`.
        Passing it in (rather than reimplementing the PID) is what keeps this
        module free of upstream code.
    brake_speed : float
        Upstream's `brake_speed`; below this the demand counts as a stop.
    speed_cap : float
        Physical speed ceiling published by the radar supervisor
        (`clre.control.radar_safety.RadarSafetyOverride.evaluate`). +inf means
        the supervisor is transparent this tick. It can only ever LOWER the
        demand, so it can never command acceleration.
    emergency : bool
        Measured-TTC emergency flag from the same supervisor.
    desired_speed_override : float or None
        Used ONLY by the route-centerline fallback, which must hold the ego
        speed LATCHED at fallback entry. CALLER CONTRACT: latch this value
        once per fallback episode; do NOT re-read the current speed every
        tick. Re-reading makes the tracking error identically zero, the PID
        window drains, throttle decays, and the target ratchets down with the
        coasting speed -- a silent coast-to-stop wearing a "hold speed" label.
    e_db, e_full : float
        Brake-onset and full-brake thresholds; `0 < e_db < e_full`.

    Returns
    -------
    (throttle, brake, telemetry) : (float, float, dict)
        `telemetry` carries the branch taken plus every intermediate demand,
        so "did the override bind?" is a subtraction rather than a guess.
    """
    assert 0.0 < float(e_db) < float(e_full), (e_db, e_full)

    desired_speed_wp = float(desired_speed_wp)
    desired_speed = (desired_speed_wp if desired_speed_override is None
                     else float(desired_speed_override))
    final_target_speed = min(desired_speed, float(speed_cap))
    override_active = bool(final_target_speed < desired_speed - 1e-6)

    e_v = float(speed) - final_target_speed

    brake_raw = None   # pre-clip proportional value; None outside branch (P)
    if emergency:
        branch = "emergency"
        brake, throttle = 1.0, 0.0
    elif final_target_speed < float(brake_speed):
        branch = "stop"
        brake, throttle = 1.0, 0.0
    elif e_v >= float(e_full):
        branch = "full_error"
        brake, throttle = 1.0, 0.0
    elif e_v > float(e_db):
        branch = "proportional"
        brake_raw = (e_v - float(e_db)) / (float(e_full) - float(e_db))
        brake, throttle = brake_raw, 0.0
    else:
        branch = "throttle_path"
        brake, throttle = 0.0, float(pid_throttle)

    telemetry = {
        'peb_branch': branch,
        'peb_branch_code': BRANCH_CODES[branch],
        'peb_e_v': float(e_v),
        'peb_brake_raw': (float(brake_raw) if brake_raw is not None
                          else float('nan')),
        'peb_e_db': float(e_db),
        'peb_e_full': float(e_full),
        # `safe_target_speed` is the ceiling the radar layer published;
        # `final_target_speed` is what the controller actually tracked.
        'safe_target_speed': float(speed_cap),
        'final_target_speed': float(final_target_speed),
        'safety_override_active': override_active,
        'safety_emergency': bool(emergency),
        # the demand BEFORE the radar cap, and the raw waypoint-derived value,
        # so fallback and override effects stay separable in analysis
        'plan_target_speed': float(desired_speed),
        'desired_speed_wp': desired_speed_wp,
    }
    return float(throttle), float(brake), telemetry
