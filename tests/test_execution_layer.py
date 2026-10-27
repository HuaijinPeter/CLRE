#!/usr/bin/env python
"""Execution-layer tests -- NO CARLA, NO checkpoint, NO GPU, no upstream code.

Derived from the frozen research suite (tests/test_e59_pebv2.py at commit
b1f509d). The radar-supervisor and route-centerline checks are carried over
unchanged, because those two modules are shipped verbatim. The braking checks
are re-expressed against `clre.control.peb_longitudinal`, since the frozen
`PEBv2Controller` subclass cannot be redistributed (docs/LICENSE_AUDIT.md);
the extracted law was verified exact against that subclass on 1260 synthetic
ticks at release time (docs/PROVENANCE.md).

Scenario coverage, matching the frozen suite:
  (a) fallback holds the latched entry speed and follows the centerline
  (b) normal mode + radar hazard    -> override activates
  (c) fallback mode + radar hazard  -> override activates
  (d) no hazard                     -> radar layer exactly transparent
  (e) severe closing hazard         -> braking increases monotonically
  (f) no planner state is reachable from either execution-layer module
  (g) ground / overhead clutter rejected, real obstacle kept
  (h) centerline ego frame == the production local_command_xy transform
  (i) non-closing objects transparent (no close-range brake latch)
  (k) emergency requires persistence; one hot frame cannot full-brake
  (l) suspect radar column order -> fail-transparent
  (m) latched hold target does not ratchet down with the coasting speed
  (n) route end -> no waypoint collapse, no arctan2(0, 0)
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from clre.control.radar_safety import (  # noqa: E402
    RadarSafetyOverride, EMERGENCY_PERSIST_N)
from clre.control.route_centerline import RouteCenterline  # noqa: E402
from clre.control.peb_longitudinal import (  # noqa: E402
    peb_v2_longitudinal, window_delta, E_DB_DEFAULT, E_FULL_DEFAULT)

# Upstream PIDController's own default; a fixture value, not CLRE code.
BRAKE_SPEED = 0.4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def radar_point(depth, closing, lateral=0.0, height_above_road=0.8,
                sensor_z=1.0):
    """One radar return, built the way the LEADERBOARD delivers it.

    INDEPENDENCE: the delivered layout is re-derived from the two upstream
    facts the implementation cites, WITHOUT importing the module's own column
    constants -- otherwise a swapped constant pair would pass every test.

      RadarData.h wire struct : [velocity, azimuth, altitude, depth]
      sensor_interface.py     : np.flip(points, 1)

    `closing` is the PHYSICAL closing speed (positive = approaching); the raw
    column is written as `-closing` because the CARLA column was measured to
    be a RANGE RATE (negative while closing). Encoding the measured
    convention here rather than importing RADAR_V_SIGN keeps the check real.
    """
    azi = np.arcsin(np.clip(lateral / max(depth, 1e-6), -1, 1))
    alt = np.arcsin(np.clip((height_above_road - sensor_z) / max(depth, 1e-6),
                            -1, 1))
    struct_order = np.array([-closing, azi, alt, depth])  # RadarData.h order
    return struct_order[::-1].copy()                      # leaderboard flip


def peb(speed, desired_wp, speed_cap=float("inf"), emergency=False,
        override=None, pid_throttle=0.0):
    """Shorthand: one tick of the longitudinal law with upstream defaults."""
    return peb_v2_longitudinal(speed, desired_wp, pid_throttle, BRAKE_SPEED,
                               speed_cap=speed_cap, emergency=emergency,
                               desired_speed_override=override)


def make_centerline(xy):
    return RouteCenterline(np.asarray(xy, float))


# ---------------------------------------------------------------------------
# (a) all-infeasible fallback -> hold entry speed, follow the centerline
# ---------------------------------------------------------------------------

def test_a_fallback_holds_speed_and_follows_centerline():
    # FRAME: with compass = 0 the ego heads toward world -y (a point 10 m
    # ahead sits at world offset (0, -10) and maps to ego (0, +10) -- forward
    # is +y in the PID frame). The route is 20 m straight along -y then a
    # hard turn along -x, i.e. an intersection.
    xy = ([[0.0, -float(s)] for s in np.arange(0.0, 20.0, 1.0)] +
          [[-float(t), -20.0] for t in np.arange(1.0, 21.0, 1.0)])
    cl = make_centerline(xy)

    v_hold = 6.0
    wp = cl.waypoints_ego(np.array([0.0, -10.0]), 0.0, v_hold)
    assert wp.shape == (6, 2)

    # longitudinal: the tracked target is EXACTLY the latched entry speed,
    # neither slowed nor accelerated by the fallback
    thr, brk, tel = peb(v_hold, desired_wp=99.0, override=v_hold,
                        pid_throttle=0.3)
    assert abs(tel["final_target_speed"] - v_hold) < 1e-9
    assert tel["safety_override_active"] is False
    assert brk == 0.0 and thr == 0.3
    assert tel["peb_branch"] == "throttle_path"

    # forward progress with no lateral wander while still on the straight
    # segment (the corner is 20 m away; the 6 x 3 m horizon reaches 18 m)
    assert np.all(np.diff(wp[:3, 1]) > 0) and wp[0][1] > 0, wp
    assert np.max(np.abs(wp[:3, 0])) < 1e-6, wp

    # lateral: the fallback must TURN at the intersection, not run straight.
    # This is what makes "follow the navigation route" structural.
    wp_corner = cl.waypoints_ego(np.array([0.0, -18.0]), 0.0, v_hold)
    assert abs(wp_corner[-1][0]) > 3.0, wp_corner

    # hold speed must be exact at standstill too: the MIN_SPACING geometry
    # floor must not leak into the longitudinal target and creep forward
    wp0 = cl.waypoints_ego(np.array([0.0, -10.0]), 0.0, 0.0)
    assert np.linalg.norm(wp0[0]) > 0.5, "geometry floor missing"
    wp0_speed = float(sum(np.linalg.norm(wp0[i + 1] - wp0[i]) * 2.0 / 5
                          for i in range(5)))
    assert wp0_speed > 1.0, "floor should be visible in the waypoints"
    thr0, brk0, tel0 = peb(0.0, desired_wp=wp0_speed, override=0.0)
    assert tel0["final_target_speed"] == 0.0
    assert brk0 == 1.0 and thr0 == 0.0 and tel0["peb_branch"] == "stop"
    print("(a) fallback holds latched speed + follows centerline OK")


# ---------------------------------------------------------------------------
# (b) normal planner mode + radar hazard -> override activates
# ---------------------------------------------------------------------------

def test_b_override_activates_in_normal_mode():
    radar = RadarSafetyOverride()
    v = 10.0
    cap_v, tel_r = radar.evaluate(np.stack([radar_point(12.0, v)]), v)
    assert cap_v < v, cap_v
    _, brk, tel = peb(v, desired_wp=v, speed_cap=cap_v,
                      emergency=tel_r["safety_emergency"])
    assert tel["safety_override_active"] is True
    assert tel["final_target_speed"] == cap_v
    assert brk > 0.0
    print("(b) override activates in normal mode OK")


# ---------------------------------------------------------------------------
# (c) fallback mode + radar hazard -> override still activates
# ---------------------------------------------------------------------------

def test_c_override_activates_in_fallback_mode():
    radar = RadarSafetyOverride()
    v = 10.0
    cap_v, tel_r = radar.evaluate(np.stack([radar_point(12.0, v)]), v)
    # the fallback latches `v` as its target; the radar cap must still bind
    _, brk, tel = peb(v, desired_wp=99.0, speed_cap=cap_v,
                      emergency=tel_r["safety_emergency"], override=v)
    assert tel["plan_target_speed"] == v
    assert tel["safety_override_active"] is True
    assert tel["final_target_speed"] == cap_v
    assert brk > 0.0
    print("(c) override activates in fallback mode OK")


# ---------------------------------------------------------------------------
# (d) no hazard -> radar layer exactly transparent
# ---------------------------------------------------------------------------

def test_d_radar_transparent_without_hazard():
    radar = RadarSafetyOverride()
    v = 7.0
    cases = {
        "no_sensor": None,
        "empty": np.zeros((0, 4)),
        "far": np.stack([radar_point(80.0, v)]),              # > max_range
        "off_corridor": np.stack([radar_point(15.0, v, lateral=4.0)]),
    }
    for name, pts in cases.items():
        cap_v, tel_r = radar.evaluate(pts, v)
        assert cap_v == float("inf"), (name, cap_v)
        assert tel_r["safety_emergency"] is False, name

        # the longitudinal decision is then identical to an uncapped tick
        _, brk, tel = peb(v, desired_wp=v, speed_cap=cap_v, pid_throttle=0.5)
        _, brk0, tel0 = peb(v, desired_wp=v, pid_throttle=0.5)
        assert tel["safety_override_active"] is False, name
        assert tel["final_target_speed"] == tel0["final_target_speed"], name
        assert (brk, tel["peb_branch"]) == (brk0, tel0["peb_branch"]), name
    print("(d) radar transparent without hazard OK")


# ---------------------------------------------------------------------------
# (e) severe closing hazard -> braking increases monotonically
# ---------------------------------------------------------------------------

def test_e_braking_monotone_in_severity():
    radar = RadarSafetyOverride()
    v = 10.0
    gaps = [45.0, 30.0, 22.0, 18.0, 16.5, 15.0, 13.5, 12.0, 8.0, 5.0, 3.0]
    caps, brakes = [], []
    for g in gaps:
        cap_v, tel_r = radar.evaluate(np.stack([radar_point(g, v)]), v)
        _, brk, _ = peb(v, desired_wp=v, speed_cap=cap_v,
                        emergency=tel_r["safety_emergency"])
        caps.append(cap_v)
        brakes.append(float(brk))

    assert all(np.diff(caps) <= 1e-9), caps        # ceiling falls with gap
    assert all(np.diff(brakes) >= -1e-9), brakes   # brake rises with severity
    assert brakes[0] == 0.0, brakes                # far: no intervention
    assert brakes[-1] == 1.0, brakes               # imminent: full brake
    # the ramp is genuinely proportional somewhere in between -- a saturated
    # ramp, not a rebuilt binary brake
    assert any(0.0 < b < 1.0 for b in brakes), brakes

    # closing FASTER at a fixed gap must brake at least as hard
    prev = -1.0
    for vc in (1.0, 3.0, 6.0, 12.0):
        cap_v, tel_r = radar.evaluate(np.stack([radar_point(14.0, vc)]), v)
        _, brk, _ = peb(v, desired_wp=v, speed_cap=cap_v,
                        emergency=tel_r["safety_emergency"])
        assert float(brk) >= prev - 1e-9, (vc, brk, prev)
        prev = float(brk)

    # and the saturated ramp matches its closed form exactly
    for dv in (0.0, 0.25, 0.5, 0.9, 1.25, 1.9, 2.0, 3.0):
        tgt = 5.0
        _, brk, tel = peb(tgt + dv, desired_wp=99.0, override=tgt)
        want = (0.0 if dv <= E_DB_DEFAULT else
                1.0 if dv >= E_FULL_DEFAULT else
                (dv - E_DB_DEFAULT) / (E_FULL_DEFAULT - E_DB_DEFAULT))
        assert abs(brk - want) < 1e-12, (dv, brk, want, tel["peb_branch"])
    print("(e) braking monotone + closed-form ramp OK")


# ---------------------------------------------------------------------------
# (f) no planner state is reachable from either execution-layer module
# ---------------------------------------------------------------------------

def _identifiers(obj):
    """Every identifier the object's CODE touches.

    Parses the AST instead of grepping the source text: both modules discuss
    the planner at length in their docstrings (that is the whole rationale
    for the redesign), so a substring test would either fail on prose or be
    weakened until it proved nothing.
    """
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
    return names


def test_f_no_planner_state_reaches_execution_layer():
    from clre.control import radar_safety as rs
    banned = {"infeasible_override", "all_infeasible", "sel_mode",
              "exec_source", "feasible", "n_feasible", "min_gap"}
    for obj in (peb_v2_longitudinal, window_delta, rs.RadarSafetyOverride):
        leaked = _identifiers(obj) & banned
        assert not leaked, (obj, leaked)
    # branch code 1 stays a permanent hole so v1/v2 traces can never be
    # confused in analysis
    from clre.control.peb_longitudinal import BRANCH_CODES
    assert 1 not in set(BRANCH_CODES.values()), BRANCH_CODES
    assert set(BRANCH_CODES) == {"emergency", "stop", "full_error",
                                 "proportional", "throttle_path"}
    print("(f) no planner state reachable; (O) branch permanently retired OK")


# ---------------------------------------------------------------------------
# (g) ground / overhead clutter rejected, real obstacle kept
# ---------------------------------------------------------------------------

def test_g_ground_clutter_rejected():
    radar = RadarSafetyOverride()
    v = 9.0
    # A 30x10 deg radar at z=1.0 with pitch 0 sprays its lower beams onto the
    # road. Those returns are static, so their closing speed is exactly ego
    # speed -- without a height gate they read as an imminent collision on a
    # completely empty road.
    ground = np.stack([radar_point(d, v, lateral=0.0, height_above_road=0.0)
                       for d in (6.0, 10.0, 15.0, 22.0, 30.0)])
    cap_v, tel = radar.evaluate(ground, v)
    assert tel["radar_n_returns"] == 5
    assert tel["radar_n_in_corridor"] == 0, tel
    assert cap_v == float("inf")
    assert tel["safety_emergency"] is False

    # a real obstacle at bumper height in the same sweep is still seen
    mixed = np.vstack([ground, radar_point(10.0, v, height_above_road=0.8)])
    cap_v2, tel2 = radar.evaluate(mixed, v)
    assert tel2["radar_n_in_corridor"] == 1
    assert cap_v2 < v

    # overhead structures (gantries, bridges) are rejected too
    over = np.stack([radar_point(20.0, v, height_above_road=4.5)])
    assert radar.evaluate(over, v)[0] == float("inf")
    print("(g) ground clutter rejected, real obstacle kept OK")


# ---------------------------------------------------------------------------
# (h) centerline ego frame == the production local_command_xy transform
# ---------------------------------------------------------------------------

def test_h_centerline_frame_matches_production():
    """Reproduce the agent's own transform on an arbitrary point and require
    the centerline module to agree."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        pos = rng.normal(0, 50, 2)
        theta = rng.uniform(-np.pi, np.pi)
        node = pos + rng.normal(0, 20, 2)

        # --- production expression, as used in the agent's run_step ---
        can0, can1 = pos[0], -pos[1]
        cmd_xy = np.array([node[0] - can0, -node[1] - can1])
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        expect = rot @ cmd_xy

        # --- the centerline module, given a polyline through `node` ---
        d = node - pos
        n = d / max(np.linalg.norm(d), 1e-9)
        s_node = float(np.linalg.norm(d))
        xy = np.stack([pos + n * s for s in np.arange(0.0, s_node + 60.0, 0.5)])
        cl = RouteCenterline(xy)
        # first waypoint at arc length exactly s_node -> the same world point
        got = cl.waypoints_ego(pos, theta, s_node / 0.5)[0]
        assert np.allclose(got, expect, atol=1e-6), (got, expect)
    print("(h) centerline frame matches production transform OK")


# ---------------------------------------------------------------------------
# (i) non-closing objects are transparent (no close-range brake latch)
# ---------------------------------------------------------------------------

def test_i_nonclosing_transparent():
    """What IS guaranteed: under the correct sign convention a receding or
    non-closing object imposes NO ceiling and NO emergency, which is what
    rules out a hard brake latch behind an object that is driving away.

    What is NOT claimed anywhere: that an inverted RADAR_V_SIGN is fail-safe.
    It is dangerous in both directions -- it silences approaching objects AND
    makes receding ones read as closing. The defense is the per-tick
    `radar_v_static_med` telemetry, which must track +ego_speed and is the
    first thing to read on any new run.
    """
    radar = RadarSafetyOverride()
    v = 10.0
    for gap in (2.5, 5.0, 30.0, 59.0):
        cap_v, tel = radar.evaluate(np.stack([radar_point(gap, -6.0)]), v)
        assert tel["radar_ttc"] == float("inf"), gap
        assert tel["safety_emergency"] is False, gap
        assert cap_v == float("inf"), (gap, cap_v)   # transparent even at 2.5 m

    # a stationary-relative return (closing ~ 0) is transparent too
    cap0, tel0 = radar.evaluate(np.stack([radar_point(1.8, 0.0)]), 0.0)
    assert cap0 == float("inf") and tel0["safety_emergency"] is False
    print("(i) non-closing objects transparent (no brake latch) OK")


# ---------------------------------------------------------------------------
# (k) emergency requires persistence; one hot frame cannot full-brake
# ---------------------------------------------------------------------------

def test_k_emergency_persistence():
    radar = RadarSafetyOverride()
    v = 10.0
    hot = np.stack([radar_point(5.0, 10.0)])      # ttc 0.5 s
    cold = np.stack([radar_point(50.0, 1.0)])     # ttc 50 s

    caps = []
    for _ in range(EMERGENCY_PERSIST_N):
        cap_v, tel = radar.evaluate(hot, v)
        caps.append((cap_v, tel["safety_emergency"],
                     tel["safety_emergency_streak"]))
    # emergency exactly at the Nth consecutive frame, not before
    assert all(e is False for _, e, _ in caps[:-1]), caps
    assert caps[-1][1] is True and caps[-1][0] == 0.0, caps
    # pre-emergency frames still cap via the stopping-distance law
    assert all(c < v for c, _, _ in caps[:-1]), caps

    # a cold frame resets the streak
    _, tel_cold = radar.evaluate(cold, v)
    assert tel_cold["safety_emergency_streak"] == 0, tel_cold
    _, tel_hot = radar.evaluate(hot, v)
    assert tel_hot["safety_emergency"] is False, tel_hot
    print("(k) emergency persistence OK")


# ---------------------------------------------------------------------------
# (l) suspect radar column order -> fail-transparent
# ---------------------------------------------------------------------------

def test_l_layout_check_fail_transparent():
    radar = RadarSafetyOverride()
    v = 8.0
    # 40 returns whose AZIMUTH spread is narrower than the ALTITUDE spread
    # (impossible under the configured 30x10 deg FOV unless the columns are
    # swapped) -- one of them a would-be hazard dead ahead
    rng = np.random.default_rng(0)
    pts = np.stack([radar_point(20.0, 1.0,
                                lateral=float(rng.uniform(-0.3, 0.3)),
                                height_above_road=float(rng.uniform(-2.0, 4.0)))
                    for _ in range(40)] + [radar_point(6.0, v)])
    cap_v, tel = radar.evaluate(pts, v)
    assert radar.layout_warning, "layout check should have fired"
    assert tel["radar_layout_disabled"] is True
    assert cap_v == float("inf"), cap_v      # transparent, not mis-gating
    # ...and it stays transparent on later, plausible frames
    cap2, tel2 = radar.evaluate(np.stack([radar_point(6.0, v)]), v)
    assert cap2 == float("inf") and tel2["radar_layout_disabled"] is True

    # sanity: a healthy instance with the same hazard DOES cap
    ok = RadarSafetyOverride()
    cap3, _ = ok.evaluate(np.stack([radar_point(6.0, v)]), v)
    assert cap3 < v
    print("(l) suspect layout -> fail-transparent OK")


# ---------------------------------------------------------------------------
# (m) latched hold target does not ratchet down with the coasting speed
# ---------------------------------------------------------------------------

def test_m_latched_hold_does_not_ratchet():
    """The caller contract, stated as a property of the demand rather than of
    the PID: with a LATCHED entry speed the tracked target is constant while
    the ego coasts, so the PID keeps a restoring error. Re-reading the
    current speed every tick instead makes the target follow the speed down
    -- a silent coast-to-stop wearing a "hold speed" label, which below
    `brake_speed` becomes a full brake.
    """
    entry = 6.0
    coasting = [6.0, 5.4, 4.7, 3.9, 2.8, 1.5, 0.6, 0.2]

    latched = [peb(v, desired_wp=99.0, override=entry)[2]["final_target_speed"]
               for v in coasting]
    assert all(abs(t - entry) < 1e-12 for t in latched), latched
    # the PID is handed a positive, growing error -> a restoring force
    deltas = [window_delta(t, v, 0.25) for t, v in zip(latched, coasting)]
    assert deltas[0] == 0.0 and all(d > 0.0 for d in deltas[1:]), deltas
    assert all(np.diff(deltas) >= -1e-12), deltas

    per_tick = [peb(v, desired_wp=99.0, override=v)[2]["final_target_speed"]
                for v in coasting]
    assert all(np.diff(per_tick) <= 1e-12), per_tick       # ratchets down
    assert all(window_delta(t, v, 0.25) == 0.0
               for t, v in zip(per_tick, coasting)), "error drains to zero"
    # ...and once the ratcheted target falls below brake_speed it full-brakes
    _, brk_end, tel_end = peb(coasting[-1], desired_wp=99.0,
                              override=coasting[-1])
    assert brk_end == 1.0 and tel_end["peb_branch"] == "stop"
    print("(m) latched hold does not ratchet; per-tick target does OK")


# ---------------------------------------------------------------------------
# (n) route end -> no waypoint collapse, no arctan2(0, 0)
# ---------------------------------------------------------------------------

def test_n_route_end_extrapolation():
    cl = make_centerline([[0.0, -float(s)] for s in np.arange(0.0, 30.0, 1.0)])
    v = 6.0
    # ego 2 m before the final vertex: the 6 x 3 m horizon extends 16 m past it
    wp = cl.waypoints_ego(np.array([0.0, -27.0]), 0.0, v)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    assert np.all(seg > 0.5), seg            # no collapsed pairs
    tail = wp[-1] - wp[-2]
    assert np.linalg.norm(tail) > 0.5, tail  # aim_last is well-defined
    # extrapolation continues the final segment direction (straight ahead)
    assert np.max(np.abs(wp[:, 0])) < 1e-6, wp
    print("(n) route-end extrapolation keeps spacing OK")


if __name__ == "__main__":
    for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
        globals()[fn]()
    print("\nALL CLRE EXECUTION-LAYER TESTS PASSED")
