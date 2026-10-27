#!/usr/bin/env python
"""E59 -- independent radar-based longitudinal safety override.

DESIGN CONTRACT (E59 task section 3)
====================================
This layer is a MEASUREMENT-space supervisor. It is deliberately and
structurally independent of the planner:

  * it consumes ONLY directly observed radar returns + ego speed;
  * it NEVER reads `all_infeasible`, `sel_mode`, `exec_source`, predicted
    OBB feasibility, or any planner state;
  * it runs identically in normal A-v3 planning mode and in fallback mode.

Rationale from this project's own diagnostics:
  * E56 measured 76.3-77.4% of the planner's "infeasible" verdicts on
    route 3178 as PHANTOM -- an oracle re-check found the space physically
    clear (pooled 76.3%, independently-verified subset 77.4%). A supervisor
    keyed on planner infeasibility inherits the very prediction defect it
    is supposed to backstop.
  * DevVal-16 found collisions occur while the controller faithfully
    executes a wrong plan (ego UNDER desired speed at zero brake at the
    contact instant). Only a plan-independent measured signal fires there.
  * E58 measured that 29% of dev-cohort collisions are with `static.*`
    props (fence/pole/wall/guardrail) absent from the VAD agent set
    entirely, so no objective-space term can ever see them. Radar is a
    ray-cast sensor and does return them.

RADAR ARRAY LAYOUT (verified against CARLA source, not comments)
================================================================
`LibCarla/source/carla/sensor/data/RadarData.h` (the copy vendored under
CARLA_0.9.15/Plugins/carlaviz -- the packaged build ships no first-party
LibCarla sources, so this is the closest authoritative source available
in this environment) defines the wire struct as

    struct RadarDetection { float velocity; azimuth; altitude; depth; };

and `RadarSerializer` memcpy's the detection vector with no header, so
`np.frombuffer(...).reshape(-1, 4)` yields columns
`[velocity, azimuth, altitude, depth]`. Bench2Drive's
`sensor_interface.py::_parse_radar_cb` then applies `np.flip(points, 1)`,
so THIS module receives

    col 0 = depth (m) | col 1 = altitude (rad) | col 2 = azimuth (rad)
                                               | col 3 = velocity (m/s)

WARNING for future readers: two nearby comments are WRONG about columns
1/2 -- Bench2Drive's own `# [depth, azimuth, altitute, velocity]` and
CARLA's `manual_control.py` example. The C++ struct is authoritative
because it is the memcpy'd layout. `assert_layout_plausible()` below is a
cheap runtime cross-check (azimuth must span the configured horizontal
FOV, which is 3x the vertical FOV, so a swap is directly observable in
the spreads), and on failure the layer goes FAIL-TRANSPARENT rather than
gating on garbage columns.

VELOCITY SIGN -- MEASURED, NOT ASSUMED (E59 smoke run, route 3785)
==================================================================
CARLA's docs describe `RadarDetection.velocity` as the velocity of the
detected object TOWARDS the sensor, which would make positive = closing.
**That is wrong for this build.** The E59 smoke run measured it directly:

    median( raw_velocity ) / ego_speed  =  -0.9935   over 653 moving frames
    99.4% of moving frames negative;  corr(-raw, speed) = +0.934

i.e. static clutter -- which by definition closes at exactly ego speed --
reads NEGATIVE with magnitude equal to ego speed (matching to 0.7%). The
column is a RANGE RATE (d(depth)/dt), negative while closing. Hence

    RADAR_V_SIGN = -1.0

The magnitude agreement to 0.7% is also an independent confirmation that
the depth and velocity COLUMNS are correctly identified (a mis-identified
column would not track ego speed at all).

History, kept deliberately: the first draft shipped `+1.0` and claimed the
layer was "fail-safe by construction" against an inversion. The adversarial
review refuted the fail-safe claim (an inverted sign silences approaching
objects AND makes receding ones read as closing, which can brake
spuriously), and the smoke run then showed the sign really was inverted:
`safety_override_active` fired on 0 of 705 ticks -- the layer was
STRUCTURALLY INERT for the whole run. Had the "fail-safe" wording survived,
that null result would have been misread as "the route needed no
intervention" instead of "the layer never ran".

Defenses retained: (1) `radar_v_static_med` is logged every tick and must
track +ego_speed under the now-corrected convention -- it stays the FIRST
item to read on any run, before any behavioural statistic; (2) the
emergency persistence requirement below, so one mis-signed or noisy frame
cannot command a full brake.

KNOWN GEOMETRIC LIMITS (measured by the adversarial review; disclosed,
not designed around)
=====================
The height gate is computed in the SENSOR frame (`sensor_z + d*sin(alt)`)
and is exact only on a flat road with zero ego pitch. Consequences:
  * road grade rising >= ~3% ahead (sag curve, ramp foot) or a nose-down
    ego pitch of ~2-3 deg (hard braking) admits ROAD-SURFACE returns into
    the corridor -> phantom proportional/emergency braking on empty road,
    with a possible brake->pitch->more-returns feedback;
  * on a ~5% downgrade the same gate REJECTS most of a lead vehicle's
    body returns at 20-30 m -> the effective detection range against a
    lead vehicle collapses exactly where stopping distance grows.
A single fixed-pitch radar cannot distinguish these without an IMU-fed
pitch correction, which is out of E59's minimal scope. The first CARLA
run must therefore watch `radar_n_in_corridor` on open road (should be
~0; persistently large on grades = the false-positive mode above).

KNOWN SENSING LIMITS (recorded honestly, not designed around)
=============================================================
The leaderboard OVERRIDES radar `range` to 100 m and `points_per_second`
to 1500 regardless of what `sensors()` requests (agent_wrapper.py:197-198).
At 20 Hz that is ~75 returns per frame spread over the whole FOV, so the
sampling of any single object is thin and stochastic. This layer is
therefore a coarse supervisor, not a tracker: it takes the CLOSEST
in-corridor return and never attempts association across frames.
"""
import numpy as np

# --- radar array columns (see module docstring; verified vs RadarData.h) ---
C_DEPTH, C_ALT, C_AZI, C_VEL = 0, 1, 2, 3

# MEASURED on the E59 smoke run (route 3785): the raw column is a range
# rate, NEGATIVE while closing -- the opposite of what CARLA's docs say.
# ratio to ego speed on static clutter = -0.9935 over 653 moving frames.
# Single point of truth; NOT fail-safe if wrong -- see VELOCITY SIGN in the
# module docstring, and re-verify radar_v_static_med on every new run.
RADAR_V_SIGN = -1.0

# Consecutive frames of TTC <= ttc_emergency required before the emergency
# full brake fires (0.15 s at 20 Hz). E51 established persistence as the
# discriminator against bursty signals in this project, and the review
# showed the closest-in-corridor pick is stochastic frame-to-frame at the
# leaderboard's forced ~75 returns/frame: one noisy (or one mis-signed)
# frame must not be able to command a full brake. The proportional ceiling
# below is NOT gated on persistence -- it is continuous, so a transient
# false positive costs a transient mild slowdown, not a brake spike.
EMERGENCY_PERSIST_N = 3


class RadarSafetyOverride:
    """Continuous speed ceiling derived from measured range + closing rate.

    Physical model -- one equation, no tuned gain schedule:

        d_stop(v) = v * t_reaction + v^2 / (2 * a_brake)

    is the distance needed to stop from speed v given an actuation latency
    `t_reaction` and usable deceleration `a_brake`. Solving
    `d_stop(v) = d_gap - d_standoff` for the largest admissible v gives the
    ceiling

        v_safe = -a*t + sqrt((a*t)^2 + 2*a*max(d_gap - d_standoff, 0))

    the standard constant-deceleration STOPPING-DISTANCE law (an earlier
    draft called it a "safe-following law"; the review correctly objected
    that a following law would be a function of relative speed -- this one
    deliberately is not, it bounds the worst case where the object ahead
    stops dead). It is applied ONLY while the nearest in-corridor return is
    actually CLOSING (v_close > 0); see the gate in `evaluate()` for why.
    Every constant below is a physical quantity with an independently
    checkable value; no shaping coefficients were fitted.

    Emergency: when the measured TTC stays below `ttc_emergency` for
    EMERGENCY_PERSIST_N consecutive frames, the ceiling is forced to 0 AND
    a full-brake flag is raised, because at that point tracking a merely-
    lowered ceiling can no longer arrest the vehicle in time.
    """

    def __init__(self,
                 a_brake=4.0,          # m/s^2 usable decel (planner bound is 6.0)
                 t_reaction=0.4,       # s, sense -> actuation latency budget
                 d_standoff=2.0,       # m, residual bumper gap to preserve
                 ttc_emergency=1.2,    # s, below this -> emergency full brake
                 corridor_half_w=1.2,  # m, |lateral| gate (ego half-width 0.93)
                 sensor_z=1.0,         # m, radar height above the road
                 h_ground_clear=0.30,  # m, ignore returns below this height
                 h_overhead=2.20,      # m, ignore returns above this height
                 max_range=60.0,       # m, ignore returns beyond this
                 min_range=0.5):       # m, ignore self/near-field clutter
        self.a_brake = float(a_brake)
        self.t_reaction = float(t_reaction)
        self.d_standoff = float(d_standoff)
        self.ttc_emergency = float(ttc_emergency)
        self.corridor_half_w = float(corridor_half_w)
        self.sensor_z = float(sensor_z)
        self.h_ground_clear = float(h_ground_clear)
        self.h_overhead = float(h_overhead)
        self.max_range = float(max_range)
        self.min_range = float(min_range)
        self._layout_checked = False
        self._emergency_streak = 0
        self.layout_warning = ""

    # ------------------------------------------------------------------
    def params(self):
        """Provenance dict for the run record."""
        return {k: v for k, v in vars(self).items()
                if not k.startswith("_") and isinstance(v, float)}

    # ------------------------------------------------------------------
    def assert_layout_plausible(self, pts):
        """One-shot runtime check of the C++-derived column order.

        The radar is configured with horizontal_fov = 3 x vertical_fov, so
        the azimuth column must have a visibly wider spread than the
        altitude column. If the two are swapped relative to the struct, this
        inequality reverses.

        On failure the layer goes FAIL-TRANSPARENT: `layout_warning` is
        recorded into provenance AND `evaluate()` publishes an infinite
        ceiling for the remainder of the route (see the guard there), so
        driving degrades to the frozen A-v3 + PID baseline instead of
        continuing to gate on garbage columns. The review showed
        warn-and-continue was not a defense: with swapped columns the
        +-5 deg column would be used as the LATERAL gate, admitting every
        return inside ~14 m, and the +-15 deg column as the height gate,
        spraying roadside returns into the accept band -- i.e. near-
        continuous phantom braking for a full campaign with only a log
        line to show for it. The residual risk of this choice is the
        opposite one -- a false-positive warning silently disables the
        safety layer -- which is why the warning is surfaced in
        cand_a_meta.json and listed as the FIRST thing to read after the
        smoke run.
        """
        if self._layout_checked or len(pts) < 30:
            return
        self._layout_checked = True
        s_alt = float(np.std(pts[:, C_ALT]))
        s_azi = float(np.std(pts[:, C_AZI]))
        if not (s_azi > s_alt):
            self.layout_warning = (
                f"radar column order suspect: std(azimuth)={s_azi:.4f} "
                f"<= std(altitude)={s_alt:.4f}; expected azimuth wider "
                f"(horizontal_fov = 3x vertical_fov)")

    # ------------------------------------------------------------------
    def _corridor_mask(self, pts):
        """Returns that the ego body would actually sweep through.

        lateral  = depth * sin(azimuth)   -- symmetric gate, +-corridor_half_w
        height   = depth * sin(altitude)  -- relative to the sensor

        The height gate is what makes this usable at all: the radar sits
        `sensor_z` above the road with a symmetric vertical FOV, so at any
        appreciable range its lower beams strike the ROAD SURFACE. Those
        returns are static, hence their closing speed equals ego speed
        exactly, and without this gate they would produce a continuous
        stream of phantom TTC-emergency brakes while driving normally on
        an empty road. Returns are kept only if they sit between
        `h_ground_clear` and `h_overhead` above the road. (Flat-road
        assumption; see KNOWN GEOMETRIC LIMITS in the module docstring for
        the measured failure modes on grades and under brake dive.)
        """
        d = pts[:, C_DEPTH]
        lateral = np.abs(d * np.sin(pts[:, C_AZI]))
        h_road = self.sensor_z + d * np.sin(pts[:, C_ALT])
        return ((lateral <= self.corridor_half_w) &
                (h_road >= self.h_ground_clear) &
                (h_road <= self.h_overhead))

    # ------------------------------------------------------------------
    def evaluate(self, radar_pts, ego_speed):
        """Returns (v_ceiling, telemetry) -- a pure MEASUREMENT.

        `radar_pts` : (N,4) array in the documented column order, or None.
        `ego_speed` : m/s, current measured ego speed (telemetry and the
                      sign self-check only; the ceiling depends solely on
                      measured geometry and closing rate, never on how fast
                      we happen to be going).

        This function deliberately does NOT know the normal/fallback target
        speed and does NOT decide whether an intervention occurred. It
        publishes a ceiling; `PEBv2Controller` applies `min(desired,
        ceiling)` and reports `safety_override_active`. That split keeps the
        hazard measurement independent of the demand it modifies, so the two
        can be attributed separately in post-hoc analysis (requirement 5).

        With nothing closing in the corridor the ceiling is +inf: the layer
        is exactly transparent. It can only ever lower a demand, never
        raise one, so it can never command acceleration.
        """
        tel = {"radar_n_returns": 0, "radar_n_in_corridor": 0,
               "radar_min_gap": float("nan"),
               "radar_closing_speed": float("nan"),
               "radar_ttc": float("inf"),
               "radar_v_ceiling": float("inf"),
               "radar_v_static_med": float("nan"),
               "safety_emergency": False,
               "safety_emergency_streak": 0,
               "radar_layout_disabled": bool(self.layout_warning)}

        if radar_pts is None or len(radar_pts) == 0:
            self._emergency_streak = 0
            return float("inf"), tel
        pts = np.asarray(radar_pts, np.float64).reshape(-1, 4)
        tel["radar_n_returns"] = int(len(pts))
        self.assert_layout_plausible(pts)
        if self.layout_warning:
            # FAIL-TRANSPARENT: column order suspect -> the gates would be
            # nonsense, so publish no ceiling for the rest of the route
            # (degrades to frozen A-v3 + PID) and surface the warning in
            # provenance. See assert_layout_plausible.
            tel["radar_layout_disabled"] = True
            self._emergency_streak = 0
            return float("inf"), tel

        rng_ok = ((pts[:, C_DEPTH] >= self.min_range) &
                  (pts[:, C_DEPTH] <= self.max_range))
        sel = pts[rng_ok & self._corridor_mask(pts)]
        tel["radar_n_in_corridor"] = int(len(sel))

        # Sign-convention self-check, computed over ALL in-range returns
        # (not just the corridor) because most of the scene is static: the
        # median signed radial velocity should read +ego_speed. Logged every
        # tick so the first CARLA run can falsify RADAR_V_SIGN directly.
        if rng_ok.any():
            tel["radar_v_static_med"] = float(
                RADAR_V_SIGN * np.median(pts[rng_ok, C_VEL]))

        if len(sel) == 0:
            self._emergency_streak = 0
            return float("inf"), tel

        i_near = int(np.argmin(sel[:, C_DEPTH]))
        d_gap = float(sel[i_near, C_DEPTH])
        v_close = RADAR_V_SIGN * float(sel[i_near, C_VEL])
        tel["radar_min_gap"] = d_gap
        tel["radar_closing_speed"] = v_close

        ttc = (d_gap / v_close) if v_close > 1e-3 else float("inf")
        tel["radar_ttc"] = float(ttc)

        # The ceiling applies only to CLOSING hazards. A non-closing return
        # (receding lead car, static object while the ego is stopped) poses
        # no kinetic threat this tick, and capping on gap alone was shown by
        # the review to hard-latch brake=1.0 behind any in-corridor return
        # inside ~2 m even as it drives AWAY -- recreating the E50 wedge
        # family this experiment exists to reduce. Approaching a static
        # obstacle still engages the ceiling (closing speed = ego speed > 0),
        # so the static-hazard coverage that motivated the radar is kept.
        if v_close > 1e-3:
            margin = max(d_gap - self.d_standoff, 0.0)
            a, t = self.a_brake, self.t_reaction
            v_safe = float(max(
                -a * t + np.sqrt((a * t) ** 2 + 2.0 * a * margin), 0.0))
        else:
            v_safe = float("inf")

        # Emergency full brake requires the TTC condition to PERSIST for
        # EMERGENCY_PERSIST_N consecutive frames (see constant above).
        if ttc <= self.ttc_emergency:
            self._emergency_streak += 1
        else:
            self._emergency_streak = 0
        tel["safety_emergency_streak"] = int(self._emergency_streak)
        if self._emergency_streak >= EMERGENCY_PERSIST_N:
            v_safe = 0.0
            tel["safety_emergency"] = True

        tel["radar_v_ceiling"] = v_safe
        return v_safe, tel
