#!/usr/bin/env python
"""E59 -- minimal route-consistent centerline fallback.

WHAT THIS REPLACES (E59 task sections 1-2)
==========================================
Frozen A-v3 selects, when NO candidate is hard-feasible, the
`all_infeasible_safest` candidate (max-min-clearance -- typically a hard
brake), and E48's PEB additionally forced full brake on that mode. E49
showed the resulting stop is not what recovers the vehicle, and E56
measured 76.3-77.4% of "infeasible" verdicts on route 3178 as PHANTOM
(prediction error, oracle says the space was clear). Treating
`all_infeasible` as a physical hazard therefore converts a *prediction*
defect into a *stop*, which E50 identified as the dominant failure form
(wedge/blocked).

PEB-v2 reinterprets `all_infeasible` as "the optimizer has nothing to
say this tick", NOT as "there is an obstacle":

  longitudinal : hold the ego speed measured at fallback ENTRY, latched
                 once per episode (no automatic slow-down, no automatic
                 acceleration; latching is the agent's job -- see the
                 CALLER CONTRACT in peb_v2_controller.control_pid)
  lateral      : follow the route centerline

Physical hazard is handled ONLY by the independent radar layer
(`radar_safety.py`), which runs in this mode too.

CENTERLINE SOURCE (no new interface, no new map query)
======================================================
The leaderboard hands every agent the interpolated global route at
`hop_resolution = 1.0 m` (`route_manipulation.interpolate_trajectory`).
`AutonomousAgent.set_global_plan` keeps the FULL dense GPS copy on
`self._plan_gps_HACK`, while `self._global_plan` is downsampled to <=50 m
spacing -- too sparse to steer on. We therefore use `_plan_gps_HACK`,
converted to world coordinates ONCE at setup via the agent's own
`RoutePlanner.gps_to_location` (same transform the production tick path
already uses for `command_near_xy`, so no new geometry convention is
introduced).

Because this IS the navigation route, it turns at intersections by
construction -- the "follow the route command rather than blindly
continuing straight" requirement is satisfied structurally, with no
command-branching logic of our own. It is also route-consistent by
definition, so the fallback cannot steer the ego off-route.

OUTPUT CONTRACT
===============
`waypoints_ego(...)` returns a (6,2) array in the SAME ego frame and the
SAME arc-length convention that `resample_wp6` produces for the planner
(6 points at 0.5 s spacing at the requested speed), so it is a drop-in
`exec_plan` for the unchanged PID `control_pid()`. Nothing downstream
needs to know which source produced the plan.
"""
import numpy as np

WP_N = 6            # matches cap.resample_wp6 output length
WP_DT = 0.5         # s, matches the 0.5 s export interval
MIN_SPACING = 1.0   # m, steering-geometry floor only (see waypoints_ego)
STALE_HINT_M = 15.0  # m, windowed-search miss distance that forces a rescan


class RouteCenterline:
    """Dense world-frame route polyline + ego-frame waypoint sampling."""

    def __init__(self, xy_world):
        self.xy = np.asarray(xy_world, np.float64).reshape(-1, 2)
        if len(self.xy) >= 2:
            seg = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
            self.s = np.concatenate([[0.0], np.cumsum(seg)])
        else:
            self.s = np.zeros(len(self.xy))
        self._i_hint = 0
        # distance from the last queried ego position to this polyline, in
        # the AGENT's own frame. This is the decisive frame check: a vehicle
        # driving its route must stay within roughly a lane half-width of
        # the centerline, and both quantities come from the same
        # `gps_to_location`, so any constant georeference offset cancels.
        # Logged every tick as `centerline_offset_m` (see the agent).
        self.last_offset_m = float("nan")

    # ------------------------------------------------------------------
    @classmethod
    def from_agent(cls, agent):
        """Build from the agent's dense GPS plan (`_plan_gps_HACK`, 1 m hop)
        using the AGENT's own `gps_to_location`.

        Using the agent's transform rather than the leaderboard's true
        lat/lon reference is deliberate: `VadAgent._init` does not read the
        map's georeference, it numerically *solves* for `lat_ref`/`lon_ref`
        with `fsolve`. Ego position (`tick_data['pos']`) is produced by that
        same solved transform, so converting the route with it puts the
        centerline and the ego in the SAME frame even if the solved
        reference carries a small offset. Using the true reference here
        would introduce a systematic centerline bias.

        Returns None if the dense plan is unavailable, so the caller can
        degrade gracefully (see agent wiring).
        """
        plan = getattr(agent, "_plan_gps_HACK", None)
        if not plan or not hasattr(agent, "gps_to_location"):
            return None
        pts = []
        for item in plan:
            g = item[0] if isinstance(item, (tuple, list)) else item
            try:
                lat, lon = float(g["lat"]), float(g["lon"])
            except Exception:
                continue
            pts.append(agent.gps_to_location(np.array([lat, lon])))
        if len(pts) < 2:
            return None
        obj = cls(np.asarray(pts, np.float64))

        # One-time SAME-FRAME self-check: the leaderboard also hands us the
        # DOWNSAMPLED route (`_global_plan`) as GPS. Converting it with the
        # SAME `gps_to_location` and requiring every point to lie on the
        # dense polyline validates the densification and the transform's
        # internal consistency.
        #
        # An earlier version compared against `_global_plan_world_coord`,
        # i.e. TRUE CARLA world coordinates. That was a CROSS-FRAME
        # comparison and it produced a false alarm on the E59 smoke run
        # (24.29 m): `VadAgent._init` does not read the map georeference,
        # it `fsolve`s for lat_ref/lon_ref, and on route 3785 that solution
        # carried a CONSTANT (-25.5, -1.5) m offset from true world --
        # removing that one translation dropped the route-shape residual to
        # 1.29 m, with the route span matching to 0.6 m (218.4 vs 219.0 m).
        # The offset is harmless precisely because ego position AND
        # centerline both pass through that same transform, so it cancels
        # in `centerline - ego`, which is all `waypoints_ego` ever uses.
        # The old metric could not distinguish that benign offset from a
        # real geometry error, so it is replaced rather than re-tuned. The
        # decisive per-tick check is `last_offset_m` below.
        obj.setup_residual_m = float("nan")
        try:
            ds = getattr(agent, "_global_plan", None)
            if ds:
                ref = []
                for item in ds:
                    g = item[0] if isinstance(item, (tuple, list)) else item
                    ref.append(agent.gps_to_location(
                        np.array([float(g["lat"]), float(g["lon"])])))
                ref = np.asarray(ref, np.float64)
                d = np.linalg.norm(obj.xy[None, :, :] - ref[:, None, :], axis=2)
                obj.setup_residual_m = float(d.min(axis=1).max())
        except Exception:
            pass
        return obj

    # ------------------------------------------------------------------
    def note_position(self, p_world):
        """Advance the progress hint. MUST be called every tick (not only
        when the fallback fires), otherwise the windowed search below can be
        arbitrarily stale by the time the fallback is first needed."""
        return self._nearest_index(p_world)

    def _nearest_index(self, p_world):
        """Nearest polyline vertex.

        Searched in a moving window around the previous hit so a route that
        doubles back on itself cannot snap backwards, and so the cost is
        O(window) per tick. If the windowed best is implausibly far (the
        vehicle was respawned, or the hint went stale), fall back to a global
        argmin -- correctness over the monotonicity heuristic.
        """
        p = np.asarray(p_world, np.float64)
        n = len(self.xy)
        lo = max(0, self._i_hint - 50)
        hi = min(n, self._i_hint + 400)
        d = np.linalg.norm(self.xy[lo:hi] - p, axis=1)
        i = int(lo + np.argmin(d))
        best = float(d.min())
        if best > STALE_HINT_M:
            dg = np.linalg.norm(self.xy - p, axis=1)
            i = int(np.argmin(dg))
            best = float(dg.min())
        self._i_hint = i
        self.last_offset_m = best
        return i

    # ------------------------------------------------------------------
    def waypoints_ego(self, p_world, compass_rad, speed_mps,
                      n=WP_N, dt=WP_DT):
        """(n,2) waypoints tracing the route ahead, in the PID's ego frame.

        Spacing is `speed * dt`, floored at MIN_SPACING. The floor exists
        purely to keep the STEERING geometry well conditioned near standstill
        (without it all six points collapse onto the ego and `arctan2(0,0)`
        feeds the turn PID garbage). It is deliberately NOT allowed to set the
        longitudinal target: the agent passes the hold speed to the controller
        explicitly via `desired_speed_override`, so the floor cannot make the
        fallback creep forward from rest.

        FRAME: this reproduces, verbatim, the transform the production tick
        path already applies to build `local_command_xy` in
        `candidate_a_agent.run_step`:

            can0, can1        = pos[0], -pos[1]
            command_near_xy   = [node[0] - can0, -node[1] - can1]
            rot               = [[cos(th), -sin(th)], [sin(th), cos(th)]]
            local_command_xy  = rot @ command_near_xy

        i.e. negate world Y (CARLA's world is left-handed; every other
        consumer in this codebase negates it -- the same convention whose
        omission was the E56 BLOCKER), then rotate by the raw compass angle.
        The result is x-right / y-forward, matching `res.wp6` and
        `out_truck`, so this array is a drop-in `exec_plan`. Reusing the
        production expression rather than writing an equivalent one is
        intentional: it makes a frame mismatch impossible by construction.
        """
        i0 = self._nearest_index(p_world)
        step = max(float(speed_mps) * float(dt), MIN_SPACING)
        s_targets = self.s[i0] + step * np.arange(1, n + 1)
        x = np.interp(s_targets, self.s, self.xy[:, 0])
        y = np.interp(s_targets, self.s, self.xy[:, 1])
        # Past the route end np.interp CLAMPS, which would collapse the
        # trailing waypoints onto the final vertex and feed arctan2(0,0)
        # into the PID's aim_last angle (review MAJOR: the exact degeneracy
        # MIN_SPACING exists to prevent, re-entering through the back door
        # over the last ~n*step metres of every route). Extrapolate along
        # the final segment direction instead, so spacing survives; the
        # leaderboard ends the route on arrival, so the extrapolated tail
        # is only ever aimed at, not driven.
        over = s_targets > self.s[-1]
        if over.any() and len(self.xy) >= 2:
            tail = self.xy[-1] - self.xy[-2]
            u = tail / max(float(np.linalg.norm(tail)), 1e-9)
            ds = s_targets[over] - self.s[-1]
            x[over] = self.xy[-1, 0] + u[0] * ds
            y[over] = self.xy[-1, 1] + u[1] * ds

        p = np.asarray(p_world, np.float64)
        can0, can1 = p[0], -p[1]
        d = np.stack([x - can0, -y - can1], axis=1)
        th = float(compass_rad)
        rot = np.array([[np.cos(th), -np.sin(th)],
                        [np.sin(th), np.cos(th)]])
        return (rot @ d.T).T
