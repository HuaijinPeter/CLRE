#!/usr/bin/env python
"""Candidate A closed-loop MVP planner (Phase 4B-2A).

This is the CLOSED-LOOP implementation of the FROZEN Phase-4B-1 Candidate-A
formulation (docs/phase4b1_candidate_a_formulation.md, run-4 weights; offline
prototype research/candidate_a/mvp_planner.py). The formulation is preserved
term-by-term; this module only adds what closed-loop use requires:

  * batched (vectorized-over-agents) objective so one solve is a fixed small
    number of tensor ops (the offline prototype looped over agents in Python);
  * device/dtype configurability (CPU float64 reproduces the frozen offline
    prototype bit-for-bit within float tolerance; GPU is for closed loop);
  * warm start across planning cycles: the previously selected control
    sequence, time-shifted, enters as ONE EXTRA seed next to the 8 frozen
    behavior seeds (the 8 basins keep competing every cycle — the brake basin
    is never dropped);
  * deterministic failure detection (NaN/Inf, hard-bound violation) so the
    calling agent can fall back to the Native plan for that cycle;
  * per-term objective components + timings for per-cycle logging.

FROZEN FORMULATION (verbatim from Phase 4B-1, run 4):
  state  x = [x, y, psi, v]   (kinematic bicycle, plan frame: x right, y fwd)
  control u = [a, kappa]      hard-clamped a in [-6, 2.5], |kappa| <= 0.2
  horizon 3.0 s, dt = 0.25 s (N = 12), v >= 0 (no reverse)

  J = 0.05 * Huber(p - p_nat)                          (tiny point anchor)
    + 1.2  * gate * crosstrack(p, native path @0.4 m)  (commitment-gated prior)
    - 1.0  * s_N                                       (terminal route progress)
    + 0.6  * (v - v_ref)_+^2                           (progress saturation)
    + 1.6  * top-1 oriented-ellipse comfort risk       (margins 0.6/0.4 m)
    + 25   * top-1 oriented-ellipse physical barrier   (margin 0.05 m)
    + 6    * visible Solid/SolidSolid boundary term
    + 0.5  * route-corridor overshoot beyond 4.0 m
    + 0.8  * smoothness (da^2 + 30 dkappa^2 + 3 kappa^2)

  gate = sigmoid(2 * (s_nat - 1.5))  — a stalled native plan carries no
  stop-authority (anti-stop-forever); v_ref caps the progress reward
  (anti-drive-into-hazard).

Known frozen-prototype quirks REPLICATED VERBATIM (fixing either would be a
formulation change and would invalidate the Phase-4B-1 falsification results;
recorded in docs/phase4b2a_candidate_a_mvp_implementation.md as revisit-only-
with-refalsification items):
  1. `_native_controls` returns a NEGATED curvature profile for the native
     seed (sign flip vs the rollout convention). Seeds only choose descent
     basins; Adam plus the point anchor / path prior corrects the path, and
     the frozen run-4 evidence was produced with exactly this seed.
  2. `interp_pred` prepends the PLAN-FRAME ORIGIN (not the agent's current
     position) as the t=0 knot before resampling agent predictions to the
     0.25 s grid, so the first ~0.5 s of every interpolated agent path is
     biased toward the ego origin. The frozen run-4 risk evidence includes
     this bias.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Frozen constants (identical to research/candidate_a/mvp_planner.py)
# ---------------------------------------------------------------------------
DT = 0.25
N_STEPS = 12
R_EGO = 1.3          # ego disk radius (half-width 1.06 + margin)
A_MIN, A_MAX = -6.0, 2.5
K_MAX = 0.2          # |curvature| bound (R >= 5 m)
V_MAX = 15.0

WEIGHTS = dict(pt=0.05, ct=1.2, prog=1.0, v=0.6, risk=1.6, bar=25.0,
               bnd=6.0, cor=0.5, sm=0.8)

SEED_NAMES = ("native", "brake", "commit", "swerve_pos_x", "brake_swerve_pos_x",
              "swerve_neg_x", "brake_swerve_neg_x", "creep_release", "warm",
              "warm_raw")

# Agent-selection constants (identical to the frozen offline window builder,
# tools/diagnosis/phase4b1_candidate_a_offline.py::build_window)
AGENT_SCORE_MIN = 0.30
AGENT_DIST_MAX = 30.0
AGENT_MAX = 8
EXT_MIN = 0.4
BOUNDARY_SCORE_MIN = 0.3
BOUNDARY_LABELS = (1, 2)          # Solid / SolidSolid

# --- Phase 4B-2A Revision 1 (A-v2) constants ---------------------------------
# R1: strictly-ADDITIVE inclusion of safety-relevant low-score dynamic actors.
# The frozen top-8 selection is never altered; actors with score in
# [R1_SCORE_MIN, AGENT_SCORE_MIN) are APPENDED (distance-ordered, at most
# R1_EXTRA_MAX) iff they are near (<= R1_DIST_MAX) and their top-1 predicted
# 3-s displacement >= R1_MOVE_MIN (moving actors only — static low-score
# clutter stays excluded). Constants justified by the audit in
# tools/diagnosis/phase4b2a_v2_repair_analysis.py (see
# docs/phase4b2a_candidate_a_v2_revision.md §R1).
R1_SCORE_MIN = 0.15
R1_DIST_MAX = 20.0
R1_MOVE_MIN = 2.0                 # m over the 3-s top-1 prediction
R1_EXTRA_MAX = 4
# --- Phase 4B-2A A-v3 constants (structural audit, E22) -----------------
# GEOM: the v1/v2 risk field models the ego as ONE disk of radius R_EGO=1.3
# (covers half-width 0.93 + margin) — it ignores the ego's LENGTH. Measured
# at the three A-v2 collision instants (E20 traces): true box contact occurs
# at normalized ellipse distance d~1.21-1.26, i.e. the "physical barrier" is
# ZERO at real longitudinal/oblique contact (understatement ~1.1 m: ego half
# length 2.41-2.45 vs disk 1.3). fix_geom evaluates the SAME ellipse field at
# THREE ego disk centers (0, +-EGO_DISC_OFF along heading; min over disks),
# which leaves lateral pricing identical (center disk) and completes the
# longitudinal geometry. lincoln.mkz_2020 half-extents: 2.45 x 0.93.
EGO_HALF_LEN = 2.45
EGO_HALF_W = 0.93
EGO_DISC_OFF = 1.633              # +-L/3: 3 disks of R_EGO cover the ego box
# FEAS: hard safety-feasibility layer (the escalation pre-registered in
# docs/phase4b1_candidate_a_formulation.md for "closed loop shows
# contact-buying", now observed twice — E20 M6). A candidate is FEASIBLE iff
# its rollout's exact oriented ego box (EGO_HALF_LEN x EGO_HALF_W, heading
# psi_k) keeps >= FEAS_EPS clearance from every agent's predicted oriented
# box (R2-anchored top-1) at every step. Selection: best-J among feasible;
# if none feasible: max-min-clearance candidate (safest, typically braking).
# Progress can no longer BUY predicted contact through the scalar objective.
# FEAS_EPS reuses the frozen formulation's own physical margin constant
# (0.05 m) — no new tuned number.
FEAS_EPS = 0.05

# R3: the native path prior becomes a true CROSS-TRACK prior: the densified
# native path is extended beyond its endpoint along its TERMINAL TANGENT
# (fallback: route direction when the path is degenerate/too short to define
# a tangent), so distance-to-path measures lateral deviation only — advancing
# along the native plan's natural continuation is free, and a short stalled
# plan can no longer convert along-track progress into fake "cross-track"
# cost (stop authority). Same 0.4 m densify step; extension covers the
# maximum rollout reach (V_MAX * 3 s). No new weights.
R3_EXT_STEP = 0.4
R3_EXT_LEN = 45.0                 # m, >= V_MAX * horizon
R3_TANGENT_MIN = 0.4              # m of path needed to define a tangent


# ---------------------------------------------------------------------------
# Frozen geometry helpers (verbatim math from the offline prototype)
# ---------------------------------------------------------------------------

def interp_pred(traj6: np.ndarray, n: int,
                pos: Optional[np.ndarray] = None) -> np.ndarray:
    """(6,2) positions @0.5s -> (n,2) @0.25s (extrapolate last segment).

    pos=None reproduces frozen v1 quirk Q-B: the t=0 knot is the plan-frame
    ORIGIN (see module doc). With the R2 fix the caller passes the actor's
    current observed position `pos` so the interpolated path is anchored to
    the actor's actual state in the same frame as its predicted future —
    a semantic bug fix, not a design choice."""
    knots = np.vstack([[0.0, 0.0] if pos is None else np.asarray(pos, float),
                       traj6])
    t_in = np.arange(7) * 0.5
    t_out = (np.arange(n) + 1) * DT
    ext = t_out > t_in[-1]
    out = np.stack([np.interp(t_out, t_in, knots[:, i]) for i in range(2)], 1)
    if ext.any():                       # hold last velocity
        vel = (knots[-1] - knots[-2]) / 0.5
        out[ext] = knots[-1] + vel[None] * (t_out[ext, None] - t_in[-1])
    return out


def native_fine(native: np.ndarray, n: int) -> np.ndarray:
    """Native 6x0.5s plan -> n x 0.25s points (hold endpoint beyond 3 s)."""
    knots = np.vstack([[0, 0], native])
    t_in = np.arange(7) * 0.5
    t_out = np.minimum((np.arange(n) + 1) * DT, t_in[-1])
    return np.stack([np.interp(t_out, t_in, knots[:, i]) for i in range(2)], 1)


def densify_path(pts: np.ndarray, step: float = 0.4) -> np.ndarray:
    """Polyline (M,2) -> dense points every `step` m along arc length."""
    pts = np.asarray(pts, np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if seg.sum() < 1e-6:
        return pts
    s = np.concatenate([[0.0], np.cumsum(seg)])
    ss = np.arange(0.0, s[-1] + step, step)
    return np.stack([np.interp(ss, s, pts[:, i]) for i in range(2)], 1)


def native_desired_speed(plan6: np.ndarray) -> float:
    """Replicates PIDController.control_pid's desired_speed formula (pure
    function of the waypoints; lets the agent obtain the native v_ref without
    stepping the stateful native PID twice per tick)."""
    wp = np.asarray(plan6, np.float64)
    num_pairs = len(wp) - 1
    return float(sum(np.linalg.norm(wp[i + 1] - wp[i]) * 2.0 / num_pairs
                     for i in range(num_pairs)))


def resample_wp6(traj12: np.ndarray) -> np.ndarray:
    """(12,2) rollout @0.25s -> (6,2) waypoints @0.5s for the unchanged PID."""
    return np.asarray(traj12, np.float64)[1::2].copy()


# ---------------------------------------------------------------------------
# Window (one planning problem in the plan frame)
# ---------------------------------------------------------------------------

@dataclass
class WindowData:
    v0: float
    native: np.ndarray               # (6,2) selected-branch cumsum plan
    route_dir: np.ndarray            # unit vector
    nat_gate: float
    v_ref: float
    nat_dense: np.ndarray            # (M,2) densified native path @0.4 m
    agents: List[dict]               # frozen dict schema: pos/yaw/ext/traj6/prob
    boundaries: List[np.ndarray]     # list of (P,2)
    s_nat: float = 0.0
    nat_len: float = 0.0             # native plan arc length (R3 prior support)
    nat_dense_ext: Optional[np.ndarray] = None  # R3: path + terminal-ray ext


def make_window(v0: float, native_plan: np.ndarray, route_target_xy: np.ndarray,
                agents: List[dict], boundaries: List[np.ndarray],
                v_ref_in: float) -> WindowData:
    """Frozen Window semantics (mvp_planner.Window + offline build_window).

    `v_ref_in` must already be max(native_desired_speed, 2.5) — the offline
    harness applied that floor at the call site, so the frozen run-4 results
    include it in BOTH gate branches; replicated exactly.
    `route_target_xy` is the raw target_xy_local; the >1 m degeneracy guard
    and normalization both replicate the frozen two-stage handling.
    """
    tgt = np.asarray(route_target_xy, np.float64)
    g_raw = tgt if np.linalg.norm(tgt) > 1.0 else np.array([0.0, 1.0])
    n = np.linalg.norm(g_raw)
    route_dir = g_raw / n if n > 1e-6 else np.array([0.0, 1.0])
    native = np.asarray(native_plan, np.float64)
    s_nat = float(native[-1] @ route_dir)
    nat_gate = float(1.0 / (1.0 + np.exp(-2.0 * (s_nat - 1.5))))
    if nat_gate > 0.5:
        v_ref = float(np.clip(v_ref_in, 0.5, 8.0))
    else:
        v_ref = float(np.clip(max(v_ref_in, 2.5), 2.5, 8.0))
    nat_dense = densify_path(np.vstack([[0.0, 0.0], native]), 0.4)
    nat_len = float(np.linalg.norm(
        np.diff(np.vstack([[0.0, 0.0], native]), axis=0), axis=1).sum())
    # R3 extended path: terminal-tangent ray beyond the native endpoint
    if nat_len >= R3_TANGENT_MIN and len(nat_dense) >= 2:
        u = nat_dense[-1] - nat_dense[-2]
        u = u / max(np.linalg.norm(u), 1e-9)
    else:
        u = route_dir
    ks = np.arange(1, int(R3_EXT_LEN / R3_EXT_STEP) + 1)
    ext = nat_dense[-1][None] + u[None] * (ks[:, None] * R3_EXT_STEP)
    nat_dense_ext = np.vstack([nat_dense, ext])
    return WindowData(v0=float(v0), native=native, route_dir=route_dir,
                      nat_gate=nat_gate, v_ref=v_ref, nat_dense=nat_dense,
                      agents=agents, boundaries=boundaries, s_nat=s_nat,
                      nat_len=nat_len, nat_dense_ext=nat_dense_ext)


def agents_from_arrays(ag_valid, ag_score, ag_box, ag_traj, ag_mode_prob,
                       ag_label, fix_r1: bool = False) -> List[dict]:
    """Frozen agent selection (offline build_window): order by distance,
    score >= 0.30, dist <= 30 m, top-1 mode, at most 8 agents.

    fix_r1 (A-v2): STRICTLY ADDITIVE second pass appending near, MOVING,
    low-score actors (score in [R1_SCORE_MIN, 0.30), dist <= R1_DIST_MAX,
    top-1 3-s displacement >= R1_MOVE_MIN), distance-ordered, at most
    R1_EXTRA_MAX. The frozen v1 set is never reordered or reduced.

    ag_box columns follow the trace schema: [cx, cy, w, l, h, yaw, cz]."""
    agents = []
    ab = np.asarray(ag_box, np.float64)
    order = np.argsort(np.hypot(ab[:, 0], ab[:, 1]))

    def build(j, cx, cy, top1, r1_added=False):
        traj6 = np.array([cx, cy]) + np.cumsum(
            np.asarray(ag_traj[j, top1], np.float64), axis=0)
        return dict(pos=np.array([cx, cy]),
                    yaw=float(ab[j, 5]),
                    ext=(max(float(ab[j, 3]) / 2, EXT_MIN),
                         max(float(ab[j, 2]) / 2, EXT_MIN)),
                    traj6=traj6,
                    prob=float(ag_mode_prob[j, top1]),
                    label=int(ag_label[j]),
                    r1_added=r1_added)

    for j in order:
        if not ag_valid[j] or float(ag_score[j]) < AGENT_SCORE_MIN:
            continue
        cx, cy = float(ab[j, 0]), float(ab[j, 1])
        if np.hypot(cx, cy) > AGENT_DIST_MAX:
            continue
        top1 = int(np.argmax(ag_mode_prob[j]))
        agents.append(build(j, cx, cy, top1))
        if len(agents) >= AGENT_MAX:
            break
    if fix_r1:
        extra = 0
        for j in order:
            if extra >= R1_EXTRA_MAX:
                break
            s = float(ag_score[j])
            if not ag_valid[j] or s >= AGENT_SCORE_MIN or s < R1_SCORE_MIN:
                continue
            cx, cy = float(ab[j, 0]), float(ab[j, 1])
            if np.hypot(cx, cy) > R1_DIST_MAX:
                continue
            top1 = int(np.argmax(ag_mode_prob[j]))
            disp = float(np.linalg.norm(
                np.asarray(ag_traj[j, top1], np.float64).sum(0)))
            if disp < R1_MOVE_MIN:
                continue
            agents.append(build(j, cx, cy, top1, r1_added=True))
            extra += 1
    return agents


def boundaries_from_map(map_pts, map_score, map_label) -> List[np.ndarray]:
    """Frozen boundary selection: Solid/SolidSolid polylines, score > 0.3."""
    lab = np.asarray(map_label, int)
    sco = np.asarray(map_score, np.float64)
    keep = (sco > BOUNDARY_SCORE_MIN) & np.isin(lab, BOUNDARY_LABELS)
    return [np.asarray(poly, np.float64)
            for poly in np.asarray(map_pts, np.float64)[keep]]


# ---------------------------------------------------------------------------
# Seeds (frozen) + warm seed
# ---------------------------------------------------------------------------

def _native_controls(win: WindowData, n: int):
    """Frozen native-seed fit — including quirk 1 (negated curvature)."""
    pts = np.vstack([[0, 0], win.native])
    seg = np.diff(pts, axis=0)
    v_prof = np.linalg.norm(seg, axis=1) / 0.5
    v_prof = np.concatenate([[win.v0], v_prof])
    a6 = np.diff(v_prof) / 0.5
    hd = np.arctan2(seg[:, 0], seg[:, 1])
    dh = np.diff(np.concatenate([[0.0], hd]))
    with np.errstate(divide="ignore", invalid="ignore"):
        k6 = np.where(v_prof[1:] > 0.3, dh / (v_prof[1:] * 0.5 + 1e-6), 0.0)
    a = np.zeros(n)
    k = np.zeros(n)
    a[: min(n, 12)] = np.repeat(a6, 2)[: min(n, 12)]
    k[: min(n, 12)] = np.repeat(k6, 2)[: min(n, 12)]
    return np.clip(a, A_MIN, A_MAX), np.clip(-k, -K_MAX, K_MAX)


def behavior_seeds(win: WindowData, n: int):
    """The 8 frozen behavior seeds, order and values verbatim."""
    a0, k0 = _native_controls(win, n)
    acc = float(np.clip((win.v_ref - win.v0) / 1.5, 0.3, A_MAX))
    S = [(a0, k0),
         (np.full(n, -3.0), k0),
         (np.full(n, acc), k0 * 0.3)]
    for s in (+1, -1):
        kk = np.concatenate([np.full(n // 3, s * 0.08),
                             np.full(n // 3, 0.0),
                             np.full(n - 2 * (n // 3), -s * 0.06)])
        S.append((np.full(n, max(acc * 0.5, 0.3)), kk))
        S.append((np.full(n, -1.5), kk))
    S.append((np.concatenate([np.full(n // 2, 0.8),
                              np.full(n - n // 2, 0.0)]), k0))
    a = np.stack([x[0] for x in S])
    k = np.stack([x[1] for x in S])
    return a, k


class WarmBuffer:
    """Persists the last accepted (a, kappa) across 20 Hz planning cycles and
    time-shifts it onto the current 0.25 s control grid (one grid step per
    5 sim ticks). Controls are frame-invariant, so no pose transform (and no
    GPS-noise injection) is involved."""

    def __init__(self, tick_dt: float = 0.05):
        self.tick_dt = float(tick_dt)
        self.reset()

    def reset(self):
        self.a: Optional[np.ndarray] = None
        self.k: Optional[np.ndarray] = None
        self.phase = 0.0

    def update(self, a: np.ndarray, k: np.ndarray):
        self.a = np.asarray(a, np.float64).copy()
        self.k = np.asarray(k, np.float64).copy()
        self.phase = 0.0

    def advance(self):
        """Advance one sim tick; shift the buffer when a full dt has passed."""
        if self.a is None:
            return
        self.phase += self.tick_dt
        while self.phase >= DT - 1e-9:
            self.phase -= DT
            self.a = np.concatenate([self.a[1:], [0.0]])       # taper: coast
            self.k = np.concatenate([self.k[1:], [self.k[-1]]])  # hold curvature

    def seed(self):
        if self.a is None:
            return None
        return (np.clip(self.a, A_MIN, A_MAX), np.clip(self.k, -K_MAX, K_MAX))


# ---------------------------------------------------------------------------
# Batched objective (identical math to the frozen prototype)
# ---------------------------------------------------------------------------

def rollout(a: torch.Tensor, k: torch.Tensor, v0: float):
    K_, n = a.shape
    x = torch.zeros(K_, dtype=a.dtype, device=a.device)
    y = torch.zeros(K_, dtype=a.dtype, device=a.device)
    psi = torch.zeros(K_, dtype=a.dtype, device=a.device)
    v = torch.full((K_,), float(v0), dtype=a.dtype, device=a.device)
    xs, ys, vs, ps = [], [], [], []
    for i in range(n):
        v = torch.clamp(v + a[:, i] * DT, min=0.0, max=V_MAX)
        psi = psi + v * k[:, i] * DT
        x = x + v * DT * torch.sin(psi)
        y = y + v * DT * torch.cos(psi)
        xs.append(x); ys.append(y); vs.append(v); ps.append(psi)
    return (torch.stack(xs, 1), torch.stack(ys, 1), torch.stack(vs, 1),
            torch.stack(ps, 1))


class _WindowTensors:
    """Per-window constants moved to the target device once per solve."""

    def __init__(self, win: WindowData, n: int, device, dtype,
                 fix_r2: bool = False):
        td = dict(device=device, dtype=dtype)
        self.g = torch.tensor(win.route_dir, **td)
        self.lat_axis = torch.tensor([win.route_dir[1], -win.route_dir[0]], **td)
        self.natf = torch.tensor(native_fine(win.native, n), **td)
        self.nat_dense = torch.tensor(win.nat_dense, **td)
        self.nat_dense_ext = (torch.tensor(win.nat_dense_ext, **td)
                              if win.nat_dense_ext is not None else None)
        self.nat_gate = win.nat_gate
        self.v_ref = win.v_ref
        self.nat_len = win.nat_len
        self.n_agents = len(win.agents)
        if win.agents:
            preds = np.stack([
                interp_pred(ag["traj6"], n, ag["pos"] if fix_r2 else None)
                for ag in win.agents])
            self.ag_pred = torch.tensor(preds, **td)              # (A,n,2)
            self.ag_cos = torch.tensor(
                [np.cos(ag["yaw"]) for ag in win.agents], **td)   # (A,)
            self.ag_sin = torch.tensor(
                [np.sin(ag["yaw"]) for ag in win.agents], **td)
            ext = np.array([ag["ext"] for ag in win.agents])      # (A,2)
            self.ax = torch.tensor(ext[:, 0] + R_EGO + 0.6, **td)
            self.bx = torch.tensor(ext[:, 1] + R_EGO + 0.4, **td)
            self.axh = torch.tensor(ext[:, 0] + R_EGO + 0.05, **td)
            self.bxh = torch.tensor(ext[:, 1] + R_EGO + 0.05, **td)
            self.wgt = torch.tensor(
                [float(ag.get("prob", 1.0)) for ag in win.agents], **td)
        if win.boundaries:
            P = np.concatenate([b.reshape(-1, 2) for b in win.boundaries])
            self.bnd = torch.tensor(P, **td)                      # (P,2)
        else:
            self.bnd = None


def objective(wt: _WindowTensors, win: WindowData, a: torch.Tensor,
              k: torch.Tensor, w: Dict[str, float] = WEIGHTS,
              fix_r3: bool = False, fix_geom: bool = False):
    """Batched over seeds AND agents; term-by-term identical to the frozen
    prototype's `objective` (verified by tests/test_candidate_a_planner.py
    against research/candidate_a/mvp_planner.py). fix_r3 (A-v2) limits the
    path prior to the native plan's own spatial support (see R3 constants).
    fix_geom (A-v3) evaluates the agent ellipse field at three ego disk
    centers (min over disks) instead of the single center disk."""
    X, Y, V, PSI = rollout(a, k, win.v0)
    dp = torch.sqrt((X - wt.natf[None, :, 0]) ** 2 +
                    (Y - wt.natf[None, :, 1]) ** 2 + 1e-9)
    j_pt = torch.where(dp < 1.0, 0.5 * dp ** 2, dp - 0.5).sum(1)
    # R3 (fix_r3): distance to the native path EXTENDED along its terminal
    # tangent — a true cross-track prior; advancing along the native plan's
    # natural continuation is free, so a short/stalled native plan carries
    # no along-track stop authority. v1: distance to the raw (short) path.
    dense = (wt.nat_dense_ext if (fix_r3 and wt.nat_dense_ext is not None)
             else wt.nat_dense)
    d2 = (X[:, :, None] - dense[None, None, :, 0]) ** 2 + \
         (Y[:, :, None] - dense[None, None, :, 1]) ** 2
    j_ct = torch.sqrt(d2.min(2).values + 1e-9).sum(1) * wt.nat_gate
    s = X * wt.g[0] + Y * wt.g[1]
    j_prog = -s[:, -1]
    j_v = ((V - wt.v_ref).clamp(min=0) ** 2).sum(1)
    if wt.n_agents:
        if fix_geom:
            # A-v3: three ego disk centers along heading; min over disks
            offs = torch.tensor([-EGO_DISC_OFF, 0.0, EGO_DISC_OFF],
                                dtype=X.dtype, device=X.device)
            cx = X[:, None, :] + offs[None, :, None] * torch.sin(PSI)[:, None, :]
            cy = Y[:, None, :] + offs[None, :, None] * torch.cos(PSI)[:, None, :]
            dx = cx[:, :, None, :] - wt.ag_pred[None, None, :, :, 0]  # (K,D,A,n)
            dy = cy[:, :, None, :] - wt.ag_pred[None, None, :, :, 1]
            du = dx * wt.ag_sin[None, None, :, None] + \
                dy * wt.ag_cos[None, None, :, None]
            dv = dx * wt.ag_cos[None, None, :, None] - \
                dy * wt.ag_sin[None, None, :, None]
            dn = torch.sqrt((du / wt.ax[None, None, :, None]) ** 2 +
                            (dv / wt.bx[None, None, :, None]) ** 2 + 1e-9
                            ).amin(dim=1)                          # (K,A,n)
            dnh = torch.sqrt((du / wt.axh[None, None, :, None]) ** 2 +
                             (dv / wt.bxh[None, None, :, None]) ** 2 + 1e-9
                             ).amin(dim=1)
        else:
            dx = X[:, None, :] - wt.ag_pred[None, :, :, 0]        # (K,A,n)
            dy = Y[:, None, :] - wt.ag_pred[None, :, :, 1]
            du = dx * wt.ag_sin[None, :, None] + dy * wt.ag_cos[None, :, None]
            dv = dx * wt.ag_cos[None, :, None] - dy * wt.ag_sin[None, :, None]
            dn = torch.sqrt((du / wt.ax[None, :, None]) ** 2 +
                            (dv / wt.bx[None, :, None]) ** 2 + 1e-9)
            dnh = torch.sqrt((du / wt.axh[None, :, None]) ** 2 +
                             (dv / wt.bxh[None, :, None]) ** 2 + 1e-9)
        risk = (wt.wgt[None, :] * torch.nn.functional.softplus(
            (1.0 - dn) * 4.0).sum(2) / 4.0).sum(1)
        bar = (wt.wgt[None, :] * torch.nn.functional.softplus(
            (1.0 - dnh) * 12.0).sum(2) / 12.0).sum(1)
    else:
        risk = torch.zeros_like(j_pt)
        bar = torch.zeros_like(j_pt)
    if wt.bnd is not None:
        d = torch.sqrt((X[:, :, None] - wt.bnd[None, None, :, 0]) ** 2 +
                       (Y[:, :, None] - wt.bnd[None, None, :, 1]) ** 2 + 1e-9)
        j_bnd = torch.nn.functional.softplus(
            (0.6 - d.min(2).values) * 8.0).sum(1) / 8.0
    else:
        j_bnd = torch.zeros_like(j_pt)
    lat = X * wt.lat_axis[0] + Y * wt.lat_axis[1]
    j_cor = ((lat.abs() - 4.0).clamp(min=0) ** 2).sum(1)
    j_sm = ((a[:, 1:] - a[:, :-1]) ** 2).sum(1) + \
           ((k[:, 1:] - k[:, :-1]) ** 2).sum(1) * 30 + (k ** 2).sum(1) * 3
    comps = dict(pt=j_pt, ct=j_ct, prog=j_prog, v=j_v, risk=risk, bar=bar,
                 bnd=j_bnd, cor=j_cor, sm=j_sm)
    # E39 (Phase 4B-2A instrumentation): `weighted` is built from the exact
    # same per-term tensors J itself sums below -- not a parallel/duplicate
    # computation. This keeps J and the returned decomposition numerically
    # identical by CONSTRUCTION (J = sum(weighted.values())), not merely by
    # a tolerance check. No existing return value, key, or math changed.
    weighted = {key: w[key] * comps[key] for key in comps}
    J = sum(weighted.values())
    return J, dict(X=X, Y=Y, V=V, PSI=PSI, s=s, comps=comps, weighted=weighted)


# ---------------------------------------------------------------------------
# A-v3 exact feasibility geometry (no gradients — selection-time check only)
# ---------------------------------------------------------------------------

def _pt_box_dist(px, py, cx, cy, ux, uy, hl, hw):
    """Distance from points (px,py) to an oriented box (center c, unit axis
    u = long direction, half dims hl/hw). Vectorized numpy; 0 inside."""
    dx = px - cx
    dy = py - cy
    du = np.abs(dx * ux + dy * uy)
    dv = np.abs(-dx * uy + dy * ux)
    return np.hypot(np.maximum(du - hl, 0.0), np.maximum(dv - hw, 0.0))


def _obb_gap(c1, u1, h1, c2, u2, h2):
    """Exact 2D OBB-OBB clearance: negative iff overlapping (SAT), else the
    min corner-to-box distance both ways (exact for separated 2D boxes)."""
    corners = []
    for c, u, h in ((c1, u1, h1), (c2, u2, h2)):
        n = np.array([-u[1], u[0]])
        corners.append(np.array([c + s1 * h[0] * u + s2 * h[1] * n
                                 for s1 in (-1, 1) for s2 in (-1, 1)]))
    # SAT overlap test on the 4 axes; if all overlap, return the negative
    # minimum-translation depth so deeper penetrations rank as worse
    min_ovl = np.inf
    separated = False
    for u in (u1, np.array([-u1[1], u1[0]]), u2, np.array([-u2[1], u2[0]])):
        p1 = corners[0] @ u
        p2 = corners[1] @ u
        ovl = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
        if ovl < 0:
            separated = True
            break
        min_ovl = min(min_ovl, ovl)
    if not separated:
        return float(-min_ovl)
    d1 = _pt_box_dist(corners[0][:, 0], corners[0][:, 1], c2[0], c2[1],
                      u2[0], u2[1], h2[0], h2[1]).min()
    d2 = _pt_box_dist(corners[1][:, 0], corners[1][:, 1], c1[0], c1[1],
                      u1[0], u1[1], h1[0], h1[1]).min()
    return float(min(d1, d2))


def feasibility_gap(traj12: np.ndarray, psi12: np.ndarray,
                    agents: List[dict], n: int = N_STEPS) -> float:
    """A-v3 hard-constraint quantity: min over steps and agents of the exact
    clearance between the ego oriented box (EGO_HALF_LEN x EGO_HALF_W,
    heading psi_k) and each agent's predicted oriented box (R2-anchored
    top-1 interpolation). Negative = predicted physical contact."""
    if not agents:
        return 99.0
    gmin = 99.0
    t_in = np.arange(7) * 0.5
    t_out = (np.arange(n) + 1) * DT
    eh = np.array([EGO_HALF_LEN, EGO_HALF_W])
    for ag in agents:
        knots = np.vstack([ag["pos"], ag["traj6"]])
        pred = np.stack([np.interp(np.minimum(t_out, 3.0), t_in, knots[:, i])
                         for i in range(2)], 1)
        au = np.array([np.sin(ag["yaw"]), np.cos(ag["yaw"])])
        ah = np.array([ag["ext"][0], ag["ext"][1]])
        for k in range(n):
            eu = np.array([np.sin(psi12[k]), np.cos(psi12[k])])
            g = _obb_gap(traj12[k], eu, eh, pred[k], au, ah)
            if g < gmin:
                gmin = g
    return gmin


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass
class PlannerConfig:
    device: str = "cpu"
    dtype: str = "float64"           # frozen prototype precision
    iters_cold: int = 70             # frozen offline budget
    iters_warm: int = 25             # closed-loop warm budget (validated offline)
    lr: float = 0.06                 # frozen
    runtime_budget_ms: float = 1000.0  # generous hard cap; exceeding it is a
    # per-cycle fallback reason (load-dependent by nature; logged either way)
    weights: Dict[str, float] = field(default_factory=lambda: dict(WEIGHTS))
    # A-v2 targeted repairs (Phase 4B-2A Revision 1); all False == exact v1
    fix_r1: bool = False             # additive low-score moving-actor inclusion
    fix_r2: bool = False             # interp_pred t=0 knot = actor position
    fix_r3: bool = False             # path prior limited to native arc support
    # A-v3 structural additions (E22); v3 == r1,r2,r3,geom,feas
    fix_geom: bool = False           # 3-disk ego geometry in the risk field
    fix_feas: bool = False           # hard safety-feasibility selection layer

    def torch_dtype(self):
        return torch.float64 if self.dtype == "float64" else torch.float32

    def fixes_str(self) -> str:
        return ",".join(n for n, f in (("r1", self.fix_r1), ("r2", self.fix_r2),
                                       ("r3", self.fix_r3),
                                       ("geom", self.fix_geom),
                                       ("feas", self.fix_feas)) if f) or "none"


def apply_fixes(cfg: PlannerConfig, spec: str) -> PlannerConfig:
    """Parse a comma list like 'r1,r2,r3,geom,feas' ('none'/'') into flags."""
    parts = {p.strip() for p in spec.split(",") if p.strip() and
             p.strip() != "none"}
    unknown = parts - {"r1", "r2", "r3", "geom", "feas"}
    assert not unknown, f"unknown fix flags: {unknown}"
    cfg.fix_r1 = "r1" in parts
    cfg.fix_r2 = "r2" in parts
    cfg.fix_r3 = "r3" in parts
    cfg.fix_geom = "geom" in parts
    cfg.fix_feas = "feas" in parts
    return cfg


@dataclass
class PlanResult:
    ok: bool
    reason: str                      # "ok" or failure reason
    traj12: Optional[np.ndarray]     # (12,2) plan-frame rollout @0.25 s
    wp6: Optional[np.ndarray]        # (6,2) @0.5 s for the unchanged PID
    a: Optional[np.ndarray]
    k: Optional[np.ndarray]
    v: Optional[np.ndarray]
    J: float
    components: Dict[str, float]
    seed_index: int
    seed_name: str
    s_N: float
    gate: float
    v_ref: float
    iters: int
    n_seeds: int
    warm_seeded: bool
    solve_ms: float
    sat_a: float = 0.0
    sat_k: float = 0.0
    # A-v3 feasibility layer (populated when cfg.fix_feas)
    min_gap: Optional[float] = None      # winner's predicted min box clearance
    feasible: Optional[bool] = None
    n_feasible: Optional[int] = None
    sel_mode: str = "unconstrained"      # feasible_best | all_infeasible_safest

    # ---------------------------------------------------------------------
    # E39 (Phase 4B-2A instrumentation, docs/final_dev_stage_master_plan.md
    # Part 5-6) -- ADDITIVE ONLY. Every field below is a read-out of a
    # quantity the solver already computes; none changes selection, the
    # objective, the optimizer, seeds, constraints, or fallback logic.
    # All fields default so no existing keyword-only construction site
    # needs to change; early-return failure branches populate what they
    # already have (native-nominal + candidate matrix are computed BEFORE
    # any failure check, so they are present on every branch that reaches
    # objective() at least once).
    # ---------------------------------------------------------------------
    weighted: Optional[Dict[str, float]] = None  # winner's w_i*c_i (exact:
    # sum(weighted.values()) == J by construction, see objective())

    # Native VAD's own unrefined rollout (seed index 0, BEFORE any Adam
    # step), evaluated once under the SAME objective() -- "how would A-v3's
    # own cost function score the Native plan as-is".
    native_nominal_J: Optional[float] = None
    native_nominal_components: Optional[Dict[str, float]] = None
    native_nominal_weighted: Optional[Dict[str, float]] = None

    # Full cross-candidate matrix at the FINAL (post-optimization,
    # pre-selection) evaluation -- the same tensors selection itself reads,
    # just also kept instead of discarded.
    all_candidates_seed_names: Optional[List[str]] = None
    all_candidates_J: Optional[List[float]] = None
    all_candidates_components: Optional[List[Dict[str, float]]] = None
    all_candidates_gap: Optional[List[Optional[float]]] = None
    all_candidates_feasible: Optional[List[Optional[bool]]] = None
    runnerup_index: Optional[int] = None
    runnerup_J: Optional[float] = None
    margin_to_runnerup: Optional[float] = None  # J(runner-up) - J(winner)

    # Timing breakdown (ms); solve_ms (existing field) is the whole solve().
    # optimizer_ms/obj_eval_ms/grad_ms/seed_ms/feas_ms/native_nominal_ms are
    # sub-intervals of solve_ms (they overlap/nest, not a disjoint partition
    # -- see docs/e39_instrumentation_schema.md for the exact nesting).
    seed_ms: float = 0.0
    native_nominal_ms: float = 0.0
    optimizer_ms: float = 0.0
    obj_eval_ms: float = 0.0
    obj_n_calls: int = 0
    grad_ms: float = 0.0
    feas_ms: float = 0.0


def _feasible(a: np.ndarray, k: np.ndarray, tol: float = 1e-6) -> bool:
    return bool(np.all(np.isfinite(a)) and np.all(np.isfinite(k)) and
                a.max() <= A_MAX + tol and a.min() >= A_MIN - tol and
                np.abs(k).max() <= K_MAX + tol)


def solve(win: WindowData, cfg: PlannerConfig,
          warm: Optional[tuple] = None,
          n_steps: int = N_STEPS,
          freeze_lateral: bool = False,
          iters: Optional[int] = None) -> PlanResult:
    """One planning cycle. `warm` is an optional (a, k) numpy pair appended as
    a 9th seed; iteration budget defaults to iters_warm when warm is given,
    iters_cold otherwise. freeze_lateral reproduces the offline Q5 probe.

    E39 (Phase 4B-2A instrumentation, docs/e39_instrumentation_schema.md):
    every `time.perf_counter()` call added below is a pure read of the OS
    monotonic clock bracketing an EXISTING computation -- no computation is
    added, reordered, skipped, batched differently, or changed as a result
    of instrumentation being present. `cfg.device` is "cpu" under this
    project's frozen runtime config (NEWBORN_CANDA_DEVICE=cpu), so these
    timers need no CUDA synchronization to be accurate; see the schema doc
    for the (documented, not implemented) GPU-sync caveat if that ever
    changes. The ONE genuinely new piece of work is the native-nominal
    evaluation below -- an extra call to the SAME `objective()` function,
    never a reimplementation -- timed separately (`native_nominal_ms`) and
    reported, not hidden, in the overhead check."""
    t0 = time.perf_counter()
    dtype = cfg.torch_dtype()
    device = torch.device(cfg.device)

    t_seed0 = time.perf_counter()
    a_np, k_np = behavior_seeds(win, n_steps)
    warm_seeded = warm is not None
    if warm_seeded:
        a_np = np.concatenate([a_np, warm[0][None]], 0)
        k_np = np.concatenate([k_np, warm[1][None]], 0)
    n_iters = iters if iters is not None else (
        cfg.iters_warm if warm_seeded else cfg.iters_cold)
    seed_ms = (time.perf_counter() - t_seed0) * 1000.0

    a = torch.tensor(a_np, dtype=dtype, device=device)
    k = torch.tensor(k_np, dtype=dtype, device=device)
    wt = _WindowTensors(win, n_steps, device, dtype, fix_r2=cfg.fix_r2)

    # E39: native-nominal decomposition. Row 0 of a/k (BEFORE any Adam step,
    # BEFORE the .clone().requires_grad_(True) below) is always the
    # native-fitted seed (behavior_seeds()'s own S[0]), independent of
    # warm_seeded. Answers "how does A-v3's own cost function score
    # Native's plan as-is" without touching selection or the optimizer.
    t_nat0 = time.perf_counter()
    with torch.no_grad():
        ac0 = a.clamp(A_MIN, A_MAX)
        kc0 = k.clamp(-K_MAX, K_MAX)
        J0, aux0 = objective(wt, win, ac0, kc0, cfg.weights,
                             fix_r3=cfg.fix_r3, fix_geom=cfg.fix_geom)
    native_nominal_ms = (time.perf_counter() - t_nat0) * 1000.0
    native_nominal_J = float(J0[0])
    native_nominal_components = {key: float(val[0])
                                 for key, val in aux0["comps"].items()}
    native_nominal_weighted = {key: float(val[0])
                               for key, val in aux0["weighted"].items()}

    if freeze_lateral:
        a = a.clone().requires_grad_(True)
        params = [a]
        k = k.clone()
    else:
        a = a.clone().requires_grad_(True)
        k = k.clone().requires_grad_(True)
        params = [a, k]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    obj_eval_ms = 0.0
    obj_n_calls = 0
    grad_ms = 0.0
    # the calling agent runs the VAD forward under torch.no_grad();
    # the solver must re-enable grad locally regardless of caller context
    t_opt0 = time.perf_counter()
    with torch.enable_grad():
        for _ in range(n_iters):
            opt.zero_grad()
            ac = a.clamp(A_MIN, A_MAX)
            kc = k.clamp(-K_MAX, K_MAX) if not freeze_lateral else k
            t_obj = time.perf_counter()
            J, _ = objective(wt, win, ac, kc, cfg.weights, fix_r3=cfg.fix_r3,
                             fix_geom=cfg.fix_geom)
            obj_eval_ms += (time.perf_counter() - t_obj) * 1000.0
            obj_n_calls += 1
            t_grad = time.perf_counter()
            J.sum().backward()
            opt.step()
            grad_ms += (time.perf_counter() - t_grad) * 1000.0
    optimizer_ms = (time.perf_counter() - t_opt0) * 1000.0
    with torch.no_grad():
        ac = a.clamp(A_MIN, A_MAX)
        kc = k.clamp(-K_MAX, K_MAX) if not freeze_lateral else k
        if warm_seeded:
            # receding-horizon monotonicity: the RAW (unrefined) warm seed is
            # evaluated alongside the refined seeds, so a warm cycle can never
            # end worse than the time-shifted previous solution (guards
            # against Adam wander under the reduced warm iteration budget;
            # solver machinery only — the frozen objective is unchanged)
            raw_a = torch.tensor(a_np[-1:], dtype=dtype, device=device)
            raw_k = torch.tensor(k_np[-1:], dtype=dtype, device=device)
            ac = torch.cat([ac, raw_a.clamp(A_MIN, A_MAX)], 0)
            kc = torch.cat([kc, raw_k if freeze_lateral
                            else raw_k.clamp(-K_MAX, K_MAX)], 0)
        t_obj = time.perf_counter()
        J, aux = objective(wt, win, ac, kc, cfg.weights, fix_r3=cfg.fix_r3,
                           fix_geom=cfg.fix_geom)
        obj_eval_ms += (time.perf_counter() - t_obj) * 1000.0
        obj_n_calls += 1
        solve_ms = (time.perf_counter() - t0) * 1000.0

        # E39: full cross-candidate matrix -- reads of the SAME `aux`/`J`
        # tensors selection itself uses below, kept instead of discarded.
        n_cand = len(J)
        all_seed_names = [SEED_NAMES[j] if j < 8 else
                          ("warm" if j == 8 else "warm_raw")
                          for j in range(n_cand)]
        all_candidates_J = [float(v) for v in J.cpu().tolist()]
        all_candidates_components = [
            {key: float(val[j]) for key, val in aux["comps"].items()}
            for j in range(n_cand)]

        if not torch.isfinite(J).any():
            return PlanResult(ok=False, reason="nonfinite_objective",
                              traj12=None, wp6=None, a=None, k=None, v=None,
                              J=float("nan"), components={}, seed_index=-1,
                              seed_name="none", s_N=0.0, gate=win.nat_gate,
                              v_ref=win.v_ref, iters=n_iters,
                              n_seeds=len(a_np), warm_seeded=warm_seeded,
                              solve_ms=solve_ms,
                              native_nominal_J=native_nominal_J,
                              native_nominal_components=native_nominal_components,
                              native_nominal_weighted=native_nominal_weighted,
                              all_candidates_seed_names=all_seed_names,
                              all_candidates_J=all_candidates_J,
                              all_candidates_components=all_candidates_components,
                              seed_ms=seed_ms, native_nominal_ms=native_nominal_ms,
                              optimizer_ms=optimizer_ms, obj_eval_ms=obj_eval_ms,
                              obj_n_calls=obj_n_calls, grad_ms=grad_ms)
        # non-finite seeds must never win the argmin
        Jsel = torch.where(torch.isfinite(J), J,
                           torch.full_like(J, float("inf")))
        gaps = None
        sel_mode = "unconstrained"
        t_feas0 = time.perf_counter()
        if cfg.fix_feas:
            # A-v3 hard safety-feasibility layer: an infeasible candidate
            # (predicted physical box contact within FEAS_EPS) can never win
            # on J; if ALL candidates are infeasible, take the safest
            # (max min-clearance) — progress cannot buy predicted contact.
            Xn = aux["X"].cpu().numpy()
            Yn = aux["Y"].cpu().numpy()
            Pn = aux["PSI"].cpu().numpy()
            gaps = np.array([feasibility_gap(
                np.stack([Xn[j], Yn[j]], 1), Pn[j], win.agents)
                for j in range(len(Jsel))])
            finite = torch.isfinite(J).cpu().numpy()
            feas_mask = (gaps >= FEAS_EPS) & finite
            if feas_mask.any():
                Jmask = np.where(feas_mask, Jsel.cpu().numpy(), np.inf)
                i = int(np.argmin(Jmask))
                sel_mode = "feasible_best"
            else:
                i = int(np.argmax(np.where(finite, gaps, -np.inf)))
                sel_mode = "all_infeasible_safest"
        else:
            i = int(Jsel.argmin())
        feas_ms = (time.perf_counter() - t_feas0) * 1000.0
        a_out = ac[i].detach().cpu().numpy()
        k_out = kc[i].detach().cpu().numpy()
        traj = np.stack([aux["X"][i].cpu().numpy(),
                         aux["Y"][i].cpu().numpy()], 1)
        v_out = aux["V"][i].cpu().numpy()
        comps = {key: float(val[i]) for key, val in aux["comps"].items()}
        weighted = {key: float(val[i]) for key, val in aux["weighted"].items()}
        s_N = float(aux["s"][i, -1])

        # E39: runner-up margin -- exact decomposition of the decision
        # (docs/final_dev_stage_master_plan.md Part 5.2 #2), among FINITE
        # candidates only, matching Jsel's own masking.
        Jsel_np = Jsel.cpu().numpy()
        order = np.argsort(Jsel_np)
        runnerup_index = (int(order[1]) if len(order) > 1 and
                          np.isfinite(Jsel_np[order[1]]) else None)
        runnerup_J = (float(Jsel_np[runnerup_index])
                      if runnerup_index is not None else None)
        margin_to_runnerup = (runnerup_J - float(Jsel_np[i])
                              if runnerup_J is not None else None)

        all_candidates_gap = ([float(g) for g in gaps.tolist()]
                              if gaps is not None else [None] * n_cand)
        all_candidates_feasible = ([bool(g >= FEAS_EPS) for g in gaps.tolist()]
                                   if gaps is not None else [None] * n_cand)

    diag_common = dict(
        native_nominal_J=native_nominal_J,
        native_nominal_components=native_nominal_components,
        native_nominal_weighted=native_nominal_weighted,
        all_candidates_seed_names=all_seed_names,
        all_candidates_J=all_candidates_J,
        all_candidates_components=all_candidates_components,
        all_candidates_gap=all_candidates_gap,
        all_candidates_feasible=all_candidates_feasible,
        runnerup_index=runnerup_index, runnerup_J=runnerup_J,
        margin_to_runnerup=margin_to_runnerup,
        seed_ms=seed_ms, native_nominal_ms=native_nominal_ms,
        optimizer_ms=optimizer_ms, obj_eval_ms=obj_eval_ms,
        obj_n_calls=obj_n_calls, grad_ms=grad_ms, feas_ms=feas_ms,
    )

    if not np.all(np.isfinite(traj)):
        return PlanResult(ok=False, reason="nonfinite_trajectory",
                          traj12=None, wp6=None, a=None, k=None, v=None,
                          J=float(J[i]), components=comps, weighted=weighted,
                          seed_index=i,
                          seed_name=SEED_NAMES[min(i, len(SEED_NAMES) - 1)],
                          s_N=s_N, gate=win.nat_gate, v_ref=win.v_ref,
                          iters=n_iters, n_seeds=len(a_np),
                          warm_seeded=warm_seeded, solve_ms=solve_ms,
                          **diag_common)
    if not _feasible(a_out, k_out):
        return PlanResult(ok=False, reason="hard_bound_violation",
                          traj12=None, wp6=None, a=None, k=None, v=None,
                          J=float(J[i]), components=comps, weighted=weighted,
                          seed_index=i,
                          seed_name=SEED_NAMES[min(i, len(SEED_NAMES) - 1)],
                          s_N=s_N, gate=win.nat_gate, v_ref=win.v_ref,
                          iters=n_iters, n_seeds=len(a_np),
                          warm_seeded=warm_seeded, solve_ms=solve_ms,
                          **diag_common)

    seed_name = SEED_NAMES[i] if i < 8 else ("warm" if i == 8 else "warm_raw")
    return PlanResult(ok=True, reason="ok", traj12=traj,
                      wp6=resample_wp6(traj), a=a_out, k=k_out, v=v_out,
                      J=float(J[i]), components=comps, weighted=weighted,
                      seed_index=i,
                      seed_name=seed_name, s_N=s_N, gate=win.nat_gate,
                      v_ref=win.v_ref, iters=n_iters, n_seeds=len(a_np),
                      warm_seeded=warm_seeded, solve_ms=solve_ms,
                      sat_a=float((np.abs(a_out) > 0.98 * abs(A_MIN)).mean()),
                      sat_k=float((np.abs(k_out) > 0.98 * K_MAX).mean()),
                      min_gap=(float(gaps[i]) if gaps is not None else None),
                      feasible=(bool(gaps[i] >= FEAS_EPS)
                                if gaps is not None else None),
                      n_feasible=(int((gaps >= FEAS_EPS).sum())
                                  if gaps is not None else None),
                      sel_mode=sel_mode,
                      **diag_common)


# ---------------------------------------------------------------------------
# E39 (Phase 4B-2A instrumentation): pure data-transformation from a
# PlanResult into fixed-shape, NaN-padded numpy fields ready for the
# per-tick npz trace. Deliberately CARLA/torch-free (numpy + stdlib only)
# so it is directly unit-testable without a CARLA/mmcv environment, and
# reused verbatim by candidate_a_agent.py (no hand-duplicated copy there).
# See docs/e39_instrumentation_schema.md for the full field dictionary.
# ---------------------------------------------------------------------------

MAX_TRACE_CANDIDATES = 10  # 8 behavior seeds + warm + warm_raw (max possible)


def plan_result_to_trace_fields(res: Optional["PlanResult"],
                                max_cand: int = MAX_TRACE_CANDIDATES):
    """Returns a dict of numpy-ready fields (winner weighted decomposition,
    native-nominal decomposition, fixed-shape candidate matrix, runner-up
    margin, solver latency breakdown) for one planning cycle's `res`.

    `res=None` (the optimizer was never invoked this tick -- NEWBORN_CANDA_
    MODE=native, or the planner raised before returning) yields the SAME
    key set, all NaN/-1/0, so every tick contributes to every trace-array
    key with a fixed shape -- this keeps `TraceWriter`'s per-key buffered
    lists tick-aligned (unlike the pre-existing `J_pt..J_sm` fields, which
    are only appended when `res is not None` and are therefore NOT
    tick-aligned; documented, pre-existing behavior, unchanged by E39 --
    see docs/e39_instrumentation_schema.md).

    Every field otherwise defaults to NaN/-1/None-safe when the underlying
    PlanResult field is None (e.g. early-return failure branches that never
    reached the feasibility layer) -- nothing is fabricated, matching this
    project's own "never invent a missing value" discipline
    (frozen_v9_regression_summary.py's docstring)."""
    term_order = list(WEIGHTS.keys())
    out = {}
    if res is None:
        for key in term_order:
            out[f"Jw_{key}"] = float("nan")
            out[f"Jnat_{key}"] = float("nan")
        out["native_nominal_J"] = float("nan")
        out["cand_J"] = np.full(max_cand, np.nan, np.float64)
        out["cand_gap"] = np.full(max_cand, np.nan, np.float64)
        out["cand_feasible"] = np.full(max_cand, -1, np.int8)
        out["cand_comps"] = np.full((max_cand, len(term_order)), np.nan, np.float64)
        out["n_candidates"] = 0
        out["runnerup_index"] = -1
        out["runnerup_J"] = float("nan")
        out["margin_to_runnerup"] = float("nan")
        out["seed_ms"] = 0.0
        out["native_nominal_ms"] = 0.0
        out["optimizer_ms"] = 0.0
        out["obj_eval_ms"] = 0.0
        out["obj_n_calls"] = 0
        out["grad_ms"] = 0.0
        out["feas_ms"] = 0.0
        return out
    for key in term_order:
        out[f"Jw_{key}"] = float((res.weighted or {}).get(key, float("nan")))
    for key in term_order:
        out[f"Jnat_{key}"] = float(
            (res.native_nominal_components or {}).get(key, float("nan")))
    out["native_nominal_J"] = float(
        res.native_nominal_J if res.native_nominal_J is not None else float("nan"))

    all_J = res.all_candidates_J or []
    n_cand = len(all_J)
    cand_J = np.full(max_cand, np.nan, np.float64)
    cand_gap = np.full(max_cand, np.nan, np.float64)
    cand_feasible = np.full(max_cand, -1, np.int8)  # -1 = n/a or padding
    cand_comps = np.full((max_cand, len(term_order)), np.nan, np.float64)
    for j in range(min(n_cand, max_cand)):
        cand_J[j] = all_J[j]
        g = (res.all_candidates_gap or [None] * n_cand)[j]
        cand_gap[j] = np.nan if g is None else g
        f = (res.all_candidates_feasible or [None] * n_cand)[j]
        cand_feasible[j] = -1 if f is None else np.int8(bool(f))
        comps_j = res.all_candidates_components[j]
        for ti, key in enumerate(term_order):
            cand_comps[j, ti] = comps_j[key]
    out["cand_J"] = cand_J
    out["cand_gap"] = cand_gap
    out["cand_feasible"] = cand_feasible
    out["cand_comps"] = cand_comps
    out["n_candidates"] = int(n_cand)
    out["runnerup_index"] = int(res.runnerup_index if res.runnerup_index is not None else -1)
    out["runnerup_J"] = float(res.runnerup_J if res.runnerup_J is not None else float("nan"))
    out["margin_to_runnerup"] = float(
        res.margin_to_runnerup if res.margin_to_runnerup is not None else float("nan"))

    out["seed_ms"] = float(res.seed_ms)
    out["native_nominal_ms"] = float(res.native_nominal_ms)
    out["optimizer_ms"] = float(res.optimizer_ms)
    out["obj_eval_ms"] = float(res.obj_eval_ms)
    out["obj_n_calls"] = int(res.obj_n_calls)
    out["grad_ms"] = float(res.grad_ms)
    out["feas_ms"] = float(res.feas_ms)
    return out
