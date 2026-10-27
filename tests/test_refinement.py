#!/usr/bin/env python
"""Unit tests for the CLRE receding-horizon refinement layer.

Copied from the frozen research suite (tests/test_candidate_a_planner.py
at commit b1f509d) with only the two import rewrites and the upstream-PID
skip guard recorded in docs/PROVENANCE.md.

No CARLA, no VAD model. Validates:
  1. formulation equivalence: batched objective == frozen prototype objective
     (clre/planning/reference_prototype.py) to float64 round-off;
  2. full-solve equivalence vs the frozen prototype on synthetic windows;
  3. commitment gate: stalled native prior does NOT trap the ego (progress),
     committing native prior keeps the plan near native (clean preservation);
  4. conflict windows: physical clearance kept, no longitudinal/lateral
     cancellation (full solve vs freeze-lateral solve);
  5. dynamic feasibility by construction (hard bounds, v >= 0);
  6. deterministic fallback signalling on non-finite inputs;
  7. warm-start buffer shift semantics;
  8. PID handoff: wp6 resampling + native_desired_speed replicates the real
     v0.0.4 PIDController formula on the real class.
"""

import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from clre.planning import refinement as cap  # noqa: E402  (RELEASE EDIT)
from clre.planning import reference_prototype as frozen  # noqa: E402  (RELEASE EDIT)

CFG64 = cap.PlannerConfig(device="cpu", dtype="float64")


# ---------------------------------------------------------------------------
# Synthetic windows
# ---------------------------------------------------------------------------

def straight_native(v=5.0):
    """Committing native plan: straight ahead at constant speed v."""
    return np.stack([np.zeros(6), v * 0.5 * np.arange(1, 7)], 1)


def stalled_native():
    """Degenerate stalled native plan (the no-commit failure signature)."""
    return np.full((6, 2), 0.05)


def agent_ahead(dist=8.0, speed=0.0, yaw=0.0, lateral=0.0):
    """Agent at (lateral, dist) moving at `speed` along +y, car extents."""
    deltas = np.tile([0.0, speed * 0.5], (6, 1))
    traj6 = np.array([lateral, dist]) + np.cumsum(deltas, 0)
    return dict(pos=np.array([lateral, dist]), yaw=yaw, ext=(2.4, 1.1),
                traj6=traj6, prob=0.9, label=0)


def make_pair(v0, native, agents, boundaries, v_ref_in, tgt=(0.0, 60.0)):
    """Same window in both representations (new planner / frozen prototype)."""
    win_new = cap.make_window(v0, native, np.array(tgt), agents, boundaries,
                              v_ref_in=v_ref_in)
    tgt_arr = np.asarray(tgt, float)
    route_dir = tgt_arr if np.linalg.norm(tgt_arr) > 1.0 else np.array([0., 1.])
    win_old = frozen.Window(v0, native, route_dir, agents, boundaries,
                            v_ref=v_ref_in)
    return win_new, win_old


def pred_min_gap(agents, traj):
    """Min oriented box gap vs top-1 predicted futures (frozen harness metric)."""
    if not agents:
        return None
    gmin = 99.0
    for ag in agents:
        pred = cap.interp_pred(ag["traj6"], len(traj))
        c, s = np.cos(ag["yaw"]), np.sin(ag["yaw"])
        d = pred - traj
        du = np.abs(d[:, 0] * s + d[:, 1] * c)
        dv = np.abs(d[:, 0] * c - d[:, 1] * s)
        out = np.hypot(np.maximum(du - ag["ext"][0], 0),
                       np.maximum(dv - ag["ext"][1], 0))
        inside = (du < ag["ext"][0]) & (dv < ag["ext"][1])
        g = np.where(inside, -0.01, out) - cap.R_EGO
        gmin = min(gmin, float(g.min()))
    return gmin


def true_min_gap(agents, traj):
    """Min oriented box gap vs CORRECTLY interpolated predicted futures
    (t=0 knot at the agent's current position, not the frozen harness's
    origin-knot quirk). Used as the physically meaningful evaluation metric;
    the planner's internal risk field keeps the frozen quirk."""
    if not agents:
        return None
    gmin = 99.0
    n = len(traj)
    t_in = np.arange(7) * 0.5
    t_out = (np.arange(n) + 1) * cap.DT
    for ag in agents:
        knots = np.vstack([ag["pos"], ag["traj6"]])
        pred = np.stack([np.interp(np.minimum(t_out, 3.0), t_in, knots[:, i])
                         for i in range(2)], 1)
        c, s = np.cos(ag["yaw"]), np.sin(ag["yaw"])
        d = pred - traj
        du = np.abs(d[:, 0] * s + d[:, 1] * c)
        dv = np.abs(d[:, 0] * c - d[:, 1] * s)
        out = np.hypot(np.maximum(du - ag["ext"][0], 0),
                       np.maximum(dv - ag["ext"][1], 0))
        inside = (du < ag["ext"][0]) & (dv < ag["ext"][1])
        g = np.where(inside, -0.01, out) - cap.R_EGO
        gmin = min(gmin, float(g.min()))
    return gmin


def boundary_wall(x=3.0):
    """Longitudinal Solid boundary polyline at lateral offset x."""
    return [np.stack([np.full(20, x), np.linspace(0, 30, 20)], 1)]


SCENARIOS = {
    "clean_commit": dict(v0=5.0, native=straight_native(5.0), agents=[],
                         boundaries=[], v_ref_in=5.0),
    "stall_open": dict(v0=0.0, native=stalled_native(), agents=[],
                       boundaries=[], v_ref_in=2.5),
    "conflict_stopped_car": dict(v0=5.0, native=straight_native(5.0),
                                 agents=[agent_ahead(9.0, 0.0)],
                                 boundaries=boundary_wall(3.0), v_ref_in=5.0),
    "conflict_crosser": dict(
        v0=4.0, native=straight_native(4.0),
        agents=[agent_ahead(12.0, 0.0, yaw=np.pi / 2, lateral=-6.0),
                agent_ahead(20.0, 4.0)],
        boundaries=boundary_wall(3.5), v_ref_in=4.0),
    "stall_with_wall": dict(v0=0.3, native=stalled_native(),
                            agents=[agent_ahead(25.0, 5.0)],
                            boundaries=boundary_wall(-3.0), v_ref_in=2.5),
}


# ---------------------------------------------------------------------------
# 1-2. Equivalence vs frozen prototype
# ---------------------------------------------------------------------------

def test_objective_equivalence():
    for name, sc in SCENARIOS.items():
        win_new, win_old = make_pair(**sc)
        a_np, k_np = cap.behavior_seeds(win_new, cap.N_STEPS)
        a = torch.tensor(a_np, dtype=torch.float64)
        k = torch.tensor(k_np, dtype=torch.float64)
        wt = cap._WindowTensors(win_new, cap.N_STEPS, torch.device("cpu"),
                                torch.float64)
        J_new, _ = cap.objective(wt, win_new, a, k)
        J_old, _ = frozen.objective(win_old, a, k)
        d = float((J_new - J_old).abs().max())
        assert d < 1e-9, f"{name}: objective mismatch {d}"
        # frozen seed generator must agree too
        a_f, k_f = frozen._seeds(win_old, cap.N_STEPS)
        assert np.abs(a_np - a_f.numpy()).max() < 1e-12
        assert np.abs(k_np - k_f.numpy()).max() < 1e-12
    print("objective_equivalence OK")


def test_solve_equivalence():
    for name, sc in SCENARIOS.items():
        win_new, win_old = make_pair(**sc)
        r_new = cap.solve(win_new, CFG64)
        r_old = frozen.solve(win_old)
        assert r_new.ok, name
        dJ = abs(r_new.J - r_old["J"]) / max(abs(r_old["J"]), 1.0)
        dtraj = float(np.abs(r_new.traj12 - r_old["traj"]).max())
        assert dJ < 1e-6, f"{name}: J mismatch {r_new.J} vs {r_old['J']}"
        assert dtraj < 1e-3, f"{name}: traj mismatch {dtraj}"
        assert r_new.seed_index == r_old["seed"], \
            f"{name}: seed {r_new.seed_index} vs {r_old['seed']}"
    print("solve_equivalence OK")


# ---------------------------------------------------------------------------
# 3. Commitment gate behavior
# ---------------------------------------------------------------------------

def test_stalled_native_recovers_progress():
    win, _ = make_pair(**SCENARIOS["stall_open"])
    assert win.nat_gate < 0.1, "stalled plan must not gate the path prior on"
    assert win.v_ref >= 2.5
    r = cap.solve(win, CFG64)
    assert r.ok
    assert r.s_N >= 3.0, f"no-commit window failed to progress: s_N={r.s_N}"
    print(f"stalled_native_recovers_progress OK (s_N={r.s_N:.2f} m, "
          f"seed={r.seed_name})")


def test_committing_native_is_preserved():
    win, _ = make_pair(**SCENARIOS["clean_commit"])
    assert win.nat_gate > 0.9
    r = cap.solve(win, CFG64)
    assert r.ok
    dense = cap.densify_path(np.vstack([[0, 0], win.native]), 0.2)
    cross = max(float(np.min(np.linalg.norm(dense - p[None], axis=1)))
                for p in r.traj12)
    assert cross <= 0.5, f"clean window deviated {cross:.2f} m cross-track"
    v_end = float(r.v[-1])
    assert abs(v_end - 5.0) <= 0.15 * 5.0, f"clean terminal speed {v_end}"
    print(f"committing_native_is_preserved OK (cross={cross:.3f} m, "
          f"v_end={v_end:.2f})")


# ---------------------------------------------------------------------------
# 4. Conflict: clearance kept; no lateral/longitudinal cancellation
# ---------------------------------------------------------------------------

def test_conflict_keeps_physical_clearance():
    for name in ("conflict_stopped_car", "conflict_crosser"):
        sc = SCENARIOS[name]
        win, _ = make_pair(**sc)
        r = cap.solve(win, CFG64)
        assert r.ok
        nat = cap.native_fine(win.native, cap.N_STEPS)
        # physically meaningful clearance (correct interpolation knots)
        tg = true_min_gap(sc["agents"], r.traj12)
        tg_nat = true_min_gap(sc["agents"], nat)
        assert tg > 0.0, f"{name}: TRUE physical contact in optimized plan ({tg})"
        assert tg >= tg_nat - 0.05, \
            f"{name}: true clearance lost vs native ({tg} < {tg_nat})"
        # frozen-harness metric: relative not-worse criterion (Q2 semantics)
        gap = pred_min_gap(sc["agents"], r.traj12)
        gap_nat = pred_min_gap(sc["agents"], nat)
        assert gap >= gap_nat - 0.05, \
            f"{name}: harness-metric clearance lost ({gap} < {gap_nat})"
        print(f"{name}: true_gap opt={tg:.2f} native={tg_nat:.2f} | "
              f"harness gap opt={gap:.2f} native={gap_nat:.2f} | "
              f"seed={r.seed_name} vN={r.v[-1]:.2f}")
    print("conflict_keeps_physical_clearance OK")


def test_no_lateral_longitudinal_cancellation():
    for name in ("conflict_stopped_car", "conflict_crosser"):
        sc = SCENARIOS[name]
        win, _ = make_pair(**sc)
        r_full = cap.solve(win, CFG64)
        r_long = cap.solve(win, CFG64, freeze_lateral=True)
        g_full = pred_min_gap(sc["agents"], r_full.traj12)
        g_long = pred_min_gap(sc["agents"], r_long.traj12)
        assert g_full >= g_long - 0.05, \
            f"{name}: full solve worse than longitudinal-only " \
            f"({g_full} vs {g_long}) — cancellation pathology"
    print("no_lateral_longitudinal_cancellation OK")


# ---------------------------------------------------------------------------
# 5. Feasibility
# ---------------------------------------------------------------------------

def test_feasibility_bounds():
    for name, sc in SCENARIOS.items():
        win, _ = make_pair(**sc)
        for warm in (None, cap.behavior_seeds(win, cap.N_STEPS)):
            w = None if warm is None else (warm[0][2], warm[1][2])
            r = cap.solve(win, CFG64, warm=w)
            assert r.ok
            assert r.a.min() >= cap.A_MIN - 1e-9
            assert r.a.max() <= cap.A_MAX + 1e-9
            assert np.abs(r.k).max() <= cap.K_MAX + 1e-9
            assert (r.v >= -1e-12).all(), f"{name}: negative speed"
            # first-step displacement consistent with v0 and a-bound
            d1 = float(np.linalg.norm(r.traj12[0]))
            assert d1 <= (win.v0 + cap.A_MAX * cap.DT) * cap.DT + 1e-6
    print("feasibility_bounds OK")


# ---------------------------------------------------------------------------
# 6. Fallback signalling
# ---------------------------------------------------------------------------

def test_fallback_on_nonfinite():
    # layer 1: NaN native plan -> window construction RAISES; the agent's
    # planner try/except catches this as planner_exception -> Native fallback
    bad_native = straight_native(5.0)
    bad_native[3] = np.nan
    raised = False
    try:
        cap.make_window(5.0, bad_native, np.array([0.0, 60.0]), [], [],
                        v_ref_in=5.0)
    except Exception:
        raised = True
    assert raised, "NaN native plan must raise in window construction"

    # layer 2: NaN inside an agent prediction -> window builds, the SOLVER
    # must detect the non-finite objective and signal fallback
    bad_agent = agent_ahead(9.0, 0.0)
    bad_agent["traj6"][2] = np.nan
    win = cap.make_window(5.0, straight_native(5.0), np.array([0.0, 60.0]),
                          [bad_agent], [], v_ref_in=5.0)
    r = cap.solve(win, CFG64)
    assert not r.ok
    assert r.reason in ("nonfinite_objective", "nonfinite_trajectory"), r.reason
    assert r.traj12 is None and r.wp6 is None
    print(f"fallback_on_nonfinite OK (solver reason={r.reason})")


# ---------------------------------------------------------------------------
# 7. Warm buffer
# ---------------------------------------------------------------------------

def test_warm_buffer_shift():
    wb = cap.WarmBuffer(tick_dt=0.05)
    assert wb.seed() is None
    a = np.arange(12, dtype=float) * 0.1
    k = np.arange(12, dtype=float) * 0.01
    wb.update(a, k)
    for _ in range(4):
        wb.advance()
    s = wb.seed()
    assert np.allclose(s[0], np.clip(a, cap.A_MIN, cap.A_MAX)), "early shift"
    wb.advance()  # 5th tick -> one full dt -> shift by one step
    s = wb.seed()
    assert np.allclose(s[0][:-1], a[1:]) and s[0][-1] == 0.0
    assert np.allclose(s[1][:-1], k[1:]) and s[1][-1] == k[-1]
    wb.reset()
    assert wb.seed() is None
    print("warm_buffer_shift OK")


def test_warm_seed_not_worse():
    """Receding-horizon monotonicity: a warm solve evaluates the RAW warm
    seed alongside the refined seeds, so on the same window it can never end
    above the cold optimum it was warmed from (up to float round-off)."""
    for name in ("stall_open", "conflict_stopped_car", "clean_commit"):
        win, _ = make_pair(**SCENARIOS[name])
        r_cold = cap.solve(win, CFG64)
        r_warm = cap.solve(win, CFG64, warm=(r_cold.a, r_cold.k))
        assert r_warm.ok
        assert r_warm.J <= r_cold.J + 1e-9, \
            f"{name}: warm J {r_warm.J} worse than cold {r_cold.J}"
    print("warm_seed_not_worse OK")


# ---------------------------------------------------------------------------
# 8. PID handoff
# ---------------------------------------------------------------------------

def test_wp6_resampling():
    traj = np.stack([np.arange(12, dtype=float),
                     np.arange(12, dtype=float) * 2], 1)
    wp6 = cap.resample_wp6(traj)
    assert wp6.shape == (6, 2)
    assert np.allclose(wp6, traj[[1, 3, 5, 7, 9, 11]])
    print("wp6_resampling OK")


def test_native_desired_speed_matches_real_pid():
    """RELEASE EDIT: requires the upstream Bench2DriveZoo controller, which
    this repository does not redistribute (see docs/LICENSE_AUDIT.md). Put
    your Bench2DriveZoo checkout on PYTHONPATH to run it; otherwise it is
    skipped rather than silently passing."""
    try:
        from Bench2DriveZoo.team_code.pid_controller import PIDController
    except ImportError:  # pragma: no cover - upstream not installed
        print("native_desired_speed_matches_real_pid SKIPPED "
              "(Bench2DriveZoo not on PYTHONPATH)")
        return
    rng = np.random.default_rng(0)
    for v0 in (0.0, 3.0, 8.0):
        plan = np.cumsum(rng.normal([0.0, v0 * 0.5], 0.3, (6, 2)), 0)
        pid = PIDController()
        _, _, _, meta = pid.control_pid(plan, np.float64(v0),
                                        np.array([0.0, 30.0]))
        ours = cap.native_desired_speed(plan)
        assert abs(ours - meta["desired_speed"]) < 1e-9, \
            f"{ours} vs {meta['desired_speed']}"
    print("native_desired_speed_matches_real_pid OK")


# ---------------------------------------------------------------------------
# 9. Informational runtime probe (full benchmark in the replay tool)
# ---------------------------------------------------------------------------

def test_runtime_probe():
    sc = SCENARIOS["conflict_crosser"]
    win, _ = make_pair(**sc)
    r = cap.solve(win, CFG64)  # jit/alloc warmup
    t0 = time.perf_counter()
    r_cold = cap.solve(win, CFG64)
    t_cold = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    r_warm = cap.solve(win, CFG64, warm=(r_cold.a, r_cold.k))
    t_warm = (time.perf_counter() - t0) * 1000
    print(f"runtime cpu/float64: cold({r_cold.iters} it)={t_cold:.0f} ms, "
          f"warm({r_warm.iters} it)={t_warm:.0f} ms")


if __name__ == "__main__":
    test_objective_equivalence()
    test_solve_equivalence()
    test_stalled_native_recovers_progress()
    test_committing_native_is_preserved()
    test_conflict_keeps_physical_clearance()
    test_no_lateral_longitudinal_cancellation()
    test_feasibility_bounds()
    test_fallback_on_nonfinite()
    test_warm_buffer_shift()
    test_warm_seed_not_worse()
    test_wp6_resampling()
    test_native_desired_speed_matches_real_pid()
    test_runtime_probe()
    print("\nALL CANDIDATE-A PLANNER TESTS PASSED")
