"""Candidate A MVP — unified sampling+gradient trajectory optimizer (research
prototype for OFFLINE falsification only; no closed-loop claim).

Formulation (docs/phase4b1_candidate_a_formulation.md):
  state  x_k = [x, y, psi, v]         (kinematic bicycle, plan frame)
  control u_k = [a, kappa]            (accel, curvature), clamped hard
  rollout x_{k+1} = f(x_k, u_k), dt = 0.25 s, default N = 12 (3.0 s horizon
          = VAD prediction horizon; W4 capability probes may use N = 24)
  v >= 0 (no reverse in MVP — interface limitation, documented)

  J = w_pt  * Huber(p - p_nat)                       (tiny point anchor)
    + w_ct  * gate(native progress) * crosstrack(p, native path)
    + w_prog* (-s_N)  + w_v * (v - v_ref)_+^2        (saturated progress)
    + w_risk* soft_clearance(top-1 pred agents, comfort margins)
    + w_bar * steep_clearance(top-1 pred agents, physical margins)
    + w_bnd * boundary_proximity(model-visible Solid/SolidSolid)
    + w_cor * corridor_overshoot(route-lateral beyond 4.0 m)
    + w_sm  * smoothness(da, dkappa, kappa)

Design rationale (anti-stop-forever without drive-into-hazard):
  * progress is LINEAR in terminal route progress but speed-saturated at
    v_ref (per-window desired speed), so "faster than sensible" is never
    rewarded — removes the progress/clearance trade observed in run 1;
  * the native anchor is a PATH prior gated by the native plan's own
    commitment: gate = sigmoid(2*(s_nat - 1.5 m)). A stalled native plan
    carries no stop-authority (otherwise imitation re-creates the
    no-commit failure); a moving native plan strongly shapes the path
    (cross-track), preserving semantic route choice on clean windows;
  * clearance barrier is steep and extent-aware, so progress cannot buy
    contact.

Solver: K parallel behavior seeds (nominal / brake / commit / swerve L,R /
brake+swerve L,R / creep-release) refined jointly by Adam, best-J returned.
Joint longitudinal+lateral objective and multi-basin seeding are the
explicit answers to the old sequential-DDP failures ("descent can't
brake", "A cancels B"): braking and swerving compete on ONE scalar J.

Agent clearance: extent-aware oriented ellipse on TOP-1 predicted mode
(Phase 4A: top-1 ~ best mode, median gap 0.07 m), agent-frame semi-axes
ext + ego disk R_EGO + margin.
"""

import numpy as np
import torch

DT = 0.25
N_DEFAULT = 12
R_EGO = 1.3          # ego disk radius (half-width 1.06 + margin)
A_MIN, A_MAX = -6.0, 2.5
K_MAX = 0.2          # |curvature| bound (R >= 5 m)

WEIGHTS = dict(pt=0.05, ct=1.2, prog=1.0, v=0.6, risk=1.6, bar=25.0,
               bnd=6.0, cor=0.5, sm=0.8)


def interp_pred(traj6, n):
    """(6,2) positions @0.5s -> (n,2) @0.25s (extrapolate last segment)."""
    knots = np.vstack([[0.0, 0.0], traj6])
    t_in = np.arange(7) * 0.5
    t_out = (np.arange(n) + 1) * DT
    ext = t_out > t_in[-1]
    out = np.stack([np.interp(t_out, t_in, knots[:, i]) for i in range(2)], 1)
    if ext.any():                       # hold last velocity
        vel = (knots[-1] - knots[-2]) / 0.5
        out[ext] = knots[-1] + vel[None] * (t_out[ext, None] - t_in[-1])
    return out


def native_fine(native, n):
    """Native 6x0.5s plan -> n x 0.25s points (hold endpoint beyond 3 s)."""
    knots = np.vstack([[0, 0], native])
    t_in = np.arange(7) * 0.5
    t_out = np.minimum((np.arange(n) + 1) * DT, t_in[-1])
    return np.stack([np.interp(t_out, t_in, knots[:, i]) for i in range(2)], 1)


def densify_path(pts, step=0.4):
    """Polyline (M,2) -> dense points every `step` m along arc length."""
    pts = np.asarray(pts, np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if seg.sum() < 1e-6:
        return pts
    s = np.concatenate([[0.0], np.cumsum(seg)])
    ss = np.arange(0.0, s[-1] + step, step)
    return np.stack([np.interp(ss, s, pts[:, i]) for i in range(2)], 1)


class Window:
    """One offline decision problem in the plan frame at a trace tick."""

    def __init__(self, v0, native_plan, route_dir, agents, boundaries,
                 v_ref=3.0):
        self.v0 = float(v0)
        self.native = np.asarray(native_plan, np.float64)        # (6,2)
        g = np.asarray(route_dir, np.float64)
        n = np.linalg.norm(g)
        self.route_dir = g / n if n > 1e-6 else np.array([0.0, 1.0])
        self.agents = agents      # list of dicts: pos,yaw,ext,traj6,prob
        self.boundaries = boundaries                              # (B,P,2)
        s_nat = float(self.native[-1] @ self.route_dir)
        self.nat_gate = float(1.0 / (1.0 + np.exp(-2.0 * (s_nat - 1.5))))
        # trust native speed intent when native commits; floor only on stall
        if self.nat_gate > 0.5:
            self.v_ref = float(np.clip(v_ref, 0.5, 8.0))
        else:
            self.v_ref = float(np.clip(max(v_ref, 2.5), 2.5, 8.0))
        self.nat_dense = densify_path(
            np.vstack([[0.0, 0.0], self.native]), 0.4)


def _native_controls(win, n):
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


def _seeds(win, n):
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
    a = torch.tensor(np.stack([x[0] for x in S]), dtype=torch.float64)
    k = torch.tensor(np.stack([x[1] for x in S]), dtype=torch.float64)
    return a, k


def rollout(a, k, v0):
    K_, n = a.shape
    x = torch.zeros(K_, dtype=a.dtype)
    y = torch.zeros(K_, dtype=a.dtype)
    psi = torch.zeros(K_, dtype=a.dtype)
    v = torch.full((K_,), float(v0), dtype=a.dtype)
    xs, ys, vs = [], [], []
    for i in range(n):
        v = torch.clamp(v + a[:, i] * DT, min=0.0, max=15.0)
        psi = psi + v * k[:, i] * DT
        x = x + v * DT * torch.sin(psi)
        y = y + v * DT * torch.cos(psi)
        xs.append(x); ys.append(y); vs.append(v)
    return torch.stack(xs, 1), torch.stack(ys, 1), torch.stack(vs, 1)


def _agent_terms(win, X, Y):
    n = X.shape[1]
    risk = torch.zeros(X.shape[0], dtype=X.dtype)
    bar = torch.zeros(X.shape[0], dtype=X.dtype)
    for ag in win.agents:
        pred = torch.tensor(interp_pred(ag["traj6"], n), dtype=X.dtype)
        c, s = np.cos(ag["yaw"]), np.sin(ag["yaw"])
        dx = X - pred[None, :, 0]
        dy = Y - pred[None, :, 1]
        du = dx * s + dy * c
        dv = dx * c - dy * s
        ax = ag["ext"][0] + R_EGO + 0.6
        bx = ag["ext"][1] + R_EGO + 0.4
        axh = ag["ext"][0] + R_EGO + 0.05
        bxh = ag["ext"][1] + R_EGO + 0.05
        dn = torch.sqrt((du / ax) ** 2 + (dv / bx) ** 2 + 1e-9)
        dnh = torch.sqrt((du / axh) ** 2 + (dv / bxh) ** 2 + 1e-9)
        wgt = float(ag.get("prob", 1.0))
        risk = risk + wgt * torch.nn.functional.softplus(
            (1.0 - dn) * 4.0).sum(1) / 4.0
        bar = bar + wgt * torch.nn.functional.softplus(
            (1.0 - dnh) * 12.0).sum(1) / 12.0
    return risk, bar


def _boundary_term(win, X, Y):
    if not win.boundaries:
        return torch.zeros(X.shape[0], dtype=X.dtype)
    P = torch.tensor(np.concatenate(
        [b.reshape(-1, 2) for b in win.boundaries]), dtype=X.dtype)
    d = torch.sqrt((X[:, :, None] - P[None, None, :, 0]) ** 2 +
                   (Y[:, :, None] - P[None, None, :, 1]) ** 2 + 1e-9)
    dmin = d.min(2).values
    return torch.nn.functional.softplus((0.6 - dmin) * 8.0).sum(1) / 8.0


def objective(win, a, k, w=WEIGHTS):
    X, Y, V = rollout(a, k, win.v0)
    n = X.shape[1]
    g = torch.tensor(win.route_dir, dtype=a.dtype)
    lat_axis = torch.tensor([win.route_dir[1], -win.route_dir[0]],
                            dtype=a.dtype)
    natf = torch.tensor(native_fine(win.native, n), dtype=a.dtype)
    dp = torch.sqrt((X - natf[None, :, 0]) ** 2 +
                    (Y - natf[None, :, 1]) ** 2 + 1e-9)
    j_pt = torch.where(dp < 1.0, 0.5 * dp ** 2, dp - 0.5).sum(1)
    nd = torch.tensor(win.nat_dense, dtype=a.dtype)
    d2 = (X[:, :, None] - nd[None, None, :, 0]) ** 2 + \
         (Y[:, :, None] - nd[None, None, :, 1]) ** 2
    j_ct = torch.sqrt(d2.min(2).values + 1e-9).sum(1) * win.nat_gate
    s = X * g[0] + Y * g[1]
    j_prog = -s[:, -1]
    j_v = ((V - win.v_ref).clamp(min=0) ** 2).sum(1)
    risk, bar = _agent_terms(win, X, Y)
    j_bnd = _boundary_term(win, X, Y)
    lat = X * lat_axis[0] + Y * lat_axis[1]
    j_cor = ((lat.abs() - 4.0).clamp(min=0) ** 2).sum(1)
    j_sm = ((a[:, 1:] - a[:, :-1]) ** 2).sum(1) + \
           ((k[:, 1:] - k[:, :-1]) ** 2).sum(1) * 30 + (k ** 2).sum(1) * 3
    J = (w["pt"] * j_pt + w["ct"] * j_ct + w["prog"] * j_prog +
         w["v"] * j_v + w["risk"] * risk + w["bar"] * bar +
         w["bnd"] * j_bnd + w["cor"] * j_cor + w["sm"] * j_sm)
    return J, dict(X=X, Y=Y, V=V, s=s)


def solve(win, n_steps=N_DEFAULT, iters=70, lr=0.06, freeze_lateral=False,
          w=WEIGHTS):
    a, k = _seeds(win, n_steps)
    if freeze_lateral:
        a = a.clone().requires_grad_(True)
        params = [a]
        k = k.clone()
    else:
        a = a.clone().requires_grad_(True)
        k = k.clone().requires_grad_(True)
        params = [a, k]
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        ac = a.clamp(A_MIN, A_MAX)
        kc = k.clamp(-K_MAX, K_MAX) if not freeze_lateral else k
        J, _ = objective(win, ac, kc, w)
        J.sum().backward()
        opt.step()
    with torch.no_grad():
        ac = a.clamp(A_MIN, A_MAX)
        kc = k.clamp(-K_MAX, K_MAX) if not freeze_lateral else k
        J, aux = objective(win, ac, kc, w)
        i = int(J.argmin())
        traj = np.stack([aux["X"][i].numpy(), aux["Y"][i].numpy()], 1)
        return dict(J=float(J[i]), traj=traj, v=aux["V"][i].numpy(),
                    s_N=float(aux["s"][i, -1]), seed=i,
                    a=ac[i].detach().numpy(), k=kc[i].detach().numpy(),
                    sat_a=float((ac[i].abs() > 0.98 * abs(A_MIN)).float().mean()),
                    sat_k=float((kc[i].abs() > 0.98 * K_MAX).float().mean()))
