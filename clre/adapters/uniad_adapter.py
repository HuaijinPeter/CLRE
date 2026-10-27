#!/usr/bin/env python
"""UniAD -> NewBorn pure data adapter (ADAPTER IMPLEMENTATION).

Authoritative specification: docs/experiments/uniad_newborn_integration_design.md
(sections 5-10, 15; PROTOCOL FREEZE section 0). This module implements the
frozen adapter contract EXACTLY; every constant below is a frozen adapter
parameter under freeze item 4 and may not be retuned in response to pilot
outcomes. Only bug fixes restoring a documented invariant are permitted.

Scope (design 22.1): pure conversion + assertions. This module
  * converts UniAD inference outputs into the frozen `ag_*` array schema
    and the frozen planner's `native_plan` / `boundaries` input forms;
  * NEVER imports the planner, NEVER duplicates objective/OBB mathematics,
    NEVER holds model state, NEVER performs I/O.
All structural violations raise UniADAdapterError (fail-loud; design 15):
adapter corruption must never masquerade as legitimate driving behavior
(e.g. as an all-infeasible tick).

Frozen upstream identity: UPSTREAM_SHA256 pins the nine v0.0.4 files whose
layout this adapter assumes (design 16 Stage A / review F8). verify_upstream()
must PASS before any integration run; a mismatch is a hard stop.

PENDING REAL-FORWARD VALIDATION (design 25; do NOT weaken to make synthetic
tests pass):
  * instance-level box column order / yaw convention (Stage B distribution
    checks: median(l) > median(w) on vehicles; yaw-vs-velocity alignment);
  * seg raster row/col <-> (x,y) orientation + half-cell offset (Gate 5
    straight-road test) — GridSpec below encodes the NOMINAL convention
    mirroring the occ-grid style (planning_head.py:226-228) at the seg
    grid's own 0.512 m resolution;
  * motion displacement origin = agent box center (Stage B traj6[0] bound).
"""
import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

ADAPTER_VERSION = ("uniad_adapter/1.1 (design 2026-09-10, freeze S0; "
                   "IL-20 2026-09-11 SDC-identity ego-ghost rule)")

# ---------------------------------------------------------------------------
# Frozen upstream pins (design 16 Stage A; review F8 file set)
# ---------------------------------------------------------------------------
UPSTREAM_SHA256: Dict[str, str] = {
    "external/v004/Bench2DriveZoo/team_code/uniad_b2d_agent.py":
        "c40f34d2f12a08659e7e4d4f55a7517fb1f6b1fc883a8eedd1f0dab3168e020c",
    "external/v004/Bench2DriveZoo/mmcv/models/dense_heads/planning_head.py":
        "b32f5ea56fe25ed89c26513040f6042cf3f2a36be56f5f21ab8358239602a168",
    "external/v004/Bench2DriveZoo/mmcv/models/dense_heads/motion_head.py":
        "4bf022d7adaa653356f2713ae0d6608bfe61cd704ea7b37bab112b813e5e9551",
    "external/v004/Bench2DriveZoo/mmcv/models/dense_heads/panseg_head.py":
        "b90ecb146ac31c20187e01587cbc9716536eb4c64044352298c45639fbf0eb8d",
    "external/v004/Bench2DriveZoo/mmcv/models/detectors/uniad_e2e.py":
        "9d1f35bdb802e283b33018ef6ba9f6648b445039eb70176b80741737e57e4c6f",
    "external/v004/Bench2DriveZoo/mmcv/models/detectors/uniad_track.py":
        "a6e142af8c319e57325a19649d89197c505cc9481b065bf1d63436f09edd31fd",
    "external/v004/Bench2DriveZoo/mmcv/core/bbox/coder/detr3d_track_coder.py":
        "2c81ee69c153b8662a173ec959e84589e2e277339081a003bdc7845780b3e40a",
    "external/v004/Bench2DriveZoo/mmcv/core/bbox/util.py":
        "d24c61941209fa7a7be18d3b4c628dc06006f692fde0e7863fb54bca50282beb",
    "external/v004/Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py":
        "ed0d6fced1b8d87b101e65781b3564e2000e12c2f38df4c1d4711e15b472c42b",
}
UNIAD_CKPT_SHA256 = \
    "8c6fb898033ef62e0f20a45ff68d554c131c0b58ca405d57bb5bfa4af1f8dcb0"
# RELEASE EDIT (see docs/PROVENANCE.md): the frozen research source hard-coded
# a machine-local checkpoint path here. Point CLRE_UNIAD_CKPT at your own copy of
# the official Bench2DriveZoo UniAD-Base checkpoint; the sha256 above is the pin
# the paper runs verified. No checkpoint is distributed with this repository.
UNIAD_CKPT_PATH = os.environ.get(
    "CLRE_UNIAD_CKPT", os.path.join("ckpts", "uniad_base_b2d.pth"))

# ---------------------------------------------------------------------------
# Frozen adapter parameters (design S0 item 4 — NOT tunable)
# ---------------------------------------------------------------------------
# Prior (design 6; review F2 conditional ratio guard)
PRIOR_N = 6
PRIOR_MAX_STEP_M = 8.0          # 0.5 s * 16 m/s — hard implausibility bound
PRIOR_FIRST_STEP_MAX_M = 5.0
# IL-4 (impl-review HIGH): the design's literal max/median ratio guard
# crashes legitimate decelerate-to-stop plans (median -> 0 while the first
# steps are large), contradicting the design's OWN F2 clause that near-stop
# plans must not crash. Restored invariant: F2 acceptance. Teleport
# protection is kept via a consecutive-step jump bound: a genuine teleport
# ([.1,.1,7.9,.1,...]) has a step-to-step delta near the 8 m cap, while any
# physically coherent accel/brake profile under the caps stays well below.
PRIOR_STEP_DELTA_MAX_M = 6.0    # derived from PRIOR_MAX_STEP_M - slack

# Agent pool constants — MUST equal the frozen VAD values
# (vad_trace_native_agent.py:36-37; asserted against file text by the tests).
K_AGENTS = 24
AG_SCORE_MIN = 0.10
# dynamic classes: identical 9-name class list in the VAD and UniAD B2D
# configs -> same indices (car,van,truck,bicycle,...,pedestrian@7). The
# launcher --check asserts the class list from the UniAD config text.
DYNAMIC_CLASSES = (0, 1, 2, 3, 7)

# SDC verify-and-drop (design 8)
SDC_CROSSKEY_TOL = 1e-4
SDC_ORIGIN_MAX_M = 2.0
EGO_LEN_M = 2 * 2.45            # frozen ego box (planner EGO_HALF_LEN/W)
EGO_W_M = 2 * 0.93
EGO_GHOST_DIST_M = 1.0          # IL-20: DIAGNOSTIC-only (audit field), never
EGO_GHOST_EXT_RTOL = 0.30       # a fail-loud condition; values unchanged

# Agent geometry invariants (design 15)
# IL-14 (Stage-E real-instance finding, route 11658 frame 0137): a REAL
# traffic_sign track box has w=0.05 m — thin static classes (sign/cone/
# light) are legitimately below the draft's blanket 0.2 m floor. The frozen
# method's own contract for small boxes is the build() clamp
# ext=max(dim/2, EXT_MIN=0.4) (candidate_a_planner.py:84,276-278), so the
# corruption-detecting lower bound applies to DYNAMIC classes only
# (vehicles/bicycle/pedestrian cannot physically be 5 cm thin).
EXTENT_MIN_M = 0.2              # dynamic classes only (IL-14)
EXTENT_MAX_M = 15.0             # all classes
YAW_MAX_RAD = math.pi + 1e-3
TRAJ0_SLACK_M = 3.0             # ||traj6[0]-pos|| <= max(2*0.5*|v|, 3.0)
MOTION_MODES = 6
MOTION_STEPS = 12
MOTION_KEEP_STEPS = 6           # first 6 x 0.5 s = 3.0 s (truncation only)

# Map / boundary extraction (design 10, frozen; review F1 channel set)
SEG_CANVAS = (200, 200)
SEG_N_CLASSES = 6               # B2D things: Broken,Solid,SolidSolid,Center,TL,SS
BOUNDARY_CHANNELS = (1, 2)      # Solid + SolidSolid == frozen BOUNDARY_LABELS
SEG_CELL_M = 102.4 / 200.0      # 0.512 m/cell (config patch_size / canvas)
BOUNDARY_MIN_LEN_M = 5.0
BOUNDARY_MIN_PTS = 6
BOUNDARY_VERTEX_SPACING_M = 0.5  # frozen instantiation of design's 0.5-1.0 m
BOUNDARY_RANGE_M = 60.0
BOUNDARY_MAX_PER_TICK = 40
# REVIEW F14 (kept visible): cv2.findContours returns blob OUTLINES — two
# lines offset by +-half the painted stroke width (~0.25-0.5 m) around the
# true lane-line centerline, at j_bnd's 0.6 m length scale. Gate 5 either
# accepts this documented offset or applies deterministic skeletonization;
# it is NOT to be "fixed" by empirical tuning here.

# All-infeasible tripwire (design 15)
TRIPWIRE_WINDOW = 500
TRIPWIRE_FRAC = 0.60
TRIPWIRE_MED_DIST_M = 10.0


class UniADAdapterError(RuntimeError):
    """Structural integration error. NEVER catch-and-continue in the agent:
    design 15 forbids converting adapter corruption into fallback driving."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _to_np(x) -> np.ndarray:
    """torch.Tensor | np.ndarray | sequence -> float64 numpy (no grad)."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, np.float64)


def _softmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def verify_upstream(root: str) -> List[Dict[str, object]]:
    """Pin check (design 16 Stage A). Returns one row per pinned file."""
    rows = []
    for rel, expect in UPSTREAM_SHA256.items():
        path = os.path.join(root, rel)
        try:
            observed = sha256_file(path)
        except OSError as e:
            observed = f"<unreadable: {e}>"
        rows.append(dict(path=rel, expected=expect, observed=observed,
                         ok=(observed == expect)))
    return rows


def require_upstream(root: str) -> None:
    rows = verify_upstream(root)
    bad = [r for r in rows if not r["ok"]]
    if bad:
        detail = "; ".join(f"{r['path']}: {str(r['observed'])[:16]}... != "
                           f"{r['expected'][:16]}..." for r in bad)
        raise UniADAdapterError(
            f"upstream provenance FAIL ({len(bad)}/{len(rows)}): {detail}. "
            "Unknown upstream variant — refusing to integrate (design 16/G1).")


# ---------------------------------------------------------------------------
# Part B — nominal prior (design 6)
# ---------------------------------------------------------------------------
def validate_prior(prior: np.ndarray) -> np.ndarray:
    """Assert the frozen prior contract on a (6,2) cumulative ego-frame plan.

    Returns a float64 COPY. Never modifies geometry (assert-only; design 6).
    """
    arr = np.asarray(prior)
    if arr.shape != (PRIOR_N, 2):
        raise UniADAdapterError(f"prior shape {arr.shape} != ({PRIOR_N}, 2)")
    arr = arr.astype(np.float64, copy=True)
    if not np.all(np.isfinite(arr)):
        raise UniADAdapterError("prior contains non-finite values")
    steps = np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], arr]), axis=0),
                           axis=1)
    if steps[0] > PRIOR_FIRST_STEP_MAX_M:
        raise UniADAdapterError(
            f"prior first step {steps[0]:.2f} m > {PRIOR_FIRST_STEP_MAX_M}")
    if steps.max() > PRIOR_MAX_STEP_M:
        raise UniADAdapterError(
            f"prior step {steps.max():.2f} m > {PRIOR_MAX_STEP_M}")
    # REVIEW F2 + IL-4: near-stop and decelerate-to-stop plans are LEGAL
    # (zero/near-zero tail spacing). Pathology is caught as a teleport:
    # a consecutive-step jump larger than any coherent accel/brake profile.
    if len(steps) > 1:
        jump = float(np.abs(np.diff(steps)).max())
        if jump > PRIOR_STEP_DELTA_MAX_M:
            raise UniADAdapterError(
                f"prior consecutive-step jump {jump:.2f} m > "
                f"{PRIOR_STEP_DELTA_MAX_M} (teleporting plan)")
    return arr


def extract_prior(out0: dict) -> np.ndarray:
    """out0 = output_data_batch[0]. sdc_traj used AS-IS: no cumsum (already
    cumulative, planning_head.py:193), no axis swap, no normalization."""
    try:
        raw = out0["planning"]["result_planning"]["sdc_traj"][0]
    except (KeyError, IndexError, TypeError) as e:
        raise UniADAdapterError(f"sdc_traj not found in output dict: {e}")
    return validate_prior(_to_np(raw))


def validate_ego_state(speed: float) -> float:
    """v0 for the ego-local optimizer state x0=[0,0,0,v0] (design 12)."""
    v = float(speed)
    if not math.isfinite(v) or v < -0.5:
        raise UniADAdapterError(f"ego speed invalid: {v}")
    return v


# ---------------------------------------------------------------------------
# Part C — agents + SDC verify-and-drop (design 7-8; review F10)
# ---------------------------------------------------------------------------
def _unpack_track_tuple(entry):
    """track_bbox_results[0] == [boxes(LiDARInstance3DBoxes), scores, labels,
    bbox_index, mask] (motion_head.py:180-183 consumption order)."""
    if not isinstance(entry, (list, tuple)) or len(entry) < 3:
        raise UniADAdapterError(
            f"track_bbox_results[0] malformed (len {len(entry) if isinstance(entry, (list, tuple)) else 'n/a'})")
    boxes_obj = entry[0]
    tensor = boxes_obj.tensor if hasattr(boxes_obj, "tensor") else boxes_obj
    boxes = _to_np(tensor)
    if boxes.ndim != 2 or boxes.shape[1] < 9:
        raise UniADAdapterError(
            f"track box tensor shape {boxes.shape} not (N, >=9) "
            "[cx,cy,cz,w,l,h,yaw,vx,vy family]")
    scores = _to_np(entry[1]).reshape(-1)
    labels = _to_np(entry[2]).reshape(-1).astype(np.int64)
    return boxes, scores, labels


def extract_agents(out0: dict,
                   motion_policy: str = "raise") -> Dict[str, np.ndarray]:
    """UniAD track+motion outputs -> frozen ag_* array schema + audit.

    Steps (design 7): alignment asserts -> SDC verify-and-drop (design 8)
    -> SDC-identity duplicate + track_ids bookkeeping (IL-20; ego-proximity
    is diagnostic only) -> geometry invariants -> top-1 mode -> first-6-step
    truncation -> cumulative-displacement -> per-step offsets -> frozen
    K=24 pool fill (dynamic classes first, score >= 0.10).

    The returned arrays feed the UNCHANGED frozen
    candidate_a_planner.agents_from_arrays; nothing here re-implements the
    frozen selection or OBB math.

    motion_policy (cold-start validity session, 2026-09-10):
      "raise" — IL-18 SYSTEMATIC motion invalidity raises (legacy /
                standalone-tool behavior).
      "flag"  — IL-18 SYSTEMATIC motion invalidity is REPORTED in
                audit["motion_systematic_invalid"] and never raises; the
                caller owns the policy (bypass: audit-only; full: input-
                validity gate -> native tick). UPSTREAM PREDICTION
                INVALIDITY is a caller decision; ADAPTER STRUCTURAL
                CORRUPTION (NaN / SDC / alignment / shape / extent /
                round-trip) raises IDENTICALLY under both policies.
    """
    if motion_policy not in ("raise", "flag"):
        raise UniADAdapterError(f"motion_policy {motion_policy!r}")
    try:
        tb = out0["track_bbox_results"][0]
        traj = _to_np(out0["traj"])
        traj_scores = _to_np(out0["traj_scores"])
        sdc_tb = out0["sdc_track_bbox_results"][0]
    except (KeyError, IndexError, TypeError) as e:
        raise UniADAdapterError(f"agent keys missing from output dict: {e}")
    boxes, scores, labels = _unpack_track_tuple(tb)
    sdc_boxes, _, _ = _unpack_track_tuple(sdc_tb)
    if sdc_boxes.shape[0] != 1:
        raise UniADAdapterError(
            f"sdc_track_bbox_results carries {sdc_boxes.shape[0]} rows != 1")

    n_all = boxes.shape[0]
    # design 15 / review F10: the (A+1) family must agree BEFORE the drop.
    if traj.ndim != 4 or traj.shape[1] != MOTION_MODES \
            or traj.shape[2] != MOTION_STEPS or traj.shape[3] < 2:
        raise UniADAdapterError(
            f"traj shape {traj.shape} != (A+1, {MOTION_MODES}, "
            f"{MOTION_STEPS}, >=2)")
    if not (len(traj) == len(traj_scores) == n_all == len(scores)
            == len(labels)):
        raise UniADAdapterError(
            f"A+1 family misaligned: boxes {n_all}, traj {len(traj)}, "
            f"traj_scores {len(traj_scores)}, scores {len(scores)}, "
            f"labels {len(labels)}")
    if n_all < 1:
        raise UniADAdapterError(
            "track_bbox_results has zero rows — the SDC row is appended "
            "unconditionally (motion_head.py:171-177); its absence means "
            "the upstream structure changed")

    # --- SDC verify (cross-key + origin) THEN drop (design 8, D3) ---
    last = boxes[-1]
    crosskey = float(np.max(np.abs(last[:7] - sdc_boxes[0][:7])))
    if crosskey > SDC_CROSSKEY_TOL:
        raise UniADAdapterError(
            f"SDC cross-key mismatch: last row differs from "
            f"sdc_track_bbox_results by {crosskey:.3e} > {SDC_CROSSKEY_TOL} "
            "— upstream SDC-append semantics changed; refusing to drop")
    sdc_center_dist = float(np.hypot(last[0], last[1]))
    if sdc_center_dist > SDC_ORIGIN_MAX_M:
        raise UniADAdapterError(
            f"SDC candidate row center {sdc_center_dist:.2f} m from origin "
            f"> {SDC_ORIGIN_MAX_M} — not the ego; refusing to drop")
    boxes, scores, labels = boxes[:-1], scores[:-1], labels[:-1]
    traj, traj_scores = traj[:-1], traj_scores[:-1]
    n = n_all - 1
    if not (len(traj) == len(traj_scores) == len(boxes) == len(scores)
            == len(labels) == n):
        raise UniADAdapterError("post-drop arrays misaligned")

    # --- IL-20 (route-2554 ParkedObstacle false positive, 2026-09-11):
    # SDC identity is STRUCTURAL, not geometric. motion_head.forward_test
    # (motion_head.py:174-177) torch.cat's the sdc_track_bbox_results row
    # itself onto track_bbox_results as the LAST row, and the top-level
    # track_ids (uniad_track.py:784-790, _track_instances2results) stay
    # length A without the SDC. The cross-key + origin checks above
    # therefore PROVE the dropped row was the SDC. After the drop, a
    # remaining row is structural corruption only if it (a) carries the
    # SDC row's identity (all 7 box dims within SDC_CROSSKEY_TOL — the
    # SDC appended twice) or (b) breaks the A/A+1 bookkeeping (exactly one
    # row dropped). A real vehicle near the ego origin with ego-like
    # extents (close pass) is a legitimate scene and is only AUDITED.
    # Gate-6 bypass evidence (route 3785, 920 ticks): len(track_ids) ==
    # n_after_drop on 807/807 conclusive ticks, 0 duplicated ids.
    n_sdc_dup = 0
    ego_like_near_n = 0
    ego_like_near_min = float("inf")
    if n:
        dup = np.max(np.abs(boxes[:, :7] - sdc_boxes[0][:7]), axis=1) \
            <= SDC_CROSSKEY_TOL
        n_sdc_dup = int(dup.sum())
        if n_sdc_dup:
            raise UniADAdapterError(
                f"duplicated SDC identity: {n_sdc_dup} remaining row(s) "
                "equal the sdc_track_bbox_results row after the drop — "
                "upstream appended the SDC more than once (structure "
                "changed)")
        d_all = np.hypot(boxes[:, 0], boxes[:, 1])
        l_like = np.abs(boxes[:, 4] - EGO_LEN_M) / EGO_LEN_M \
            <= EGO_GHOST_EXT_RTOL
        w_like = np.abs(boxes[:, 3] - EGO_W_M) / EGO_W_M <= EGO_GHOST_EXT_RTOL
        near = (d_all < EGO_GHOST_DIST_M) & l_like & w_like
        ego_like_near_n = int(near.sum())
        if ego_like_near_n:
            ego_like_near_min = float(d_all[near].min())
    track_ids_n = -1
    ids = out0.get("track_ids") if hasattr(out0, "get") else None
    if ids is not None:
        try:
            track_ids_n = int(len(ids))
        except TypeError as e:
            raise UniADAdapterError(f"track_ids not sized: {e}")
        if track_ids_n != n:
            raise UniADAdapterError(
                f"SDC drop bookkeeping impossible: {n} rows after the drop "
                f"!= top-level track_ids length {track_ids_n} (exactly one "
                "appended SDC row is expected; structure changed)")

    # --- geometry invariants (design 15) ---
    if n:
        if not np.all(np.isfinite(boxes)):
            raise UniADAdapterError("non-finite values in agent boxes")
        if not np.all(np.isfinite(scores)):
            raise UniADAdapterError(
                "non-finite track scores (would silently empty the pool)")
        if not np.all(np.isfinite(traj_scores)):
            raise UniADAdapterError(
                "non-finite traj_scores (softmax would produce NaN probs)")
        ext = boxes[:, 3:6]
        if ext.min() <= 0 or ext.max() > EXTENT_MAX_M:
            raise UniADAdapterError(
                f"agent extents outside (0, {EXTENT_MAX_M}] m: "
                f"min {ext.min():.2f}, max {ext.max():.2f}")
        dyn_rows = np.isin(labels, DYNAMIC_CLASSES)
        if dyn_rows.any() and ext[dyn_rows].min() < EXTENT_MIN_M:
            raise UniADAdapterError(
                f"dynamic-class agent extent {ext[dyn_rows].min():.2f} m < "
                f"{EXTENT_MIN_M} (corrupted dims; thin static classes are "
                "exempt — IL-14)")
        if np.abs(boxes[:, 6]).max() > YAW_MAX_RAD:
            raise UniADAdapterError(
                f"agent |yaw| {np.abs(boxes[:, 6]).max():.2f} > pi "
                "(degrees? unwrapped?)")
        if not np.all(np.isfinite(traj[:, :, :MOTION_KEEP_STEPS, :2])):
            raise UniADAdapterError("non-finite values in motion futures")

    # --- top-1 mode (design 7 / D2), truncation, offsets (design 5) ---
    ag_valid = np.zeros(K_AGENTS, bool)
    ag_score = np.zeros(K_AGENTS, np.float32)
    ag_label = np.full(K_AGENTS, -1, np.int8)
    ag_box = np.zeros((K_AGENTS, 7), np.float32)
    ag_vel = np.zeros((K_AGENTS, 2), np.float32)
    ag_traj = np.zeros((K_AGENTS, MOTION_MODES, MOTION_KEEP_STEPS, 2),
                       np.float32)
    ag_prob = np.zeros((K_AGENTS, MOTION_MODES), np.float32)

    min_agent_dist = float("inf")
    if n:
        probs = np.stack([_softmax(traj_scores[a]) for a in range(n)])
        disp6 = traj[:, :, :MOTION_KEEP_STEPS, :2]            # cumulative
        # per-step offsets so the frozen build() pos+cumsum(offsets)
        # reproduces pos+disp EXACTLY (design 5 round-trip contract)
        offs6 = np.diff(
            np.concatenate([np.zeros((n, MOTION_MODES, 1, 2)), disp6],
                           axis=2), axis=2)
        rt = np.max(np.abs(np.cumsum(offs6, axis=2) - disp6))
        if rt > 1e-6:
            raise UniADAdapterError(f"offsets round-trip error {rt:.2e}")
        # future-indexing bound (design 9). IL-18 (Gate-6 attempt-3 crash,
        # route 3785 tick 2): a SINGLE weakly-tracked agent (|v|=0.66)
        # emitted a legitimate-but-garbage 61.9 m first step from the
        # unbounded regression head; ticks 0-1 (all agents) and Stage-E
        # (20 agents) were clean, so this is network output, NOT the
        # systematic conversion error this bound exists to catch (a
        # double-cumsum / origin error hits essentially EVERY moving
        # agent). Restored invariants: (a) the frozen method's unbounded
        # future pass-through (VAD ag_traj was never bounded — bounding
        # per-agent would CHANGE frozen selection semantics), (b) design
        # 15's fail-loud for STRUCTURAL corruption, now keyed on the
        # systematic signature: >=50% of >=3 agents violating. Per-agent
        # outliers pass through UNCHANGED and are counted in the audit /
        # dump so the Stage-B distribution half still detects systematic
        # drift post-hoc. Bound VALUE unchanged (max(2*0.5*|v|, 3.0)).
        v_abs = np.hypot(boxes[:, 7], boxes[:, 8])
        top1 = probs.argmax(axis=1)
        first = np.linalg.norm(disp6[np.arange(n), top1, 0, :], axis=1)
        bound = np.maximum(2 * 0.5 * v_abs, TRAJ0_SLACK_M)
        bad = first > bound
        n_viol = int(bad.sum())
        # Root cause (tensor-proven 2026-09-10, UPSTREAM CONFIRMED, review
        # 0 MAJOR): the systematic signature is a UniAD cold-start
        # transient — tracker mass-ID-re-association with motion offsets
        # still anchored at each query's previous reference position (raw
        # pre-adapter tensors replicate 7/7 @ 61.61 m; adapter round-trip
        # 0.000e+00). It is upstream PREDICTION invalidity, not adapter
        # corruption, so under motion_policy="flag" it is reported, not
        # fatal; "raise" retains the legacy fail-loud for standalone tools.
        motion_systematic = bool(n >= 3 and 2 * n_viol >= n)
        if motion_systematic and motion_policy == "raise":
            raise UniADAdapterError(
                f"motion first-step bound violated on {n_viol}/{n} agents "
                f"(max {first.max():.2f} m) — systematic upstream motion "
                "invalidity (IL-18; cold-start stale-anchor transient) or "
                "displacement-origin error")
        min_agent_dist = float(np.hypot(boxes[:, 0], boxes[:, 1]).min())

        # frozen pool fill: identical logic to candidate_a_agent.py:339-343
        order = np.argsort(-scores)
        order = order[scores[order] >= AG_SCORE_MIN]
        dyn = np.isin(labels[order], DYNAMIC_CLASSES)
        sel = np.concatenate([order[dyn], order[~dyn]])[:K_AGENTS]
        m = len(sel)
        if m:
            b = boxes[sel]
            ag_valid[:m] = True
            ag_score[:m] = scores[sel]
            ag_label[:m] = labels[sel]
            # [cx,cy,cz,w,l,h,yaw,vx,vy] -> ag_box [cx,cy,w,l,h,yaw,cz]
            ag_box[:m] = np.stack(
                [b[:, 0], b[:, 1], b[:, 3], b[:, 4], b[:, 5], b[:, 6],
                 b[:, 2]], 1)
            ag_vel[:m] = b[:, 7:9]
            ag_traj[:m] = offs6[sel]
            ag_prob[:m] = probs[sel]

    audit = dict(
        motion_bound_violations=(n_viol if n else 0),
        max_first_step_m=(float(first.max()) if n else 0.0),
        motion_systematic_invalid=bool(n and motion_systematic),
        n_raw=int(n_all), n_after_drop=int(n),
        sdc_center_dist_m=sdc_center_dist,
        sdc_crosskey_max_abs=crosskey,
        sdc_duplicate_rows=int(n_sdc_dup),
        track_ids_n=int(track_ids_n),
        ego_like_near_origin_n=int(ego_like_near_n),
        ego_like_near_origin_min_dist_m=ego_like_near_min,
        min_agent_dist_m=min_agent_dist,
        n_pool=int(ag_valid.sum()),
        n_dynamic=int(np.isin(ag_label[ag_valid], DYNAMIC_CLASSES).sum())
        if ag_valid.any() else 0,
    )
    return dict(ag_valid=ag_valid, ag_score=ag_score, ag_label=ag_label,
                ag_box=ag_box, ag_vel=ag_vel, ag_traj=ag_traj,
                ag_prob=ag_prob, audit=audit)


# ---------------------------------------------------------------------------
# Part E — map raster -> boundary polylines (design 10; frozen params)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridSpec:
    """Seg-raster grid convention. NOMINAL until the Gate-5 straight-road
    orientation test (design 10.4): the occ path (planning_head.py:226-228)
    maps col->x, row->y with a half-cell offset on ITS OWN 0.5 m grid; we
    mirror that STYLE at the seg grid's 0.512 m resolution.
    PENDING REAL-FORWARD VALIDATION — do not treat as verified."""
    h: int = SEG_CANVAS[0]
    w: int = SEG_CANVAS[1]
    cell_m: float = SEG_CELL_M

    def px_to_m(self, rowcol: np.ndarray) -> np.ndarray:
        """(P,2) [row, col] pixel coords -> (P,2) [x, y] metres (ego frame)."""
        rc = np.asarray(rowcol, np.float64)
        x = (rc[:, 1] - self.w / 2.0) * self.cell_m + self.cell_m / 2.0
        y = (rc[:, 0] - self.h / 2.0) * self.cell_m + self.cell_m / 2.0
        return np.stack([x, y], axis=1)


def _resample_polyline(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Arc-length resampling (deterministic vertex decimation; design 10.5)."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return pts[:1]
    n_out = max(int(math.floor(s[-1] / spacing)) + 1, 2)
    st = np.linspace(0.0, s[-1], n_out)
    return np.stack([np.interp(st, s, pts[:, 0]),
                     np.interp(st, s, pts[:, 1])], axis=1)


def _split_at_border(px: np.ndarray, h: int, w: int) -> List[np.ndarray]:
    """Split a closed contour into open runs at image-border touches
    (design 10.5: border touches are artificial closures, not geometry)."""
    on_border = ((px[:, 0] <= 0) | (px[:, 0] >= h - 1)
                 | (px[:, 1] <= 0) | (px[:, 1] >= w - 1))
    if not on_border.any():
        return [px]
    n = len(px)
    runs, cur = [], []
    # rotate so we start on a border pixel -> interior runs are contiguous
    start = int(np.argmax(on_border))
    idx = np.roll(np.arange(n), -start)
    for i in idx:
        if on_border[i]:
            if len(cur) >= 2:
                runs.append(np.asarray(cur))
            cur = []
        else:
            cur.append(px[i])
    if len(cur) >= 2:
        runs.append(np.asarray(cur))
    return runs


def extract_boundaries(lane: np.ndarray,
                       grid: GridSpec = GridSpec()) -> List[np.ndarray]:
    """(6,200,200) B2D lane raster -> list[(P,2)] ego-frame boundary
    polylines for the frozen planner's `boundaries` input.

    Frozen pipeline (design 10.5): channels {1,2} (Solid+SolidSolid ==
    frozen BOUNDARY_LABELS semantics; review F1) -> cv2.findContours ->
    border split -> px->m -> length/vertex filters -> 0.5 m resampling.
    `drivable` is NEVER consumed (untrained with num_stuff_classes=0).
    """
    import cv2  # deferred: keeps module import light for non-map tests
    lane = np.asarray(lane)
    if lane.shape != (SEG_N_CLASSES, grid.h, grid.w):
        raise UniADAdapterError(
            f"lane raster shape {lane.shape} != "
            f"({SEG_N_CLASSES}, {grid.h}, {grid.w}) [review F1 contract]")
    mask = ((lane[BOUNDARY_CHANNELS[0]] > 0)
            | (lane[BOUNDARY_CHANNELS[1]] > 0)).astype(np.uint8)
    if mask.sum() == 0:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    out: List[np.ndarray] = []
    for c in contours:
        px = c.reshape(-1, 2)[:, ::-1]          # cv2 gives (x=col, y=row)
        for run in _split_at_border(px.astype(np.float64), grid.h, grid.w):
            pts = grid.px_to_m(run)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            if len(pts) < BOUNDARY_MIN_PTS or seg.sum() < BOUNDARY_MIN_LEN_M:
                continue
            poly = _resample_polyline(pts, BOUNDARY_VERTEX_SPACING_M)
            if len(poly) < BOUNDARY_MIN_PTS:
                continue
            if np.abs(poly).max() > BOUNDARY_RANGE_M:
                raise UniADAdapterError(
                    f"boundary polyline extends to {np.abs(poly).max():.1f} m"
                    f" > {BOUNDARY_RANGE_M} (grid transform suspect)")
            out.append(poly)
    if len(out) > BOUNDARY_MAX_PER_TICK:
        raise UniADAdapterError(
            f"{len(out)} boundary polylines > {BOUNDARY_MAX_PER_TICK} "
            "per tick (raster noise / extraction defect)")
    return out


# ---------------------------------------------------------------------------
# Runtime-mode + shadow helpers (used by the wrapped agent; pure, testable)
# ---------------------------------------------------------------------------
VALID_MODES = ("bypass", "full")

# IL-8: stable, decodable fallback-reason codes (Python str hash is salted
# per process and was useless for post-hoc attribution).
FALLBACK_REASON_CODE = {
    "": 0,
    "runtime_budget_exceeded": 1,
    "nonfinite_objective": 2,
    "nonfinite_trajectory": 3,
    "hard_bound_violation": 4,
    "planner_exception": 5,     # any planner_exception:<T> maps here
}


def fallback_reason_code(reason: str) -> int:
    if reason.startswith("planner_exception"):
        return FALLBACK_REASON_CODE["planner_exception"]
    return FALLBACK_REASON_CODE.get(reason, 9)


def parse_mode(env: Optional[dict] = None) -> str:
    """NEWBORN_UNIAD_MODE parser. `refine_only` is dead code under the S0
    freeze (item 1) and is rejected explicitly, not silently."""
    env = os.environ if env is None else env
    mode = env.get("NEWBORN_UNIAD_MODE", "")
    if mode == "refine_only":
        raise RuntimeError(
            "NEWBORN_UNIAD_MODE=refine_only is NOT a runnable arm "
            "(design S0 freeze item 1: no formal refine_only experiment). "
            "Definition retained in design 13 for the record only.")
    if mode not in VALID_MODES:
        raise RuntimeError(
            f"NEWBORN_UNIAD_MODE must be one of {VALID_MODES} "
            f"(got {mode!r}); 'bypass' is engineering-only (Gate 6).")
    return mode


def parse_map_option(env: Optional[dict] = None) -> str:
    """NEWBORN_UNIAD_MAP_OPTION: 'A' (raster boundaries) or 'B' (j_bnd
    inert, disclosed). REQUIRED in full mode — no default, so the Gate-5
    decision must be recorded explicitly (design 10/17-task)."""
    env = os.environ if env is None else env
    opt = env.get("NEWBORN_UNIAD_MAP_OPTION", "")
    if opt not in ("A", "B"):
        raise RuntimeError(
            "NEWBORN_UNIAD_MAP_OPTION must be 'A' or 'B' (Gate-5 decision, "
            f"design 10); got {opt!r}. No default: the decision must be "
            "explicit and recorded in provenance.")
    return opt


def stock_postprocess(steer_raw: float, throttle_raw: float,
                      brake_raw: float, speed: float
                      ) -> Tuple[float, float, float]:
    """EXACT stock UniAD post-processing chain (uniad_b2d_agent.py:378-387):
    used ONLY by the bypass shadow-equality check (design 17). Not applied
    in full mode (D8: the quirk is native execution law)."""
    steer_traj, throttle_traj, brake_traj = steer_raw, throttle_raw, brake_raw
    if brake_traj < 0.05:
        brake_traj = 0.0
    if throttle_traj > brake_traj:
        brake_traj = 0.0
    if speed > 5:
        throttle_traj = 0
    steer = float(np.clip(float(steer_traj), -1, 1))
    throttle = float(np.clip(float(throttle_traj), 0, 0.75))
    brake = float(np.clip(float(brake_traj), 0, 1))
    return steer, throttle, brake


def newborn_postprocess(steer_raw: float, throttle_raw: float,
                        brake_raw: float) -> Tuple[float, float, float]:
    """EXACT frozen NewBorn execution post-processing
    (candidate_a_agent.py:489-498): brake deadzone, throttle-priority brake
    zero, clips. NO speed>5 throttle cut (design D8)."""
    if brake_raw < 0.05:
        brake_raw = 0.0
    if throttle_raw > brake_raw:
        brake_raw = 0.0
    steer = float(np.clip(float(steer_raw), -1, 1))
    throttle = float(np.clip(float(throttle_raw), 0, 0.75))
    brake = float(np.clip(float(brake_raw), 0, 1))
    return steer, throttle, brake


class AllInfeasibleTripwire:
    """Campaign tripwire (design 15): all-infeasible fraction > 60% over a
    500-tick window WITH median min-agent-distance > 10 m == the ego-ghost /
    corruption signature -> raise. A crash, never a silent switch."""

    def __init__(self, window: int = TRIPWIRE_WINDOW,
                 frac: float = TRIPWIRE_FRAC,
                 med_dist_m: float = TRIPWIRE_MED_DIST_M):
        self.window = int(window)
        self.frac = float(frac)
        self.med_dist_m = float(med_dist_m)
        self._flags: List[bool] = []
        self._dists: List[float] = []

    def update(self, all_infeasible: bool, min_agent_dist_m: float) -> None:
        self._flags.append(bool(all_infeasible))
        self._dists.append(float(min_agent_dist_m))
        if len(self._flags) > self.window:
            self._flags.pop(0)
            self._dists.pop(0)
        if len(self._flags) < self.window:
            return
        frac = sum(self._flags) / len(self._flags)
        if frac <= self.frac:
            return
        ai_dists = [d for f, d in zip(self._flags, self._dists) if f]
        med = float(np.median(ai_dists)) if ai_dists else float("inf")
        if med > self.med_dist_m:
            raise UniADAdapterError(
                f"all-infeasible tripwire: fraction {frac:.2f} > "
                f"{self.frac} over {self.window} ticks with median "
                f"min-agent-distance {med:.1f} m > {self.med_dist_m} — "
                "adapter-corruption signature (design 15), not traffic")


def frozen_adapter_params() -> Dict[str, object]:
    """Provenance snapshot of every frozen adapter parameter (S0 item 4 /
    G8 freeze-compliance check)."""
    return dict(
        adapter_version=ADAPTER_VERSION,
        prior=dict(n=PRIOR_N, max_step_m=PRIOR_MAX_STEP_M,
                   first_step_max_m=PRIOR_FIRST_STEP_MAX_M,
                   step_delta_max_m=PRIOR_STEP_DELTA_MAX_M,
                   guard="IL-4 consecutive-step jump (replaced the "
                         "ratio/zero-median guard that crashed legitimate "
                         "decel-to-stop plans)"),
        agents=dict(k_agents=K_AGENTS, ag_score_min=AG_SCORE_MIN,
                    ag_array_dtype="float32 (documented deviation: frozen "
                                   "VAD trace schema stores float16; both "
                                   "UniAD arms unaffected)",
                    dynamic_classes=list(DYNAMIC_CLASSES),
                    sdc_crosskey_tol=SDC_CROSSKEY_TOL,
                    sdc_origin_max_m=SDC_ORIGIN_MAX_M,
                    ego_ghost_dist_m=EGO_GHOST_DIST_M,
                    ego_ghost_ext_rtol=EGO_GHOST_EXT_RTOL,
                    ego_ghost_rule=("IL-20 (2026-09-11): fail-loud only on "
                                    "exact SDC-identity duplicate (7 dims "
                                    "within sdc_crosskey_tol) or top-level "
                                    "track_ids != n_after_drop; ego_ghost_"
                                    "dist_m/ext_rtol are DIAGNOSTIC-only "
                                    "(audit ego_like_near_origin_*)"),
                    extent_min_m=EXTENT_MIN_M, extent_max_m=EXTENT_MAX_M,
                    keep_steps=MOTION_KEEP_STEPS),
        map=dict(canvas=list(SEG_CANVAS), n_classes=SEG_N_CLASSES,
                 boundary_channels=list(BOUNDARY_CHANNELS),
                 cell_m=SEG_CELL_M, min_len_m=BOUNDARY_MIN_LEN_M,
                 min_pts=BOUNDARY_MIN_PTS,
                 vertex_spacing_m=BOUNDARY_VERTEX_SPACING_M,
                 range_m=BOUNDARY_RANGE_M,
                 max_per_tick=BOUNDARY_MAX_PER_TICK,
                 grid_orientation="NOMINAL-PENDING-GATE5"),
        tripwire=dict(window=TRIPWIRE_WINDOW, frac=TRIPWIRE_FRAC,
                      med_dist_m=TRIPWIRE_MED_DIST_M),
    )
