#!/usr/bin/env python
"""DriveTransformer-Large -> NewBorn pure data adapter (ADAPTER IMPLEMENTATION).

Authoritative specification:
    docs/experiments/drivetransformer_newborn_integration_design.md
(sections 2-6 adapter contracts, 9 file responsibilities, 11 trace codes,
12.F upstream pins, 13 stages B/E/F, 15 frozen/adapter/prohibited table).
Every constant below is either INHERITED unchanged from the frozen VAD /
UniAD contract (imported from `uniad_adapter`, never re-typed) or a
structural fact of the pinned DriveTransformer release (shapes, keys,
class lists, coder ranges). None is a tunable.

Scope (design 9): pure functions + assertions. This module
  * converts DriveTransformer `forward_test` outputs into the frozen
    `ag_*` array schema and the frozen planner's `native_plan` /
    `boundaries` input forms;
  * replicates the released coder's top-k index law so the motion-mode
    logits captured by the head hook are row-aligned with `boxes_3d`;
  * NEVER imports the planner (`candidate_a_planner`), NEVER duplicates
    objective/OBB mathematics, NEVER holds model state, NEVER performs
    trace/provenance I/O (reading the pinned upstream files for the
    sha256 gate and path-loading the pinned `denormalize_bbox` are the
    only file reads).
All structural violations raise DTAdapterError (fail-loud; design 2/3/4
"never fall back"): adapter corruption must never masquerade as driving
behaviour.

Validity policy (design 5, FROZEN Option A): the auxiliary motion head is
gated per tick by the INHERITED IL-18 systematic signature; the caller
withholds the agent set (`ag_valid[:] = False`) on gated ticks. The gate
itself (`pred_terms_gate`) is a stateless pure function.

PENDING REAL-FORWARD VALIDATION (design 3 step 9 / 5 diagnostics 1-5;
do NOT weaken to make synthetic tests pass): instance-level l > w on
vehicle rows; decode-rotation/frame check (yaw-vs-velocity alignment of
the first displacement); `||traj6[0] - pos||` bound distribution; map
polyline quality of the released checkpoint.
"""
import importlib.util
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from clre.adapters import uniad_adapter as ua  # RELEASE EDIT: package renamed

ADAPTER_VERSION = ("drivetransformer_adapter/1.0 (design 2026-09-11; "
                   "stages B-F implemented 2026-09-12)")

# ---------------------------------------------------------------------------
# Frozen upstream identity (design 12.F; all 15 pins re-verified 2026-09-12)
# ---------------------------------------------------------------------------
DT_COMMIT = "19702015eb700e804bafb53e63b04571c7b8604a"
DT_REL_ROOT = "external/v004/DriveTransformer"
_ADZ = f"{DT_REL_ROOT}/adzoo/drivetransformer"
UPSTREAM_SHA256: Dict[str, str] = {
    f"{DT_REL_ROOT}/team_code/drivetransformer_b2d_agent.py":
        "f957af41019b65db541c19fe3e0bfaf6bcb2e59c8175ce8d606bdde331f1a9c9",
    f"{DT_REL_ROOT}/team_code/pid_controller.py":
        "99b3a75c9b1fd0a3503243f6bdfa7bc64d95d7ef018d23d8953c835c7f923e8f",
    f"{DT_REL_ROOT}/team_code/planner.py":
        "99178efb83af3d02068d7346732948cef445ad76a756caf2f95d489147aece31",
    f"{_ADZ}/configs/drivetransformer/drivetransformer_large.py":
        "3c96ad00eb1f7b3f84b70b7ae41114a5486a71f518f7bcdeb1fd4d6aa96ffa4e",
    f"{_ADZ}/mmdet3d_plugin/ours/drivetransformer.py":
        "03c69acbcca0d7cb39e7fa005ed80a88fc5865b37e0cd68c47f9b41acbc5aa05",
    f"{_ADZ}/mmdet3d_plugin/ours/drivetransformer_head.py":
        "744f859e834a7c14c1df41cb6457a9b1bda52f68925bfaf2d9816a42d950d227",
    f"{_ADZ}/mmdet3d_plugin/ours/drivetransformer_layers.py":
        "5270f2551f44135d8231409d9b98b7fc9c82e5b9f83cc787e0eab1a6eae29c44",
    f"{_ADZ}/mmdet3d_plugin/core/bbox/coders/fut_nms_free_coder.py":
        "5b5a46d14af16d541ee511aa4479202497b322df7bc50c5385be221f3921804a",
    f"{_ADZ}/mmdet3d_plugin/core/bbox/coders/map_nms_free_coder.py":
        "2d7f288d799a16f0e5a448034154122535283d8870a5772143c15501252310d7",
    f"{_ADZ}/mmdet3d_plugin/datasets/bench2drive_drivetransformer_dataset.py":
        "425b003747ddf52d7ef2d6dd3b1a5ad7a8969b406d373296e049d42a9712dab1",
    f"{_ADZ}/mmdet3d_plugin/datasets/pipelines/traj_class_assign.py":
        "739cd2fa9fd248846ec33481d81f6797af88116d9a081b557bdda9ae6aa003fc",
    f"{DT_REL_ROOT}/mmcv/core/bbox/util.py":
        "d24c61941209fa7a7be18d3b4c628dc06006f692fde0e7863fb54bca50282beb",
    f"{DT_REL_ROOT}/mmcv/core/bbox/transforms.py":
        "58d2b87ebbe7d18b550aa13c901378e9f76cb6a414fe5d073610f43ecf3f057f",
    f"{DT_REL_ROOT}/setup.py":
        "60599722ce86c849cde24dedcb8944afe849924d25e6a633d1c77acd5af018da",
    f"{DT_REL_ROOT}/requirements.txt":
        "32b5aa35a6f742912364a559fde3be8dc7dee4d980a8a49b319ecaeaa65176a3",
}
DT_CONFIG_REL = f"{_ADZ}/configs/drivetransformer/drivetransformer_large.py"
DT_STOCK_AGENT_REL = f"{DT_REL_ROOT}/team_code/drivetransformer_b2d_agent.py"
DT_BBOX_UTIL_REL = f"{DT_REL_ROOT}/mmcv/core/bbox/util.py"
# `mmcv/core/bbox/util.py` is byte-identical to the Zoo v0.0.4 file pinned
# by the UniAD adapter (design 3 table, row `boxes_3d`).
assert UPSTREAM_SHA256[DT_BBOX_UTIL_REL] == ua.UPSTREAM_SHA256[
    "external/v004/Bench2DriveZoo/mmcv/core/bbox/util.py"]

# Checkpoint: no official hash is published; the sha256 observed at Stage A
# (artifacts/drivetransformer_ckpt_sha256.txt, 2026-09-12) is the project pin.
# RELEASE EDIT (see docs/PROVENANCE.md): machine-local path replaced by an env var.
# Point CLRE_DT_CKPT at your own copy of the official DriveTransformer-Large
# checkpoint. No checkpoint is distributed with this repository.
DT_CKPT_PATH = os.environ.get(
    "CLRE_DT_CKPT", os.path.join("ckpts", "drivetransformer_large.pth"))
DT_CKPT_SHA256 = \
    "982f2024cb17f59541d0333a1b48ad74267ee0ba91eca3abe880afc60ed8c2f5"
DT_CKPT_SHA256_RECORD_REL = "artifacts/drivetransformer_ckpt_sha256.txt"
DT_PARAMS_M = 520.18        # Stage A: built model 520.18 M / ckpt 520.24 M values
DT_FUT_EGO_MODE = 1         # CFG:200 (single-mode release; A:486 selected_mode=0)

# ---------------------------------------------------------------------------
# Nominal prior contract (design 2)
# ---------------------------------------------------------------------------
PRIOR_DT_S = 0.1                              # fix_time row spacing
PRIOR_N_ROWS = 30                             # rows 0..29 <-> t = 0.1..3.0 s
PRIOR_TIMES_S = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
PRIOR_ROWS = [4, 9, 14, 19, 24, 29]           # row k <-> t = (k+1)*0.1 s
FIX_TIME_KEY = "ego_fut_preds_fix_time"
FIX_DIST_KEY = "ego_fut_preds_fix_dist"
EGO_CLS_KEY = "ego_traj_cls_scores"
FIX_TIME_SHAPE = (1, DT_FUT_EGO_MODE, PRIOR_N_ROWS, 2)
# frozen NewBorn floor at the make_window call site (VA:389 / UW:594)
V_REF_FLOOR_MPS = 2.5

# ---------------------------------------------------------------------------
# Agent contract (design 3) — pool/validity constants INHERITED from UA
# ---------------------------------------------------------------------------
CLASS_NAMES = ('car', 'van', 'truck', 'bicycle', 'traffic_sign',
               'traffic_cone', 'traffic_light', 'pedestrian', 'others')
N_CLASSES = len(CLASS_NAMES)                  # 9
BOX_DIM = 9                                   # [cx,cy,cz_bottom,w,l,h,yaw,vx,vy]
N_BOXES_MAX = 100                             # coder max_num
DYNAMIC_CLASSES = ua.DYNAMIC_CLASSES          # (0, 1, 2, 3, 7)
K_AGENTS = ua.K_AGENTS                        # 24
AG_SCORE_MIN = ua.AG_SCORE_MIN                # 0.10
MOTION_MODES = ua.MOTION_MODES                # 6
MOTION_STEPS = ua.MOTION_KEEP_STEPS           # 6 (DriveTransformer emits 6)
MOTION_DT_S = 0.5
TRAJ_FLAT = MOTION_STEPS * 2                  # 12: trajs_3d is (N, 6, 12)
TRAJ0_SLACK_M = ua.TRAJ0_SLACK_M              # 3.0 m
EXTENT_MIN_M = ua.EXTENT_MIN_M                # 0.2 (dynamic classes only, IL-14)
EXTENT_MAX_M = ua.EXTENT_MAX_M                # 15.0
YAW_MAX_RAD = ua.YAW_MAX_RAD
VALIDITY_MIN_AGENTS = 3                       # systematic = (N >= 3) and 2*n_viol >= N
AG_ARRAY_DTYPE = np.float32

# ---------------------------------------------------------------------------
# Coder replication constants (design 4; CFG:349-355 / FC:54-91)
# ---------------------------------------------------------------------------
COD_PC_RANGE = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
COD_POST_CENTER_RANGE = [-20, -35, -10.0, 20, 35, 10.0]
COD_MAX_NUM = 100
COD_NUM_CLASSES = 9
AGENT_QUERIES = 900                           # agent_num_query_sifted (CFG:128)
BBOX_CODE_SIZE = 10                           # normalized [cx,cy,w,l,cz,h,sin,cos,vx,vy]
HOOK_SCORE_TOL = 1e-5
HOOK_XY_TOL = 1e-4

# ---------------------------------------------------------------------------
# Map contract (design 6)
# ---------------------------------------------------------------------------
MAP_CLASS_NAMES = ('Broken', 'Solid', 'SolidSolid', 'Center', 'TrafficLight',
                   'StopSign')
MAP_N_CLASSES = len(MAP_CLASS_NAMES)          # 6
MAP_MAX = 50
MAP_PTS = 20
BOUNDARY_LABELS = (1, 2)                      # == frozen CAP.BOUNDARY_LABELS
BOUNDARY_SCORE_MIN = 0.3                      # == frozen CAP.BOUNDARY_SCORE_MIN
BOUNDARY_RANGE_M = ua.BOUNDARY_RANGE_M        # 60 m grid/units guard
BOUNDARY_MAX_PER_TICK = ua.BOUNDARY_MAX_PER_TICK   # 40

# ---------------------------------------------------------------------------
# Protocol / parity constants (design 1, 8, 10)
# ---------------------------------------------------------------------------
TWIN_PID_GAINS = dict(speed_k_p=2.0, speed_k_i=0.8, speed_k_d=1.5,
                      steer_k_p=1.5, steer_k_i=0.2, steer_k_d=0.2)  # A:44
STARTUP_BRAKE_LAST_STEP = 20                  # A:494: step <= 20 -> brake
STARTUP_CONTROL = (0.0, 0.0, 1.0)             # (steer, throttle, brake)
CADENCE_PROTOCOL = ("Option A: 10 Hz decisions, odd-tick hold, 20-tick "
                    "start-up brake in both arms")
SIM_TICK_S = 0.05
MODEL_TICK_S = 0.10

# ---------------------------------------------------------------------------
# Trace codes (design 11)
# ---------------------------------------------------------------------------
SEL_MODE_CODE = {"feasible_best": 0, "all_infeasible_safest": 1,
                 "unconstrained": 2}
EXEC_SOURCE_CODE = {"dt_prior": 0, "candidate_a": 1, "e59_fallback": 2,
                    "startup_brake": 3, "hold": 4, "dt_native": 5}

VALID_MODES = ("bypass", "full")
VALID_PRED_TERMS = ("off", "gated")

# ---------------------------------------------------------------------------
# Passive diagnostics constants (design 5 items 1-5; structural, non-tuned)
# ---------------------------------------------------------------------------
DIAG_MOVING_SPEED_MPS = 1.0       # "moving agents": ||(vx, vy)|| > 1 m/s
DIAG_MIN_AGENT_TICKS = 100        # decode-rotation check needs >= 100 agent-ticks
DIAG_ROT_MEDIAN_DEG = 30.0        # median |wrap(atan2(dx1,dy1) - atan2(vx,vy))| < 30 deg
DIAG_VEHICLE_CLASSES = (0, 1, 2)  # car, van, truck for the l > w readout
DIAG_ASSOC_DIST_M = 1.0           # tick-to-tick top-1 consistency association

# re-exports (inherited helpers used by the wrapper; imported, never copied)
validate_prior = ua.validate_prior
validate_ego_state = ua.validate_ego_state
newborn_postprocess = ua.newborn_postprocess
fallback_reason_code = ua.fallback_reason_code
AllInfeasibleTripwire = ua.AllInfeasibleTripwire
sha256_file = ua.sha256_file


class DTAdapterError(RuntimeError):
    """Structural integration error. NEVER catch-and-continue in the agent:
    adapter corruption must not be converted into fallback driving."""


ADAPTER_ERRORS = (DTAdapterError, ua.UniADAdapterError)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _to_np(x) -> np.ndarray:
    return ua._to_np(x)


def _softmax_rows(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    v = v - v.max(axis=1, keepdims=True)
    e = np.exp(v)
    return e / e.sum(axis=1, keepdims=True)


def _sigmoid(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return 1.0 / (1.0 + np.exp(-v))


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, np.float64) + np.pi) % (2 * np.pi) - np.pi


def verify_upstream(root: str) -> List[Dict[str, object]]:
    """Pin check (design 12.F). One row per pinned file."""
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
        raise DTAdapterError(
            f"upstream provenance FAIL ({len(bad)}/{len(rows)}): {detail}. "
            "Unknown DriveTransformer variant — refusing to integrate.")


def read_recorded_ckpt_sha(root: str) -> Optional[str]:
    """The Stage-A record `artifacts/drivetransformer_ckpt_sha256.txt`
    (`sha256sum` format). None if absent."""
    path = os.path.join(root, DT_CKPT_SHA256_RECORD_REL)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        first = f.readline().strip()
    return first.split()[0] if first else None


_DENORM_CACHE: Dict[str, object] = {}


def load_pinned_denormalize(root: str):
    """Path-load `denormalize_bbox` from the PINNED DriveTransformer
    `mmcv/core/bbox/util.py` (torch-only file). Used when the caller does
    not hand over the fork's own imported function (tests); the wrapper
    passes the fork's `mmcv.core.bbox.util.denormalize_bbox` (identical
    file by sha256, see UPSTREAM_SHA256)."""
    path = os.path.join(root, DT_BBOX_UTIL_REL)
    if path in _DENORM_CACHE:
        return _DENORM_CACHE[path]
    if sha256_file(path) != UPSTREAM_SHA256[DT_BBOX_UTIL_REL]:
        raise DTAdapterError("pinned bbox util sha256 mismatch")
    spec = importlib.util.spec_from_file_location("_dt_pinned_bbox_util",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _DENORM_CACHE[path] = mod.denormalize_bbox
    return mod.denormalize_bbox


# ---------------------------------------------------------------------------
# Design 2 — nominal prior
# ---------------------------------------------------------------------------
def extract_prior(out0: dict, return_raw: bool = False):
    """`out0 = output_data_batch[0]`. Reads the RAW `ego_fut_preds_fix_time`
    tensor (BEFORE the stock column swap at A:493), asserts the single-mode
    (1,1,30,2) contract, selects PRIOR_ROWS -> (6,2) absolute positions at
    0.5 s..3.0 s (x right / y forward, no interpolation, no swap, no
    scaling) and runs the INHERITED `validate_prior`. Never falls back."""
    if not hasattr(out0, "get"):
        raise DTAdapterError("out0 is not a dict")
    if FIX_TIME_KEY not in out0:
        raise DTAdapterError(f"{FIX_TIME_KEY} missing from output dict")
    if out0.get(EGO_CLS_KEY) is not None:
        raise DTAdapterError(
            f"{EGO_CLS_KEY} is not None: multi-mode ego head — the released "
            "single-mode contract (CFG fut_ego_mode=1) does not hold")
    if FIX_DIST_KEY not in out0 or out0[FIX_DIST_KEY] is None:
        raise DTAdapterError(
            f"{FIX_DIST_KEY} missing: needed by the stock steering law and "
            "the twin-controller parity")
    t = out0[FIX_TIME_KEY]
    shape = tuple(t.shape) if hasattr(t, "shape") else None
    if shape != FIX_TIME_SHAPE:
        raise DTAdapterError(
            f"{FIX_TIME_KEY} shape {shape} != {FIX_TIME_SHAPE} "
            "(a (1, M>1, 30, 2) tensor means a multi-mode checkpoint/config; "
            "mode 0 is never taken silently)")
    raw = _to_np(t)[0, 0]                          # (30, 2) float64
    if not np.all(np.isfinite(raw)):
        raise DTAdapterError("non-finite values in ego_fut_preds_fix_time "
                             "(all 30 rows are checked, not only the 6 kept)")
    prior = raw[PRIOR_ROWS, :].copy()
    prior = ua.validate_prior(prior)               # inherited, unchanged
    if return_raw:
        return prior, raw
    return prior


# ---------------------------------------------------------------------------
# Design 4 — coder index replication for the head-hook logits
# ---------------------------------------------------------------------------
def replicate_coder_indices(outs: dict, coder, denormalize_fn=None,
                            root: Optional[str] = None) -> Dict[str, object]:
    """Replicates `CustomNMSFreeCoder.decode_single` (FC:54-91) index law
    on the head's `outs` (hook capture): sigmoid -> flat top-100 ->
    labels/bbox_index -> denormalize -> post-centre mask; returns the
    row-aligned motion-mode LOGITS `outs['all_traj_cls_scores'][-1][0]
    [bbox_index][mask]` plus the replicated labels/scores/xy for the
    cross-key assertion (`assert_hook_alignment`). fp32 exactly as
    `force_fp32` does. Hard stop if the coder carries a score threshold
    (the FC:67-75 loop is NOT replicated by design)."""
    import torch  # runtime dependency of the hook path only
    if getattr(coder, "score_threshold", None) is not None:
        raise DTAdapterError(
            "coder.score_threshold is not None — released config has none; "
            "the FC:67-75 threshold loop is deliberately not replicated")
    max_num = int(getattr(coder, "max_num"))
    num_classes = int(getattr(coder, "num_classes"))
    if max_num != COD_MAX_NUM or num_classes != COD_NUM_CLASSES:
        raise DTAdapterError(
            f"coder (max_num={max_num}, num_classes={num_classes}) != pinned "
            f"({COD_MAX_NUM}, {COD_NUM_CLASSES})")
    pc_range = getattr(coder, "pc_range")
    pcr_list = getattr(coder, "post_center_range")
    if pcr_list is None:
        raise DTAdapterError("coder.post_center_range is None")
    for key in ("all_cls_scores", "all_bbox_preds", "all_traj_cls_scores"):
        if key not in outs:
            raise DTAdapterError(f"hook outs missing {key}")
    cls_all = outs["all_cls_scores"]
    bbox_all = outs["all_bbox_preds"]
    tcls_all = outs["all_traj_cls_scores"]
    if not (torch.is_tensor(cls_all) and torch.is_tensor(bbox_all)
            and torch.is_tensor(tcls_all)):
        raise DTAdapterError("hook outs entries are not tensors")
    if cls_all.dim() != 4 or bbox_all.dim() != 4 or tcls_all.dim() != 4:
        raise DTAdapterError(
            f"hook outs rank: cls {tuple(cls_all.shape)}, bbox "
            f"{tuple(bbox_all.shape)}, traj_cls {tuple(tcls_all.shape)}")
    if cls_all.shape[1] != 1:
        raise DTAdapterError(f"batch {cls_all.shape[1]} != 1")
    if not (cls_all.shape[0] == bbox_all.shape[0] == tcls_all.shape[0]
            and cls_all.shape[2] == bbox_all.shape[2] == tcls_all.shape[2]):
        raise DTAdapterError("hook outs layer/query dims disagree")
    if (cls_all.shape[2] != AGENT_QUERIES or cls_all.shape[3] != N_CLASSES
            or bbox_all.shape[3] != BBOX_CODE_SIZE
            or tcls_all.shape[3] != MOTION_MODES):
        raise DTAdapterError(
            f"hook outs shapes cls {tuple(cls_all.shape)} bbox "
            f"{tuple(bbox_all.shape)} traj_cls {tuple(tcls_all.shape)} != "
            f"[L,1,{AGENT_QUERIES},{N_CLASSES}] / [..,{BBOX_CODE_SIZE}] / "
            f"[..,{MOTION_MODES}]")
    if denormalize_fn is None:
        if root is None:
            raise DTAdapterError("denormalize_fn or root required")
        denormalize_fn = load_pinned_denormalize(root)
    with torch.no_grad():
        cls = cls_all[-1][0].float().sigmoid()                     # (900, 9)
        scores, idx = cls.view(-1).topk(max_num)                   # FC:56
        labels = idx % num_classes                                 # FC:57
        bbox_index = idx // num_classes                            # FC:58
        boxes = denormalize_fn(bbox_all[-1][0].float()[bbox_index],
                               pc_range)                           # FC:59-61
        pcr = torch.as_tensor(pcr_list, device=boxes.device,
                              dtype=boxes.dtype)                   # FC:78-79
        mask = (boxes[:, :3] >= pcr[:3]).all(1) & \
               (boxes[:, :3] <= pcr[3:]).all(1)                    # FC:80-83
        logits = tcls_all[-1][0].float()[bbox_index][mask]         # (N, 6)
        rep = dict(
            logits=logits.cpu().numpy().astype(np.float32),
            labels=labels[mask].cpu().numpy().astype(np.int64),
            scores=scores[mask].cpu().numpy().astype(np.float32),
            boxes_xy=boxes[mask][:, :2].cpu().numpy().astype(np.float32),
            bbox_index=bbox_index[mask].cpu().numpy().astype(np.int64),
            n_top=int(max_num), n_kept=int(mask.sum().item()),
        )
    if not np.all(np.isfinite(rep["logits"])):
        raise DTAdapterError("non-finite motion-mode logits from the hook")
    return rep


def assert_hook_alignment(out0: dict, rep: Dict[str, object]
                          ) -> Dict[str, float]:
    """Structural cross-key (design 4): the replicated rows must equal the
    coder's returned rows exactly: N, labels (exact), scores (<=1e-5),
    (cx, cy) (<=1e-4; untouched by the cz lowering). Any mismatch = the
    coder semantics changed -> hard fail."""
    boxes_obj = out0.get("boxes_3d")
    tensor = boxes_obj.tensor if hasattr(boxes_obj, "tensor") else boxes_obj
    b = _to_np(tensor)
    scores = _to_np(out0["scores_3d"]).reshape(-1)
    labels = _to_np(out0["labels_3d"]).reshape(-1).astype(np.int64)
    n = int(len(scores))
    if b.ndim != 2 or b.shape[0] != n or len(labels) != n:
        raise DTAdapterError(
            f"out0 family misaligned: boxes {b.shape}, scores {n}, "
            f"labels {len(labels)}")
    if int(rep["n_kept"]) != n or len(rep["logits"]) != n:
        raise DTAdapterError(
            f"hook cross-key: replicated N {rep['n_kept']} != coder N {n}")
    d_lab = int(np.sum(np.asarray(rep["labels"]) != labels))
    d_sc = float(np.max(np.abs(np.asarray(rep["scores"], np.float64)
                               - scores))) if n else 0.0
    d_xy = float(np.max(np.abs(np.asarray(rep["boxes_xy"], np.float64)
                               - b[:, :2]))) if n else 0.0
    if d_lab or d_sc > HOOK_SCORE_TOL or d_xy > HOOK_XY_TOL:
        raise DTAdapterError(
            f"hook cross-key mismatch: labels differ on {d_lab} rows, "
            f"max|score diff| {d_sc:.2e} (tol {HOOK_SCORE_TOL}), max|xy diff| "
            f"{d_xy:.2e} (tol {HOOK_XY_TOL}) — coder index law changed")
    return dict(hook_n=n, hook_label_mismatch=d_lab, hook_score_maxdiff=d_sc,
                hook_xy_maxdiff=d_xy)


# ---------------------------------------------------------------------------
# Design 3 — agents
# ---------------------------------------------------------------------------
def extract_agents(out0: dict, logits) -> Dict[str, object]:
    """DriveTransformer det+motion outputs + hook logits -> frozen `ag_*`
    schema + audit (design 3 steps 1-8).

    Steps: unpack/align asserts -> geometry invariants -> ag_box mapping
    [cx,cy,w,l,h,yaw,cz_bottom] -> softmax mode probability (D2) ->
    cumulative -> per-step offsets with round-trip identity -> validity
    signal (inherited IL-18: first-step bound max(2*0.5*|v|, 3.0),
    systematic = >=50% of >=3 agents; per-agent outliers pass through
    UNBOUNDED) -> frozen pool fill (K=24, score >= 0.10, dynamic first).
    The validity signal is REPORTED here (audit); the withholding policy
    (design 5, Option A) is the caller's, via `pred_terms_gate`."""
    for key in ("boxes_3d", "scores_3d", "labels_3d", "trajs_3d"):
        if key not in out0:
            raise DTAdapterError(f"agent key {key} missing from output dict")
    boxes_obj = out0["boxes_3d"]
    tensor = boxes_obj.tensor if hasattr(boxes_obj, "tensor") else boxes_obj
    b = _to_np(tensor)
    scores = _to_np(out0["scores_3d"]).reshape(-1)
    labels = _to_np(out0["labels_3d"]).reshape(-1)
    trajs_raw = _to_np(out0["trajs_3d"])
    if b.ndim != 2 or b.shape[1] != BOX_DIM:
        raise DTAdapterError(
            f"boxes_3d.tensor shape {b.shape} != (N, {BOX_DIM}) "
            "[cx,cy,cz_bottom,w,l,h,yaw,vx,vy]")
    n = int(b.shape[0])
    if n > N_BOXES_MAX:
        raise DTAdapterError(f"{n} boxes > coder max_num {N_BOXES_MAX}")
    if not (len(scores) == len(labels) == n):
        raise DTAdapterError(
            f"det family misaligned: boxes {n}, scores {len(scores)}, "
            f"labels {len(labels)}")
    if n == 0:
        trajs = np.zeros((0, MOTION_MODES, MOTION_STEPS, 2))
    else:
        if trajs_raw.ndim != 3 or tuple(trajs_raw.shape[1:]) != \
                (MOTION_MODES, TRAJ_FLAT):
            raise DTAdapterError(
                f"trajs_3d shape {trajs_raw.shape} != (N, {MOTION_MODES}, "
                f"{TRAJ_FLAT})")
        if trajs_raw.shape[0] != n:
            raise DTAdapterError(
                f"trajs_3d rows {trajs_raw.shape[0]} != boxes {n}")
        trajs = trajs_raw.reshape(n, MOTION_MODES, MOTION_STEPS, 2)
    lg = np.asarray(logits, np.float64) if logits is not None else None
    if lg is None:
        raise DTAdapterError("motion-mode logits missing (hook not wired)")
    lg = lg.reshape(-1, MOTION_MODES) if lg.size else \
        np.zeros((0, MOTION_MODES))
    if lg.shape[0] != n:
        raise DTAdapterError(
            f"hook logits rows {lg.shape[0]} != boxes {n} (row alignment)")
    if not np.all(np.isfinite(lg)):
        raise DTAdapterError("non-finite motion-mode logits")

    if n:
        if not np.all(np.isfinite(b)):
            raise DTAdapterError("non-finite values in agent boxes")
        if not np.all(np.isfinite(scores)):
            raise DTAdapterError("non-finite detection scores")
        if scores.min() < 0.0 or scores.max() > 1.0:
            raise DTAdapterError(
                f"detection scores outside [0,1]: {scores.min():.3f}.."
                f"{scores.max():.3f} (sigmoid expected)")
        lab_i = labels.astype(np.int64)
        if np.any(lab_i != labels) or lab_i.min() < 0 \
                or lab_i.max() >= N_CLASSES:
            raise DTAdapterError(
                f"labels outside 0..{N_CLASSES - 1}: {labels.min()}.."
                f"{labels.max()}")
        labels = lab_i

        # Frozen pool eligibility (VA:339-357 / UA:489-506), computed HERE
        # (not only at pool-fill time below) so the physical-size asserts
        # can be scoped to it: a row AG_SCORE_MIN / the K=24 cap discards
        # anyway never reaches ag_box, so its geometry cannot "masquerade
        # as legitimate driving behavior" and must not crash the tick on
        # synthetic/out-of-distribution input (Stage-E finding, 2026-09-12:
        # step 0 of the first real forward raised on a top-100 row decoded
        # to a tiny dynamic-class extent; the row's score was NOT captured
        # — this fix does not depend on knowing it, only on the fact that
        # physical-size validation ran before pool-eligibility filtering).
        # Computed ONCE; the pool-fill block below reuses it verbatim so
        # there is exactly one definition of "eligible". No threshold
        # changed: EXTENT_MIN_M stays 0.2, EXTENT_MAX_M stays 15,
        # AG_SCORE_MIN stays 0.10, K_AGENTS stays 24.
        order = np.argsort(-scores)
        order = order[scores[order] >= AG_SCORE_MIN]
        dyn = np.isin(labels[order], DYNAMIC_CLASSES)
        sel = np.concatenate([order[dyn], order[~dyn]])[:K_AGENTS]

        # ---- physical-size assertions: pool-eligible rows only ----
        if len(sel):
            ext = b[sel][:, 3:6]
            lab_sel = labels[sel]
            if ext.min() <= 0 or ext.max() > EXTENT_MAX_M:
                raise DTAdapterError(
                    f"agent extents outside (0, {EXTENT_MAX_M}] m: min "
                    f"{ext.min():.3f}, max {ext.max():.2f}")
            dyn_rows = np.isin(lab_sel, DYNAMIC_CLASSES)
            if dyn_rows.any() and ext[dyn_rows].min() < EXTENT_MIN_M:
                raise DTAdapterError(
                    f"dynamic-class agent extent {ext[dyn_rows].min():.3f} m "
                    f"< {EXTENT_MIN_M} (corrupted dims; thin static classes "
                    "exempt, IL-14)")
        # ---- structural asserts: ALL rows (unchanged) ----
        if np.abs(b[:, 6]).max() > YAW_MAX_RAD:
            raise DTAdapterError(
                f"agent |yaw| {np.abs(b[:, 6]).max():.2f} > pi (degrees? "
                "unwrapped?)")
        if not np.all(np.isfinite(trajs)):
            raise DTAdapterError("non-finite values in trajs_3d futures")
    else:
        labels = labels.astype(np.int64)
        sel = np.zeros(0, np.int64)

    ag_valid = np.zeros(K_AGENTS, bool)
    ag_score = np.zeros(K_AGENTS, AG_ARRAY_DTYPE)
    ag_label = np.full(K_AGENTS, -1, np.int8)
    ag_box = np.zeros((K_AGENTS, 7), AG_ARRAY_DTYPE)
    ag_vel = np.zeros((K_AGENTS, 2), AG_ARRAY_DTYPE)
    ag_traj = np.zeros((K_AGENTS, MOTION_MODES, MOTION_STEPS, 2),
                       AG_ARRAY_DTYPE)
    ag_prob = np.zeros((K_AGENTS, MOTION_MODES), AG_ARRAY_DTYPE)

    n_viol = 0
    max_first = 0.0
    motion_systematic = False
    min_agent_dist = float("inf")
    top1 = np.zeros(0, np.int64)
    first = np.zeros(0)
    bound = np.zeros(0)
    top1_sig = np.zeros(0)
    probs = np.zeros((0, MOTION_MODES))
    if n:
        probs = _softmax_rows(lg)                          # D2: softmax
        sig = _sigmoid(lg)
        top1 = lg.argmax(axis=1)                           # invariant choice
        top1_sig = sig[np.arange(n), top1]
        disp6 = trajs                                      # cumulative (N,6,6,2)
        offs6 = np.diff(
            np.concatenate([np.zeros((n, MOTION_MODES, 1, 2)), disp6],
                           axis=2), axis=2)
        rt = float(np.max(np.abs(np.cumsum(offs6, axis=2) - disp6)))
        if rt > 1e-6:
            raise DTAdapterError(f"offsets round-trip error {rt:.2e}")
        v_abs = np.hypot(b[:, 7], b[:, 8])
        first = np.linalg.norm(disp6[np.arange(n), top1, 0, :], axis=1)
        bound = np.maximum(2 * MOTION_DT_S * v_abs, TRAJ0_SLACK_M)
        bad = first > bound
        n_viol = int(bad.sum())
        max_first = float(first.max())
        motion_systematic = bool(n >= VALIDITY_MIN_AGENTS and 2 * n_viol >= n)
        min_agent_dist = float(np.hypot(b[:, 0], b[:, 1]).min())

        # frozen pool fill (VA:339-357 / UA:489-506); `sel` computed above
        m = len(sel)
        if m:
            bb = b[sel]
            ag_valid[:m] = True
            ag_score[:m] = scores[sel]
            ag_label[:m] = labels[sel]
            # [cx,cy,cz_bottom,w,l,h,yaw,vx,vy] -> ag_box [cx,cy,w,l,h,yaw,cz]
            ag_box[:m] = np.stack(
                [bb[:, 0], bb[:, 1], bb[:, 3], bb[:, 4], bb[:, 5], bb[:, 6],
                 bb[:, 2]], 1)
            ag_vel[:m] = bb[:, 7:9]
            ag_traj[:m] = offs6[sel]
            ag_prob[:m] = probs[sel]

    audit = dict(
        n_boxes=int(n),
        n_pool=int(ag_valid.sum()),
        n_dynamic=(int(np.isin(ag_label[ag_valid], DYNAMIC_CLASSES).sum())
                   if ag_valid.any() else 0),
        motion_bound_violations=int(n_viol),
        max_first_step_m=float(max_first),
        motion_systematic_invalid=bool(motion_systematic),
        motion_top1_sigmoid_med=(float(np.median(top1_sig)) if n
                                 else float("nan")),
        min_agent_dist_m=float(min_agent_dist),
        # per-row arrays (row-aligned with out0) for the passive dump
        top1=top1.astype(np.int8), first_step_m=first.astype(np.float32),
        first_bound_m=bound.astype(np.float32),
        top1_sigmoid=top1_sig.astype(np.float32),
        softmax_entropy=(-(probs * np.log(probs + 1e-12)).sum(1)
                         ).astype(np.float32),
    )
    return dict(ag_valid=ag_valid, ag_score=ag_score, ag_label=ag_label,
                ag_box=ag_box, ag_vel=ag_vel, ag_traj=ag_traj,
                ag_prob=ag_prob, audit=audit)


# ---------------------------------------------------------------------------
# Design 6 — map unpack (the frozen CAP.boundaries_from_map does the rest)
# ---------------------------------------------------------------------------
def extract_boundaries_inputs(out0: dict
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat-dict unpack + assertions -> (map_pts (K,20,2) f64, map_scores
    (K,) f64, map_labels (K,) int). The caller passes these UNCHANGED to
    the frozen `candidate_a_planner.boundaries_from_map` (labels {1,2},
    score > 0.3) and then `check_boundary_count`."""
    for key in ("map_pts_3d", "map_scores_3d", "map_labels_3d"):
        if key not in out0:
            raise DTAdapterError(f"map key {key} missing from output dict")
    pts = _to_np(out0["map_pts_3d"])
    sco = _to_np(out0["map_scores_3d"]).reshape(-1)
    lab = _to_np(out0["map_labels_3d"]).reshape(-1)
    if pts.ndim != 3 or tuple(pts.shape[1:]) != (MAP_PTS, 2):
        if pts.size == 0 and len(sco) == 0:
            pts = np.zeros((0, MAP_PTS, 2))
        else:
            raise DTAdapterError(
                f"map_pts_3d shape {pts.shape} != (K, {MAP_PTS}, 2)")
    k = int(pts.shape[0])
    if k > MAP_MAX:
        raise DTAdapterError(f"{k} map instances > coder max_num {MAP_MAX}")
    if not (len(sco) == len(lab) == k):
        raise DTAdapterError(
            f"map family misaligned: pts {k}, scores {len(sco)}, labels "
            f"{len(lab)}")
    if k:
        if not (np.all(np.isfinite(pts)) and np.all(np.isfinite(sco))):
            raise DTAdapterError("non-finite map values")
        lab_i = lab.astype(np.int64)
        if np.any(lab_i != lab) or lab_i.min() < 0 \
                or lab_i.max() >= MAP_N_CLASSES:
            raise DTAdapterError(
                f"map labels outside 0..{MAP_N_CLASSES - 1}")
        if sco.min() < 0.0 or sco.max() > 1.0:
            raise DTAdapterError("map scores outside [0,1] (sigmoid expected)")
        if np.abs(pts).max() > BOUNDARY_RANGE_M:
            raise DTAdapterError(
                f"map polyline extends to {np.abs(pts).max():.1f} m > "
                f"{BOUNDARY_RANGE_M} (grid/units suspect)")
        lab = lab_i
    else:
        lab = lab.astype(np.int64)
    return pts.astype(np.float64), sco.astype(np.float64), lab


def check_boundary_count(n_boundaries: int) -> int:
    """Inherited per-tick kept-boundary bound (UA BOUNDARY_MAX_PER_TICK)."""
    if int(n_boundaries) > BOUNDARY_MAX_PER_TICK:
        raise DTAdapterError(
            f"{n_boundaries} boundary polylines > {BOUNDARY_MAX_PER_TICK} "
            "per tick (map defect)")
    return int(n_boundaries)


# ---------------------------------------------------------------------------
# Modes / policy (design 5, 9)
# ---------------------------------------------------------------------------
def parse_mode(env: Optional[dict] = None) -> str:
    """NEWBORN_DT_MODE in {bypass, full}. No third arm exists."""
    env = os.environ if env is None else env
    mode = env.get("NEWBORN_DT_MODE", "")
    if mode == "refine_only":
        raise RuntimeError(
            "NEWBORN_DT_MODE=refine_only is NOT a runnable arm (design 1: "
            "two arms only; UniAD S0 item 1 applies verbatim).")
    if mode not in VALID_MODES:
        raise RuntimeError(
            f"NEWBORN_DT_MODE must be one of {VALID_MODES} (got {mode!r}); "
            "'bypass' is engineering-only (parity, design 10).")
    return mode


def parse_pred_terms(env: Optional[dict] = None) -> str:
    """NEWBORN_DT_PRED_TERMS in {off, gated} (design 5 staging switch).
    REQUIRED in full mode — no default, so the policy is recorded
    explicitly in provenance. `off` = agents always withheld; `gated` =
    Option-A withholding on IL-18-systematic ticks only."""
    env = os.environ if env is None else env
    pol = env.get("NEWBORN_DT_PRED_TERMS", "")
    if pol not in VALID_PRED_TERMS:
        raise RuntimeError(
            f"NEWBORN_DT_PRED_TERMS must be one of {VALID_PRED_TERMS} (got "
            f"{pol!r}). No default: the staging policy must be explicit.")
    return pol


def pred_terms_gate(policy: str, motion_systematic_invalid: bool) -> bool:
    """Stateless Option-A gate: True iff the auxiliary futures may enter
    the frozen chain this tick. `off` -> never; `gated` -> unless the
    inherited IL-18 systematic signature fired. No thresholds here."""
    if policy not in VALID_PRED_TERMS:
        raise DTAdapterError(f"pred-terms policy {policy!r}")
    if policy == "off":
        return False
    return not bool(motion_systematic_invalid)


def withhold_agents(ag: Dict[str, object]) -> Dict[str, object]:
    """Option-A withholding: a COPY of the ag_* dict with ag_valid all
    False, so the frozen `agents_from_arrays` returns [] (open-road
    regime: risk = bar = 0, feasibility_gap = 99). Arrays otherwise
    untouched (trace keeps the pool statistics)."""
    out = dict(ag)
    out["ag_valid"] = np.zeros_like(np.asarray(ag["ag_valid"], bool))
    return out


# ---------------------------------------------------------------------------
# Provenance snapshot (design 11 meta json)
# ---------------------------------------------------------------------------
def frozen_adapter_params() -> Dict[str, object]:
    return dict(
        adapter_version=ADAPTER_VERSION,
        backbone=dict(commit=DT_COMMIT, rel_root=DT_REL_ROOT,
                      config=DT_CONFIG_REL, ckpt_path=DT_CKPT_PATH,
                      ckpt_sha256=DT_CKPT_SHA256, params_M=DT_PARAMS_M,
                      fut_ego_mode=DT_FUT_EGO_MODE,
                      pins=len(UPSTREAM_SHA256)),
        protocol=dict(cadence=CADENCE_PROTOCOL, sim_tick_s=SIM_TICK_S,
                      model_tick_s=MODEL_TICK_S,
                      startup_brake_last_step=STARTUP_BRAKE_LAST_STEP,
                      startup_control=list(STARTUP_CONTROL),
                      twin_pid_gains=dict(TWIN_PID_GAINS),
                      disclosed=("PEB-v2 PID windows 40 samples = 4.0 s at "
                                 "10 Hz; radar EMERGENCY_PERSIST_N=3 = 0.3 s; "
                                 "planner re-solves every 0.1 s on the 12x0.25 "
                                 "s grid; ego_lcf_feat[8] = executed steer two "
                                 "model calls ago")),
        prior=dict(rows=list(PRIOR_ROWS), dt_s=PRIOR_DT_S,
                   times_s=list(PRIOR_TIMES_S), shape=list(FIX_TIME_SHAPE),
                   frame="x right / y forward, absolute, ego origin, "
                         "read BEFORE the stock A:493 column swap",
                   inherited_validate_prior=dict(
                       first_step_max_m=ua.PRIOR_FIRST_STEP_MAX_M,
                       max_step_m=ua.PRIOR_MAX_STEP_M,
                       step_delta_max_m=ua.PRIOR_STEP_DELTA_MAX_M),
                   v_ref_floor_mps=V_REF_FLOOR_MPS),
        agents=dict(class_names=list(CLASS_NAMES),
                    dynamic_classes=list(DYNAMIC_CLASSES),
                    k_agents=K_AGENTS, ag_score_min=AG_SCORE_MIN,
                    ag_array_dtype="float32 (UA's documented deviation "
                                   "from the VAD float16 trace dtype)",
                    box_mapping="[cx,cy,w,l,h,yaw,cz_bottom] <- "
                                "[b0,b1,b3,b4,b5,b6,b2]; cz never read by "
                                "the frozen planner",
                    modes=MOTION_MODES, steps=MOTION_STEPS, dt_s=MOTION_DT_S,
                    offsets="diff of cumulative displacements; round-trip "
                            "<= 1e-6 asserted",
                    mode_prob="D2: softmax(logits); top-1 invariant; sigmoid "
                              "top-1 logged only",
                    validity=dict(bound="max(2*0.5*|v|, 3.0) m",
                                  slack_m=TRAJ0_SLACK_M,
                                  systematic="N>=3 and 2*n_viol>=N (IL-18)",
                                  per_agent="unbounded pass-through"),
                    extent_min_m=EXTENT_MIN_M, extent_max_m=EXTENT_MAX_M,
                    r1_note="DriveTransformer top-100 has no score floor: "
                            "the additive R1 band [0.15,0.30) is LIVE"),
        hook=dict(target="model.pts_bbox_head (forward hook, read-only)",
                  coder=dict(pc_range=COD_PC_RANGE,
                             post_center_range=COD_POST_CENTER_RANGE,
                             max_num=COD_MAX_NUM, num_classes=COD_NUM_CLASSES,
                             agent_queries=AGENT_QUERIES),
                  crosskey=dict(score_tol=HOOK_SCORE_TOL, xy_tol=HOOK_XY_TOL)),
        map=dict(class_names=list(MAP_CLASS_NAMES),
                 boundary_labels=list(BOUNDARY_LABELS),
                 boundary_score_min=BOUNDARY_SCORE_MIN, max_instances=MAP_MAX,
                 pts_per_instance=MAP_PTS, range_m=BOUNDARY_RANGE_M,
                 max_per_tick=BOUNDARY_MAX_PER_TICK,
                 consumer="frozen CAP.boundaries_from_map, every model tick"),
        validity_policy=dict(option="A: withhold agents on systematic-invalid "
                                    "ticks (ag_valid[:]=False); no "
                                    "whole-native fallback implemented",
                             switch="NEWBORN_DT_PRED_TERMS in {off, gated}"),
        diagnostics=dict(moving_speed_mps=DIAG_MOVING_SPEED_MPS,
                         min_agent_ticks=DIAG_MIN_AGENT_TICKS,
                         rot_median_deg=DIAG_ROT_MEDIAN_DEG,
                         vehicle_classes=list(DIAG_VEHICLE_CLASSES),
                         assoc_dist_m=DIAG_ASSOC_DIST_M),
        codes=dict(sel_mode=dict(SEL_MODE_CODE),
                   exec_source=dict(EXEC_SOURCE_CODE),
                   fallback_reason=dict(ua.FALLBACK_REASON_CODE)),
        tripwire=dict(window=ua.TRIPWIRE_WINDOW, frac=ua.TRIPWIRE_FRAC,
                      med_dist_m=ua.TRIPWIRE_MED_DIST_M),
    )


# ---------------------------------------------------------------------------
# Design 5 — passive diagnostics over a bypass dump (pure functions)
# ---------------------------------------------------------------------------
def _valid_rows(n_boxes: np.ndarray, n_max: int) -> np.ndarray:
    """(T, n_max) bool mask of real rows in a NaN/-1 padded per-tick dump."""
    t = len(n_boxes)
    idx = np.arange(n_max)[None, :]
    return idx < np.asarray(n_boxes, np.int64)[:, None]


def passive_diagnostics(dump: Dict[str, np.ndarray]) -> Dict[str, object]:
    """Design 5 items 1-5 over a bypass dump (keys as written by the
    wrapper's `_nb_dump_row`: dump_n_boxes (T,), dump_boxes (T,100,9),
    dump_labels (T,100), dump_trajs (T,100,6,6,2), dump_logits (T,100,6),
    dump_top1 (T,100), dump_crosskey_ok (T,), dump_motion_invalid (T,),
    dump_n_viol (T,), dump_first_step_m (T,100), dump_first_bound_m
    (T,100), dump_n_map (T,), dump_map_scores (T,50), dump_map_labels
    (T,50), dump_map_pts (T,50,20,2), dump_n_boundaries (T,)).
    Every item is a structural PASS/FAIL (1, 2, 3, 5) or a reported number
    (4, map). No threshold is tuned here; INSUFFICIENT_DATA is distinct
    from FAIL and never counts as PASS."""
    d = {k: np.asarray(v) for k, v in dump.items()}
    n_boxes = d["dump_n_boxes"].astype(np.int64).reshape(-1)
    T = int(len(n_boxes))
    rows = _valid_rows(n_boxes, d["dump_boxes"].shape[1]) if T else \
        np.zeros((0, N_BOXES_MAX), bool)

    def _sel(key, tail, dtype):
        # an EMPTY dump (T == 0) carries 1-D arrays: shape the empty
        # selection explicitly instead of masking a 1-D array
        if not T:
            return np.zeros((0,) + tail, dtype)
        return d[key][rows].astype(dtype)
    boxes = _sel("dump_boxes", (BOX_DIM,), np.float64)          # (A, 9)
    labels = _sel("dump_labels", (), np.int64)
    trajs = _sel("dump_trajs", (MOTION_MODES, MOTION_STEPS, 2),
                 np.float64)                                    # (A, 6, 6, 2)
    logits = _sel("dump_logits", (MOTION_MODES,), np.float64)   # (A, 6)
    top1 = _sel("dump_top1", (), np.int64)
    A = int(len(boxes))
    out: Dict[str, object] = dict(n_model_ticks=T, n_agent_ticks=A)

    # 1. finite ratios (fail-loud otherwise)
    fin_traj = float(np.isfinite(trajs).mean()) if A else 1.0
    fin_log = float(np.isfinite(logits).mean()) if A else 1.0
    out["d1_finite"] = dict(trajs_ratio=fin_traj, logits_ratio=fin_log,
                            status=("PASS" if (fin_traj == 1.0 and
                                               fin_log == 1.0) else "FAIL"))
    # 2. hook alignment cross-key on 100 % of model ticks
    ck = d["dump_crosskey_ok"].astype(np.int64) if T else np.zeros(0, int)
    ck_frac = float(ck.mean()) if T else 0.0
    out["d2_crosskey"] = dict(ok_fraction=ck_frac, ticks=T,
                              status=("PASS" if (T and ck_frac == 1.0)
                                      else ("INSUFFICIENT_DATA" if not T
                                            else "FAIL")))
    # 3. decode-rotation / frame check on moving agents
    if A:
        v = np.hypot(boxes[:, 7], boxes[:, 8])
        moving = v > DIAG_MOVING_SPEED_MPS
        d1 = trajs[np.arange(A), top1, 0, :]                 # first displacement
        ang_d = np.arctan2(d1[:, 0], d1[:, 1])
        ang_v = np.arctan2(boxes[:, 7], boxes[:, 8])
        diff = np.abs(np.degrees(_wrap_angle(ang_d - ang_v)))[moving]
    else:
        diff = np.zeros(0)
    n_mov = int(len(diff))
    med = float(np.median(diff)) if n_mov else float("nan")
    out["d3_decode_rotation"] = dict(
        n_moving_agent_ticks=n_mov, median_abs_deg=med,
        p90_abs_deg=(float(np.percentile(diff, 90)) if n_mov else float("nan")),
        threshold_deg=DIAG_ROT_MEDIAN_DEG, min_agent_ticks=DIAG_MIN_AGENT_TICKS,
        status=("INSUFFICIENT_DATA" if n_mov < DIAG_MIN_AGENT_TICKS else
                ("PASS" if med < DIAG_ROT_MEDIAN_DEG else "FAIL")))
    # 4. reported only
    mi = d["dump_motion_invalid"].astype(np.int64) if T else np.zeros(0, int)
    nv = d["dump_n_viol"].astype(np.int64) if T else np.zeros(0, int)
    first = _sel("dump_first_step_m", (), np.float64)
    if A:
        v = np.hypot(boxes[:, 7], boxes[:, 8])
        mov = v > DIAG_MOVING_SPEED_MPS
        ratio = first[mov] / (MOTION_DT_S * v[mov]) if mov.any() else \
            np.zeros(0)
        probs = _softmax_rows(logits)
        ent = -(probs * np.log(probs + 1e-12)).sum(1)
        sig1 = _sigmoid(logits)[np.arange(A), top1]
    else:
        ratio = np.zeros(0)
        ent = np.zeros(0)
        sig1 = np.zeros(0)
    # tick-to-tick top-1 consistency by nearest-neighbour association
    consist_n = 0
    consist_same = 0
    if T >= 2:
        per_tick_boxes = d["dump_boxes"]
        per_tick_top1 = d["dump_top1"]
        for t in range(1, T):
            na, nb = int(n_boxes[t - 1]), int(n_boxes[t])
            if na == 0 or nb == 0:
                continue
            pa = per_tick_boxes[t - 1, :na, :2].astype(np.float64)
            pb = per_tick_boxes[t, :nb, :2].astype(np.float64)
            dist = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
            j = dist.argmin(axis=1)
            ok = dist[np.arange(na), j] <= DIAG_ASSOC_DIST_M
            consist_n += int(ok.sum())
            consist_same += int((per_tick_top1[t - 1, :na][ok]
                                 == per_tick_top1[t, j][ok]).sum())

    def _q(x):
        x = np.asarray(x, np.float64)
        if not len(x):
            return dict(n=0)
        return dict(n=int(len(x)), median=float(np.median(x)),
                    p10=float(np.percentile(x, 10)),
                    p90=float(np.percentile(x, 90)), max=float(x.max()))
    out["d4_reported"] = dict(
        motion_systematic_invalid_fraction=(float(mi.mean()) if T else
                                            float("nan")),
        n_viol_histogram={int(k): int(c) for k, c in
                          zip(*np.unique(nv, return_counts=True))} if T else {},
        first_over_half_v_ratio=_q(ratio),
        softmax_entropy=_q(ent), sigmoid_top1=_q(sig1),
        top1_consistency=dict(associated_pairs=consist_n,
                              same_top1_fraction=(consist_same / consist_n
                                                  if consist_n else
                                                  float("nan")),
                              association="nearest box centre <= "
                                          f"{DIAG_ASSOC_DIST_M} m (approx.)"),
    )
    # 5. median(l) > median(w) on vehicle rows
    veh = np.isin(labels, DIAG_VEHICLE_CLASSES) if A else np.zeros(0, bool)
    n_veh = int(veh.sum())
    med_l = float(np.median(boxes[veh, 4])) if n_veh else float("nan")
    med_w = float(np.median(boxes[veh, 3])) if n_veh else float("nan")
    out["d5_vehicle_lw"] = dict(
        n_vehicle_rows=n_veh, median_l=med_l, median_w=med_w,
        status=("INSUFFICIENT_DATA" if n_veh == 0 else
                ("PASS" if med_l > med_w else "FAIL")))
    # map readout (reported)
    if T and "dump_n_map" in d:
        n_map = d["dump_n_map"].astype(np.int64)
        mrows = _valid_rows(n_map, d["dump_map_scores"].shape[1])
        msc = d["dump_map_scores"][mrows].astype(np.float64)
        mlab = d["dump_map_labels"][mrows].astype(np.int64)
        mpts = d["dump_map_pts"][mrows].astype(np.float64)
        bnd = np.isin(mlab, BOUNDARY_LABELS) & (msc > BOUNDARY_SCORE_MIN)
        # straight-road readout: kept boundary polylines flanking the ego
        # (points ahead, |x| < 6 m) on the left (x<0) and right (x>0)
        tick_of_row = np.repeat(np.arange(T), n_map)
        left = np.zeros(T, bool)
        right = np.zeros(T, bool)
        for r in np.flatnonzero(bnd):
            p = mpts[r]
            ahead = p[p[:, 1] > 0]
            if len(ahead) == 0:
                continue
            xm = float(np.median(ahead[:, 0]))
            if -6.0 < xm < 0.0:
                left[tick_of_row[r]] = True
            elif 0.0 < xm < 6.0:
                right[tick_of_row[r]] = True
        out["map_reported"] = dict(
            kept_boundaries_per_tick=_q(d["dump_n_boundaries"]),
            boundary_score=_q(msc[np.isin(mlab, BOUNDARY_LABELS)]),
            all_map_score=_q(msc),
            label_histogram={int(k): int(c) for k, c in
                             zip(*np.unique(mlab, return_counts=True))}
            if len(mlab) else {},
            flanked_both_sides_fraction=float((left & right).mean()),
            left_only_fraction=float((left & ~right).mean()),
            right_only_fraction=float((~left & right).mean()))
    return out


def enablement_rule(diag: Dict[str, object]) -> Dict[str, object]:
    """FROZEN enablement rule (design 13 Stage F; executed at Stage G):
    diagnostics 1, 2, 3 PASS and 5 PASS on the bypass dump  =>  paper arm
    runs `NEWBORN_DT_PRED_TERMS=gated`; any non-PASS (FAIL or
    INSUFFICIENT_DATA)  =>  bug-fix loop or `off` with disclosure. Item 4
    and the map readout are reported, never gated."""
    checks = {k: diag[k]["status"] for k in
              ("d1_finite", "d2_crosskey", "d3_decode_rotation",
               "d5_vehicle_lw")}
    all_pass = all(v == "PASS" for v in checks.values())
    return dict(decision=("gated" if all_pass else "off"),
                checks=checks,
                rule="d1,d2,d3,d5 all PASS -> gated; otherwise off "
                     "(disclosed) or bug-fix; d4/map reported only")
