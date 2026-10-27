#!/usr/bin/env bash
# Example CLRE evaluation commands -- REFERENCE ONLY, NOT A RUNNABLE SCRIPT.
#
# This file records the shape of the paper's evaluation so it can be
# reconstructed. It is deliberately not executable as-is:
#
#   * every path below is a placeholder you must replace;
#   * the agent file is one YOU write, following integration/vad_integration.md
#     (the published wrapper embeds upstream code we may not redistribute --
#     see docs/LICENSE_AUDIT.md);
#   * each closed-loop command needs CARLA 0.9.15, a GPU and a checkpoint;
#   * one route takes minutes to tens of minutes; the full 220-route protocol
#     is a multi-day campaign, which is why the paper ran it in batches.
#
# It is guarded below so that running it by accident does nothing.
set -euo pipefail

if [[ "${CLRE_I_HAVE_EDITED_THIS_FILE:-0}" != "1" ]]; then
    cat <<'MSG'
This file is documentation, not a launcher.

Edit the placeholder paths for your machine, then re-run with
CLRE_I_HAVE_EDITED_THIS_FILE=1 if you really want to execute it.

Read first:
  README.md                       section 7, Example commands
  docs/REPRODUCIBILITY.md         protocol, hashes, mandatory configuration
  integration/vad_integration.md  how to build the agent file
MSG
    exit 0
fi

# ---------------------------------------------------------------------------
# 0. Paths -- replace all of these
# ---------------------------------------------------------------------------
CARLA_ROOT=/path/to/CARLA_0.9.15
B2D_ROOT=/path/to/Bench2Drive                 # commit 7ec25d1c...
ZOO_ROOT=/path/to/Bench2DriveZoo              # commit d9caa0af...
CLRE_ROOT=/path/to/CLRE                       # this repository
AGENT=/path/to/my_agents/clre_vad_agent.py    # YOU write this
VAD_CONFIG="${ZOO_ROOT}/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py"
VAD_CKPT=/path/to/ckpts/vad_b2d_base.pth      # from Bench2DriveZoo
OUT_DIR=/path/to/results

export PYTHONPATH="${CLRE_ROOT}:${ZOO_ROOT}:${B2D_ROOT}/leaderboard:${B2D_ROOT}/scenario_runner:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# 1. Confirm you have the same inputs as the paper (cheap, do this first)
# ---------------------------------------------------------------------------
sha256sum "${VAD_CKPT}"
#   expect 0f2d48b5c6cdb06d0dffb21612791fe740f3f5c7273b3b5e78e1f19e4a527eab
sha256sum "${B2D_ROOT}/leaderboard/data/bench2drive_0.0.4_val.xml"
#   expect 885debbc3d87f14351b521d8da0dc9d25fcee1f0d926ed1c92cc9ff62504ca45
sha256sum "${CLRE_ROOT}/clre/planning/refinement.py"
#   expect ff1df4f1f3055be09edd17c0776d9f02d2122beec09304fc20505a28b4c10236
git -C "${B2D_ROOT}" rev-parse HEAD   # 7ec25d1c9f7522d923ce5f3420986cef1cb2d956
git -C "${ZOO_ROOT}" rev-parse HEAD   # d9caa0af805ec3c435533aa268e2723d80c20017

# There are TWO different 220-route files in this ecosystem. bench2drive220.xml
# is NOT the official validation set and shares no routes with it. Check the
# hash above before trusting any comparison.

# ---------------------------------------------------------------------------
# 2. Offline checks -- no CARLA, no GPU, no checkpoint
# ---------------------------------------------------------------------------
( cd "${CLRE_ROOT}" && python -m pytest tests/ -q )

# ---------------------------------------------------------------------------
# 3. The frozen CLRE configuration -- every arm in the paper used exactly this
# ---------------------------------------------------------------------------
export NEWBORN_CANDA_MODE=exec
export NEWBORN_CANDA_DEVICE=cpu
export NEWBORN_CANDA_DTYPE=float64
export NEWBORN_CANDA_ITERS_COLD=70
export NEWBORN_CANDA_ITERS_WARM=25
export NEWBORN_CANDA_BUDGET_MS=1000.0
export NEWBORN_CANDA_FIXES=r1,r2,r3,geom,feas   # MANDATORY -- see below
export NEWBORN_PEBV2=1

# NEWBORN_CANDA_FIXES defaults to "none", which is NOT the paper's method:
# without `feas` there is no OBB hard feasibility gate at all, and without
# `geom` the ego is a single disk instead of three.

# ---------------------------------------------------------------------------
# 4. Main protocol: the official 220 validation routes, CLRE arm
# ---------------------------------------------------------------------------
# Run this in batches rather than as one invocation; the paper used 18 routes
# per batch in original XML order, resuming on routes that already have a
# summary. Expect infrastructure failures (CARLA segfaults) on a campaign this
# long: re-run the affected route, and never re-run a route for a better score.
python "${B2D_ROOT}/leaderboard/leaderboard/leaderboard_evaluator.py" \
    --agent="${AGENT}" \
    --agent-config="${VAD_CONFIG}" \
    --routes="${B2D_ROOT}/leaderboard/data/bench2drive_0.0.4_val.xml" \
    --checkpoint="${OUT_DIR}/clre_vad_official220.json" \
    --track=SENSORS \
    --port=2000 --traffic-manager-port=8000

# ---------------------------------------------------------------------------
# 5. Matched baseline: same routes, upstream planner alone
# ---------------------------------------------------------------------------
# Use the STOCK Bench2DriveZoo agent, unmodified. It is the matched baseline
# precisely because nothing of ours is in it.
#
# NOTE: the baseline carries no forward radar, so a CLRE arm with the execution
# layer enabled has one sensor more. That comparison is not sensor-matched;
# say so when you report it.
python "${B2D_ROOT}/leaderboard/leaderboard/leaderboard_evaluator.py" \
    --agent="${ZOO_ROOT}/team_code/vad_b2d_agent.py" \
    --agent-config="${VAD_CONFIG}" \
    --routes="${B2D_ROOT}/leaderboard/data/bench2drive_0.0.4_val.xml" \
    --checkpoint="${OUT_DIR}/native_vad_official220.json" \
    --track=SENSORS \
    --port=2000 --traffic-manager-port=8000

# ---------------------------------------------------------------------------
# 6. Ablation protocol: the first 36 routes of the SAME file
# ---------------------------------------------------------------------------
# Restrict to routes 1-36 in original XML order. The cohort was fixed before
# any run in the campaign and not selected on any outcome. Published arms:
#
#   no-progress-term     drop the forward-progress term from the objective
#   no-hard-OBB-gate     NEWBORN_CANDA_FIXES=r1,r2,r3,geom   (omit `feas`)
#   stock-PID execution  NEWBORN_PEBV2=0, i.e. CLRE refinement + upstream PID
#
# Each ablation is one factor changed against the full arm, same routes, paired
# by route.

# ---------------------------------------------------------------------------
# 7. Reading the results
# ---------------------------------------------------------------------------
# Driving Score is the primary endpoint. Closed-loop runs here are NOT
# deterministic: same-logic re-runs were measured to differ by up to 28 DS on
# individual routes. Compare distributions over the cohort, paired by route,
# with the sign test / bootstrap CI described in docs/REPRODUCIBILITY.md -- and
# never interpret a single-route difference on its own.
