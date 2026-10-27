"""CLRE -- Closed-Loop Refinement and Execution for learned driving planners.

Layout
------
clre.planning   receding-horizon refinement: objective, candidate seeds,
                warm start, OBB feasibility gate, candidate selection
clre.control    execution layer: forward-radar speed bound, saturated
                proportional-error braking, route-centerline fallback
clre.adapters   pure data adapters mapping a learned planner's raw head
                outputs onto the refinement layer's frozen array schema
clre.tracing    buffered per-tick npz trace writer used by the analyses

CLRE never retrains or modifies the upstream planner; it consumes that
planner's nominal trajectory and its predicted agents, and returns the
trajectory that is actually executed. See integration/ for how to attach it
to VAD, UniAD or DriveTransformer, none of which are redistributed here.
"""

__version__ = "1.0.0"
__all__ = ["planning", "control", "adapters", "tracing"]
