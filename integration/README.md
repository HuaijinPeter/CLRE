# Integration guides

CLRE attaches to a frozen learned planner through **one agent file that you
write**, in your own checkout. These guides tell you exactly what that file
has to do.

## Why you write it instead of cloning it

The published version of each agent subclasses the upstream Bench2Drive agent
and reuses some of its lines verbatim — input construction, the native control
path, the controller's steering block. Bench2Drive and Bench2DriveZoo are
licensed CC BY-NC-ND 4.0, whose NoDerivatives term does not grant the right to
redistribute modified copies, and DriveTransformer publishes no licence at
all. So those files cannot ship here. See
[../docs/LICENSE_AUDIT.md](../docs/LICENSE_AUDIT.md).

What these guides give you instead: the insertion points, the data contract at
each boundary, and the invariants that must hold for a run to be a valid
comparison. The upstream lines they refer to are already in your checkout.

## Guides

| Guide | Backbone |
| --- | --- |
| [`vad_integration.md`](vad_integration.md) | **VAD** — the paper's main evaluation backbone. Read this one first; the others are deltas against it. |
| [`uniad_integration.md`](uniad_integration.md) | UniAD |
| [`drivetransformer_integration.md`](drivetransformer_integration.md) | DriveTransformer-Large |
| [`peb_v2_controller_patch.md`](peb_v2_controller_patch.md) | The execution layer's controller, for any backbone. |

## The five invariants

These hold for every backbone. Break any one of them and the arm is no longer
the method the paper evaluates.

1. **Everything upstream of planning is untouched.** Same input construction,
   one forward pass, same branch selection. The only control-path change is
   *which trajectory reaches the controller*.
2. **One forward pass per tick.** Read the planner's agent predictions from a
   forward hook on the existing pass; never run the model twice.
3. **The refinement layer fails to the prior, never to a stop.** Non-finite
   objective, hard-bound violation, budget overrun or any exception ⇒ execute
   the upstream plan this tick and log the reason.
4. **The speed PID is stepped exactly once per tick**, unconditionally, on
   every branch. See [`peb_v2_controller_patch.md`](peb_v2_controller_patch.md).
5. **The fallback hold speed is latched at episode entry**, not re-read each
   tick. Re-reading turns "hold speed" into a silent coast-to-stop.

## Checking your wiring without CARLA

Before spending GPU hours, confirm the offline suite passes against your copy
of CLRE, and that the frame convention you implemented matches the one the
centerline module assumes:

```bash
python -m pytest tests/ -q
python -m pytest tests/test_execution_layer.py -q -k "centerline_frame"
```

The frame check is the one that catches the most expensive class of mistake:
a sign error in the world→ego transform produces a plausible-looking run whose
fallback steers the wrong way.
