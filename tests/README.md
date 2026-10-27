# Tests

Everything here runs offline: **no CARLA, no checkpoint, no GPU, no upstream
repository required.**

```bash
python -m pytest tests/ -q          # ~12 s
```

| File | What it covers |
| --- | --- |
| `test_refinement.py` | The CLRE core, carried over from the frozen research suite. Batched objective vs the reference implementation to float64 round-off; full-solve equivalence; commitment-gate behaviour (a stalled prior must not trap the ego, a committing prior must be preserved); physical clearance kept in conflict windows with no lateral/longitudinal cancellation; dynamic feasibility by construction; deterministic fallback signalling on non-finite inputs; warm-buffer shift semantics; warm seed never worse; waypoint resampling. |
| `test_execution_layer.py` | Radar supervisor (transparency, ground and overhead clutter rejection, TTC emergency persistence, fail-transparent layout check, non-closing transparency), route-centerline backup (ego-frame transform, route-end extrapolation, turning at intersections), and the PEB-v2 braking law (monotone in severity, closed-form ramp, latch contract, no planner state reachable). |
| `test_adapters_smoke.py` | UniAD / DriveTransformer adapters import standalone; upstream pins well-formed; no machine-local checkpoint path; adapters never import the refinement layer. |
| `test_release_hygiene.py` | Repo self-check: no machine-local paths, no credential-shaped strings, no large or binary artifacts, no vendored upstream source, every Python file compiles. |

## Tests that skip, and why

`test_refinement.py::test_native_desired_speed_matches_real_pid` compares
CLRE's reference-speed formula against the real upstream `PIDController`,
which is not redistributed here. It **skips with a printed reason** when
Bench2DriveZoo is absent from `PYTHONPATH`. To run it:

```bash
PYTHONPATH=/path/to/Bench2DriveZoo:$PYTHONPATH python -m pytest tests/ -q -s
```

## Not in this release

The adapters' full offline suites (24 fixture groups each, plus AST parity
checks) assert against `sha256`-pinned upstream files that cannot be
redistributed. See `docs/LICENSE_AUDIT.md`.

Nothing here validates closed-loop driving. Any Driving Score number requires
CARLA, a GPU and a checkpoint — see `docs/REPRODUCIBILITY.md` §6.
