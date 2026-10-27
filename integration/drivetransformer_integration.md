# Integrating CLRE with DriveTransformer-Large

Read [`vad_integration.md`](vad_integration.md) first, then
[`uniad_integration.md`](uniad_integration.md) — this adapter inherits several
contracts from the UniAD one rather than re-declaring them.

**Status.** Additional demonstration. Evaluated on a small route subset, not
the full benchmark.

> ### Licence warning — read before you redistribute anything
>
> The DriveTransformer repository at
> `19702015eb700e804bafb53e63b04571c7b8604a` contains **no `LICENSE` file and
> no licence statement**. With no grant, the default is all rights reserved.
>
> * The adapter in this repository is original CLRE code and contains no
>   DriveTransformer source, so it is safe to publish.
> * The **agent wrapper is not published**, and you should not publish yours
>   either: ~20 % of its lines come from
>   `team_code/drivetransformer_b2d_agent.py`.
> * Obtaining and using the upstream model is between you and its authors.
>
> See [../docs/LICENSE_AUDIT.md](../docs/LICENSE_AUDIT.md).

---

## 1. Environment

DriveTransformer ships its **own fork** of the merged `mmcv`, with its own
compiled CUDA ops. It is not interchangeable with Bench2DriveZoo's.

* Build DriveTransformer's in-place CUDA extensions per its own instructions.
* Put the DriveTransformer root on `PYTHONPATH`, and **keep the
  Bench2DriveZoo root off it** for these runs. Both define a top-level `mmcv`;
  whichever appears first wins, and a mixed resolution fails in ways that look
  like model bugs rather than import bugs.
* Use a separate Python environment from the VAD/UniAD one.

## 2. What the adapter does

[`clre/adapters/drivetransformer_adapter.py`](../clre/adapters/drivetransformer_adapter.py)
converts `forward_test` outputs into the frozen `ag_*` schema and the
refinement layer's input forms. Every constant in it is either inherited
unchanged from the UniAD contract (imported from `uniad_adapter`, never
re-typed) or a structural fact of the pinned release — shapes, keys, class
lists, coder ranges. None is a tunable.

It also **replicates the released coder's top-k index law**, so the
motion-mode logits captured by your forward hook stay row-aligned with
`boxes_3d`. This is the part most likely to break silently on a different
release: a misalignment attaches one vehicle's predicted motion to another's
box, and the refinement layer will dutifully avoid the wrong future.

```python
from clre.adapters import drivetransformer_adapter as dta

dta.require_upstream(root)                  # 15 sha256-pinned files, hard stop
prior   = dta.extract_prior(out0)           # (6, 2) nominal plan
rep     = dta.replicate_coder_indices(outs, coder, denormalize_fn=...)
dta.assert_hook_alignment(out0, rep)        # fail loud on misalignment
agents  = dta.extract_agents(out0, logits)
bnds_in = dta.extract_boundaries_inputs(out0)
```

`load_pinned_denormalize(root)` path-loads `denormalize_bbox` from the pinned
DriveTransformer tree at runtime. That is a dynamic import of code you already
have, not a copy — which is exactly why the adapter is publishable.

## 3. The ego prior

DriveTransformer's ego head emits a **`fix_time` prior**: 30 rows at 0.1 s
spacing, covering 0.1–3.0 s. The refinement layer wants 6 points at 0.5 s, so
the adapter selects rows `[4, 9, 14, 19, 24, 29]` ↔ `t = 0.5 … 3.0 s`. This is
**row selection, never interpolation**.

The release is single-mode for the ego head (`DT_FUT_EGO_MODE = 1`), so there
is no mode-selection step on the prior.

There is also a `fix_dist` head. Do not use it as the prior: the time-indexed
rows are what the 0.5 s contract requires.

## 4. Motion-head validity gating

DriveTransformer's auxiliary motion head is gated **per tick** by an inherited
systematic-signature check. When the gate trips, the caller withholds the
agent set entirely:

```python
if not dta.pred_terms_gate(policy, motion_systematic_invalid):
    agents = dta.withhold_agents(agents)     # sets ag_valid[:] = False
```

`pred_terms_gate` is a stateless pure function. Withholding is deliberate and
conservative: with no trustworthy predictions, the refinement layer keeps its
trajectory-level and ego-motion terms and simply has no interaction term that
tick — which is honest. It does **not** mean "no obstacles", and it does not
suppress the radar supervisor, which runs on measured returns regardless.

## 5. Downstream is unchanged

The OBB gate, selection, centerline backup, radar bound and PEB-v2 are
backbone-agnostic. Wire them exactly as in the VAD guide.

## 6. Diagnostics and enablement

The adapter exposes `passive_diagnostics(dump)` and `enablement_rule(diag)`.
Run them on a recorded forward dump **before** any closed-loop campaign: they
check the properties that were validated against real forward passes rather
than synthetic fixtures — instance-level `l > w` on vehicle rows, yaw-versus-
velocity alignment of the first displacement, the `‖traj6[0] − pos‖`
distribution, and map-polyline quality.

If you port this to another DriveTransformer release, re-check all four, and
do not weaken the assertions to make synthetic tests pass. A loud adapter
failure costs you a run; a quiet one costs you a conclusion.
