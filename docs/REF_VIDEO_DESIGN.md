# Reference-video conditioning for Stage-1 identity anchoring

## Problem
Long runs (7+ chunks) morph: objects change identity across Stage-1 chunks even
though every boundary is smooth (boundary_sim ≥ 0.95, S2 fidelity ≥ 0.96 — the
drift is created in S1 and faithfully preserved downstream).  The SLB carries
*local* continuity; nothing carries *global* identity.  The anchor bank only
corrects statistics (AdaIN), not content.

## Mechanism (proven for audio)
ComfyUI's `LTXAVModel` already implements exactly the needed mechanism — for
audio only ("ID-LoRA in-context conditioning"):

1. **Inject** (`_process_input`, av_model.py ~685-712): prepend `ref_audio["tokens"]`
   to the token sequence with temporal RoPE coords offset to end just before
   t=0 (`ref_pos = positions − (end_of_last + time_per_latent)`), record
   `ref_audio_seq_len` + `target_audio_seq_len` in `additional_args`.
2. **Timestep zero** (forward, ~754-761): expand the per-token timestep and
   concat `zeros(B, ref_seq_len)` in front — reference tokens are "clean".
3. **Trim** (`_process_output`, ~1000-1005): drop the first `ref_seq_len`
   tokens (and their embedded timesteps) before unpatchification.

`ltx-core` has the video twin as `reference_video_cond.py`; ComfyUI does not.

## Video port — the three patch sites

| Site | Audio (exists) | Video (to add) | Complication |
|---|---|---|---|
| inject | after `a_patchifier.patchify` | after `super()._process_input` returns `vx, v_pixel_coords` | video coords are (t,h,w) pairs — offset ONLY the t component negative; spatial coords must match the chunk's H×W grid |
| timestep | `a_timestep` expand+concat | video uses `CompressedTimestep(..., v_patches_per_frame, per_frame=...)` built from `ts_input` | must prepend one zero-frame entry to `ts_input` BEFORE compression, and bump the frame count consistently; naive token-concat breaks the per-frame compression |
| trim | `ax[:, ref_len:]` | `vx[:, ref_len:]` before `super()._process_output` | `super()._process_output` also strips keyframe tokens via `keyframe_idxs`; ref tokens must be removed FIRST and must not shift keyframe token indices |

## What to feed as reference
Per chunk ≥ 2: the active anchor-bank frame(s) (nearest anchor to the current
scene) patchified at the chunk's spatial size — 1-2 latent frames ≈ 250-500
tokens.  For i2v, chunk-0's guide-adhering first frame is the natural primary
anchor.  Update the reference when the bank banks a new anchor (intended scene
changes keep working; identity within a scene is pinned).

## Risk assessment / why this is NOT shipped blind
- `CompressedTimestep` per-frame packing and the keyframe trim ordering are the
  two places a blind patch most likely breaks silently (wrong RoPE → identity
  tokens land inside the chunk's timeline → ghosting; wrong trim → shifted
  output frames).
- Needs iterative on-machine testing: start with `ref_video_seq_len` logged,
  a 2-chunk run, and verify (a) output frame count unchanged, (b) chunk-2
  identity_sim rises vs baseline, (c) no ghost of the anchor visible at t=0.

## Implementation plan
1. Monkey-patch module `ref_video_patch.py`: wrap `LTXAVModel._process_input`,
   `forward` (timestep region via a targeted helper if separable — else patch
   `ts_input` construction), `_process_output`.  Feature flag
   `transformer_options["clss_ref_video"]` carrying `{"tokens", "pixel_coords"}`.
2. Node side: CLSSStreamingSampler builds the reference from the anchor bank,
   passes it via model_options for chunks ≥ 2 (flagged input, default off until
   validated).
3. Validation run matrix: 7 chunks × {off, on}; read identity_sim@chunk-7 and
   eyeball the tree.
