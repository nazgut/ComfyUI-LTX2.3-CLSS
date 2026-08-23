# AGENTS.md

Guidance for AI coding agents working in this repository. Read this before making any change.

## Project overview

**ComfyUI-LTX-CLSS** is a **ComfyUI custom-node package** implementing CLSS (Closed-Loop
Streaming Synthesis) — arbitrary-length audio-video generation with the LTX-2.3 22B model on
consumer 16 GB VRAM hardware. It is loaded by a ComfyUI installation (this repo lives at
`ComfyUI/custom_nodes/ComfyUI-LTX-CLSS`; `../../main.py` is the ComfyUI entry point).

CLSS generates video in short temporal **chunks** sharing an **SLB** (streaming latent buffer)
overlap, keeping latent memory O(overlap) instead of O(length). Between chunks it applies
closed-loop corrections that fight exposure-bias drift **without modifying transformer weights**:

- **§2.1** calibrated context re-noising (`tau_c`, default 0.05)
- **§2.3** EMA per-channel AdaIN drift correction (`beta`, default 0.4)
- **§2.5** dynamic anchor bank (long-range identity tracking)
- Two-band spatial detail anchor; hot audio SLB re-noise schedule (3× `tau_c`, ceiling 0.35)
  kills the audio "metronome" **at 16-lf chunks (the regime where the lock was ear-confirmed);
  at 52-lf chunks the env_corr climb is NOT a metronome — the 2026-08-15 continuation run
  (tau_mult=1.0 + ref 8s + audio_stg=0) is free of the gesture by ear: the audible failure
  there is WIND (broadband airy noise), see the split section below** — the per-chunk overlap
  phase-jitter lever was built and removed (it fought the SLB's structural assumptions);
  cross-chunk audio content context is the SLB + `ref_audio` at negative RoPE (fixed
  one-overlap window — the `ref_audio_seconds` knob was removed 2026-08-19; §2.4
  frequency-band soft shrinkage was removed the same day: no purpose)

Audio and video are jointly modelled; audio needs higher CFG (~7) than video (~4), and audio
drift is a recurring failure mode (see the extensive per-input tooltips in `nodes.py`).

The theory and results are in `paper/clss_paper.pdf`; `paper/*_logs.txt` are captured
per-chunk telemetry from real runs (16-lf matched pair, split validation, open/closed-loop
arms, wind continuation), used to calibrate the simulations and corrections.

## Technology stack and runtime architecture

- **Language:** Python (node layer targets the host ComfyUI's interpreter; the inner repo targets
  Python 3.11 via ruff config).
- **No package manifest at the outer layer** — there is no `pyproject.toml`/`package.json` in the
  repo root. ComfyUI imports the package directly.
- `__init__.py` prepends `Ltx-2-CLSS/packages/{ltx-core,ltx-pipelines}/src` to `sys.path`, then
  exports `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` from `nodes.py`. **All node code is
  in that single ~2700-line file.**
- The CLSS algorithm itself lives in the submodule at
  `Ltx-2-CLSS/packages/ltx-pipelines/src/ltx_pipelines/streaming/clss.py` (`CLSSConfig`,
  `CLSSState`); `nodes.py` imports it and orchestrates it against ComfyUI's sampler/guider
  infrastructure (`comfy.samplers`, `comfy.model_patcher`, `comfy_extras.nodes_lt`, …).
- **Hardware model:** ~16 GB VRAM, ~48 GB system RAM. The 22B checkpoint is dequantized to BF16 in
  pinned CPU RAM and transformer blocks are streamed to the GPU (GGUF Q4_K_S + CPU
  block-streaming). `nodes.py` sets `PYTORCH_ALLOC_CONF=expandable_segments:True` before importing
  torch — keep that line above the torch import.
- **The 20 s native-window wall (measured + code-verified):** AV RoPE positions are in SECONDS,
  computed per forward pass, with `max_pos=20` for both modalities
  (`comfy/ldm/lightricks/av_model.py:394-395`, video converted at `model.py:898`). Any single
  window longer than ~20 s (audio >500 af, video >60 lf at 24 fps) runs off the positional grid:
  audio glitches at ~20.5 s (measured 20.46 s transient, latent peak af 512-513/660, same
  position across seeds and video resolutions) and video goes near-static. This is why chunked
  CLSS works (every window <20 s) and why single-window draft runs are capped at ~20 s —
  longer soundtracks need chained ≤20 s drafts via upstream `LTXVReferenceAudio` (ref tokens at
  negative RoPE, av_model.py:686-709) + `LatentConcat` on the audio latents. Keep Stage-2's auto
  chunking under 60 lf for the same reason (53 lf at current budget — fine).
- **The sampler AUTO-ADJUSTS the per-chunk window to the wall (2026-08-22).** `CLSSStreamingSampler`
  used to only WARN when overlap+new exceeded ~20 s — the user's 121-px (5 s) chunks worked, longer
  chunk windows broke (the 409-px production config sits right ON the 20.0 s edge). Now the window
  is enforced automatically (`_ROPE_WALL_S`/`_ROPE_WALL_MARGIN_S`/`_MIN_OVERLAP_LF` in nodes.py):
  (1) the overlap is clamped down so overlap+new ≤ 19.5 s (the user's chunk length and chunk count
  are preserved, only the SLB context shrinks); (2) if even a zero overlap exceeds the wall (one
  chunk alone > ~19.5 s), each chunk is AUTO-SPLIT into the fewest uniform sub-chunks that fit —
  any requested length works. The per-chunk plan (`_chunk_plan`) feeds video/audio new-lengths,
  cumulative noise-slice positions and the SLB/ref ledger; `clss_config.overlap_latent_frames` is
  replaced so `update_buffer` stores exactly what each chunk places. At the production 409-px/8-lf
  config the overlap is clamped 8→6 (window 19.3 s). Stage 2's AUTO chunking is likewise capped by
  the wall (16 = max auto s2_overlap); a manual over-wall `frames_per_chunk` is kept but loudly
  warned.

## Repository layout

```
nodes.py              # all 7 ComfyUI node implementations (~3450 lines)
__init__.py           # sys.path injection + ComfyUI node-mapping exports
workflow/             # exactly two canonical workflows: i2v_LTX_CLSS.json and
                      # t2v_LTX_CLSS.json, both with the validated production config
                      # baked in (4×52 lf, split 0.394 AUTO, tau_mult 1.0, audio_stg=0,
                      # dense sigmas; ref_audio_seconds / audio_refine removed
                      # 2026-08-19).  RULE: every
                      # experiment copies the canonical file into its own workflow,
                      # runs it, and deletes it after the result is recorded in this
                      # file — never mutate a canonical workflow in place.
paper/                # CLSS paper (LaTeX + PDF) and captured run telemetry
simulations/          # standalone pure-math diagnostic scripts (no GPU needed)
Ltx-2-CLSS/           # git submodule (github.com/nazgut/Ltx-2-CLSS) — a fork of
                      # Lightricks' LTX-2 monorepo with the CLSS additions:
                      #   packages/ltx-pipelines/.../streaming/  ← the CLSS algorithm
                      #   generate_clss.py                     ← standalone CLI
                      #   convert_gguf.py                      ← GGUF→BF16 pre-conversion
```

### `Ltx-2-CLSS/` is a git submodule with its own history

It is a registered submodule (see `.gitmodules`) pointing at
`github.com/nazgut/Ltx-2-CLSS.git` — a fork of Lightricks' LTX-2. Clone with
`--recurse-submodules` (or run `git submodule update --init`). **Commit ComfyUI-node changes in
this repo and algorithm changes in the submodule**, then bump the submodule pointer here. The
outer repo records only the submodule's commit pointer (a `m Ltx-2-CLSS` in `git status` means
the inner repo has unpushed/uncommitted changes). Sub-package architecture is documented in
`Ltx-2-CLSS/packages/ltx-pipelines/CLAUDE.md`.

## The nodes and how they wire together (Stage 1)

```
LTXVideo Loader → MODEL, VAE, CLIP
CLSSScenePrompts(CLIP, prompts)          → CONDITIONING   (one entry per scene, split by a line of '---')
LTXVConditioning(positive, negative, fps)→ positive, negative
CLSSAVGuiderV2(model, pos, neg, ...)     → GUIDER         (split video/audio CFG + modality + STG; replaces CFGGuider→CLSSAVGuider)
EmptyLTXVLatentVideo + audio latent + LTXVConcatAVLatent → LATENT  (per-chunk AV template)
CLSSConfig                               → CLSS_CONFIG
CLSSStreamingSampler(GUIDER, SAMPLER, SIGMAS, NOISE, LATENT, CLSS_CONFIG, num_chunks, [image, vae]) → LATENT
```

Two-stage pipeline: Stage 1 (`CLSSStreamingSampler`) → `LTXVLatentUpsampler` (2× spatial) →
`CLSSStage2` (chunked distilled-LoRA refinement, same SLB continuity mechanism) →
`CLSSVideoDecodeSave` (streaming temporal-slice decode → PNG frames; never materializes the
whole decoded video — the old VAEDecodeTiled path pre-allocated ~18 GB for a 4×52-lf run).
See the two files in `workflow/` for the canonical wiring (i2v with guide image, t2v text-only).

Optional sampler inputs worth knowing: (`audio_slb_tau_mult` and the hot audio SLB
re-noise schedule were REMOVED 2026-08-20 — the audio SLB is now placed FROZEN,
mask=0, no tau_c on audio at all); `audio_shift_mult`
(default 0 = AUTO: audio shift = min(the connected video shift, 1.844) — the
EAR-VALIDATED target (2026-08-11 A/B; the 2.05/2.059 target failed live 3× —
grainy / deflated / noise-dominant, so the 2026-08-20 "reference parity" 2.05
ceiling was REVERTED 2026-08-22 even though the reference LTX2Scheduler runs
2.05 with a disconnected latent); at ≤121 px frames the video shift is ≤1.844
so the shared schedule is kept byte-identical, above it the AUDIO alone is
pinned to 1.844 via the v3 split while the VIDEO
keeps the token law — its motion needs the higher shift at long windows (a
whole-schedule 2.05 cap static-killed 31-lf video, reverted 2026-08-20); manual
values ≠1 = raw multiplier); 1.0 = off (audio uncapped);
`video_slb_tau_mult` (default 1.0) scales the VIDEO overlap re-noise: 0.0 = frozen
clean seam (built against the 2026-08-16 seam-clustered layout events — chunk 2→3,
t≈34.7 s, Δlayout −0.32 in 3 frames, visible as jumping/morphing; UNVALIDATED);
`auto_seam_pin` (default off) = EVENT-TRIGGERED seam pin: when a chunk contains a
coarse-layout JUMP (max per-frame layout-sim drop > 0.25 or min < 0.05), the NEXT
chunk's overlap is placed FROZEN — continue from the LAST GOOD FRAMES (the fresh
SLB), not the 1-2-chunk-old anchor bank.  Chunk 1 is excluded from the trigger
(scene establishment, not a jump).  Cannot protect the final chunk.
**The layout metric was rebuilt 2026-08-17** (sign-free coarse power-map cosine:
channel-mean → square → hard 3×3 low-pass) after the pre-v2 pooled-cosine metric
FALSE-POSITIVED: the pin-on runs fired on single-frame dips (t≈35.7 s, Δ−0.55)
that are ABSENT from the decoded video (author-verified; origin_sim at the same
frames shows no excursion).  Pre-v2 layout-drift event tables therefore
OVER-REPORT — entries are candidate events, not confirmed ones.  The pin's
behavior against a REAL visible event (like the 2026-08-16 t≈35 s / t≈63 s
jumps, which were real) remains UNVALIDATED; thresholds are uncalibrated on the
new metric.  Telemetry: paper/t2v_auto_seam_pin.txt, paper/i2v_auto_seam_pin.txt.
`object_identity` (production: on) adds the decoded seam-grid identity metric.
Soundtrack-draft mode (`soundtrack`, `soundtrack_tau`, `CLSSDraftLength`) was
REMOVED (2026-08-16) — a frozen draft has no audio feedback of the video; do not
re-add it.

### Per-modality audio shift — KEEP (production)

`audio_shift_mult` gives the audio its own sigma schedule (the audio repair: without it
the audio inherits the video's top-heavy 4.68-shift schedule at 52-lf chunks and comes
out dull — the dull-audio week of 2026-08-13/14).  The 1.844 audio target was
audible-validated in audio-only mode (2026-08-11); the 2.059 target failed live 3×
(grainy / deflated / noise-dominant).  A 2026-08-13 AV joint run with mult=0.39
deflated the audio (S1 std 0.43→0.30) — but that run also carried audio_stg=1, and
the clean retest (2026-08-14, i2v 2×52 lf, mult=0.394 + audio_stg=0) is the VALIDATED
production config: audio raw std 0.497 (vs 0.42 on the shared 4.68), video untouched
(vid_std 1.013, vid_bnd 0.967, byte-identical 4.68 sigmas), S1 aud_bnd 0.692, S2
aud_refine_sim 0.70–0.72, final audio std 0.540, ear-accepted.  Telemetry
`paper/i2v_audiosplit_logs.txt`.

Open/closed A/B at 52 lf (telemetry `paper/i2v_openloop_split_logs.txt`,
`paper/i2v_closedloop_split_logs.txt`): without corrections drift returns (vid_std
+0.118/5 chunks, aud_rms +34%, aud_hf →1.60× chunk-1 ref, aud_env −0.05→+0.59); with
corrections vid_std +0.001, object identity 0.980–0.995 (min cell 0.79 @ chunk-2 seam),
aud_rms +0.087 (upward-only anchor by design), g_SLB 24.8/28.8/42.4 (the 15–87 band of
the 16-lf paper table).  A/B caveat: same-seed runs at different total lengths do NOT
share a prefix — the pre-generated noise layout depends on total length.

The continuation experiment (2026-08-15, audio_slb_tau_mult=1.0 + ref_audio_seconds=8.0
+ audio_stg=0, 4×52 lf, split 0.394, dense S2 join 0.55) validated the anti-metronome
lever set: no metronome by ear — the audible failure is WIND (broadband airy noise);
aud_env still climbs (−0.078→0.682, below the 0.7 flag) but at 52-lf chunks env_corr
tracks the continuity context's shared noise-like envelope, not a replayed gesture —
the hot tau_c schedule was chasing a non-gesture, and the metronome is a 16-lf-regime
diagnosis.  **#1 audio failure at >3 chunks is now WIND, unisolated** (candidate
carriers: S2 join re-invention — aud_refine_sim 0.58–0.64 ≈ 40% re-roll per window
[eliminated by construction 2026-08-19: audio_refine removed, S2 audio is frozen];
the spectral-pinned continuity context; S1-native half-scale audio, per-chunk std
0.345–0.395).  The rest of the run is clean (vid_std Δ+0.000, vid_bnd 0.974–0.981,
vid_obj 0.993–0.996, aud_rms 0.358→0.362, SLB honored 1.0, aud_hf 1.095→0.902 falling,
peaks unlocked); S2 seams aud_bnd 0.295–0.468 stay WARN.  Telemetry
`paper/i2v_continuation_wind_logs.txt`.

**ROOT CAUSE of the dull-audio week** was the 52-lf template pushing the LTXVScheduler
shift to 4.68 (vs 1.844 at 16 lf — see the convention below): feeding the scheduler a
separate 16-lf template fixed the audio but STATIC-killed the video (2026-08-14) — the
split gets both.  A LEAD CAP (`audio_shift_lead`) was built against the grain and
REMOVED the same day after its first live validation DEFLATED the audio (std 0.42 →
0.24): capping how far the audio runs ahead compresses exactly the low-σ phase where
the joint model builds audio structure.  Do NOT rebuild a lead cap; the audio needs its
full low-σ descent (`simulations/audio_joint_schedule_sim.py` records it as a dead end).
Chunk-1 onset handling: smooth 3.5σ→4σ tanh limiter + 8-frame fade (the old ±4σ clamp
square-topped the onset and let a 5.7σ cluster through as a click).

### Two-pass audio repair (frozen video) — REJECTED, do not rebuild (2026-08-13)

A `frozen_video` input that re-ran the audio against the frozen main-pass video
was built and removed the same day: it doubles Stage-1 compute (~2× total
pipeline), and the user rejected the cost.  The asymmetry it attacked is real —
the video gets a dense low-σ descent in Stage 2 and the audio does not — but
the Stage-2 audio-refine lever built on exactly that idea was itself removed
2026-08-19 (see below); the asymmetry is currently UNADDRESSED.

### Stage-2 audio refine — REMOVED 2026-08-19 ("not working")

`audio_refine` / `audio_join_sigma` on `CLSSStage2` and the `_make_s2_join_sampler`
custom per-step-mask sampler were deleted; Stage-2 audio is frozen passthrough
(mask=0), bit-exact Stage-1 output.  Why it existed: the VIDEO gets a dense low-σ
descent in Stage 2 (re-noised to the manual sigmas and re-denoised) while the audio
passed through FROZEN, and Stage 1's own low-σ tail at the 4.68 shift is only ~2
steps (…→0.5372→0.1→0) — the measured dull-audio asymmetry.  Why it failed: full
refinement from σ0=0.909 DESTROYED the audio (aud_refine_sim 0.07–0.16 — a
melody/timbre/voice is not in the text prompt, so re-inventing it from 91% noise
loses it; the video survives because its composition is prompt-specified and the
detail anchor re-anchors it), and the join-sigma fix (audio frozen above the join,
explicit noise-up at it; 0.725 = 27.5% Stage-1 signal kept / 2 matched steps,
0.55 = 45% / 3 on dense sigmas) reached aud_refine_sim 0.58–0.64 in the 2026-08-15
run but never earned an ear win — the wind failure remained.  Do not rebuild
without a live ear-validated win.

### Audio-continuity port audited against the reference (2026-08-13)

The ComfyUI port of the reference streaming pipeline's audio continuity
(`Ltx-2-CLSS/.../streaming/pipeline.py:480-850`) was diffed line by line after two
bad-audio live runs: **slice offsets, ref window, ref token layout, negative-RoPE
convention and boundary smoothing are FAITHFUL — no transcription bug.**  What the
port ADDs (no reference counterpart) is a stack that structurally pins all audio to
chunk-1's statistics — it does not create the dullness (chunk-1's RAW output is
already std ≈0.3–0.45; that is upstream, the joint-pass schedule), it locks it in:

1. RMS anchor (`audio_anchor` — since 2026-08-19 FIXED to rms_only, the knob and the
   rms_dc arm are gone) WAS a hard bidirectional
   gain to the capped EMA target; the reference (pipeline.py:741-757) is soft,
   upward-only — it never attenuates a louder chunk.  **RELAXED 2026-08-14 to the
   reference direction: boost quiet chunks, keep louder chunks raw** (measured cuts
   g=0.93/0.97 on brighter chunks in the 2026-08-13/14 logs).  Exercised live in the
   2026-08-15 continuation run (4 chunks, aud_rms 0.358→0.362, one upward trim
   g=1.04) — the RMS×4 runaway the hardening was built for (15-chunk run) remains
   unsuppressed; watch the aud_rms trend on long runs.
2. The RMS reference is measured post-fade/post-limiter, onset-excluded — a low target.
3. SLB spectral normalization re-imposes chunk-1's per-bin spectrum on the context
   every chunk (nodes.py:1999-2011).
4. `_flatten_audio_env` (gains [0.6, 1.25]) compresses the dynamics of all SLB/ref
   context — loud frames attenuated up to ×0.6.  Kept on CONTEXT (video protection,
   2026-08-05), **removed from chunk-1's OUTPUT 2026-08-14** (the chunk-1 boundary
   correction's step 4 hit the ×0.6 clamp on every logged run — a direct dynR cut).
5. The final assembly spectral norm re-imposed chunk-1's spectrum AND RMS on the
   whole concatenated output (nodes.py:2087-2122).  **RELAXED 2026-08-14 to
   upward-only** (per-bin gains floored at 1.0, RMS pin boosts only): the
   2026-08-13/14 runs had chunks boosting mid bins 1.27–1.54× and this norm cut
   them to chunk-1's duller spectrum (gains 0.82–0.86).  Anti-drone direction
   (HF decay → boost back) preserved; a genuine LF rise now passes through —
   gate on aud_hf.

### Audio latent scale — measure before you tune (2026-08-09)
The audio VAE's `AudioLatentNormalizer` is calibrated so **real audio lands at unit
variance**: encoding four real music tracks gives overall latent std 0.98–1.26 with the
per-frequency-bin std flat at 1.00 (0.94–1.07 across all 16 bins). Generated audio does not
get there, and the shortfall predicts how bad it sounds:

| source | latent std | ro99 | L/R corr |
|---|---|---|---|
| real music (4 tracks) | 0.98–1.26 | 8–12 kHz | 0.69–0.87 |
| gen 02744 (per-chunk, 26 s) | 0.84 | 6.1 kHz | 0.93 |
| gen 02742 (per-chunk, 20 s) | 0.94 | 4.0 kHz | 0.98 |
| gen 02740 (single window, 106 s) | 0.63 | 3.7 kHz | 0.999 |
| draft_00001 (single window, 104 s) | 0.48 | 3.1 kHz | 1.00000 |

Half scale in a log-mel space decodes to exactly the reported complaint: quiet (−31.6 dBFS
against −21…−16 for the reference tracks), low-passed near 3 kHz although the mel ceiling is
8 kHz, and at the longest windows L/R collapsed to *identical* channels.

Contrast is **not** the deficit, and assuming it is sends you the wrong way: the draft's
0.25 s envelope range (8.1 dB) already exceeds real music's (4.3–6.3 dB) and its log-mel
contrast (1.35) exceeds real music's (1.00). Only level and treble are short. Measured in
mel space, a per-(channel, freq-bin) MEAN shift toward real music fixes both at once
(log-mel mean −4.39 → −3.26, tilt −2.49 → −1.97, against real music at −2.87 / −1.55), while
matching the latent STD does literally nothing (−4.40 / −2.54). Pooling the profile over
channels loses most of the effect — the channels differ enormously.

**Do not go hunting for a decode or sample-rate bug** — the VAE is innocent. Real music
round-trips through it with bandwidth (9.2 → 7.6 kHz) and stereo intact. The chain is
16 kHz / 64-mel (fmax 8 kHz) → vocoder → BWE → 48 kHz, and the written sample rate is correct.

The trade this exposes: the single-window soundtrack draft buys global planning and does kill
the metronome loop, but the two longest-window runs are also the two most mean-regressed,
while the run that scores highest on these measures came from a ~26 s per-chunk window.
(Scores highest — nobody has A/B'd them by ear; latent metrics localize failures, they never
prove a quality win.)
A mid-length draft window is the untested middle ground.

Measure any take with `python simulations/audio_latent_probe.py FILE --reference` (audio VAE
only, CPU, no transformer — runs while ComfyUI holds the GPU). Gate on latent std (<0.60 =
hedged, >0.80 = good), L/R corr (>0.999 = mono collapse) and ro99.

**A post-hoc statistical repair does NOT work — do not rebuild it.** A `CLSSAudioRestore`
node shifting the audio latent's per-(channel,bin) mean toward a real-music profile was
built and deleted the same day (2026-08-10). Offline it looked like a clear win: ro99
3141 -> 4477 Hz, level +4 dB, mel spectral tilt -2.49 -> -1.97. The live run was
unlistenable noise. What the metrics missed: the extra "bandwidth" was broadband noise
(energy above 4 kHz 0.22% -> 1.31%) and the loudness envelope collapsed from 9.1 dB to
2.4 dB, because a static per-(channel,bin) offset of up to +/-0.7 on a latent whose own
std is only 0.39 swamps the content with a constant spectral bed. Bandwidth and level are
not quality. Structure lost to the RoPE wall cannot be restored downstream.

The 7 nodes (`NODE_CLASS_MAPPINGS`, end of nodes.py): `CLSSConfig`, `CLSSScenePrompts`,
`CLSSStreamingSampler`, `CLSSStage2`, `CLSSAVGuider` (split-CFG patch over an existing guider),
`CLSSAVGuiderV2` (all-in-one Stage-1 guider), `CLSSVideoDecodeSave` (streaming temporal-slice
VAE decode → PNG frames on disk — added 2026-08-19 to kill the ~18 GB whole-video decode
allocation; canonical workflows wire it instead of VAEDecodeTiled).
(`CLSSDraftLength` was deleted 2026-08-16 together with the rejected soundtrack mode.)
When the guider's positive has N scene entries, the
sampler auto-unpacks one scene per chunk proportionally across `num_chunks`. The nodes interoperate
with upstream ComfyUI LTXV nodes (`comfy_extras.nodes_lt`, `nodes_custom_sampler`).

## Build, run, and test commands

There is **no build/lint/test step at the custom-node layer** — ComfyUI imports the package
directly. Ways to exercise the code:

- **Run in ComfyUI** (the ground-truth path): `cd ../.. && python main.py`, then load
  `workflow/i2v_LTX_CLSS.json` (with guide image) or `workflow/t2v_LTX_CLSS.json`
  (text-only) — both carry the validated production config. The generation path can
  only be validated by a live run.
- **Standalone 16 GB-VRAM CLI** (no ComfyUI; loads GGUF transformer + Gemma text encoder):
  inside `Ltx-2-CLSS/` after `uv sync --frozen`,
  `python generate_clss.py --gguf-path ... --embeddings-path ... --audio-vae-path ...
  --video-vae-path ... --gemma-gguf ... --gemma-tokenizer ./gemma-tokenizer/ --prompt "..."`
  (see its module docstring for the full flag set and the block-streaming / CLSS memory strategy).
- **Offline math validation** (no model, no ComfyUI): `python simulations/<name>.py`. These replay
  *measured* failure trajectories from live runs through the exact correction math to sanity-check
  a control law **before** paying for a live generation. They validate the math, never perception.
- **Paper**: `cd paper && latexmk -pdf clss_paper.tex` (auxiliary `.aux/.fls/.log/.fdb_latexmk`
  are committed build artifacts).
- **Inner repo only** (`Ltx-2-CLSS/`): ruff + pytest are configured in its `pyproject.toml`
  (dev group: `ruff`, `pytest~=9.0`, `pre-commit`). The node layer has no linter config.

## Code style guidelines

- Node layer (`nodes.py`): plain Python with `from __future__ import annotations`, heavy use of
  docstrings and section comments that cite paper sections (§2.1, §2.3, …) and measured evidence.
  Match that style — every non-obvious constant should carry its justification.
- Inner repo (`Ltx-2-CLSS/`): ruff with line-length 120, target py311, strict rule set
  (annotations, bugbear, pylint, isort with `ltx_core`/`ltx_pipelines`/`ltx_trainer` as
  first-party). See `Ltx-2-CLSS/pyproject.toml`.

## Conventions specific to this codebase

- **Node inputs are experiment knobs, not user settings.** Most `optional` inputs on the sampler /
  Stage 2 / guiders exist because a specific failure was measured live; the tooltips record the
  evidence (e.g. "RMS +58% over 7 chunks"). Defaults are baked to the validated production config.
  **Read the tooltip before changing a default.**
- **Removing a failed experiment means deleting its input + code**, not defaulting it off. A knob's
  presence implies it is still a live lever.
- **Latent metrics (cosine sims, RMS, band energies logged per chunk) measure structure only.**
  They localize failures; they never prove a quality win. The user's eyes/ears on a live decode are
  the only ground truth.
- **The denoising/generation path is high-risk.** Never ship a change to the chunk loop, noise
  construction, or correction math without a user-validated live run. Noise edits are only
  seed-safe if they preserve the exact N(0,1) marginal (that is the design constraint behind
  `noise_temporal_corr` and the audio-noise simulations).
- **LTXVScheduler's shift scales with the connected latent's token count** — the exact
  law (comfy_extras/nodes_lt.py:646): `sigma_shift = tokens×(1.1/3072)+0.583`, tokens =
  template_lf×H×W of the connected latent. At 11×20: **16 lf → 1.844 (the validated
  config — this IS where "ear-validated shift 1.84" comes from), 52 lf → 4.680**, both
  verified numerically against the printed sigmas (max err 5e-5).  The dull-audio runs
  of 2026-08-13/14 were all 52 lf chunks (node Length=409, num_chunks=2/5): raw S1
  audio halved (std 0.95→0.42 vs the 2026-08-07 t2v_logs at 16 lf; i2v_logs at 16 lf:
  0.58–0.62), decoded output −34 dB, L/R 0.999, ro99 4.4 kHz (probe: 00037/00039).
  **The decoupling fix (separate 16-lf template wired to the scheduler, chunks kept
  at 52 lf) was built and FAILED live the same day (2026-08-14): the flat shared
  schedule at 52-lf chunks destroyed the VIDEO** — the second confirmation of the
  2026-08-11 "shared 1.84 destroys video" measurement.  The dilemma is real: at
  52-lf chunks the video wants 4.68 and the audio wants 1.844, and NO shared
  schedule serves both; the per-modality split deflated the audio live (2×, see
  the shift section above).  **CONFIRMED BOTH DIRECTIONS (2026-08-14 live run):
  flat 1.844 @ 52-lf chunks → audio GOOD by ear (raw std 0.49, aud_refine_sim
  0.72–0.78, best ever) but video STATIC (vid_std 1.99, origin_sim→0.999 by
  frame 3 — every frame is the guide image).**  **Soundtrack mode REMOVED entirely (2026-08-16): the `soundtrack` /
  `soundtrack_tau` inputs and the `CLSSDraftLength` node were deleted — a
  frozen draft has no audio feedback of the video; do not re-add.**  The resolution is the one-run per-modality
  split — `audio_shift_mult=0.394` at 52-lf/11×20 → audio shift 1.844 while the
  video keeps the 4.68 sigmas byte-identical — wired in both canonical workflows.
  The split path was static-audited against the model code before the retest
  (2026-08-14): `LTXAV.process_timestep` returns per-token a_timestep = mask×σ
  (model_base.py:1193) which the patch scales to mask×a_i ✓; `calculate_denoised`
  converts with the GLOBAL sigma (model_sampling.py:35) so the sampler's
  mask-aware re-conversion is required and correct ✓; the CFGGuider pin
  (samplers.py:633-641) blends clean latent linearly in mask, consistent with
  the told mask×a_i ✓.  The earlier live deflation (0.43→0.30) was confounded
  (that run also had audio_stg=1); the retest runs stg=0.
  Long single-window runs (drafts, ceiling tests) run away the same way (61 lf →
  shift e^5.39, measured 2026-08-07): disconnect the scheduler's `latent` there
  (tokens default 4096 → shift = 2.05, the reference value).  The sampler logs a
  top-heavy-schedule WARNING at startup.

## Testing strategy

- There is no unit-test suite at the node layer. The `simulations/` scripts are the fast check:
  they assert control-law statistics (PASS/FAIL output) against measured failure trajectories and
  have caught real production defects before live runs. Run the relevant simulation after touching
  correction or noise math.
- Final validation is always a live ComfyUI or `generate_clss.py` run, judged by the user.
- The inner repo has pytest configured; use it for changes inside `Ltx-2-CLSS/packages/`.

## Security considerations

- This package executes inside ComfyUI's Python process with full user privileges; it downloads and
  loads multi-GB model weights (GGUF/safetensors) from local paths. Never add network fetches or
  dynamic code loading at import time.
- `nodes.py` mutates global process state at import (`PYTORCH_ALLOC_CONF`, `sys.path`) — keep these
  minimal and additive (`os.environ.setdefault`, existence-checked `sys.path.insert`).
- Do not commit model files, generated videos, or secrets. `.gitignore` covers Python artifacts,
  venvs, and `*.log`.
- The inner submodule has its own `LICENSE` (forked from Lightricks' LTX-2) — respect it when
  redistributing algorithm changes.
