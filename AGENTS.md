# AGENTS.md

Guidance for AI coding agents working in this repository. Read this before making any change.

## Project overview

**ComfyUI-LTX-CLSS** is a **ComfyUI custom-node package** implementing CLSS (Closed-Loop
Streaming Synthesis) — arbitrary-length audio-video generation with the LTX-2.3 22B model on
consumer 16 GB VRAM hardware. It is loaded by a ComfyUI installation (this repo lives at
`ComfyUI/custom_nodes/<name>`; `../../main.py` is the ComfyUI entry point). There is no
package manifest at the outer layer (no `pyproject.toml`/`package.json` in the repo root) —
ComfyUI imports the package directly.

CLSS generates video in short temporal **chunks** sharing an **SLB** (streaming latent buffer)
overlap, keeping latent memory O(overlap) instead of O(length). Between chunks it applies
closed-loop corrections that fight exposure-bias drift **without modifying transformer
weights**:

- **§2.1** calibrated context re-noising (`tau_c`, default 0.05; per-chunk schedule rises
  toward `_VIDEO_TAU_C_CEILING = 0.10` with a 5-chunk half-life — see `_tau_c_eff` in `nodes.py`)
- **§2.3** EMA-tracked per-channel AdaIN drift correction (`beta`, default 0.4; the EMA
  reference resets at every scene change — the first chunk of a new scene is uncorrected
  and re-anchors it, see `CLSSState.reset_drift_refs`)
- **§2.5** dynamic anchor bank (long-range identity tracking with scene-change-triggered commits)
- Two-band spatial detail anchor; hot audio SLB re-noise schedule (3× `tau_c`, ceiling 0.35,
  `_AUDIO_TAU_C_BASE_MULT` / `_AUDIO_TAU_C_CEILING` in `nodes.py`) against the chunked-audio
  "metronome" fixed-point repetition

Audio and video are jointly modelled; audio wants higher CFG (~7) than video (~4), and audio
drift is a recurring failure mode.

## Repository layout

```
nodes.py              # all 7 ComfyUI node implementations (~1475 lines)
__init__.py           # sys.path injection + ComfyUI node-mapping exports
workflow/             # exactly two canonical workflows: i2v_LTX_CLSS.json and
                      # t2v_LTX_CLSS.json, with the validated production config baked in.
                      # RULE: every experiment copies the canonical file into its own
                      # workflow — never mutate a canonical workflow in place.
Ltx-2-CLSS/           # git submodule (github.com/nazgut/Ltx-2-CLSS) — a fork of
                      # Lightricks' LTX-2 monorepo with the CLSS additions:
                      #   packages/ltx-pipelines/src/ltx_pipelines/streaming/  ← the CLSS algorithm
                      #   generate_clss.py                                   ← standalone CLI
                      #   convert_gguf.py                                    ← GGUF→BF16 pre-conversion
```

### `Ltx-2-CLSS/` is a git submodule with its own history

Registered in `.gitmodules`, pointing at `github.com/nazgut/Ltx-2-CLSS.git` — a fork of
Lightricks' LTX-2. Clone with `--recurse-submodules` (or `git submodule update --init`).
**Commit ComfyUI-node changes in this repo and algorithm changes in the submodule**, then bump
the submodule pointer here. A `m Ltx-2-CLSS` in `git status` means the inner repo has
uncommitted/unpushed changes. The inner repo's own docs:
`Ltx-2-CLSS/README.md` (upstream LTX-2), `Ltx-2-CLSS/packages/ltx-pipelines/CLAUDE.md`
(pipeline catalog, sigma schedules, guidance), `Ltx-2-CLSS/packages/ltx-trainer/AGENTS.md`
(trainer architecture — read before touching the trainer package).

## Runtime architecture

- **Language:** Python (the node layer targets the host ComfyUI's interpreter; the inner repo
  targets Python 3.11 per its ruff config, packages require >=3.10).
- `__init__.py` prepends `Ltx-2-CLSS/packages/{ltx-core,ltx-pipelines}/src` to `sys.path`
  (existence-checked), then exports `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` from
  `nodes.py`. All node code is in that single file.
- The CLSS algorithm itself lives in the submodule at
  `Ltx-2-CLSS/packages/ltx-pipelines/src/ltx_pipelines/streaming/clss.py` (`CLSSConfig` dataclass
  with all hyperparameters, `CLSSState` holding the SLB / EMA references / anchor bank, per-chunk
  `get_overlap_conditioning` / `get_anchor_conditioning` / `post_process` / `update_buffer`).
  `nodes.py` imports it and orchestrates it against ComfyUI's sampler/guider infrastructure
  (`comfy.samplers`, `comfy.model_patcher`, `comfy.nested_tensor`,
  `comfy_extras.nodes_lt`, `comfy_extras.nodes_custom_sampler`).
- **Hardware model:** ~16 GB VRAM, ~48 GB system RAM. The 22B checkpoint is dequantized to BF16
  in pinned CPU RAM and transformer blocks are streamed to the GPU (GGUF Q4_K_S + CPU
  block-streaming). `nodes.py` sets `PYTORCH_ALLOC_CONF=expandable_segments:True` before
  importing torch — keep that line above the torch import.
- The inner repo is a **uv workspace** (`Ltx-2-CLSS/pyproject.toml`): members `ltx-core`
  (model/inference stack), `ltx-pipelines` (high-level pipelines incl. the CLSS streaming
  pipeline `streaming/pipeline.py`), `ltx-trainer` (LoRA / full fine-tuning). Setup:
  `uv sync --frozen` inside `Ltx-2-CLSS/`.

### The nodes and how they wire together (Stage 1)

```
LTXVideo Loader → MODEL, VAE, CLIP
CLSSScenePrompts(CLIP, prompts)          → CONDITIONING   (one entry per scene, split by a line of '---')
LTXVConditioning(positive, negative, fps)→ positive, negative
CLSSAVGuiderV2(model, pos, neg, ...)     → GUIDER         (split video/audio CFG + modality scale + STG)
EmptyLTXVLatentVideo + audio latent + LTXVConcatAVLatent → LATENT  (per-chunk AV template)
CLSSConfig                               → CLSS_CONFIG
CLSSStreamingSampler(GUIDER, SAMPLER, SIGMAS, NOISE, LATENT, CLSS_CONFIG, num_chunks, [image, vae]) → LATENT
```

Two-stage pipeline: Stage 1 (`CLSSStreamingSampler`) → `LTXVLatentUpsampler` (2× spatial) →
`CLSSStage2` (chunked distilled-LoRA refinement with the same SLB continuity mechanism; audio is
frozen passthrough) → `CLSSVideoDecodeSave` (streaming temporal-slice VAE decode straight to PNG
frames on disk — never materializes the whole decoded video in RAM). See the two files in
`workflow/` for the canonical wiring.

The 7 nodes (`NODE_CLASS_MAPPINGS`, end of `nodes.py`): `CLSSConfig` (CLSS hyperparameters),
`CLSSScenePrompts` (per-scene prompts via the LTX2 system-prompt template), `CLSSStreamingSampler`
(main chunked Stage-1 sampler with full telemetry), `CLSSStage2` (2× refinement pass),
`CLSSAVGuider` (split-CFG patch over an existing guider), `CLSSAVGuiderV2` (all-in-one Stage-1
guider: video_cfg 4.0 / audio_cfg 7.0 / modality_scale 3.0 / rescale 0.7 / STG defaults,
stg_block 28 for LTX-2.3), `CLSSVideoDecodeSave`. When the guider's positive has N scene
entries, the sampler auto-unpacks one scene per chunk proportionally across `num_chunks`.
The boundary hand-off is selected by the sampler's `scene_handoff` input:
`transition_chunk` (default) runs a two-step crossfade straddling each boundary — the
outgoing scene's last chunk is guided by a 25%-incoming embedding blend, the incoming
scene's first chunk by 75%-incoming (`_blend_scene_cond` in `nodes.py`) — needs every
scene block ≥2 chunks, i.e. `num_chunks ≥ 2×scenes` (3 scenes → 6 chunks:
S0, T01-25%, T01-75%, T12-25%, T12-75%, S2); a single 50/50 transition chunk was
measured to drift off-manifold for far-apart scenes (anchor-sim 0.24) and poison the
next scene's SLB. `blend` applies a single 50/50 blend to the first chunk of the new
scene; `hard` is the plain text swap. A crossfade chunk statistically belongs to the
scene its text leans toward (EMA/refs reset on the first incoming-leaning chunk). A
scene with many described beats still needs multiple pure chunks — one chunk cannot
complete a long arc and reads as a jump cut.

Notable sampler knobs (defaults are the validated production config): `detail_anchor` on;
`video_slb_tau_mult` 1.0 (scales video overlap re-noise; 0 = frozen clean seam);
`audio_slb_tau_mult` 0.0 (0 = audio SLB placed frozen, no tau_c on audio; > 0 = SLB
re-noised with tau_c×mult; < 0 = overlap regenerates freely but the last |value|
SECONDS of the previous tail stay pinned frozen at the end of the overlap — keeps the
vocal phrase glued across the seam; fully free overlap cuts words mid-phoneme);
`audio_shift_mult` 0.0 = AUTO (audio shift = min(connected video shift, 1.844) — the
ear-validated target; manual values ≠ 0 are raw multipliers, 1.0 = off);
`auto_seam_pin` off (event-triggered frozen seam on coarse-layout jumps).
`CLSSConfigNode` additionally exposes `noise_temporal_corr` (default 0.3): mixes a
run-constant shared frame into every video noise frame, `n_t = sqrt(1-a)·eps_t +
sqrt(a)·eps_shared`, keeping each frame's marginal exactly N(0,1) while raising
frame-to-frame noise correlation (FreeNoise/PYoCo family).

## Measured constraints that shape the code

- **The 20 s native-window wall.** AV RoPE positions are in SECONDS with `max_pos=20` for both
  modalities (`comfy/ldm/lightricks/av_model.py`). Any single window longer than ~20 s runs off
  the positional grid (audio glitches, video goes near-static). `CLSSStreamingSampler` enforces
  this automatically (`_ROPE_WALL_S = 20.0`, `_ROPE_WALL_MARGIN_S = 0.5`, `_MIN_OVERLAP_LF = 2`
  in `nodes.py`): the overlap is clamped so overlap+new ≤ 19.5 s, and a chunk that alone exceeds
  the wall is auto-split into the fewest uniform sub-chunks that fit. Keep Stage-2 auto chunking
  under the same cap. This is why chunked CLSS works at all.
- **LTXVScheduler's shift scales with the connected latent's token count:** `sigma_shift =
  tokens×(1.1/3072)+0.583` (`comfy_extras/nodes_lt.py`). At 11×20 latent: 16 lf → shift 1.844,
  52 lf → shift 4.680. At long chunks the video wants the high shift and the audio wants 1.844 —
  no shared schedule serves both; `audio_shift_mult` exists for exactly this reason.
- **Audio latent scale:** the audio VAE normalizer is calibrated so real audio sits at unit
  variance; generated audio falls short (lower std, less treble, stereo collapse at long
  windows) and the shortfall predicts how bad it sounds. A post-hoc per-(channel,bin) latent
  statistics repair was tried and **rejected** (it produced a broadband noise bed and collapsed
  dynamics) — do not rebuild it; generate inside the wall instead.

## Build, run, and test commands

There is **no build/lint/test step at the custom-node layer** — ComfyUI imports the package
directly. Ways to exercise the code:

- **Run in ComfyUI** (the ground-truth path): `cd ../.. && python main.py`, then load
  `workflow/i2v_LTX_CLSS.json` (with guide image) or `workflow/t2v_LTX_CLSS.json` (text-only).
  The generation path can only be validated by a live run.
- **Standalone 16 GB-VRAM CLI** (no ComfyUI; loads GGUF transformer + Gemma text encoder):
  inside `Ltx-2-CLSS/` after `uv sync --frozen`,
  `python generate_clss.py --gguf-path ... --embeddings-path ... --audio-vae-path ...
  --video-vae-path ... --gemma-gguf ... --gemma-tokenizer ./gemma-tokenizer/ --prompt "..."`
  (see its module docstring for the full flag set and the block-streaming memory strategy).
- **Inner repo only** (`Ltx-2-CLSS/`): ruff + pytest are configured in its `pyproject.toml`
  (dev group: `ruff`, `pytest~=9.0`, `pre-commit`) — `uv run ruff check .`,
  `uv run ruff format .`, `uv run pytest`. Note the currently checked-out submodule contains
  no test files, so pytest is aspirational config; there is no test suite anywhere in either
  repo today.

## Code style guidelines

- Node layer (`nodes.py`): plain Python with `from __future__ import annotations`, heavy use of
  docstrings and section comments that cite paper sections (§2.1, §2.3, …) and measured
  evidence. Match that style — every non-obvious constant should carry its justification.
- Inner repo (`Ltx-2-CLSS/`): ruff with line-length 120, target py311, strict rule set
  (annotations, bugbear, pylint, isort with `ltx_core`/`ltx_pipelines`/`ltx_trainer` as
  first-party). See `Ltx-2-CLSS/pyproject.toml`.

## Conventions specific to this codebase

- **Node inputs are experiment knobs, not user settings.** Most `optional` inputs on the
  sampler / Stage 2 / guiders exist because a specific failure was measured live. Defaults are
  baked to the validated production config. **Read the tooltip/docstring before changing a
  default.**
- **Removing a failed experiment means deleting its input + code**, not defaulting it off. A
  knob's presence implies it is still a live lever.
- **Latent metrics (cosine sims, RMS, band energies logged per chunk) measure structure
  only.** They localize failures; they never prove a quality win. The user's eyes/ears on a
  live decode are the only ground truth.
- **The denoising/generation path is high-risk.** Never ship a change to the chunk loop, noise
  construction, or correction math without a user-validated live run. Noise edits are only
  seed-safe if they preserve the exact N(0,1) marginal (that is the design constraint behind
  `noise_temporal_corr`).

## Security considerations

- This package executes inside ComfyUI's Python process with full user privileges; it loads
  multi-GB model weights (GGUF/safetensors) from local paths. Never add network fetches or
  dynamic code loading at import time.
- `nodes.py` mutates global process state at import (`PYTORCH_ALLOC_CONF`, `sys.path`) — keep
  these minimal and additive (`os.environ.setdefault`, existence-checked `sys.path.insert`).
- Do not commit model files, generated videos, or secrets. `.gitignore` covers Python
  artifacts, venvs, `*.log`, `tmp/`, and `output_clss_checkpoints/`; the submodule's
  `.gitignore` additionally excludes checkpoints and media files.
- The inner submodule has its own `LICENSE` (forked from Lightricks' LTX-2) — respect it when
  redistributing algorithm changes.
