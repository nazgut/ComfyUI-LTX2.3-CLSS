# AGENTS.md

Guidance for AI coding agents working in this repository. The reader is assumed to know nothing about the project.

## Project Overview

**ComfyUI-LTX-CLSS** is a ComfyUI custom-node package for long-form video generation with **CLSS (Closed-Loop Streaming Synthesis)** — an extension of Latent Streaming Synthesis (LSS) that adds closed-loop drift corrections between temporal chunks so that arbitrarily long videos can be generated without exposure-bias drift accumulation.

The repository contains two nested git repositories:

- **Outer repo** (`git@github.com:nazgut/ComfyUI-LTX-CLSS.git`) — the ComfyUI custom node itself:
  - `__init__.py` — ComfyUI entry point. Injects the vendored `Ltx-2-CLSS/packages/{ltx-core,ltx-pipelines}/src` directories into `sys.path`, then exports `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` from `nodes.py`.
  - `nodes.py` (~2700 lines) — all ComfyUI node implementations.
  - `workflow/i2v_LTX_CLSS.json` — example ComfyUI workflow (image-to-video).
  - `simulations/` — standalone pure-math diagnostic scripts (no model loaded; synthesize tensors matching logged behavior to prove claims about audio/video drift).
  - `paper/` — LaTeX source of the CLSS paper (`clss_paper.tex`), experiment logs (`t2v_logs.txt`, `i2v_logs.txt`), and a sample output video.

- **Inner repo** (`Ltx-2-CLSS/`, `git@github.com:nazgut/Ltx-2-CLSS.git`) — a fork of Lightricks' LTX-2 monorepo with CLSS additions. This is **not a submodule**; it is a vendored, independently-committed copy loaded via `sys.path` (see "Runtime architecture" below). Key CLSS-specific files:
  - `generate_clss.py` — standalone CLI entry point for CLSS generation with GGUF checkpoints and 16 GB VRAM optimizations (CPU block offloading, `max_batch_size=1`, streaming latents).
  - `convert_gguf.py` — pre-converts a GGUF checkpoint to a BF16 safetensors directory to skip per-run dequantization.
  - `packages/ltx-pipelines/src/ltx_pipelines/streaming/` — the CLSS algorithm:
    - `clss.py` — `CLSSConfig` (hyperparameters) and `CLSSState` (orchestrates per-chunk state). The four corrections:
      - §2.1 Calibrated context re-noising (`tau_c`) via `VideoConditionByLatentIndex(strength=1−τc)`
      - §2.3 EMA-tracked per-channel distribution reference + AdaIN renormalization (`beta`, `ema_lambda`)
      - §2.4 Frequency-band soft shrinkage via 3-D FFT (`n_freq_bands`, `freq_gamma`)
      - §2.5 Dynamic anchor bank with LRU eviction and forced periodic insertion
    - `pipeline.py` — `CLSSStreamingPipeline`: chunked denoising loop that applies the corrections between chunks.
  - `packages/ltx-core/` — core LTX-2 model implementation (transformer, VAE, text encoder, scheduler).
  - `packages/ltx-trainer/` — training toolkit (LoRA, full fine-tune, IC-LoRA). Has its own `AGENTS.md` and `CLAUDE.md` — read those before editing trainer code.

## ComfyUI Nodes

`nodes.py` defines six nodes (see `NODE_CLASS_MAPPINGS` at the bottom of the file):

| Node ID | Display name | Purpose |
|---|---|---|
| `CLSSConfig` | CLSS Config | Builds the `CLSS_CONFIG` object (wraps `CLSSConfig` hyperparameters) |
| `CLSSScenePrompts` | CLSS Scene Prompts | Encodes per-scene prompts into flat CONDITIONING (one entry per scene) |
| `CLSSStreamingSampler` | CLSS Streaming Sampler | The main chunked sampler; drives the per-chunk CLSS loop inside ComfyUI. Anti-loop lever: `overlap_jitter` (default `on`) cycles the per-chunk overlap through [full, half, ¾] so no two consecutive chunks share a window phase — breaks the fixed-point repetition the audio/video lock into (video+audio overlaps stay proportional, so no A/V offset) |
| `CLSSStage2` | CLSS Stage 2 | Second-stage (refinement) pass |
| `CLSSAVGuider` | CLSS AV Guider (Split CFG) | Guider with separate video/audio CFG |
| `CLSSAVGuiderV2` | CLSS AV Guider V2 (Split CFG + Modality) | Adds modality scaling and STG |

The intended workflow (documented in the `nodes.py` module docstring) is:

```
LTXVideo Loader → MODEL, VAE, CLIP
CLSSScenePrompts(CLIP, prompts) → CONDITIONING
LTXVConditioning(positive, negative, frame_rate) → positive, negative
CFGGuider(model, positive, negative) → GUIDER
EmptyLTXVLatentVideo + audio latent + LTXVConcatAVLatent → LATENT
CLSSConfig → CLSS_CONFIG
CLSSStreamingSampler(GUIDER, SAMPLER, SIGMAS, NOISE, LATENT, CLSS_CONFIG, ...) → LATENT
LTXVSeparateAVLatent → video_latent, audio_latent
VAE Decode → IMAGE
```

The sampler interoperates with upstream ComfyUI LTXV nodes (`comfy_extras.nodes_lt`, `comfy_extras.nodes_custom_sampler`); it re-uses helpers such as `LTXVAddGuide` and `SamplerCustomAdvanced` rather than re-implementing them.

## Technology Stack & Runtime Architecture

- **Language:** Python (requires ≥ 3.10; ruff target-version `py311`).
- **ML stack:** PyTorch (~2.7), torchaudio, einops, transformers (Gemma 3 text encoder), safetensors, accelerate; `av` and `openimageio` for video I/O.
- **Models:** LTX-2.3 22B DiT audio-video model (dev or distilled checkpoints, safetensors or GGUF), Gemma 3 12B text encoder (GGUF + separate tokenizer directory), LTX video/audio VAEs, spatial upscalers, distilled LoRA. All downloaded from HuggingFace (`Lightricks/LTX-2.3`, `google/gemma-3-12b-it`).
- **Packaging:** The outer ComfyUI node has **no pyproject.toml or build system** — it is a plain directory dropped into `ComfyUI/custom_nodes/`. The inner `Ltx-2-CLSS/` is a `uv` workspace (`uv sync --frozen`) with members `packages/*` (`ltx-core`, `ltx-pipelines`, `ltx-trainer`), built with `uv_build`.
- **Import mechanism:** `__init__.py` (and `nodes.py`) prepend `Ltx-2-CLSS/packages/ltx-core/src` and `Ltx-2-CLSS/packages/ltx-pipelines/src` to `sys.path`. The packages are used from source, **not pip-installed**. Edits under those `src/` trees take effect on the next ComfyUI restart.
- **VRAM strategy:** designed for 16 GB VRAM — GGUF checkpoints dequantized to BF16 in pinned CPU RAM, transformer blocks streamed to GPU (`OffloadMode.CPU`), `PYTORCH_ALLOC_CONF=expandable_segments:True` is set at import time in `nodes.py`. Requires ~48 GB system RAM for the dequantized 22B model.

## Build, Run, and Test Commands

There is no automated test suite in this repository (no `tests/` directory, no test files). Validation is empirical: run generation and inspect outputs and the always-on `[CLSS]` telemetry printed per chunk.

- **ComfyUI node:** place this directory under `ComfyUI/custom_nodes/` and restart ComfyUI; load `workflow/i2v_LTX_CLSS.json` for a working example.
- **Standalone generation** (inside `Ltx-2-CLSS/`, after `uv sync --frozen`):
  ```bash
  python generate_clss.py \
      --gguf-path ltx-2.3-22b-dev-UD-Q4_K_S.gguf \
      --embeddings-path ltx-2.3-22b-dev_embeddings_connectors.safetensors \
      --audio-vae-path ltx-2.3-22b-dev_audio_vae.safetensors \
      --video-vae-path ltx-2.3-22b-dev_video_vae.safetensors \
      --gemma-gguf gemma-3-12b-it-qat-UD-Q4_K_XL.gguf \
      --gemma-tokenizer ./gemma-tokenizer/ \
      --prompt "..."
  ```
  (See the `generate_clss.py` docstring for the full VRAM/RAM strategy and the `convert_gguf.py` pre-conversion flow.)
- **Lint** (inside `Ltx-2-CLSS/`): `ruff check` / `ruff format` — dev dependency group, config in `Ltx-2-CLSS/pyproject.toml`.
- **Simulations:** `python simulations/<name>.py` — pure-math scripts, no model or GPU needed.
- **Paper:** `paper/clss_paper.tex` builds with a standard LaTeX toolchain (e.g. `latexmk -pdf`).

## Code Style Guidelines

- The inner LTX-2 repo follows its `pyproject.toml` ruff config: line length 120, extensive rule set (pycodestyle, pyflakes, isort, pep8-naming, annotations, bugbear, comprehensions, simplify, pylint, etc.), first-party packages `ltx_core`, `ltx_pipelines`, `ltx_trainer`. Note `T20` (no `print`) is selected there — but the CLSS code deliberately uses `print()` for always-on `[CLSS]` telemetry; keep that convention in CLSS-related files.
- `nodes.py` and `clss.py` use `from __future__ import annotations`, type hints, and dense docstrings that reference the paper's section numbers (§2.1, §2.3, §2.4, §2.5, §3.7). **Preserve these section references** — they are the primary cross-reference between code and `paper/clss_paper.tex`.
- Algorithm constants are not magic numbers: each carries a comment explaining its measured/empirical justification (e.g. why `tau_c` defaults to 0.05 instead of the paper's 0.15–0.20 range). When changing a default, update the comment with the reason and evidence.
- Keep comments describing *why*, including failed alternatives (several docstrings document approaches that were tried and rejected — e.g. adaptive β, redundant-anchor injection). Do not delete these warnings.

## Project-Specific Conventions & Gotchas

- **Chunk-local coordinates:** anchors are injected at `frame_idx=0` of each chunk (chunks use local frame indices). Do not "fix" this to absolute frame indices — `clss.py` documents why that would be wrong.
- **Corrections apply to new frames only:** the overlap region is never post-processed twice (it was corrected when generated).
- **Per-chunk telemetry:** `CLSSState.post_process` / `update_buffer` print per-channel EMA stats, band gains/energies, and anchor-bank events every chunk. These logs feed the paper (`paper/t2v_logs.txt`, `i2v_logs.txt`); keep them intact and greppable as `[CLSS] chunk=...`.
- **Dead-parameter warnings:** some node inputs are intentionally inert in certain modes (e.g. `s2_audio_denoise` / `s2_audio_slb_af` when `audio_mode='refine'`). The code warns about these rather than failing — preserve that pattern.
- **Overlap clamp:** `overlap_latent_frames` is clamped against `new_latent_frames` to prevent a seam time-skip; do not remove the clamp.
- **Two nested repos:** commit ComfyUI-node changes in the outer repo and `Ltx-2-CLSS/` changes in the inner repo separately. The inner repo has its own history forked from Lightricks/LTX-2.
- **Generated artifacts:** `__pycache__/`, `*.log`, and LaTeX build artifacts are gitignored; `paper/clss_paper.pdf` and logs are checked in deliberately as experiment records.

## Security Considerations

- Model checkpoints and the Gemma tokenizer are downloaded from HuggingFace; verify URLs against the official `Lightricks/LTX-2.3` and `google/gemma-3-12b-it` repos.
- `nodes.py` sets `PYTORCH_ALLOC_CONF` via `os.environ.setdefault` at import time and inserts paths into `sys.path` — both are process-global side effects; be aware when integrating with other code.
- The inner repo's git remote uses SSH (`git@github.com:...`); pushing requires the user's credentials.
- No secrets, API keys, or network calls beyond model downloads exist in the codebase.
