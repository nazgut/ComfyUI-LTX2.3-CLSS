# ComfyUI-LTX2.3-CLSS

**Closed-Loop Streaming Synthesis (CLSS)** — arbitrary-length audio-video generation with LTX-2.3 22B in [ComfyUI](https://github.com/comfyanonymous/ComfyUI), on consumer **16 GB VRAM** hardware.

[![Support on Patreon](https://img.shields.io/badge/Patreon-Support%20this%20project-f96854?logo=patreon)](https://www.patreon.com/c/AleksanderM)
[![Model](https://img.shields.io/badge/HuggingFace-LTX--2.3-orange?logo=huggingface)](https://huggingface.co/Lightricks/LTX-2.3)

---

## What is CLSS?

Video diffusion transformers generate only a few seconds per pass. The naive remedy — chunking the timeline and conditioning each chunk on the previous one — fails within a few hundred frames: the model keeps consuming its own slightly off-distribution output, and exposure-bias drift compounds into scene collapse or grain amplification.

CLSS treats the chunk hand-off as a **feedback loop** and controls it. Chunks share a streaming latent buffer (**SLB**) overlap, keeping latent memory O(overlap) instead of O(length), and between chunks CLSS applies lightweight corrections that fight drift **without modifying any transformer weights**:

- **Calibrated context re-noising** (τc) — the overlap context is re-projected toward the data manifold instead of being accepted verbatim; per-chunk schedule rises from 0.05 toward a 0.10 ceiling with a 5-chunk half-life
- **EMA-tracked per-channel AdaIN** (β) — suppresses fast statistical drift while letting intentional scene evolution through; the EMA reference **resets at every scene change**, so the first chunk of a new scene re-anchors it instead of being dragged toward the previous scene's statistics
- **Dynamic anchor bank** — long-range identity tracking with scene-change-triggered commits
- **Two-band spatial detail anchor** — counters the measured progressive high-frequency decay on long runs (symmetric per-band gains, scene-first referenced)
- **Audio seam control** — the audio SLB can be placed frozen, re-noised (up to 3× τc, ceiling 0.35, against the chunked-audio "metronome" fixed-point repetition), or regenerated freely with the last N seconds of the previous tail pinned frozen at the seam so vocal phrases aren't cut mid-phoneme
- **Optional temporally-correlated noise** — a run-constant shared frame mixed into every video noise frame (FreeNoise/PYoCo family), keeping each frame's marginal exactly N(0,1)

Audio and video are jointly modelled; audio wants higher CFG (~7) than video (~4), and the guider splits them.

## Multi-scene prompts

`CLSS Scene Prompts` takes one prompt per scene, separated by a line containing only `---`, and emits one CONDITIONING entry per scene. The sampler unpacks scenes proportionally across `num_chunks` and hands off at block boundaries with a **two-step crossfade** (`scene_handoff="transition_chunk"`, default): the outgoing scene's last chunk is guided by a 25%-incoming embedding blend, the incoming scene's first chunk by 75%-incoming. A single 50/50 transition chunk measured off-manifold for far-apart scenes and poisoned the next scene's SLB — the two-step version holds boundary similarity at 0.82–0.96 in the same scenario.

Rule of thumb: every scene block needs ≥ 2 chunks, i.e. `num_chunks ≥ 2 × scenes`. `blend` (single 50/50 chunk) and `hard` (plain text swap) remain available for comparison.

## Measured constraints that shape the defaults

- **The 20 s native-window wall.** AV RoPE positions are in seconds with `max_pos=20` for both modalities; any single window longer than ~20 s runs off the positional grid (audio glitches, video goes near-static). The sampler enforces this automatically: the overlap is clamped so overlap+new ≤ 19.5 s, and an over-long chunk is auto-split into uniform sub-chunks. Note the wall's *edge* is already degraded — the validated config uses ~13 s windows so the tail that becomes the next chunk's SLB is healthy; 19.3 s windows produced progressive detail fade over a run.
- **The scheduler's shift scales with the connected latent's token count.** At long chunks the video wants a high shift and the audio wants 1.844 — no shared schedule serves both, which is what `audio_shift_mult` (0 = AUTO: min(video shift, 1.844)) exists for.
- **Audio latent scale.** The audio VAE normalizer is calibrated so real audio sits at unit variance; generated audio falls short (lower std, less treble, stereo collapse at long windows) and the shortfall predicts how bad it sounds. A post-hoc per-(channel,bin) latent statistics repair was tried and **rejected** — it produced a broadband noise bed and collapsed dynamics. There is no post-hoc fix for a draft generated past the wall; generate inside it.

## Results

Matched T2V and I2V runs (identical seed and guidance stack, 10 chunks each, ~53 s clips):

| Metric | Result |
|---|---|
| Video boundary similarity | 0.79 → **0.98** (I2V, rising — no error compounding) |
| High-frequency energy share | **flat end-to-end** in both modes (no progressive softening) |
| Audio loudness | held to chunk-1 reference (±5 %) against raw drifts up to +72 % |
| Audio envelope repetition (metronome) | env_corr 0.76–0.96 → **≤ 0.19** (eliminated) |
| Audio boundary similarity | 0.07–0.53 → **0.85–0.92** |
| Scene-boundary continuity (two-step crossfade) | **0.82–0.96** (vs 0.24 for a single 50/50 transition chunk) |
| Hardware | single 16 GB GPU (GGUF Q4_K_S + CPU block-streaming) |

## Installation

1. Clone into your ComfyUI `custom_nodes` directory (**with submodules** — the CLSS
   algorithm lives in the `Ltx-2-CLSS` submodule):

   ```bash
   cd ComfyUI/custom_nodes
   git clone --recurse-submodules https://github.com/nazgut/ComfyUI-LTX2.3-CLSS.git
   ```

   Already cloned without `--recurse-submodules`? Run
   `git submodule update --init` inside the repo.

2. Restart ComfyUI. The `Ltx-2-CLSS/packages` sources are injected into `sys.path` automatically — no pip install step.

3. Download the models (HuggingFace):
   - `Lightricks/LTX-2.3` — 22B checkpoint (GGUF Q4_K_S recommended for 16 GB VRAM), video/audio VAEs, spatial upscaler, distilled LoRA
   - `google/gemma-3-12b-it` — text encoder (GGUF + tokenizer directory)

4. Load a canonical workflow: [`workflow/t2v_LTX_CLSS.json`](workflow/t2v_LTX_CLSS.json) (text-only) or [`workflow/i2v_LTX_CLSS.json`](workflow/i2v_LTX_CLSS.json) (with guide image). Both carry the validated production config — copy them for your own experiments rather than editing in place.

**Hardware requirements:** ~16 GB VRAM, ~48 GB system RAM (the 22B checkpoint is dequantized to BF16 in pinned CPU RAM; transformer blocks are streamed to GPU).

## Nodes

Every input on every node carries an in-UI tooltip explaining what it does, its default, and the measured evidence behind it.

| Node | Purpose |
|---|---|
| **CLSS Config** | CLSS hyperparameters (τc, β, overlap, noise temporal correlation) |
| **CLSS Scene Prompts** | Per-scene prompts (split on `---`) encoded into multi-entry CONDITIONING |
| **CLSS Streaming Sampler** | The main chunked Stage-1 sampler — CLSS corrections, scene crossfade, audio SLB seam modes, per-modality audio shift (auto by default), RoPE-wall enforcement, seeded noise, per-chunk telemetry + end-of-run trend summary |
| **CLSS Stage 2** | 2× refinement pass (distilled LoRA, same SLB continuity; audio frozen passthrough) |
| **CLSS AV Guider** | Split per-modality CFG patch over an existing guider |
| **CLSS AV Guider V2** | All-in-one Stage-1 guider: split CFG + modality scale + STG (video_cfg 4.0 / audio_cfg 7.0, stg_block 28 validated for LTX-2.3) |
| **CLSS Video Decode+Save** | Streaming temporal-slice VAE decode straight to PNG frames on disk — never materializes the whole decoded video in RAM |

```
LTXVideo Loader → CLSSScenePrompts → LTXVConditioning → CLSSAVGuiderV2
EmptyLTXVLatentVideo + audio latent → LTXVConcatAVLatent
CLSSConfig + CLSSStreamingSampler → LTXVLatentUpsampler → CLSSStage2 → CLSS Video Decode+Save
```

The nodes interoperate with upstream ComfyUI LTXV nodes (`comfy_extras.nodes_lt`, `nodes_custom_sampler`).

## Repository layout

```
nodes.py              # all ComfyUI node implementations
__init__.py           # sys.path injection + node-mapping exports
workflow/             # the two canonical workflows (t2v, i2v)
Ltx-2-CLSS/           # git submodule (github.com/nazgut/Ltx-2-CLSS) — a fork of
                      # Lightricks' LTX-2 monorepo with the CLSS additions:
                      #   packages/ltx-pipelines/.../streaming/  ← the CLSS algorithm
                      #   generate_clss.py                     ← standalone CLI
                      #   convert_gguf.py                      ← GGUF→BF16 pre-conversion
```

Note: `Ltx-2-CLSS` is a **submodule with its own history** — commit ComfyUI-node
changes in this repo and algorithm changes in the submodule (then bump the
submodule pointer here).

## Standalone generation (no ComfyUI)

Inside `Ltx-2-CLSS/` (after `uv sync --frozen`):

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

## Support this project

This is independent, unfunded research developed on consumer hardware. Longer-horizon experiments (more chunks, ablations, perceptual metrics) are compute-bound — your support directly buys GPU time.

**[❤️ Support on Patreon](https://www.patreon.com/c/AleksanderM)**

## Acknowledgements

Built on [LTX-2](https://github.com/Lightricks/LTX-2) by Lightricks and the ComfyUI ecosystem. The inner `Ltx-2-CLSS/` tree is a fork of the LTX-2 monorepo; see its own README and LICENSE.
