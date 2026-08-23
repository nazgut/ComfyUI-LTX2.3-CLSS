# ComfyUI-LTX-CLSS

**Closed-Loop Streaming Synthesis (CLSS)** — arbitrary-length audio-video generation with LTX-2.3 22B in [ComfyUI](https://github.com/comfyanonymous/ComfyUI), on consumer **16 GB VRAM** hardware.

[![Support on Patreon](https://img.shields.io/badge/Patreon-Support%20this%20project-f96854?logo=patreon)](https://www.patreon.com/c/AleksanderM)
[![Model](https://img.shields.io/badge/HuggingFace-LTX--2.3-orange?logo=huggingface)](https://huggingface.co/Lightricks/LTX-2.3)

---

## What is CLSS?

Video diffusion transformers generate only a few seconds per pass. The naive remedy — chunking the timeline and conditioning each chunk on the previous one — fails within a few hundred frames: the model keeps consuming its own slightly off-distribution output, and exposure-bias drift compounds into scene collapse or grain amplification.

CLSS treats the chunk hand-off as a **feedback loop** and controls it. Between chunks it applies lightweight corrections that bound the closed-loop gain below one — **without modifying any transformer weights**:

- **Calibrated context re-noising** (τc) — the overlap context is re-projected toward the data manifold instead of being accepted verbatim
- **EMA-tracked per-channel AdaIN** — suppresses fast statistical drift while letting intentional scene evolution through
- **Dynamic anchor bank** — long-range identity tracking with scene-change-triggered commits and forced periodic insertion
- **Two-band spatial detail anchor** — counters the measured progressive high-frequency decay on long runs (symmetric per-band gains, scene-first referenced)
- **Hot audio SLB re-noise schedule** — the audio overlap is re-noised on a 3× schedule (τc ≈ 0.15→0.28, ceiling 0.35) with envelope-flattened context; breaks the fixed-point repetition ("metronome") of chunked autoregressive audio *chunk-natively*, at any length. (A per-chunk overlap phase-jitter lever was built against the same failure and removed: it scattered the peak but fought the SLB's structural assumptions.)

All of it is grounded in a **linear stability analysis** of the feedback loop in latent space (ρ_loop = 0.57), with per-chunk telemetry that localizes every drift event in wall-clock time.

## Results

Matched T2V and I2V runs (identical seed and guidance stack, 10 chunks each, ~53 s clips):

| Metric | Result |
|---|---|
| Video boundary similarity | 0.79 → **0.98** (I2V, rising — no error compounding) |
| High-frequency energy share | **flat end-to-end** in both modes (no progressive softening) |
| Audio loudness | held to chunk-1 reference (±5 %) against raw drifts up to +72 % |
| Audio envelope repetition (metronome) | env_corr 0.76–0.96 → **≤ 0.19** (eliminated) |
| Audio boundary similarity | 0.07–0.53 → **0.85–0.92** |
| Hardware | single 16 GB GPU (GGUF Q4_K_S + CPU block-streaming) |
| Time | ≈ 96–97 min per ~53 s clip (Stage 1 + Stage 2) |

See [`paper/clss_paper.pdf`](paper/clss_paper.pdf) for the full analysis, per-chunk metrics, and the failure-mode history.

## Installation

1. Clone into your ComfyUI `custom_nodes` directory (**with submodules** — the CLSS
   algorithm lives in the `Ltx-2-CLSS` submodule):

   ```bash
   cd ComfyUI/custom_nodes
   git clone --recurse-submodules https://github.com/nazgut/ComfyUI-LTX-CLSS.git
   ```

   Already cloned without `--recurse-submodules`? Run
   `git submodule update --init` inside the repo.

2. Restart ComfyUI. The `Ltx-2-CLSS/packages` sources are injected into `sys.path` automatically — no pip install step.

3. Download the models (HuggingFace):
   - `Lightricks/LTX-2.3` — 22B checkpoint (GGUF Q4_K_S recommended for 16 GB VRAM), video/audio VAEs, spatial upscaler, distilled LoRA
   - `google/gemma-3-12b-it` — text encoder (GGUF + tokenizer directory)

4. Load the example workflow: [`workflow/i2v_LTX_CLSS.json`](workflow/i2v_LTX_CLSS.json)

**Hardware requirements:** ~16 GB VRAM, ~48 GB system RAM (the 22B checkpoint is dequantized to BF16 in pinned CPU RAM; transformer blocks are streamed to GPU).

## Nodes

| Node | Purpose |
|---|---|
| **CLSS Config** | CLSS hyperparameters (τc, β, overlap, …) |
| **CLSS Scene Prompts** | Per-scene prompts encoded into flat CONDITIONING |
| **CLSS Streaming Sampler** | The main chunked Stage-1 sampler — CLSS corrections, audio SLB + ref_audio continuity (fixed one-overlap window), per-modality audio shift (auto by default), seeded noise, full telemetry |
| **CLSS Stage 2** | 2× refinement pass (3-step distilled LoRA, per-window detail anchor; audio frozen passthrough) |
| **CLSS AV Guider / V2** | Split per-modality CFG (+ modality scaling and STG in V2) |
| **CLSS Video Decode+Save** | Streaming temporal-slice VAE decode straight to PNG frames on disk — never materializes the whole decoded video in RAM |

```
LTXVideo Loader → CLSSScenePrompts → LTXVConditioning → CFGGuider
EmptyLTXVLatentVideo + audio latent → LTXVConcatAVLatent
CLSSConfig + CLSSStreamingSampler → LTXVSeparateAVLatent → CLSS Video Decode+Save (frames to disk)
```

The nodes interoperate with upstream ComfyUI LTXV nodes (`comfy_extras.nodes_lt`, `nodes_custom_sampler`).

### Audio latent scale

The LTX-2 audio VAE's normalizer is calibrated so **real audio sits at unit variance**
(measured over four real tracks: overall latent std 0.98–1.26, per-frequency-bin std flat
at 1.00). Generated audio falls short of that, and how far short predicts how bad it
sounds — a 104 s single-window soundtrack draft measured std 0.48, bandwidth 3.1 kHz and
L/R correlation 1.00000 (literally mono), while the strongest per-chunk run on these
measures reached std 0.84,
6.1 kHz and real stereo. Half scale in a log-mel space means quiet (−31.6 dBFS against
−21…−16 for the reference tracks) and low-passed far below the 8 kHz mel ceiling. The VAE
is not at fault: real music round-trips through it with bandwidth and stereo intact.

Note what is *not* wrong: the draft's dynamic range (8.1 dB) already exceeds real music's
(4.3–6.3 dB), and its log-mel contrast (1.35) exceeds real music's (1.00). The deficit is
level and treble, not contrast — so scaling the latent's variance up is the wrong lever
and is deliberately not offered.

Measure any take with `python simulations/audio_latent_probe.py FILE --reference` — audio
VAE only, CPU, no transformer, so it runs while ComfyUI holds the GPU.

A latent-statistics repair was tried and **rejected** (2026-08-10). Shifting the audio
latent's per-(channel, bin) mean toward a real-music profile raised every metric I was
watching — bandwidth ro99 3141 → 4477 Hz, level +4 dB — and sounded like noise on a live
run. The added "bandwidth" was a broadband noise bed (energy above 4 kHz went 0.22% →
1.31%) while the loudness envelope collapsed to 2.4 dB. There is no post-hoc fix for a
draft generated past the wall; generate inside it.

## Repository layout

```
nodes.py              # all ComfyUI node implementations
workflow/             # example ComfyUI workflow (i2v)
paper/                # CLSS paper (LaTeX + PDF) and experiment logs
simulations/          # standalone pure-math diagnostic scripts (no GPU needed)
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
