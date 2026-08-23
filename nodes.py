"""
ComfyUI nodes for CLSS streaming video generation.

Workflow:
  LTXVideo Loader → MODEL, VAE, CLIP
  CLSSScenePrompts(CLIP, prompts) → CONDITIONING  (flat, one entry per scene)
  LTXVConditioning(positive, negative, frame_rate) → positive, negative
  CFGGuider(model, positive, negative) → GUIDER  (positive has N scene entries)
  EmptyLTXVLatentVideo + audio latent + LTXVConcatAVLatent → LATENT (chunk template)
  CLSSConfig → CLSS_CONFIG
  CLSSStreamingSampler(GUIDER, SAMPLER, SIGMAS, NOISE, LATENT, CLSS_CONFIG, ...) → LATENT
  LTXVSeparateAVLatent → video_latent, audio_latent
  VAE Decode → IMAGE
"""

from __future__ import annotations

import copy
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).parent / "Ltx-2-CLSS"
for _pkg in ("ltx-core", "ltx-pipelines"):
    _src = _REPO_ROOT / "packages" / _pkg / "src"
    if _src.exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

import comfy.model_management
import comfy.model_patcher
import comfy.nested_tensor
import comfy.sampler_helpers
import comfy.samplers
import comfy.utils
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_lt import LTXVAddGuide
from comfy_extras.nodes_textgen import LTX2_T2V_SYSTEM_PROMPT

from ltx_pipelines.streaming.clss import CLSSConfig, CLSSState


def _unconvert_cond(converted: list) -> list:
    """Reverse comfy.sampler_helpers.convert_cond: [dict, ...] → [[tensor, dict], ...].

    guider.original_conds stores already-converted conditionings (plain dicts).
    LTXVAddGuide helpers (add_keyframe_index, conditioning_set_values, etc.) expect
    the raw [[tensor, dict], ...] format.  Un-converting lets us call those helpers,
    after which we re-convert via comfy.sampler_helpers.convert_cond.
    """
    raw = []
    for c in converted:
        tensor = c.get("cross_attn", None)
        d = {k: v for k, v in c.items() if k not in ("cross_attn", "uuid")}
        raw.append([tensor, d])
    return raw


# ---------------------------------------------------------------------------
# Metric helpers (used by both Stage 1 and Stage 2 logging)
# ---------------------------------------------------------------------------

def _frame_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean-pooled channel-feature cosine similarity between two [B, C, H, W] latent frames.

    Mean-pools over spatial dims first (H×W → scalar per channel) so the feature
    is only [B, C] — cheap even at Stage-2 resolution (H=44, W=80).
    """
    with torch.no_grad():
        fa = F.normalize(a.float().reshape(a.shape[0], a.shape[1], -1).mean(-1), dim=1)
        fb = F.normalize(b.float().reshape(b.shape[0], b.shape[1], -1).mean(-1), dim=1)
        return (fa * fb).sum(dim=1).mean().item()


def _frame_chunk_of(frame_idx: int, plan: list) -> int:
    """Map a global video-frame index to its effective chunk number (1-based).

    Uses the per-chunk plan (video new frames per effective chunk) so the label
    stays correct even after a RoPE-wall auto-split produced uneven sub-chunks.
    """
    _d = int(frame_idx)
    for _c, (_v, _a) in enumerate(plan):
        if _d < _v:
            return _c + 1
        _d -= _v
    return len(plan)


def _grid_cell_cos(img_a: torch.Tensor, img_b: torch.Tensor, grid: int = 8) -> tuple[float, float]:
    """Per-cell cosine similarity between two decoded frames.

    Convention: img_a/img_b are [H, W, C] (the ComfyUI IMAGE per-frame layout —
    one frame of `vae.decode` output).  H and W are each split into `grid`
    slices (edge cells absorb the remainder when the side is not divisible),
    every cell is flattened over (h, w, C) and compared by cosine similarity.
    Returns (mean, min) over the grid×grid cells.

    Why: mean-pooled identity_sim can stay high when a single object is
    replaced (the pooled feature is dominated by the unchanged background);
    the per-cell MIN localizes exactly that failure.  Pure torch, CPU-safe.
    """
    with torch.no_grad():
        a = img_a.float()
        b = img_b.float()
        sims: list[float] = []
        for rows_a, rows_b in zip(torch.tensor_split(a, grid, dim=0),
                                  torch.tensor_split(b, grid, dim=0)):
            for cell_a, cell_b in zip(torch.tensor_split(rows_a, grid, dim=1),
                                      torch.tensor_split(rows_b, grid, dim=1)):
                sims.append(float(F.cosine_similarity(
                    cell_a.flatten(), cell_b.flatten(), dim=0)))
        return sum(sims) / len(sims), min(sims)


def _aud_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two audio latent tensors (flatten everything except batch).

    Trims to the shorter temporal dim before comparison so frames-vs-single-frame works.
    """
    with torch.no_grad():
        min_t = min(a.shape[2], b.shape[2])
        fa = F.normalize(a[:, :, :min_t].float().reshape(a.shape[0], -1), dim=1)
        fb = F.normalize(b[:, :, :min_t].float().reshape(b.shape[0], -1), dim=1)
        return (fa * fb).sum(dim=1).mean().item()


def _aud_within_chunk_sims(new_aud: torch.Tensor, n_seg: int = 3) -> list[float]:
    """Sequential cosine similarities across N equal temporal segments of a new audio chunk.

    new_aud: [B, C_a, T, freq] — new audio frames only (SLB already dropped).
    Returns n_seg-1 values.  Empty list when T is too short to split.
    Detects within-chunk audio coherence degradation — §4.3 / §5.4 claim.
    """
    T = new_aud.shape[2]
    if T < n_seg * 2:
        return []
    seg_len = T // n_seg
    sims: list[float] = []
    with torch.no_grad():
        for i in range(n_seg - 1):
            s1 = new_aud[:, :, i * seg_len:(i + 1) * seg_len].float().mean(dim=2)  # [B, C_a, freq]
            s2 = new_aud[:, :, (i + 1) * seg_len:(i + 2) * seg_len].float().mean(dim=2)
            f1 = F.normalize(s1.reshape(new_aud.shape[0], -1), dim=1)  # [B, C_a*freq=128]
            f2 = F.normalize(s2.reshape(new_aud.shape[0], -1), dim=1)
            sims.append((f1 * f2).sum(dim=1).mean().item())
    return sims


def _flatten_audio_env(x: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Flatten the per-frame energy envelope of an audio-context tensor.

    x: [B, C_a, T, freq].  Each frame is rescaled toward the window-mean RMS —
    spectral content and timbre are preserved, the loudness ARC is removed.
    Returns (flattened, min_gain, max_gain).

    Applied ALWAYS-ON to the audio context (SLB + ref_audio) — the output
    audio is untouched.  As an anti-repetition lever it is a recorded dead
    end (loop unchanged; see the dead-end log at the ref_audio site), but
    same-seed live runs showed it protects the VIDEO: two runs with the
    context flattened had no morphing, the same-prompt run without it morphed
    (2026-08-05) — plausible in the joint AV transformer, where over-loud
    audio context tokens distort shared self-attention.

    GAIN CAP [0.6, 1.25], asymmetric by design.  The first (unbounded)
    version amplified chunk-2's quietest ref frames ×2.98 — boosting their
    NOISE FLOOR into the conditioning — and freq bin 9 sat at 1.8-2.0× the
    reference for the entire run afterwards (audible hiss).  Attenuating
    loud frames is the mechanism's purpose; amplifying quiet frames only
    injects noise, so the up-gain is capped hard while down-gain stays
    loose enough to remove any real crescendo peak (measured peaks needed
    ~0.54-0.86).
    """
    with torch.no_grad():
        env = x.float().pow(2).mean(dim=(1, 3), keepdim=True).sqrt()   # [B,1,T,1]
        tgt = env.mean(dim=2, keepdim=True)
        g = (tgt / env.clamp(min=1e-6)).clamp(min=0.6, max=1.25)
        out = (x.float() * g).to(x.dtype)
        return out, float(g.min()), float(g.max())


def _chunk1_boundary_correction(audio_lat: torch.Tensor, freq_ref: list[float],
                                rms_ref: float) -> tuple[torch.Tensor, float, float]:
    """Apply to chunk-1's OUTPUT the exact correction that runs on chunk-1's
    latent when it is ADDED to chunk 2 (2026-08-11 — the code reason chunk 2
    sounds better than chunk 1):

      1. DC removal — subtract the per-(channel,freq) temporal mean
         (the SLB-save step: `_slb - _slb.mean(dim=2, keepdim=True)`)
      2. per-freq-bin energy matched to the reference spectrum, clamped [0.5, 2.0]
         (the SLB-save per-bin gain)
      3. RMS matched to the reference (the SLB-save RMS gain)

    Chunk 2 is generated FROM chunk 1's audio AFTER this pipeline runs on it —
    that corrected audio is what the model continues.  Chunk 1's own raw output
    never receives it, which is the measured asymmetry.  This makes chunk 1's
    own output receive the same treatment.

    Step 4 (env-flatten on chunk-1's OUTPUT) was REMOVED 2026-08-14: it hit the
    [0.6, 1.25] gain clamps on every logged run, compressing chunk-1's loudness
    arc by up to 40% — a direct cut on the ear-validated dynR axis, on stored
    output audio, with no reference counterpart and no video-protection
    justification (that argument covers CONTEXT only, where env-flatten stays).
    Returns (corrected, 1.0, 1.0) — the gain tuple is kept for the call-site log.
    """
    x = audio_lat.float()
    x = x - x.mean(dim=2, keepdim=True)                     # 1. DC removal
    x_freq = x.abs().mean(dim=(0, 1, 2))
    gains = torch.tensor([
        min(max(r / max(c, 1e-8), 0.5), 2.0)                # 2. per-bin spectral match
        for r, c in zip(freq_ref, x_freq.tolist())
    ], device=x.device, dtype=x.dtype)
    x = x * gains.view(1, 1, 1, -1)
    x_rms = x.pow(2).mean().sqrt()
    if x_rms > 0:
        x = x * (rms_ref / x_rms)                            # 3. RMS match
    return x.to(audio_lat.dtype), 1.0, 1.0


def _post_process_audio_latent(
    audio_lat: torch.Tensor,
    chunk_ends: list[int],
    smooth_half: int = 2,
    energy_beta: float = 0.3,
    label: str = "",
) -> torch.Tensor:
    """Normalize per-chunk audio energy and smooth chunk-boundary transitions.

    Two steps, mirroring the reference CLSS pipeline (pipeline.py):

    1. Per-chunk RMS normalization — computes median RMS across all chunks as
       target, then soft-blends each chunk toward that target with factor
       energy_beta.  Symmetric: corrects both chunk-1 loudness (common with i2v,
       no prior audio context) and quiet drift in later chunks.

    2. Boundary smoothing — linearly blends smooth_half frames on each side of
       every chunk boundary to remove clicks caused by independently-generated
       chunk edges.

    audio_lat: [B, C, T, freq] (CPU tensor, cloned internally)
    chunk_ends: cumulative audio frame counts at end of each chunk
    """
    if not chunk_ends:
        return audio_lat

    audio_lat = audio_lat.clone()
    T = audio_lat.shape[2]
    boundaries = [0] + list(chunk_ends)
    n = len(chunk_ends)

    # 1. Per-chunk RMS normalization
    if n >= 2:
        chunk_rms = []
        for i in range(n):
            seg = audio_lat[:, :, boundaries[i]:boundaries[i + 1]].float()
            chunk_rms.append(seg.pow(2).mean().sqrt().item())
        median_rms = sorted(chunk_rms)[n // 2]
        if median_rms > 1e-6:
            for i in range(n):
                if chunk_rms[i] < 1e-6:
                    continue
                raw_gain = median_rms / chunk_rms[i]
                soft_gain = 1.0 + energy_beta * (raw_gain - 1.0)
                if abs(soft_gain - 1.0) > 0.005:
                    audio_lat[:, :, boundaries[i]:boundaries[i + 1]] = (
                        audio_lat[:, :, boundaries[i]:boundaries[i + 1]] * soft_gain
                    )
                    rms_after = (
                        audio_lat[:, :, boundaries[i]:boundaries[i + 1]]
                        .float().pow(2).mean().sqrt().item()
                    )
                    print(f"[CLSS] audio_post{label}: chunk {i + 1} "
                          f"rms {chunk_rms[i]:.4f}→{rms_after:.4f} "
                          f"(soft_gain={soft_gain:.4f}  raw={raw_gain:.4f})")

    # 2. Boundary smoothing (skip the very last boundary — it's the end of the video)
    for boundary in chunk_ends[:-1]:
        b = boundary
        if b < smooth_half or b + smooth_half > T:
            continue
        for i in range(1, smooth_half + 1):
            alpha = i / (smooth_half + 1)
            prev = b - i
            nxt  = b + i - 1
            audio_lat[:, :, prev] = (
                (1.0 - alpha) * audio_lat[:, :, prev] + alpha * audio_lat[:, :, b]
            )
            audio_lat[:, :, nxt] = (
                (1.0 - alpha) * audio_lat[:, :, nxt] + alpha * audio_lat[:, :, b - 1]
            )

    return audio_lat


# ---------------------------------------------------------------------------
# Node 1: CLSSConfig
# ---------------------------------------------------------------------------

class CLSSConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tau_c":   ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5,  "step": 0.01,
                                      "tooltip": "Overlap re-noising level. 0=frozen, 0.05=paper default. "
                                                 "The one continuity/freshness trade-off worth exposing."}),
                "beta":    ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0,  "step": 0.05,
                                      "tooltip": "AdaIN drift correction strength. 0=off, 0.4=paper default."}),
                "overlap": ("INT",   {"default": 8,    "min": 1,   "max": 32,
                                      "tooltip": "Overlap latent frames shared between chunks."}),
            },
            "optional": {
                "noise_temporal_corr": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.8, "step": 0.05,
                                      "tooltip": "Mix a run-constant shared frame into every S1 video "
                                                 "noise frame, making the initial noise temporally "
                                                 "correlated (marginals stay exactly N(0,1); frame-to-frame "
                                                 "noise correlation = this value). 0.3 is the validated "
                                                 "production value (paper i2v/t2v runs); 0 = off (independent "
                                                 "white noise per frame — degrades cross-chunk consistency). "
                                                 "Changes the generated video like a seed change would; "
                                                 "too high → static content."}),
                "measure_g": (["off", "on"], {
                    "default": "off",
                    "tooltip": "Measure the open-loop transformer gain g_SLB (paper Eq. 8): "
                               "runs a SECOND denoising pass per chunk with a perturbed "
                               "overlap (~2x Stage-1 compute) and logs g_SLB per chunk + in "
                               "the trend summary.  Diagnostic only — the perturbed pass is "
                               "discarded, generation is completely unaffected.  The number "
                               "this produces was previously described (§3.5) but never "
                               "measured; enable on a validation run to record it."}),
                "measure_g_epsilon": ("FLOAT", {
                    "default": 0.01, "min": 0.0001, "max": 0.1, "step": 0.001,
                    "tooltip": "Perturbation magnitude for measure_g, as a fraction of the "
                               "overlap-latent norm.  Used only when measure_g=on."}),
            },
        }
        # Everything else is fixed or derived automatically:
        #   ema_lambda=0.10, sigma_max_drift=0.05, adain_max_amplification=1.2
        #   (validated internals — wrong values silently corrupt the video);
        #   anchor_force_every is derived from num_chunks inside the sampler.

    RETURN_TYPES = ("CLSS_CONFIG",)
    RETURN_NAMES = ("clss_config",)
    FUNCTION = "build"
    CATEGORY = "LTX-CLSS"

    def build(self, tau_c, beta, overlap, noise_temporal_corr=0.3,
              measure_g="off", measure_g_epsilon=0.01):
        return (CLSSConfig(
            tau_c=tau_c,
            beta=beta,
            ema_lambda=0.10,                 # fixed: validated EMA rate
            ema_sigma_max_drift=0.05,        # fixed: prevents late-chunk amplification
            anchor_force_every=0,            # sentinel: auto-derived in the sampler
            overlap_latent_frames=overlap,
            adain_max_amplification=1.2,     # fixed: caps AdaIN grain boost
            measure_g=(measure_g == "on"),   # default off: diagnostic-only (was hard-coded False)
            measure_g_epsilon=float(measure_g_epsilon),
            noise_temporal_corr=noise_temporal_corr,
        ),)


# ---------------------------------------------------------------------------
# Node 2: CLSSScenePrompts
# ---------------------------------------------------------------------------

class CLSSScenePrompts:
    """Multi-scene version of 'Generate LTX2 Prompt'.

    Write scene descriptions separated by a line containing only '---'.
    Each scene is Gemma-enhanced identically to 'Generate LTX2 Prompt'.
    Output is a flat CONDITIONING — one entry per scene, concatenated.
    Connect: CLSSScenePrompts → LTXVConditioning → CFGGuider → CLSSStreamingSampler.
    The sampler unpacks per-scene entries from the guider's positive automatically.

    Example input:
        A calm forest at dawn, birds singing
        ---
        A stormy ocean, waves crashing, lightning
        ---
        A peaceful mountain sunset, golden hour
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip":       ("CLIP",   {"tooltip": "LTX CLIP / Gemma — same as Generate LTX2 Prompt."}),
                "prompts":    ("STRING", {"multiline": True, "dynamicPrompts": False,
                                          "default": "Scene 1 description\n---\nScene 2 description",
                                          "tooltip": "Scene descriptions separated by a line containing only '---'."}),
                "max_length": ("INT",    {"default": 512, "min": 1, "max": 32768}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "generate"
    CATEGORY = "LTX-CLSS"

    def generate(self, clip, prompts: str, max_length: int):
        scenes = [s.strip() for s in prompts.split("\n---\n") if s.strip()]
        if not scenes:
            scenes = [prompts.strip()]

        flat_conditioning = []
        for scene in scenes:
            # Gemma enhancement — identical to TextGenerateLTX2Prompt
            formatted = (
                f"<start_of_turn>system\n{LTX2_T2V_SYSTEM_PROMPT.strip()}<end_of_turn>\n"
                f"<start_of_turn>user\nUser Raw Input Prompt: {scene}.<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            tokens = clip.tokenize(formatted, skip_template=True, min_length=1)
            generated_ids = clip.generate(tokens, do_sample=False, max_length=max_length)
            enhanced = clip.decode(generated_ids)

            scene_cond = clip.encode_from_tokens_scheduled(clip.tokenize(enhanced))
            # scene_cond is [[tensor, dict]] — extend flat list with this scene's entry
            flat_conditioning.extend(scene_cond)

        return (flat_conditioning,)


# ---------------------------------------------------------------------------
# Node 3: CLSSStreamingSampler
# ---------------------------------------------------------------------------

def _tau_c_eff(base: float, ceiling: float, chunk_idx: int, half_life: float = 5.0) -> float:
    """Effective overlap re-noising level for chunk `chunk_idx` (0-indexed among
    chunks that HAVE an overlap, i.e. chunk_idx=0 is the second physical chunk).

    ZERO GUARD (2026-08-20): base <= 0 (tau_c=0 in CLSSConfig or
    video_slb_tau_mult=0) means FROZEN — return 0.0.  Previously the
    ceiling ramp below resurrected re-noising the user explicitly zeroed:
    a live run with audio_slb_tau_mult=0.0 still ramped the audio SLB
    tau_c_eff 0.0000→0.2345 over 10 chunks (and the video 0→0.067 at
    tau_c=0.0), feeding the cross-chunk loudness loop the run was meant
    to exclude (aud_rms 0.84→1.12, aud_env_corr→0.85).

    Diagnosis (confirmed against measured logs, not assumed): stacking every
    continuity mechanism we've built -- SLB at fixed tau_c, ref_audio at full
    strength forever, and a hard energy anchor to a single fixed point -- switches
    on as a THRESHOLD, not a gradual drift.  Video intra_chunk_sim jumped from
    0.83 to 0.96 the instant full-strength SLB+ref turned on (chunk 3); audio
    within-chunk sim collapsed from ~0.85 to ~0.49 the instant ref_audio reached
    its full 67-frame window.  Once full-strength continuity conditioning is
    locked in, self-attention lets the frozen/anchored region "leak" into the
    whole chunk, and the model stops advancing content -- it re-renders a
    stabilised loop.  Repetition, not degradation.

    Fix: let tau_c (freedom) relax from its strong starting value toward a
    CAPPED ceiling as conditioning-chunks accumulate -- exactly the LTX IC-LoRA
    pattern of an adjustable attention_strength on conditioning tokens, applied
    here as a decay-with-floor schedule.  Never fully open (that reintroduces
    the original open-loop drift problem CLSS exists to prevent) -- "auto
    slowing down but don't disappear."  Video keeps a conservative ceiling
    (tau_c=0.05 was hard-won against boundary morphing); audio can afford more
    room since it has the EMA energy anchor as an independent stability backstop.
    """
    if base <= 0.0:
        return 0.0
    decay = 0.5 ** (chunk_idx / half_life)
    return ceiling - (ceiling - base) * decay


_VIDEO_TAU_C_CEILING = 0.10   # conservative: half the empirically-unstable 0.20
# Audio SLB re-noise REMOVED (2026-08-20, user directive): the audio overlap is
# placed FROZEN (mask=0) — no tau_c on audio at all.  The hot-schedule machinery
# (_AUDIO_TAU_C_BASE_MULT/_AUDIO_TAU_C_CEILING, the audio_slb_tau_mult knob) is
# gone; the reference pipeline never had an in-window audio re-noise either.
# Sigma schedule (2026-08-23, user-verified live): the model needs ONE shared
# sigma per step (the per-modality split and the 2.05 cap both broke something
# live), and the VIDEO needs the connected token law (nodes_lt.py:646:
# tokens×(1.1/3072)+0.583).  BUT the token law at long templates (52 lf → 4.68)
# is extremely top-heavy: only 7/20 steps below σ=0.9, and the audio comes out
# dull/LF-heavy/under-developed — while the 121-px config (16 lf → 1.844,
# 12/20 steps below 0.9) is good in both modalities (user-verified 2026-08-23).
# The video collapse that historically blocked flat schedules was traced to the
# UNBOUNDED rescale + euler_ancestral + temporal-corr noise — all removed — so
# AUTO now re-schedules an over-shifted template onto the shared KNOWN-GOOD
# 1.844 curve for BOTH modalities (same steps, same terminal, same stock path).
# 121 px is left byte-identical.  Manual audio_shift_mult values (≠0, ≠1) still
# engage the v3 split as an experiment lever.
_KNOWN_GOOD_SHIFT = 1.844

# RoPE wall (2026-08-22): AV RoPE positions are in SECONDS, max_pos=20 for both
# modalities (av_model.py positional_embedding_max_pos=[20] / max_pos=[20,2048,
# 2048]); the phase argument is indices*(t/max_pos*2 - 1) (model.py), so training
# only ever saw t/20 in [0,1].  A single model window — overlap + new — must stay
# inside ~20 s or the audio glitches (~20.5 s measured) and the video goes
# near-static (>60 lf at 24 fps).  The per-chunk window is AUTO-ADJUSTED to fit
# inside the wall (see CLSSStreamingSampler): the overlap is clamped, and if a
# chunk alone exceeds the wall it is split into uniform sub-chunks.
_ROPE_WALL_S = 20.0
_ROPE_WALL_MARGIN_S = 0.5     # stay inside the trained grid, not on its edge
_MIN_OVERLAP_LF = 2           # below this the SLB is too thin — split instead


class CLSSStreamingSampler:
    """CLSS streaming sampler — compatible with LTXVConcatAVLatent output.

    The `latent` input is a per-chunk AV template that defines new-frame shape.
    Build it with EmptyLTXVLatentVideo + LTXVConcatAVLatent.

    When the guider's positive conditioning has N > 1 entries (i.e. you connected
    CLSSScenePrompts → LTXVConditioning → CFGGuider), the sampler automatically
    unpacks one entry per chunk proportionally across num_chunks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider":      ("GUIDER",      {}),
                "sampler":     ("SAMPLER",     {}),
                "sigmas":      ("SIGMAS",      {}),
                "noise":       ("NOISE",       {}),
                "latent":      ("LATENT",      {"tooltip": "AV chunk template from LTXVConcatAVLatent. "
                                                            "Defines new-frames shape for each chunk."}),
                "clss_config": ("CLSS_CONFIG", {}),
                "num_chunks":  ("INT",         {"default": 10, "min": 1, "max": 500,
                                                "tooltip": "Total chunks. Output frames = num_chunks × new_frames × time_scale."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional guide image for image-to-video (i2v). "
                                               "First frame of the first chunk is fully conditioned on this image. "
                                               "Resized automatically to match the latent spatial dimensions."}),
                "vae":   ("VAE",   {"tooltip": "VAE for encoding the i2v guide image. "
                                               "Connect the VAE from LTXVideo Loader. Required when image is connected."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Frame rate used for audio accounting and time labels in "
                               "the log.  Must match the frame_rate set on LTXVConditioning."}),
                "detail_anchor": (["on", "off"], {
                    "default": "on",
                    "tooltip": "Scene-referenced detail-band anchor.  Counters the measured "
                               "long-run drift where coarse-structure (low-band) energy "
                               "inflates while high-frequency detail is only ever shrunk — "
                               "the mechanism behind progressive detail loss.  Symmetric "
                               "per-band gains sqrt(E_ref/E), hard-capped to [0.90, 1.10] "
                               "(low) / [0.90, 1.12] (high) per chunk, re-baselined at "
                               "scene changes.  'off' restores previous behaviour exactly "
                               "(the vid_hf metric is still logged)."}),
                "video_slb_tau_mult": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.25,
                    "tooltip": "Multiplier on the VIDEO SLB re-noise schedule (base = "
                               "tau_c × this; 1.0 = the validated default, overlap placed "
                               "at ~0.95 strength).  LOWER values pin the seam harder: "
                               "0.0 = frozen clean overlap (the open-loop baseline's "
                               "placement — strongest continuity; long-run drift risk "
                               "moves to vid_std).  Built against the 2026-08-16 run "
                               "where coarse-layout swing events clustered at chunk seams "
                               "(chunk 2→3, t≈34.7 s: Δlayout −0.32 within 3 frames, "
                               "visible as jumping/morphing) — the re-noised overlap lets "
                               "the model re-lay-out the scene right after the seam.  "
                               "UNVALIDATED: gate on vid_bnd/vid_std and the layout-drift "
                               "event table, validate by eye."}),
                "audio_shift_mult": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "PER-MODALITY AUDIO SHIFT (manual experiment lever only): "
                               "**0 = AUTO (default)**: BOTH modalities share the "
                               "connected token-law schedule — the model's training "
                               "distribution.  The video needs the token law (a 2.05 "
                               "whole-schedule cap broke video live) and the audio "
                               "needs the same sigma as the video at each step (the "
                               "per-modality split broke audio live) — 2026-08-23.  "
                               "For audio quality at long templates, raise the "
                               "scheduler steps instead (reference default 30).  "
                               "1.0 = off (same as AUTO).  Manual values ≠1 set "
                               "audio shift = value × video shift (v3 split: audio "
                               "keeps σ0 = 1.0 full-noise start; only the schedule "
                               "shape changes)."}),
                "open_loop": (["off", "on"], {
                    "default": "off",
                    "tooltip": "Open-loop BASELINE arm for ablation: naive sliding-window "
                               "generation — disables ALL closed-loop corrections (overlap "
                               "re-noising is forced to tau_c=0 frozen SLB, AdaIN skipped, "
                               "detail/std anchors skipped, audio RMS "
                               "anchor treated as 'off').  SLB/anchor-bank telemetry keeps "
                               "running so trend lines stay comparable against closed-loop "
                               "runs.  Expect vid_std / aud_rms drift to return — that is "
                               "the measurement."}),
                "object_identity": (["off", "on"], {
                    "default": "off",
                    "tooltip": "Spatially-resolved identity metric: decodes a short clip "
                               "around each chunk seam (requires vae connected) and reports "
                               "per-grid-cell (8x8) cosine — mean AND min — between the two "
                               "pixel frames straddling the seam.  Catches localized object "
                               "replacement that the mean-pooled identity_sim averages "
                               "away; watch the MIN.  Logged as vid_obj in the trend "
                               "summary.  Metric only, generation unaffected."}),
                "auto_seam_pin": (["off", "on"], {
                    "default": "off",
                    "tooltip": "EVENT-TRIGGERED SEAM PIN: when a chunk contains a coarse "
                               "layout JUMP (max per-frame layout-sim drop > 0.25 or layout "
                               "sim < 0.05), the NEXT chunk's video "
                               "overlap is placed FROZEN (tau_c=0) so generation "
                               "continues from the LAST GOOD FRAMES exactly, instead of "
                               "re-laying-out from the re-noised context.  The corrector "
                               "source is the fresh SLB (0.3 s old), not the anchor bank "
                               "(1-2 chunks old).  Chunk 1 is excluded (scene "
                               "establishment, not a jump).  Cannot protect the final chunk.  "
                               "UNVALIDATED — and NOTE the layout metric was rebuilt "
                               "2026-08-17 (sign-free coarse power map) after the "
                               "pre-v2 pooled-cosine metric false-positived: the pin-on "
                               "runs fired on single-frame dips ABSENT from the decoded "
                               "video.  Thresholds are uncalibrated against a real event: "
                               "treat a trigger as a candidate and confirm by eye."}),
                "audio_corrections": (["on", "off"], {
                    "default": "on",
                    "tooltip": "BARE-METAL A/B (2026-08-20): 'off' disables EVERY hardcoded "
                               "audio statistical correction — SLB/ref env-flatten, the RMS "
                               "anchor, chunk-1 boundary fix (DC+spectral+RMS), SLB spectral "
                               "normalization, and the final assembly spectral norm.  What "
                               "remains is the bare model + SLB continuity + ref_audio.  "
                               "Built after the 103 s live run decoded as a broadband NOISE "
                               "BED (8-16 kHz the loudest band at t=70 s) while every "
                               "latent-sim metric read green — the corrections were tuned on "
                               "telemetry that cannot see perception.  (The audio SLB is "
                               "always frozen now — no re-noise knob remains.)  "
                               "Judge BY EAR on a short run — latent metrics are not "
                               "evidence."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "LTX-CLSS"

    @torch.inference_mode()
    def generate(
        self,
        guider,
        sampler,
        sigmas,
        noise,
        latent,
        clss_config: CLSSConfig,
        num_chunks: int,
        image=None,
        vae=None,
        fps: float = 24.0,
        detail_anchor: str = "on",
        video_slb_tau_mult: float = 1.0,
        audio_shift_mult: float = 0.0,
        open_loop: str = "off",
        object_identity: str = "off",
        auto_seam_pin: str = "off",
        audio_corrections: str = "on",
    ):
        import dataclasses
        import math

        # ── Full settings dump (raw inputs, unconditional) ──────────────────
        # Printed BEFORE any auto-derivation so two runs with identical widget
        # values produce byte-identical text here — diff this block first when
        # two "same settings" runs disagree.  Auto-derived values
        # (anchor_force_every) are logged separately below, at the point
        # they're computed.
        print("[CLSS] ══════════ SETTINGS: CLSSStreamingSampler (Stage 1) ══════════")
        print(f"[CLSS]   num_chunks={num_chunks}  fps={fps}")
        print(f"[CLSS]   image={'connected' if image is not None else 'none'}  "
              f"vae={'connected' if vae is not None else 'none'}")
        print(f"[CLSS]   audio SLB: on (auto)  detail_anchor={detail_anchor!r}")
        print(f"[CLSS]   audio_anchor='rms_only' (fixed)  audio SLB=frozen (no tau_c)  "
              f"video_slb_tau_mult={video_slb_tau_mult}  "
              f"audio_shift_mult={audio_shift_mult}")
        print(f"[CLSS]   open_loop={open_loop!r}  object_identity={object_identity!r}  auto_seam_pin={auto_seam_pin!r}  audio_corrections={audio_corrections!r}")
        print(f"[CLSS]   clss_config={dataclasses.asdict(clss_config)}")
        print(f"[CLSS]   noise.seed={getattr(noise, 'seed', 'unknown')}  "
              f"guider.cfg={getattr(guider, 'cfg', getattr(guider, 'cfg_scale', 'unknown'))}  "
              f"guider.audio_cfg={getattr(guider, 'audio_cfg', 'unknown')}")
        # The sigma schedule was the one input NOT dumped — and it is a prime
        # suspect for audio-vs-original quality gaps: the reference
        # LTX2Scheduler (ltx-core schedulers.py) is shifted (e^2.05) AND
        # stretched to terminal sigma 0.1 (30-step reference ends ...0.388,
        # 0.266, 0.1, 0.0 — almost no low-sigma sampling), unlike generic
        # ComfyUI schedulers which fill the low-sigma tail densely.
        print(f"[CLSS]   sigmas: n={len(sigmas)}  "
              f"values={[round(float(s), 4) for s in sigmas]}")
        # Over-shift guard (measured 2026-08-07): LTXVScheduler derives its shift
        # from the connected latent's token count (nodes_lt.py:646), so a LONG
        # single-window run (draft / ceiling test) silently gets a runaway
        # top-heavy schedule — 61 lf → shift e^5.39 → 16/20 steps above σ=0.8 →
        # audio latent deflated and off-manifold (RMS 0.38 vs healthy ~0.8,
        # min/max −6.08/+1.78) while video stays fine.  Verified numerically
        # against both logs (formula reproduces the logged σ exactly).  The
        # reference LTX2Scheduler uses FIXED shift e^2.05; the validated chunked
        # config lands at e^1.84 via the 16-lf template (8/20 steps above 0.9).
        # Fix for long-window runs: disconnect the scheduler's latent input
        # (tokens default 4096 → shift = max_shift = 2.05, the reference value).
        _n_steps = max(1, len(sigmas) - 1)
        _n_hi = int(sum(1 for s in sigmas[:-1] if float(s) >= 0.9))
        if _n_hi > _n_steps // 2:
            print(f"[CLSS]   WARNING: top-heavy sigma schedule ({_n_hi}/{_n_steps} "
                  f"steps ≥ σ=0.9) — the connected template's token count (52 lf → "
                  f"shift 4.68, nodes_lt.py:646) starves the AUDIO of low-σ steps "
                  f"(7/20 below σ=0.9 vs 12/20 at the good 121-px config).  AUTO "
                  f"re-schedules both modalities onto the shared {_KNOWN_GOOD_SHIFT} "
                  f"curve below (see the audio_shift_mult line).")
        print("[CLSS] ═════════════════════════════════════════════════════════════")

        # ── Ablation arms (off by default — off = byte-exact previous behaviour) ──
        # open_loop: naive sliding-window baseline.  All closed-loop corrections
        # are gated on `_open_loop` at their application sites below (re-noising
        # forced to tau_c=0, post_process / detail anchor / std anchor skipped,
        # audio anchor treated as 'off'); the SLB is still placed and
        # update_buffer still runs, so telemetry stays comparable.
        _open_loop = (open_loop == "on")
        _auto_pin = (auto_seam_pin == "on")
        _pin_next = False   # auto seam pin: next chunk's video overlap placed FROZEN
        # audio_anchor fixed to "rms_only" (2026-08-19): scalar RMS gain toward the
        # capped EMA target, upward-only — matches the reference streaming pipeline.
        # The rms_dc arm did per-channel DC surgery (our largest un-reference-justified
        # audio edit) and was removed; open_loop still forces the anchor off.
        audio_anchor = "rms_only"
        if _open_loop:
            print("[CLSS S1] open_loop=on: all closed-loop corrections disabled (baseline arm)")
            audio_anchor = "off"
        # audio_corrections=off (2026-08-20) = BARE audio path A/B: the 103 s live
        # run decoded as a broadband noise bed while every latent-sim metric read
        # green — the statistical pins below were all tuned on telemetry that cannot
        # see perception.  This arm strips them all (env-flatten, RMS anchor,
        # chunk-1 boundary fix, SLB spectral norm, final spectral norm); the audio
        # SLB is always frozen (no tau_c on audio since 2026-08-20).
        _audio_corr = (audio_corrections == "on")
        if not _audio_corr:
            audio_anchor = "off"
            print("[CLSS S1] audio_corrections=off: BARE audio path — env-flatten, "
                  "RMS anchor, chunk-1 boundary fix, SLB spectral norm and final "
                  "spectral norm ALL disabled (A/B diagnostic — judge by ear)")
        # object_identity: decoded seam-frame grid metric — needs the VAE.
        _obj_id = (object_identity == "on")
        if _obj_id and vae is None:
            print("[CLSS S1] WARNING: object_identity=on requires vae connected — "
                  "metric disabled.  Connect the VAE output from LTXVideo Loader.")
            _obj_id = False

        # ── Auto-derived settings (not user knobs) ───────────────────────
        # anchor_force_every: force a bank entry roughly every quarter of the run
        # so the anchor bank actually grows on long videos (7 chunks previously
        # produced bank_size=2 with the fixed default of 5) while never anchoring
        # more often than every 2 chunks.
        if clss_config.anchor_force_every <= 0:
            _auto_anchor = max(2, min(5, math.ceil(num_chunks / 4)))
            clss_config = dataclasses.replace(clss_config, anchor_force_every=_auto_anchor)
            print(f"[CLSS] auto: anchor_force_every={_auto_anchor} (num_chunks={num_chunks})")

        samples = latent["samples"]

        # Split AV template → video [1,128,F_v,H,W] + optional audio [1,C,F_a,freq]
        is_av = isinstance(samples, comfy.nested_tensor.NestedTensor)
        if is_av:
            vid_tmpl, aud_tmpl = samples.unbind()
        else:
            vid_tmpl = samples
            aud_tmpl = None

        B, C_v, new_lf, H, W = vid_tmpl.shape
        overlap_lf = clss_config.overlap_latent_frames
        device = vid_tmpl.device

        # Pre-encode i2v guide image (once, before the chunk loop)
        img_guide_latent: torch.Tensor | None = None
        if image is not None and vae is not None:
            i2v_scale_factors = vae.downscale_index_formula
            _, img_guide_latent = LTXVAddGuide.encode(vae, W, H, image[:1], i2v_scale_factors)
            print(f"[CLSS] i2v: guide image encoded, latent shape={list(img_guide_latent.shape)}")
        elif image is not None:
            print("[CLSS] WARNING: image connected without vae — i2v skipped. "
                  "Connect the VAE output from LTXVideo Loader.")

        if new_lf > 21:
            print(
                f"[CLSS] chunk length: new_lf={new_lf} (~{new_lf * 8 / fps:.1f}s per chunk at "
                f"{fps:.0f} fps). VRAM and per-step time grow with chunk length, and the CLSS "
                f"correction constants (tau_c schedule, AdaIN caps, anchors) were tuned at 13 lf. "
                f"See the RoPE-window check below for the hard ceiling."
            )

        # Audio overlap proportional to video overlap — carries speech/dialog across chunks.
        # Without this, each chunk starts from pure noise → incoherent audio, broken dialog.
        if aud_tmpl is not None:
            B_a, C_a, new_af, freq = aud_tmpl.shape
            # Audio timeline accounting — continuation pixel mapping.
            # The causal discount (lf−1)·8+1 applies ONCE, at the very first video
            # frame of the whole sequence.  Every subsequent latent frame covers a
            # full 8 px.  The audio template (new_af for a standalone new_lf chunk)
            # defines the af-per-px rate: af_per_px = new_af / ((new_lf−1)·8+1).
            #   chunk 1 keeps new_af (covers (new_lf−1)·8+1 px)
            #   chunks 2+ keep new_af_cont = round(new_lf·8·af_per_px)  (cover new_lf·8 px)
            #   overlap (a continuation) covers overlap_lf·8 px → audio_overlap_af af
            # The old accounting kept new_af per chunk regardless, undercounting each
            # non-first chunk by 7 px ≈ 0.28 s — a cumulative A/V desync (~1.7 s at
            # 7 chunks: audio ended ~2 s before the video).
            _first_px  = (new_lf - 1) * 8 + 1
            # ── overlap_lf > new_lf guard (root cause of the 'quick jump forward
            # at every seam' run) ────────────────────────────────────────────
            # update_buffer stores min(overlap, F) frames — a chunk produces
            # new_lf, so the SLB buffer can NEVER hold more than new_lf.  Any
            # overlap_lf beyond that allocates window slots the replacement
            # cannot fill: they stay pure noise, get freely generated as
            # continuation, and the accounting then DISCARDS them — skipping
            # (overlap_lf−new_lf) frames of motion at every seam.  Measured at
            # 16/13: video_SLB shape 13 vs overlap=16, biggest drift event at
            # the first seam (t=4.2s, Δorigin_sim −0.4263), 135af dropped vs a
            # 109af audio buffer, chunk-2 ref_audio lost.  Clamping here (before
            # the audio accounting) fixes video window, audio ledger, and ref
            # saving in one place.
            if overlap_lf > new_lf:
                print(f"[CLSS] overlap_lf={overlap_lf} > new_lf={new_lf}: the SLB buffer "
                      f"only carries the {new_lf} frames a chunk produces — the extra "
                      f"{overlap_lf - new_lf} slots would be free-generated then DISCARDED, "
                      f"skipping ~{(overlap_lf - new_lf) * 8 / fps:.1f}s of motion at every "
                      f"seam.  Clamping effective overlap to {new_lf}.  For longer context, "
                      f"raise new_lf as well.")
                overlap_lf = new_lf
            _af_per_px = new_af / _first_px if _first_px > 0 else 0.0
            audio_overlap_af = round(overlap_lf * 8 * _af_per_px)
            new_af_cont      = round(new_lf * 8 * _af_per_px)
            # ref_audio window length (the clean negative-RoPE context the model
            # composes the next chunk against): exactly one overlap — the legacy
            # default (the ref_audio_seconds extension knob was removed 2026-08-19).
            _ref_len_af = audio_overlap_af
            print(f"[CLSS] audio accounting: af_per_px={_af_per_px:.4f}  "
                  f"chunk1={new_af}af  chunks2+={new_af_cont}af  overlap={audio_overlap_af}af  "
                  f"total={new_af + (num_chunks - 1) * new_af_cont}af for "
                  f"{(num_chunks * new_lf - 1) * 8 + 1}px")
            print(f"[CLSS] audio ref window: {_ref_len_af}af "
                  f"(~{_ref_len_af / (fps * _af_per_px):.1f}s, 1x overlap)")
            # Audio SLB is always on: the per-chunk energy anchor (applied before
            # new_aud feeds the SLB/ref) keeps the fed-forward context from
            # drifting, and without the SLB, ref_audio alone is too weak for
            # cross-chunk audio continuity.  ref_audio length = the audio overlap.
            print(f"[CLSS] audio SLB: on (auto) — {audio_overlap_af}f, cross-chunk "
                  f"audio continuity")

            # ── 20 s AV RoPE wall ───────────────────────────────────────────
            # AV RoPE positions are in SECONDS, normalized by max_pos=20 for
            # both modalities (av_model.py positional_embedding_max_pos=[20] /
            # max_pos=[20,2048,2048]); the phase argument is
            # indices*(t/max_pos*2 - 1) (model.py), so training only ever saw
            # t/20 in [0,1], i.e. the argument in [-1,+1].  What matters is the
            # WINDOW THE MODEL SEES PER CHUNK — overlap + new — not the total
            # output length: chunking is exactly what keeps arbitrarily long AV
            # inside the grid.
            #
            # EAR-VALIDATED (2026-08-10), identical seed/prompt/settings:
            #     window    latent std   dynR     verdict
            #     19.8 s    0.62         9.8 dB   "much better"
            #     104.3 s   0.49         3.3 dB   "drone and shity"
            # A 6x17.3 s chained run then reached std 0.63 and sounded ok.
            # Note dynR — NOT bandwidth — is the axis that tracked the ear; the
            # better-sounding take had LOWER ro99.
            # Chunk 1 carries no SLB overlap, so a single-chunk run sees only new_af.
            # (Measured here; ENFORCED by the "RoPE-wall AUTO-ADJUSTMENT" block
            # right below.)
            _win_af = (new_af if num_chunks == 1
                       else audio_overlap_af + max(new_af, new_af_cont))
            _win_s = _win_af / (fps * _af_per_px) if _af_per_px > 0 else 0.0
            if num_chunks == 1:
                print(f"[CLSS] audio window: {_win_s:.1f}s per chunk "
                      f"(single chunk: new {new_af}af, no overlap)")
            else:
                print(f"[CLSS] audio window: {_win_s:.1f}s per chunk "
                      f"(overlap {audio_overlap_af}af + new {max(new_af, new_af_cont)}af)")
        else:
            B_a = C_a = new_af = freq = audio_overlap_af = new_af_cont = _ref_len_af = 0
            _win_af = 0
            _win_s = 0.0

        # ── RoPE-wall AUTO-ADJUSTMENT (2026-08-22) ─────────────────────────
        # The 20 s wall applies to BOTH modalities and to the WINDOW the model
        # sees per chunk (overlap + new).  The code above only MEASURED it; the
        # older code merely WARNED when it was exceeded — the user's longer-chunk
        # runs (e.g. 409 px ≈ 17 s new + 8 lf overlap ≈ 20.0 s, right on the
        # edge) still broke.  This block makes the run FIT inside the wall:
        #   1. the overlap is clamped so overlap+new ≤ wall−margin — the user's
        #      chunk length and chunk count are preserved, only the continuity
        #      context shrinks;
        #   2. if even a zero overlap exceeds the wall (one chunk alone is too
        #      long), each chunk is AUTO-SPLIT into the fewest uniform
        #      sub-chunks that fit, so any requested length works.
        _max_win_lf = max(1, int((_ROPE_WALL_S - _ROPE_WALL_MARGIN_S) * fps / 8))
        _eff_overlap = overlap_lf
        _eff_num_chunks = num_chunks
        _split_into = 1                       # sub-chunks per user chunk (1 = no split)
        _win_v_s = (new_lf if num_chunks == 1 else overlap_lf + new_lf) * 8 / fps
        _target_s = _ROPE_WALL_S - _ROPE_WALL_MARGIN_S
        _ov_orig = overlap_lf
        if max(_win_v_s, _win_s) > _target_s:
            _win_orig_s = max(_win_v_s, _win_s)
            # Shrink the overlap (the window is overlap + new) until it fits —
            # but never below _MIN_OVERLAP_LF: a thinner SLB is not worth
            # keeping, and the split branch below takes over instead.
            while max(_win_v_s, _win_s) > _target_s and _eff_overlap > _MIN_OVERLAP_LF:
                _eff_overlap -= 1
                _win_v_s = (new_lf if num_chunks == 1 else _eff_overlap + new_lf) * 8 / fps
                if aud_tmpl is not None:
                    audio_overlap_af = round(_eff_overlap * 8 * _af_per_px)
                    _ref_len_af = audio_overlap_af
                    _win_af = (new_af if num_chunks == 1
                               else audio_overlap_af + max(new_af, new_af_cont))
                    _win_s = _win_af / (fps * _af_per_px) if _af_per_px > 0 else 0.0
            if max(_win_v_s, _win_s) > _target_s:
                # Even with zero overlap a single chunk exceeds the wall — the
                # chunk content alone is longer than the window allows.  Split
                # each user chunk into the fewest uniform sub-chunks that fit,
                # keeping a meaningful overlap for continuity.
                _eff_overlap = min(overlap_lf, max(_MIN_OVERLAP_LF, _max_win_lf // 5))
                _max_new = max(1, _max_win_lf - _eff_overlap)
                _split_into = max(2, math.ceil(new_lf / _max_new))
                _eff_num_chunks = num_chunks * _split_into
                if aud_tmpl is not None:
                    audio_overlap_af = round(_eff_overlap * 8 * _af_per_px)
                    _ref_len_af = audio_overlap_af
            # Re-derive the anchor cadence from the effective chunk count.
            if _eff_num_chunks != num_chunks:
                _auto_anchor = max(2, min(5, math.ceil(_eff_num_chunks / 4)))
                clss_config = dataclasses.replace(clss_config, anchor_force_every=_auto_anchor)
            print(f"[CLSS] ⚠ RoPE WALL: window was {_win_orig_s:.1f}s "
                  f"(target ≤ {_target_s:.1f}s, max_pos={_ROPE_WALL_S:.0f}s) — "
                  + (f"overlap clamped {_ov_orig}→{_eff_overlap} lf"
                     if _split_into == 1 else
                     f"chunk too long: auto-split each {new_lf}-lf chunk into "
                     f"{_split_into} sub-chunks ({_eff_num_chunks} total), "
                     f"overlap {_eff_overlap} lf"))
        # Propagate the effective overlap into the CLSS config so the SLB buffer
        # (update_buffer) stores exactly the frames each chunk places.
        if _eff_overlap != overlap_lf:
            clss_config = dataclasses.replace(clss_config, overlap_latent_frames=_eff_overlap)

        # Per-chunk plan: (video new frames, audio new frames).  In the normal
        # case every user chunk is new_lf and chunk 1's audio keeps new_af (the
        # causal (lf−1)·8+1 px discount), continuations new_af_cont.  After a
        # split, each user chunk's audio budget (new_af / new_af_cont) is shared
        # across its sub-chunks proportionally to their frames.
        if _split_into == 1:
            _chunk_plan = ([(new_lf, new_af)] + [(new_lf, new_af_cont)] * (num_chunks - 1)
                           if aud_tmpl is not None
                           else [(new_lf, 0)] * num_chunks)
        else:
            _base, _extra = new_lf // _split_into, new_lf % _split_into
            _lens = [_base + (1 if _j < _extra else 0) for _j in range(_split_into)]
            _chunk_plan = []
            for _ci in range(num_chunks):
                _budget = (new_af if _ci == 0 else new_af_cont) if aud_tmpl is not None else 0
                if _budget > 0:
                    _cum, _prev = 0, 0
                    for _v in _lens:
                        _cum += _v
                        _cur = round(_cum * _budget / new_lf)
                        _chunk_plan.append((_v, _cur - _prev))
                        _prev = _cur
                else:
                    for _v in _lens:
                        _chunk_plan.append((_v, 0))
        _max_new_planned = max(p[0] for p in _chunk_plan)
        print(f"[CLSS] RoPE-adjusted: chunks={_eff_num_chunks}  "
              f"per-chunk new=[{', '.join(str(p[0]) for p in _chunk_plan[:16])}"
              + (", …" if len(_chunk_plan) > 16 else "")
              + f"]  overlap={_eff_overlap} lf  "
              f"max window={(_max_new_planned + (0 if num_chunks == 1 else _eff_overlap)) * 8 / fps:.1f}s"
              + (f"  audio new=[{', '.join(str(p[1]) for p in _chunk_plan[:16])}]"
                 if aud_tmpl is not None else ""))

        # Read scene conditionings already stored inside the guider.
        # original_conds["positive"] is a list of converted cond dicts (one per scene
        # after convert_cond ran inside CFGGuider.set_conds). N > 1 means scene prompts.
        pos_conds = guider.original_conds.get("positive", [])
        num_scenes = len(pos_conds)

        _cfg_val = getattr(guider, "cfg", getattr(guider, "cfg_scale", "unknown"))
        _aud_cfg_val = getattr(guider, "audio_cfg", None)
        _cfg_str = (f"video_cfg={_cfg_val} audio_cfg={_aud_cfg_val} (split)"
                    if _aud_cfg_val is not None else f"guider_cfg={_cfg_val} (shared, no split)")
        print(f"[CLSS] Starting — chunks={_eff_num_chunks}"
              + (f" (requested {num_chunks}, auto-split ×{_split_into})" if _split_into > 1 else "")
              + f", new_lf={new_lf}, overlap_lf={_eff_overlap}, "
              f"scenes={num_scenes}, tau_c={clss_config.tau_c}, beta={clss_config.beta}, "
              f"mode={'AV' if is_av else 'video-only'}, {_cfg_str}"
              + (f", new_af={new_af}, audio_overlap_af={audio_overlap_af}" if is_av else ""))

        # §item-7/8: corrections active + reproducibility metadata
        _corrections = {
            "renoise": clss_config.tau_c > 0,
            "adain":   clss_config.beta > 0,
            "anchor":  clss_config.anchor_max_size > 0,
        }
        _rho_loop = (1.0 - clss_config.beta) * (1.0 - clss_config.tau_c)
        _seed = getattr(noise, "seed", "unknown")
        print(
            f"[CLSS] Config: corrections={_corrections}"
            f"  rho_loop={_rho_loop:.4f}  seed={_seed}"
            f"  beta={clss_config.beta}"
            f"  ema_lambda={clss_config.ema_lambda}"
        )

        clss_state = CLSSState(clss_config)
        acc_video: list[torch.Tensor] = []
        acc_audio: list[torch.Tensor] = []
        audio_chunk_ends: list[int] = []   # cumulative audio frame count per chunk

        # audio_slb_latent: overlap-time audio SLB placed at lat_aud[:,0:audio_overlap_af]
        # with mask=tau_c.  Needed because model_base.py process_timestep() multiplies
        # audio_denoise_mask × sigma → per-token a_timestep.  Without tau_c on overlap
        # audio, those tokens get full-sigma a_timestep → a2v treats them as maximally
        # noisy → video discontinuity.  Content: last audio_overlap_af frames of new_aud
        # (same temporal period as video SLB).
        audio_slb_latent:     torch.Tensor | None = None
        # audio_overlap_latent: pre-overlap frames injected as ref_audio at negative RoPE
        # positions (av_model.py line 708).  Temporal context for what preceded the chunk.
        audio_overlap_latent: torch.Tensor | None = None

        # Tracking state for per-chunk coherence metrics (§items 1,2,6)
        _s1_prev_last:       torch.Tensor | None = None  # [B, C_v, H, W] last corrected frame
        _s1_prev_tail:       torch.Tensor | None = None  # [B, C_v, ≤2, H, W] last frames (object_identity seam decode only)
        _s1_vid_std_ref:     float | None = None          # chunk-0 global video std (creep anchor)
        _prev_scene_idx:     int | None = None            # scene of the previous chunk (stat-anchor re-baseline)
        _s1_band_ref:        tuple[float, float] | None = None  # scene-first (E_low, E_high) detail-band reference
        _origin_ref:         torch.Tensor | None = None   # FIXED scene-first frame (origin-drift telemetry)
        _origin_layout:      torch.Tensor | None = None   # its low-band spatial map
        _origin_track:       list = []                    # per-output-frame origin_sim (whole run)
        _layout_track:       list = []                    # per-output-frame layout_sim (whole run)
        _layout_argmin_track: list = []                   # per-chunk frame index of the layout minimum (phase-lock check)
        _aud_peak_track:     list = []                    # per-chunk audio energy peak frame (phase-lock check)
        _prev_aud_env:       torch.Tensor | None = None   # previous chunk's audio energy envelope
        # Per-chunk trend accumulators → compact end-of-run summary so drift is
        # readable at a glance instead of scraping N chunks by hand.
        _trend = {
            "vid_std":   [],  # post-correction video global std (creep check)
            "vid_ident": [],  # identity_sim vs nearest anchor (content drift)
            "vid_intra": [],  # intra-chunk sim — repetition signal (0.73 healthy, 0.97+ = looping)
            "vid_bnd":   [],  # video boundary_sim (chunk seam)
            "vid_obj":   [],  # decoded seam 8x8 grid-cell cosine MEAN (object_identity only)
            "vid_hf":    [],  # high-frequency energy share (detail retention)
            "vid_origin": [], # per-chunk floor of frame-vs-scene-first similarity (drift)
            "aud_env":   [],  # chunk-to-chunk loudness-gesture correlation (repetition)
            "aud_rms":   [],  # audio RMS AFTER anchor (energy stability)
            "aud_bnd":   [],  # audio boundary_sim (content seam)
            "aud_slb":   [],  # audio SLB honored (continuity mechanism health)
            "aud_wc":    [],  # audio within-chunk END sim (intra-chunk audio drift)
            "aud_hf":    [],  # audio high-freq energy ratio (spectral drift)
            "g_slb":     [],  # measured open-loop transformer gain (Eq. 8, measure_g only)
        }
        _s1_aud_prev_last:   torch.Tensor | None = None  # [B, C_a, 1, freq] last audio frame
        _s1_audio_ema_rms:   float | None = None          # slow-drifting RMS anchor target (capped vs origin)
        _s1_audio_rms_ref:   float | None = None         # chunk-0 scalar RMS (onset-excluded) — correction target
        _s1_audio_freq_ref:  list[float]  | None = None  # chunk-0 per-bin energy reference (fixed; USED — SLB spectral norm at save + final assembly spectral norm)
        # Rolling audio tail (reference pipeline.py:771-806): last overlap+ref frames of
        # accumulated output, kept across chunks.  Lets ref_audio be a FULL window ending
        # immediately before the next overlap, even when that window spans
        # a chunk boundary (with new_af=102, ov=60 the within-chunk pre-overlap region is
        # only 42f — the tail restores the missing frames from the previous chunk).
        # The ref window is one overlap, so the tail keeps ov + _ref_len_af = 2×ov frames.
        _s1_audio_tail:      torch.Tensor | None = None
        # Note: identity_sim is computed vs nearest bank anchor (not fixed chunk-1) so it tracks
        # within-scene identity; with a single-anchor bank it equals vs-chunk-1 and is flagged.

        # Pre-generate full-video noise once — ComfyUI's RandomNoise seeds from noise.seed, so
        # two chunks with the same latent shape (e.g. chunks 2 and 3, both new_lf frames) produce
        # IDENTICAL noise.  Generating a [B, C_v, num_chunks*new_lf, H, W] field here and slicing
        # per chunk gives each chunk's new frames a distinct, spatially-coherent noise region.
        _noise_seed_s1 = getattr(noise, "seed", 0)
        _noise_tmpl_s1 = torch.zeros(B, C_v, num_chunks * new_lf, H, W, device=device)
        _full_noise_vid_s1: torch.Tensor = noise.generate_noise({"samples": _noise_tmpl_s1})
        del _noise_tmpl_s1
        print(
            f"[CLSS] S1 noise: pre-generated shape={list(_full_noise_vid_s1.shape)} "
            f"seed={_noise_seed_s1} fingerprint={_full_noise_vid_s1.flatten()[:4].tolist()}"
        )
        # Temporally-correlated noise prior (EXPERIMENTAL, see CLSSConfig).
        # Mixes one run-constant shared frame into every video noise frame:
        # n_t = sqrt(1-a)·eps_t + sqrt(a)·eps_shared.  Marginals stay N(0,1);
        # temporal noise correlation becomes a at all lags.  Targets the ~4 s
        # layout oscillation (i.i.d. noise gives each ~4 s span an independent
        # low-frequency content suggestion → a fresh motion arc).  Applied to
        # the full field BEFORE slicing, so cross-chunk consistency of the
        # correlated field is automatic.  seed+2 (audio field uses seed+1).
        # Video-only: the audio arc is window-locked, a different mechanism.
        _ntc = float(getattr(clss_config, "noise_temporal_corr", 0.0))
        if _ntc > 0.0:
            _g_shared = torch.Generator(device="cpu").manual_seed(
                (int(_noise_seed_s1) + 2) % (2 ** 63))
            _eps_shared = torch.randn(
                _full_noise_vid_s1.shape[0], _full_noise_vid_s1.shape[1], 1,
                _full_noise_vid_s1.shape[3], _full_noise_vid_s1.shape[4],
                generator=_g_shared, dtype=_full_noise_vid_s1.dtype,
            ).to(_full_noise_vid_s1.device)
            _full_noise_vid_s1 = (
                math.sqrt(1.0 - _ntc) * _full_noise_vid_s1
                + math.sqrt(_ntc) * _eps_shared
            )
            print(
                f"[CLSS] S1 noise: EXPERIMENTAL temporal-corr mix a={_ntc:.2f} "
                f"(shared-frame seed={(int(_noise_seed_s1) + 2) % (2 ** 63)}) "
                f"fingerprint={_full_noise_vid_s1.flatten()[:4].tolist()}"
            )
        # S1 AUDIO noise field.  Without this, _SlicedNoise had no audio field in
        # Stage 1 and fell back to torch.randn_like — GLOBAL RNG, unseeded: the
        # noise seed controlled video only, and every run rolled fresh audio noise.
        # Proven by two runs with byte-identical SETTINGS blocks (same seed) whose
        # chunk-1 outputs diverged (audio mean 0.0431 vs -0.0017; video band_E
        # differed too, because joint AV attention lets the differing audio perturb
        # the video).  Every A/B comparison before this fix had audio noise as an
        # uncontrolled variable.  One seeded field (seed+1, the Stage-2 convention)
        # sliced per chunk makes runs reproducible AND gives chunks
        # distinct-but-coherent audio noise.
        _full_noise_aud_s1: torch.Tensor | None = None
        _s1_a_noise_pos = 0
        if aud_tmpl is not None:
            _aud_seed_s1 = (int(_noise_seed_s1) + 1) % (2 ** 63)   # +1 = S2 convention; mod: max-seed overflow guard
            _g_aud_s1 = torch.Generator(device="cpu").manual_seed(_aud_seed_s1)
            _total_af_s1 = new_af + (num_chunks - 1) * new_af_cont
            _full_noise_aud_s1 = torch.randn(
                B_a, C_a, _total_af_s1, freq, generator=_g_aud_s1, dtype=aud_tmpl.dtype)
            print(
                f"[CLSS] S1 audio noise: pre-generated shape={list(_full_noise_aud_s1.shape)} "
                f"seed={_aud_seed_s1} "
                f"fingerprint={_full_noise_aud_s1.flatten()[:4].tolist()}"
            )

        # ── Shared known-good schedule (2026-08-23, user-verified live) ────
        # AUTO re-schedules an over-shifted template onto the shared 1.844
        # curve (the good 121-px regime) for BOTH modalities — see the note at
        # _KNOWN_GOOD_SHIFT above.  The per-modality split (v3) below is the
        # experiment lever that manual audio_shift_mult values still engage.

        # ── Per-modality audio shift (2026-08-12, v3; manual only) ─────────
        # The user's 2026-08-11 finding — pinning the SHARED scheduler shift to
        # e^1.84 helped audio but destroyed video — proves the two modalities
        # want DIFFERENT schedules.  v3 gives the AUDIO its OWN sigma SCHEDULE —
        # generated with audio sigma_shift = audio_shift_mult × (the video's
        # sigma_shift, recovered from the sigmas) — while the video keeps the
        # original sigmas byte-identical.  The audio still STARTS at full noise
        # (σ0 = 1.0: the coarse-structure phase is preserved); only the schedule
        # SHAPE changes (less top-heavy), which is exactly what the healthy
        # reference (shift 2.05, latent disconnected) does for audio.
        #
        # v2 lesson (2026-08-12, live): the previous linear scaling
        # (audio_sigma = mult×sigma) re-mixed the audio initial state to
        # mult×σ0 < 1, so the model was told it was already (1−mult) denoised
        # while the input was still mostly noise — it skipped the σ∈[mult,1]
        # coarse-structure phase and produced a quiet, under-developed latent
        # (full_aud std 0.48 → 0.24 at mult=0.6 = 'no audio').  v3 removes the
        # re-mix entirely.
        #
        # NB (2026-08-12, why the first version silently no-op'd): guider.inner_model
        # is None until prepare_sampling() runs INSIDE the sampler call — it is
        # assigned there (samplers.py:1233) and returns model.model, i.e. the SAME
        # base object as guider.model_patcher.model (sampler_helpers.py:202).  Patch
        # that base object, which is exactly what the sampling path calls.
        _base_pts = None
        _base_mod = getattr(getattr(guider, "model_patcher", None), "model", None)
        _a_sig = None
        _remap_fn = None
        _mult = float(audio_shift_mult)
        if (_mult <= 0.0 and aud_tmpl is not None
                and sigmas is not None and len(sigmas) >= 3):
            # AUTO (default 0.0) = SHARED KNOWN-GOOD SCHEDULE (2026-08-23):
            # the connected template's token law over-shifts long templates
            # (52 lf → 4.68) and starves the AUDIO of low-σ steps — 7/20 below
            # σ=0.9 vs 12/20 at the 121-px config (1.844), which is good in
            # both modalities (user-verified).  The video blockers that
            # historically made flat schedules collapse are gone (unbounded
            # rescale is clamped, euler_ancestral → euler, temporal-corr noise
            # → white).  AUTO therefore re-schedules an over-shifted template
            # onto the shared 1.844 curve for BOTH modalities — the model sees
            # one sigma per step either way, and the audio's low-σ tail is
            # preserved.  121 px stays byte-identical.  Manual audio_shift_mult
            # values (≠0, ≠1) still engage the v3 split as an experiment lever.
            _term_auto = max(float(sigmas[-2]), 1e-3)
            _s_v_auto, _ = _estimate_video_sigma_shift(sigmas.double(), terminal=_term_auto)
            _mult = 1.0   # shared schedule — no per-modality split
            if _s_v_auto > _KNOWN_GOOD_SHIFT + 1e-6:
                sigmas = _shifted_shared_schedule(sigmas, _KNOWN_GOOD_SHIFT, terminal=_term_auto)
                _n_lo = int(sum(1 for s in sigmas[:-1] if float(s) < 0.9))
                print(f"[CLSS S1]   audio_shift_mult=auto → SHARED {_KNOWN_GOOD_SHIFT} "
                      f"schedule for BOTH modalities: the connected template over-shifted "
                      f"({_s_v_auto:.3f}) and starved the audio of low-σ steps.  "
                      f"{_n_lo}/{len(sigmas) - 1} steps below σ=0.9.  "
                      f"New sigmas: {[round(float(s), 4) for s in sigmas]}")
            else:
                _n_lo = int(sum(1 for s in sigmas[:-1] if float(s) < 0.9))
                print(f"[CLSS S1]   audio_shift_mult=auto → connected schedule kept "
                      f"(shift {_s_v_auto:.3f} = the known-good 121-px regime, "
                      f"{_n_lo}/{len(sigmas) - 1} steps below σ=0.9)")
        _s_a_eff = _mult
        if (_mult != 1.0 and aud_tmpl is not None
                and sigmas is not None and len(sigmas) >= 3):
            _a_sig, _remap_fn, _s_v_est = _audio_shift_schedule(sigmas, _mult)
            _s_a_eff = _mult * _s_v_est
            print(f"[CLSS S1]   audio_shift_mult={_mult:.4f} — audio "
                  f"sigma_shift={_s_a_eff:.3f} (video sigma_shift≈{_s_v_est:.3f}); "
                  f"audio keeps σ0=1.0 FULL-noise start, schedule-shape remap only "
                  f"(v3 — v2's re-mix start deflated the audio, removed)")
        if _base_mod is not None and hasattr(_base_mod, "process_timestep"):
            _base_pts = _base_mod.process_timestep
            if _remap_fn is not None:
                def _pts(timestep, x, denoise_mask=None, audio_denoise_mask=None, **kw):
                    _ret = _base_pts(timestep, x, denoise_mask=denoise_mask,
                                     audio_denoise_mask=audio_denoise_mask, **kw)
                    if isinstance(_ret, tuple) and len(_ret) >= 2 and _ret[1] is not None:
                        # scale the audio timestep so mask==1 tokens read the REMAPPED
                        # sigma: mask·sigma·(a_i/sigma) = mask·a_i — the model is told
                        # the audio's real (remapped) noise level, never 'cleaner than
                        # reality' (the v1 bug).  mask<1 (SLB/frozen) tokens are handled
                        # by the mask itself and the mask-aware conversion.
                        _ts = timestep.float()
                        _scale = torch.where(
                            _ts > 1e-6, _remap_fn(_ts) / _ts.clamp(min=1e-6),
                            torch.ones_like(_ts))
                        return _ret[0], _ret[1] * _scale
                    return _ret
                _base_mod.process_timestep = _pts
                print(f"[CLSS S1]   per-modality audio sigma: audio follows its own "
                      f"{_s_a_eff:.3f}-shift schedule (custom AV sampler active, "
                      f"full-noise start)")
            elif _mult != 1.0:
                print(f"[CLSS S1]   WARNING: audio_shift_mult={_mult} requested "
                      f"but patch target unavailable (guider.model_patcher.model="
                      f"{type(_base_mod).__name__ if _base_mod is not None else None}) — "
                      f"audio shift NOT applied.")
        if _a_sig is not None and _mult != 1.0 and aud_tmpl is not None:
            sampler = copy.copy(sampler)
            # Honor the SELECTED sampler's determinism (2026-08-22): the wrapper
            # used to always replicate euler_ancestral_RF — silently overriding an
            # 'euler' selection with per-step noise injection.  The reference
            # pipeline samples deterministic Euler and never injects per-step
            # noise into the audio latent.
            _fname = getattr(getattr(sampler, "sampler_function", None), "__name__", "")
            _ancestral = "ancestral" in _fname
            sampler.sampler_function = _make_av_permodality_sampler(_a_sig, ancestral=_ancestral)
            if _ancestral:
                print("[CLSS S1]   NOTE: euler_ancestral injects fresh noise into the audio "
                      "latent at every step; the reference pipeline samples deterministic "
                      "euler (EulerDiffusionStep).  If the audio sounds noisy/warbly, "
                      "select 'euler' in KSamplerSelect.")

        _vid_pos = 0   # cumulative video-frame position across effective chunks
        for chunk_idx in range(_eff_num_chunks):
            is_first = chunk_idx == 0
            _cur_new_lf = _chunk_plan[chunk_idx][0]
            chunk_overlap = 0 if is_first else _eff_overlap
            total_lf = chunk_overlap + _cur_new_lf

            scene_idx = 0
            if num_scenes > 1:
                scene_idx = min(int(chunk_idx * num_scenes / _eff_num_chunks), _eff_num_chunks - 1)

            # Scene change → re-baseline the STATISTICS anchors.  These anchors
            # (video global-std, audio RMS/DC EMA) exist to stop autoregressive
            # drift WITHIN a scene; they are scene-blind by construction and, on
            # the first multi-scene run, measurably fought intended scene changes:
            # the std anchor amplified scene-2 by +3.6% and scene-3 by +7.4%
            # toward scene-1's contrast (log: g=1.0358, g=1.0741), and the audio
            # EMA forced scenes that legitimately run at RMS ~1.01-1.05 down
            # toward scene-1's 0.82, pinned at its ±5% cap the whole run.
            # Setting the refs to None makes the first chunk of each scene the
            # new baseline (its own init path re-fires), exactly as chunk 0 did
            # for scene 1.  Single-scene runs: no scene change ever fires, so
            # behaviour is byte-identical.  The CONTENT continuity mechanisms
            # (SLB, ref_audio, anchor bank) are untouched — the bank already has
            # its own scene-change handling (scene_change_streak).
            if chunk_idx > 0 and scene_idx != _prev_scene_idx:
                _s1_vid_std_ref    = None
                _s1_audio_rms_ref  = None
                _s1_audio_ema_rms  = None
                _s1_band_ref       = None
                _origin_ref        = None
                _origin_layout     = None
                print(f"[CLSS S1]   scene {(_prev_scene_idx or 0) + 1}→{scene_idx + 1}: "
                      f"statistics anchors re-baselined (video std, audio RMS now "
                      f"anchor to this scene's first chunk; content continuity "
                      f"mechanisms unchanged)")
            _prev_scene_idx = scene_idx

            has_slb     = not is_first and clss_state._overlap_latent is not None
            has_aud_slb = not is_first and audio_slb_latent is not None
            has_aud_ref = not is_first and audio_overlap_latent is not None
            print(f"[CLSS S1] ── Chunk {chunk_idx + 1}/{_eff_num_chunks} ── "
                  f"t=[{_vid_pos * 8 / fps:.2f}s:{(_vid_pos + _cur_new_lf) * 8 / fps:.2f}s] "
                  f"──────────────────")
            print(f"[CLSS S1]   video lf total={total_lf} (overlap={chunk_overlap}+new={_cur_new_lf}) "
                  f"scene={scene_idx + 1}/{num_scenes} "
                  f"video_SLB={'yes(tau_c=' + str(clss_config.tau_c) + ')' if has_slb else 'no(first)'}"
                  + (f"  audio_ref={'yes' if has_aud_ref else 'no(first)'}" if is_av else ""))

            # Per-chunk guider: unpack the right scene from the guider's positive.
            # Must be created before the i2v block so we can update its conditionings.
            guider_chunk = copy.copy(guider)
            if num_scenes > 1:
                guider_chunk.original_conds = {
                    **guider.original_conds,
                    "positive": [pos_conds[scene_idx]],
                }

            # Video latent: zeros + noise_mask = 1 (fully noisy)
            lat_vid = torch.zeros(B, C_v, total_lf, H, W, device=device)
            mask_vid = torch.ones(B, 1, total_lf, 1, 1, device=device)

            # §2.1 Place SLB at overlap frames with noise_mask = tau_c_eff (decaying
            # strength schedule -- see _tau_c_eff docstring).  chunk_idx-1 because the
            # schedule counts chunks THAT HAVE an overlap (first chunk has none).
            if has_slb:
                _tau_c_v = _tau_c_eff(clss_config.tau_c * video_slb_tau_mult,
                                      _VIDEO_TAU_C_CEILING, chunk_idx - 1)
                if _open_loop:
                    # Baseline arm: no calibrated re-noising — the SLB is placed
                    # FROZEN (strength 1.0), exactly the existing tau_c=0 path.
                    _tau_c_v = 0.0
                if _pin_next:
                    _tau_c_v = 0.0
                    print("[CLSS S1]   auto seam pin: previous chunk flagged a layout "
                          "event — overlap placed FROZEN (continue from the last good frames)")
                _pin_next = False
                lat_vid, mask_vid = LTXVAddGuide.replace_latent_frames(
                    lat_vid, mask_vid,
                    guiding_latent=clss_state._overlap_latent.to(device),
                    latent_idx=0,
                    strength=1.0 - _tau_c_v,
                )
                print(f"[CLSS S1]   video tau_c_eff={_tau_c_v:.4f} "
                      f"(base={clss_config.tau_c * video_slb_tau_mult:.3f}, "
                      f"ceiling={_VIDEO_TAU_C_CEILING})")

            # §2.5 Dynamic anchor bank: telemetry-only (identity tracking in the
            # end-of-chunk block), NOT wired into conditioning.  (The nudge
            # experiment was removed after validation runs; see
            # clss_experimental_changes.patch.)

            # i2v: in-place first-frame conditioning — the canonical LTX i2v path.
            # ComfyUI's LTXVAddGuide itself uses replace_latent_frames for frame_idx=0;
            # append_keyframe is the pathway for NON-aligned keyframes only.
            #
            # The previous append_keyframe approach added an extra video token block at
            # the END of the sequence (RoPE pointing back to t=0).  In AV mode the audio
            # tokens attend to that out-of-place block for the entire chunk — the same
            # contamination class that forced skipping guide_attention_entries.  Chunk-1
            # audio (which seeds the SLB/ref chain for the whole video) came out as
            # noise/drone regardless of guidance settings.  In-place replacement keeps
            # the token sequence clean: no appended block, no post-sample stripping, and
            # audio temporal coverage matches video exactly.
            if is_first and img_guide_latent is not None:
                lat_vid, mask_vid = LTXVAddGuide.replace_latent_frames(
                    lat_vid, mask_vid,
                    guiding_latent=img_guide_latent.to(device),
                    latent_idx=0,
                    strength=1.0,   # noise_mask=0 → frame 0 fully conditioned
                )
                print(f"[CLSS] i2v: guide placed in-place at frame 0, "
                      f"lat_vid={list(lat_vid.shape)} (no appended tokens)")

            # §2.5 Dynamic anchor bank: telemetry-only (identity tracking in the
            # end-of-chunk block), NOT wired into conditioning.
            #
            # History: an in-place anchor nudge at strength 0.35 was REVERTED
            # 2026-07-21 after visible content morphing / "jump in time"
            # (chunk 7 nudged to anchor@frame35 after chunk 5 used anchor@frame47).
            # The failure was recorded as "retrieval picks the LEAST-similar
            # anchor" — but the library retrieve() has returned the MOST-similar
            # non-redundant anchor since the first commit (verified 2026-08-05),
            # so the July diagnosis was wrong and the correct wiring was never
            # tested.  A 2026-08-05 retry (most-similar, strength 0.20) was
            # implemented and then removed again pending isolated validation;
            # it lives in clss_experimental_changes.patch.  The append-style
            # VideoConditionByKeyframeIndex alternative corrupts AV audio (see
            # the i2v guide note above) and remains off the table.

            if aud_tmpl is not None:
                # Audio latent covers same temporal span as video (overlap + new frames).
                # cur_new_af comes from the per-chunk plan: chunk-1 keeps the causal
                # (new_lf−1)·8+1 px discount, continuations cover new_lf·8 px, and a
                # RoPE-wall split shares each chunk's audio budget proportionally.
                cur_new_af = _chunk_plan[chunk_idx][1]
                chunk_af = (audio_overlap_af if not is_first else 0) + cur_new_af
                lat_aud  = torch.zeros(B_a, C_a, chunk_af, freq, device=device)
                # [B, 1, T, 1] broadcasts correctly through reshape_mask → [B, C, T, freq]
                mask_aud = torch.ones(B_a, 1, chunk_af, 1, device=device)

                # Audio SLB (always on): place previous chunk's overlap-time audio
                # FROZEN — mask=0, verbatim passthrough, no tau_c on audio at all
                # (2026-08-20, user directive; the hot re-noise schedule and its
                # audio_slb_tau_mult knob were removed).
                # Required: model_base.process_timestep multiplies audio_denoise_mask×sigma
                # → per-token a_timestep, so mask=0 tells the model these tokens are
                # CLEAN (a_timestep=0) — consistent with the frozen placement.
                _slb_ctx_used: torch.Tensor | None = None   # what was actually PLACED (for the honored check)
                if has_aud_slb:
                    slb = audio_slb_latent.to(device)
                    n   = min(audio_overlap_af, slb.shape[2], chunk_af)
                    _slb_ctx = slb[:, :, :n]
                    if _audio_corr:
                        _slb_ctx, _fg_lo, _fg_hi = _flatten_audio_env(_slb_ctx)
                    else:
                        _fg_lo, _fg_hi = 1.0, 1.0
                    lat_aud[:, :, :n]  = _slb_ctx
                    mask_aud[:, :, :n] = 0.0
                    _slb_ctx_used = _slb_ctx.detach().cpu()
                    print(f"[CLSS S1]   audio SLB: {n}f  FROZEN (mask=0)  "
                          f"mean={_slb_ctx.float().mean():.4f}  "
                          f"env-flatten gain=[{_fg_lo:.3f}, {_fg_hi:.3f}]")

                # ref_audio at negative RoPE positions: temporal context for what
                # preceded this chunk (av_model.py line 708 prepends ref tokens).
                if has_aud_ref:
                    ref_slb   = audio_overlap_latent.to(device)   # [B, C_a, T_ov, freq]
                    # Faithful to reference pipeline.py: the previous chunk's
                    # pre-overlap audio tail is the negative-RoPE conditioning,
                    # passed at full length, envelope-flattened (always on —
                    # context only; protects the video side, see
                    # _flatten_audio_env docstring).
                    #
                    # DEAD-END LOG (2026-07-23, as ANTI-LOOP levers — do NOT
                    # re-add for that purpose): every attempt to break the
                    # audio metronome by PERTURBING this reference was tested
                    # live on the user's ears and FAILED identically —
                    #   • ref-length decay (0.85^chunk, floored)      → loop unchanged
                    #   • white-noise blend (ramp 0.05/chunk to a cap) → loop unchanged
                    #                                                    + injected HF hiss
                    #   • env-flatten                                  → loop unchanged
                    # The env_corr metronome still locks (→0.9) the chunk the ref
                    # reaches full window, independent of these.  The loop's
                    # carrier is the near-verbatim SLB tail; the raised audio
                    # tau_c schedule that used to address it was itself removed
                    # 2026-08-20 — the audio SLB is now frozen (mask=0).
                    if _audio_corr:
                        ref_slb, _fg_lo, _fg_hi = _flatten_audio_env(ref_slb)
                        print(f"[CLSS S1]   audio ref env-flattened: gain=[{_fg_lo:.3f}, "
                              f"{_fg_hi:.3f}]")
                    b_r, c_r, t_r, f_r = ref_slb.shape
                    ref_tokens = ref_slb.permute(0, 2, 1, 3).reshape(b_r, t_r, c_r * f_r)
                    ref_audio_dict = {"tokens": ref_tokens}
                    # Unconvert → add ref_audio to every conditioning entry → reconvert.
                    pos_raw = _unconvert_cond(guider_chunk.original_conds.get("positive", []))
                    neg_raw = _unconvert_cond(guider_chunk.original_conds.get("negative", []))
                    for entry in pos_raw:
                        entry[1]["ref_audio"] = ref_audio_dict
                    for entry in neg_raw:
                        entry[1]["ref_audio"] = ref_audio_dict
                    guider_chunk.original_conds = {
                        **guider_chunk.original_conds,
                        "positive": comfy.sampler_helpers.convert_cond(pos_raw),
                        "negative": comfy.sampler_helpers.convert_cond(neg_raw),
                    }
                    print(f"[CLSS S1]   audio ref_audio injected: {t_r} tokens "
                          f"mean={ref_slb.float().mean():.4f} "
                          f"std={ref_slb.float().std():.4f} "
                          f"nan={ref_slb.isnan().any().item()} "
                          f"inf={ref_slb.isinf().any().item()}")
                else:
                    print(f"[CLSS S1]   audio: no ref_audio (first chunk — generating unconditioned)")

                _n_slb = min(audio_overlap_af, audio_slb_latent.shape[2]) if has_aud_slb else 0
                _ov_gen = (audio_overlap_af - _n_slb) if not is_first else 0
                print(f"[CLSS S1]   audio in: chunk_af={chunk_af} "
                      f"(slb={_n_slb}f + overlap_rest={_ov_gen}f + new={cur_new_af}f) "
                      f"mask_mean={mask_aud.mean():.3f}")
                av_samples = comfy.nested_tensor.NestedTensor((lat_vid, lat_aud))
                av_mask    = comfy.nested_tensor.NestedTensor((mask_vid, mask_aud))
                chunk_latent = {"samples": av_samples, "noise_mask": av_mask}
            else:
                chunk_latent = {"samples": lat_vid, "noise_mask": mask_vid}

            # Denoise — slice consistent noise per chunk so chunks 2+ get distinct noise
            # (not a repeated realisation caused by same seed + same tensor shape).
            _s1_noise_pos = _vid_pos
            _s1_chunk_noise = _SlicedNoise(
                _full_noise_vid_s1, _s1_noise_pos, chunk_overlap, seed=_noise_seed_s1,
                full_noise_aud=_full_noise_aud_s1,
                a_pos=_s1_a_noise_pos,
                a_overlap=(audio_overlap_af if not is_first else 0),
            )
            print(
                f"[CLSS S1]   noise pos={_s1_noise_pos}"
                + (f" a_pos={_s1_a_noise_pos}" if aud_tmpl is not None else "")
                + f" fingerprint={_full_noise_vid_s1[:, :, _s1_noise_pos:_s1_noise_pos+1].flatten()[:4].tolist()}"
            )
            if aud_tmpl is not None:
                _s1_a_noise_pos += cur_new_af
            _, denoised = SamplerCustomAdvanced().sample(
                noise=_s1_chunk_noise,
                guider=guider_chunk,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=chunk_latent,
            )

            # Separate AV output
            denoised_samples = denoised["samples"]
            if is_av:
                vid_out, aud_out = denoised_samples.unbind()
            else:
                vid_out = denoised_samples
                aud_out = None
            # i2v: nothing to strip — the guide is conditioned in-place at frame 0.
            # Frame 0 of the output IS the (denoised-around) guide frame; log adherence.
            if is_first and img_guide_latent is not None:
                _guide_sim = _frame_cos(vid_out[:, :, 0], img_guide_latent.to(device)[:, :, 0])
                print(f"[CLSS S1]   i2v guide adherence: {_guide_sim:.4f}")

            # Drop video overlap, apply CLSS corrections to new video frames
            new_vid   = vid_out[:, :, chunk_overlap:]
            mu_pre    = new_vid.mean().item()
            std_pre   = new_vid.std().item()

            # §3.5 / Eq. 8 g_SLB measurement — OPTIONAL second denoising pass
            # with a perturbed overlap latent, isolating the transformer's OWN
            # sensitivity to its SLB-mediated boundary input.  Compares against
            # new_vid (the RAW, pre-AdaIN output) on both sides, so the
            # measurement reflects f_theta itself, decoupled from the CLSS
            # correction stack it composes with in the actual closed loop —
            # this was previously described (§3.5) but never implemented, so
            # no g_SLB number ever reached the paper's results.  Purely
            # diagnostic: the perturbed pass never touches clss_state,
            # acc_video/acc_audio, or the SLB pushed forward; its output is
            # discarded once the norm ratio is computed.  has_slb guards
            # against chunk 1, which has no overlap to perturb.  Costs one
            # extra full denoise per chunk when enabled (measure_g defaults
            # off — see CLSSConfigNode tooltip).
            if clss_config.measure_g and has_slb:
                _eps = float(clss_config.measure_g_epsilon)
                _g_gen = torch.Generator(device="cpu").manual_seed(
                    (int(_noise_seed_s1) % (2 ** 31)) * 131 + 9973 * chunk_idx)
                _ov_clean = clss_state._overlap_latent.to(device)
                _delta_dir = torch.randn(_ov_clean.shape, generator=_g_gen,
                                          dtype=torch.float32).to(device)
                _delta_norm = (_eps * _ov_clean.float().norm()
                               / _delta_dir.norm().clamp(min=1e-12))
                _delta = (_delta_dir * _delta_norm).to(lat_vid.dtype)
                _lat_vid_p = lat_vid.clone()
                _lat_vid_p[:, :, :chunk_overlap] = _lat_vid_p[:, :, :chunk_overlap] + _delta
                if is_av:
                    _chunk_latent_p = {
                        "samples": comfy.nested_tensor.NestedTensor((_lat_vid_p, lat_aud)),
                        "noise_mask": chunk_latent["noise_mask"],
                    }
                else:
                    _chunk_latent_p = {"samples": _lat_vid_p, "noise_mask": mask_vid}
                _, _denoised_p = SamplerCustomAdvanced().sample(
                    noise=_s1_chunk_noise,
                    guider=guider_chunk,
                    sampler=sampler,
                    sigmas=sigmas,
                    latent_image=_chunk_latent_p,
                )
                _samples_p = _denoised_p["samples"]
                _vid_out_p = _samples_p.unbind()[0] if is_av else _samples_p
                _new_vid_p = _vid_out_p[:, :, chunk_overlap:]
                _n_tail = min(overlap_lf, new_vid.shape[2], _new_vid_p.shape[2])
                _out_diff_norm = (_new_vid_p[:, :, -_n_tail:].float()
                                   - new_vid[:, :, -_n_tail:].float()).norm().item()
                _g_slb = _out_diff_norm / max(_delta.float().norm().item(), 1e-12)
                _trend["g_slb"].append(_g_slb)
                print(f"[CLSS S1]   g_SLB={_g_slb:.4f}  (eps={_eps:.3f}  "
                      f"||delta||={_delta.float().norm().item():.4f}  "
                      f"||out_diff||={_out_diff_norm:.4f}  tail={_n_tail}f)")

            # open_loop baseline: skip AdaIN (post_process)
            # but keep update_buffer below so SLB/anchor telemetry still runs.
            corrected = new_vid if _open_loop else clss_state.post_process(new_vid)

            # ── Detail-band anchor (scene-first-referenced, symmetric, capped) ──
            # Built against the measured progressive detail loss on long runs:
            # coarse-structure (low-band) energy inflates (+19.6% over 7 chunks,
            # +15% over 15 chunks) while the high-frequency SHARE of energy
            # falls ~20%+ → progressively smoother, less detailed output.  (The
            # §2.4 shrinkage this anchor used to complement was removed
            # 2026-08-19 — shrink-only gains could never RESTORE band energy;
            # this anchor is now the only band-energy correction in the loop.)
            # A two-band (spatial low/high) equalizer with SYMMETRIC gains
            # sqrt(E_ref/E), hard-capped per chunk, referenced to the scene's
            # first chunk (re-baselined on scene change, 0019 philosophy).
            # With detail_anchor="off" behaviour is exactly as before (the hf
            # metric is still logged).
            _da_x = corrected.float()
            _da_b, _da_c, _da_t, _da_h, _da_w = _da_x.shape
            _da_flat = _da_x.permute(0, 2, 1, 3, 4).contiguous().reshape(
                _da_b * _da_t, _da_c, _da_h, _da_w)
            _da_low = torch.nn.functional.avg_pool2d(_da_flat, 3, stride=1, padding=1)
            _da_high = _da_flat - _da_low
            _e_low = float(_da_low.pow(2).mean())
            _e_high = float(_da_high.pow(2).mean())
            _hf_share = _e_high / max(_e_low + _e_high, 1e-12)
            if detail_anchor == "on" and not _open_loop:  # open_loop baseline: no detail anchor
                if _s1_band_ref is None:
                    _s1_band_ref = (_e_low, _e_high)
                    print(f"[CLSS S1]   detail anchor: reference captured "
                          f"E_low={_e_low:.4f} E_high={_e_high:.4f} "
                          f"hf_share={_hf_share:.4f}")
                else:
                    _g_lo = min(1.10, max(0.90, (_s1_band_ref[0] / max(_e_low, 1e-12)) ** 0.5))
                    _g_hi = min(1.12, max(0.90, (_s1_band_ref[1] / max(_e_high, 1e-12)) ** 0.5))
                    if abs(_g_lo - 1.0) > 0.005 or abs(_g_hi - 1.0) > 0.005:
                        corrected = (_da_low * _g_lo + _da_high * _g_hi).reshape(
                            _da_b, _da_t, _da_c, _da_h, _da_w
                        ).permute(0, 2, 1, 3, 4).contiguous().to(corrected.dtype)
                        _e_low_p, _e_high_p = _e_low * _g_lo ** 2, _e_high * _g_hi ** 2
                        _hf_p = _e_high_p / max(_e_low_p + _e_high_p, 1e-12)
                        print(f"[CLSS S1]   detail anchor: E_low {_e_low:.4f}→{_e_low_p:.4f} "
                              f"(g={_g_lo:.4f})  E_high {_e_high:.4f}→{_e_high_p:.4f} "
                              f"(g={_g_hi:.4f})  hf_share {_hf_share:.4f}→{_hf_p:.4f} "
                              f"(ref={_s1_band_ref[1] / (_s1_band_ref[0] + _s1_band_ref[1]):.4f})")
                        _hf_share = _hf_p
            _trend["vid_hf"].append(_hf_share)

            # ── Origin-drift telemetry (0027) ───────────────────────────────
            # Every prior video metric compares ADJACENT things (frames, seams)
            # or DRIFTING references (nearest banked anchor) — smooth morphs
            # are invisible to all of them (measured: the 25s morph sits inside
            # chunk 7 which reads boundary 0.9905 / intra 0.9602 / identity
            # 0.9643).  This tracks every output frame against the FIXED
            # scene-first frame, so cumulative drift shows as a staircase and
            # each morph localizes as a per-frame DROP with a timestamp
            # (end-of-run events table).  layout_sim isolates coarse scene
            # layout from texture.  METRIC v2 (2026-08-17): the original
            # pooled-cosine layout map false-positived — single-frame dips
            # (e.g. −0.55 at t≈35.7 s) that are ABSENT from the decoded video
            # (author-verified), while origin_sim against the same reference
            # frame shows no excursion at those frames.  The lightly blurred
            # signed map is dominated by fine latent phase, which the VAE
            # decoder integrates away.  The fixed metric is a SIGN-FREE
            # coarse POWER map: channel-mean → square → hard 3×3 low-pass
            # (stride 3) → cosine.  Power maps are non-negative, so the
            # cosine cannot sign-flip, and the hard low-pass keeps only the
            # spatial scale that survives decoding.
            if _origin_ref is None:
                _origin_ref = corrected[:, :, -1:].detach().float().cpu()
                _origin_layout = torch.nn.functional.avg_pool2d(
                    _origin_ref[0].mean(0).square(), 3, stride=3).flatten()
            _oc = corrected.detach().float().cpu()
            _o_flat = _origin_ref.flatten()
            _osims, _lsims = [], []
            for _fi in range(_oc.shape[2]):
                _fr = _oc[:, :, _fi:_fi + 1]
                _osims.append(float(torch.nn.functional.cosine_similarity(
                    _fr.flatten(), _o_flat, dim=0)))
                _fl = torch.nn.functional.avg_pool2d(
                    _fr[0].mean(0).square(), 3, stride=3).flatten()
                _lsims.append(float(torch.nn.functional.cosine_similarity(
                    _fl, _origin_layout, dim=0)))
                _origin_track.append(_osims[-1])
                _layout_track.append(_lsims[-1])
            print(f"[CLSS S1]   origin_sim/frame: {[round(s, 3) for s in _osims]}")
            print(f"[CLSS S1]   layout_sim/frame: {[round(s, 3) for s in _lsims]}"
                  f"  (coarse layout power map vs scene-first)")
            _layout_argmin_track.append(int(_lsims.index(min(_lsims))))
            _trend["vid_origin"].append(min(_osims))

            # Auto seam pin (event-triggered): the 2026-08-16 runs showed
            # visible coarse-layout JUMP events (jumping/morphing content).
            # When this chunk contains such an event, the NEXT chunk's
            # overlap is placed FROZEN (tau_c=0): the model continues from
            # the LAST GOOD FRAMES exactly, instead of re-laying-out the
            # scene from a re-noised context.  Corrector source is the fresh
            # SLB, not the anchor bank.  Chunk 1 is EXCLUDED (scene
            # establishment, not a jump).  NOTE 2026-08-17: the trigger runs
            # on the layout metric ABOVE, and the pre-v2 metric
            # false-positived — the pin-on runs fired on dips that are
            # absent from the decoded video.  With metric v2 the thresholds
            # are UNRECALIBRATED: treat a trigger as a candidate event and
            # confirm it in the decoded video.
            if _auto_pin:
                _dd_max = (max(_lsims[i] - _lsims[i + 1]
                               for i in range(len(_lsims) - 1))
                           if len(_lsims) > 1 else 0.0)
                _hit = chunk_idx >= 1 and (
                    (_dd_max > 0.25) or (min(_lsims) < 0.05))
                if _hit and chunk_idx + 1 < num_chunks:
                    _pin_next = True
                    print(f"[CLSS S1]   layout event in this chunk (max drop "
                          f"{_dd_max:.3f}, min {min(_lsims):.3f}) → next chunk's "
                          f"overlap placed FROZEN (auto seam pin)")

            # Gentle global-std anchor to chunk-0.  The EMA AdaIN corrects per-channel
            # stats but its sigma cap still permits ~+5% cumulative std growth over a
            # long run (measured 1.008→1.054 over 15 chunks); late chunks run hot, and
            # Stage 2 inherits progressively hotter latents, dropping fidelity toward
            # the end.  Here we only correct the GLOBAL std when it drifts beyond ±4%
            # of chunk-0, and only partially (blend 0.5) — enough to stop the monotonic
            # creep without flattening legitimate per-scene contrast changes.  Mean is
            # left untouched (it carries scene evolution).
            # open_loop baseline: the whole std-anchor block (ref capture +
            # correction) is skipped — the anchor is a closed-loop correction.
            if not _open_loop:
                if _s1_vid_std_ref is None:
                    _s1_vid_std_ref = corrected.float().std().item()
                else:
                    _cur_vstd = corrected.float().std().item()
                    _ratio = _s1_vid_std_ref / max(_cur_vstd, 1e-6)
                    if _ratio < 0.96 or _ratio > 1.04:      # only when drift exceeds ±4%
                        _g_v = 1.0 + 0.5 * (_ratio - 1.0)   # partial pull toward chunk-0
                        _m = corrected.float().mean()
                        corrected = ((corrected.float() - _m) * _g_v + _m).to(corrected.dtype)
                        print(f"[CLSS S1]   video std anchor: {_cur_vstd:.4f}→"
                              f"{corrected.float().std().item():.4f} (ref={_s1_vid_std_ref:.4f}, g={_g_v:.4f})")
            mu_post   = corrected.mean().item()
            std_post  = corrected.std().item()
            clss_state.update_buffer(corrected)
            acc_video.append(corrected.cpu())

            print(f"[CLSS S1]   video done: pre_AdaIN mean={mu_pre:.4f} std={std_pre:.4f} | "
                  f"post_AdaIN mean={mu_post:.4f} std={std_post:.4f} | "
                  f"video_SLB updated shape={clss_state._overlap_latent.shape if clss_state._overlap_latent is not None else 'None'}")

            # §item-1: intra-chunk cosine — first vs last new frame (corrected latent)
            _intra = _frame_cos(corrected[:, :, 0], corrected[:, :, -1])
            _trend["vid_intra"].append(_intra)
            # §item-2: boundary cosine — last frame of previous chunk vs first new frame
            if _s1_prev_last is not None:
                _bnd = _frame_cos(_s1_prev_last.to(device), corrected[:, :, 0])
                print(f"[CLSS S1]   boundary_sim={_bnd:.4f}  intra_chunk_sim={_intra:.4f}")
                _trend["vid_bnd"].append(_bnd)
            else:
                print(f"[CLSS S1]   boundary_sim=N/A(first)  intra_chunk_sim={_intra:.4f}")
            # §object_identity: spatially-resolved seam identity (DECODED pixel frames).
            # Decodes a short clip around the seam — last ≤2 corrected latent frames of
            # the previous chunk + first ≤2 of this chunk — via vae.decode, the exact
            # path the workflow's VAEDecode node uses (LTXV latent format has no
            # process_out normalization; sampler-space latents are decoder-ready).
            # LTXV is a causal 3D VAE: latent frame 0 → 1 px frame, every subsequent
            # latent frame → 8 px frames, so with _kp previous frames the two px frames
            # straddling the seam are at indices 8*(_kp-1) and 8*(_kp-1)+1.  The metric
            # is a diagnostic: a decode failure must never kill a generation, so it is
            # caught and the chunk is skipped.
            if _obj_id and _s1_prev_tail is not None:
                try:
                    _kp = _s1_prev_tail.shape[2]
                    _kn = min(2, corrected.shape[2])
                    _seam_clip = torch.cat(
                        [_s1_prev_tail, corrected[:, :, :_kn].cpu()], dim=2)
                    _px = vae.decode(_seam_clip)   # [B, T_px, H_px, W_px, C], [0,1]
                    _si = 8 * (_kp - 1)
                    _obj_mean, _obj_min = _grid_cell_cos(_px[0, _si], _px[0, _si + 1])
                    print(f"[CLSS S1]   object_identity: cell_cos mean={_obj_mean:.3f} "
                          f"min={_obj_min:.3f} (8x8 grid, decoded)")
                    _trend["vid_obj"].append(_obj_mean)
                    del _px, _seam_clip
                except Exception as _e:
                    print(f"[CLSS S1]   object_identity: seam decode FAILED "
                          f"({type(_e).__name__}: {_e}) — metric skipped for this chunk")
            _trend["vid_std"].append(std_post)
            # §item-6: identity-retention — cosine vs nearest bank anchor.
            # Comparing vs the NEAREST anchor (not always chunk-1) separates within-scene
            # identity from intended scene changes: if the bank grew, the nearest anchor
            # should be the active scene's reference.  If bank_size=1 the metric reduces
            # to vs-chunk-1 and is flagged "(bank=1, equiv chunk-1)".
            # identity_sim: for the first chunk, we ARE the reference (the anchor bank was
            # just seeded from this chunk's last frame; comparing first-vs-last would measure
            # intra-chunk coherence, already reported above). From chunk 2 onwards, compare
            # the first new frame against the nearest bank anchor to track identity retention.
            if is_first:
                print(f"[CLSS S1]   identity_sim=1.0000 (reference)")
            else:
                _cur_feat = F.normalize(corrected[:, :, 0].float().reshape(B, C_v, -1).mean(-1), dim=1)
                _bank = clss_state._anchor_bank
                if _bank.anchors:
                    _anchor_sims = [
                        F.cosine_similarity(
                            _cur_feat,
                            F.normalize(a.feature.unsqueeze(0).to(device), dim=1),
                        ).item()
                        for a in _bank.anchors
                    ]
                    _best_sim = max(_anchor_sims)
                    _best_idx = _anchor_sims.index(_best_sim)
                    _best_fid = _bank.anchors[_best_idx].frame_idx
                    _note = "(bank=1, equiv chunk-1)" if len(_bank.anchors) == 1 else f"(bank_size={len(_bank.anchors)})"
                    print(f"[CLSS S1]   identity_sim={_best_sim:.4f} {_note} vs anchor@frame{_best_fid}")
                    _trend["vid_ident"].append(_best_sim)
                else:
                    print(f"[CLSS S1]   identity_sim=N/A (bank empty)")
            _s1_prev_last = corrected[:, :, -1].cpu()
            # object_identity: keep the last ≤2 corrected frames for next chunk's
            # seam decode (None when the metric is off — zero cost).
            _s1_prev_tail = corrected[:, :, -2:].detach().cpu() if _obj_id else None
            # Per-frame adjacent sim for the last chunk — locates visual breaks precisely.
            if chunk_idx == num_chunks - 1 and corrected.shape[2] > 1:
                _adj = [_frame_cos(corrected[:, :, i], corrected[:, :, i + 1])
                        for i in range(corrected.shape[2] - 1)]
                print(f"[CLSS S1]   per-frame adj sims (last chunk): "
                      f"[{', '.join(f'{s:.3f}' for s in _adj)}]")

            if aud_out is not None:
                # Drop the audio overlap-time region (covers the same time as the video SLB).
                # Non-first chunks generate chunk_af = audio_overlap_af + new_af frames;
                # we keep only the new_af portion.  First chunk: no drop (chunk_af = new_af).
                aud_drop = audio_overlap_af if not is_first else 0
                if aud_drop > 0 and aud_out.shape[2] < aud_drop:
                    print(f"[CLSS S1]   audio ERROR: aud_out.shape={list(aud_out.shape)} "
                          f"but aud_drop={aud_drop} — model returned fewer audio frames than "
                          f"expected ({chunk_af}).  Setting aud_drop=0 to avoid empty new_aud.")
                    aud_drop = 0
                new_aud = aud_out[:, :, aud_drop:]
                aud_acc_start = sum(a.shape[2] for a in acc_audio)
                aud_acc_end   = aud_acc_start + new_aud.shape[2]
                print(f"[CLSS S1]   audio out: aud_out shape={list(aud_out.shape)} "
                      f"mean={aud_out.float().mean():.4f} std={aud_out.float().std():.4f} "
                      f"min={aud_out.float().min():.4f} max={aud_out.float().max():.4f} "
                      f"nan={aud_out.isnan().any().item()} inf={aud_out.isinf().any().item()}")
                print(f"[CLSS S1]   audio acc: new_aud af=[{aud_acc_start}:{aud_acc_end}] "
                      f"({new_aud.shape[2]}f kept, {aud_drop}f overlap-time dropped)")
                # SLB-honored check: compares the overlap audio AS PLACED
                # (env-flattened) against the denoised output.  The audio SLB is
                # FROZEN (mask=0) since 2026-08-20, so honored should read ~1.0 —
                # a near-zero reading = mask not applied.
                if not is_first and _slb_ctx_used is not None and audio_overlap_af > 0:
                    _slb_sim = _aud_cos(_slb_ctx_used.to(device),
                                        aud_out[:, :, :audio_overlap_af])
                    print(f"[CLSS S1]   audio SLB honored: {_slb_sim:.4f} "
                          f"(frozen placement; expect ~1.0, ~0 = mask failure)")
                    _trend["aud_slb"].append(_slb_sim)
                # Per-channel max-abs for first 8 frames (diagnose onset spike in chunk 1)
                with torch.no_grad():
                    _n8 = min(8, new_aud.shape[2])
                    _ch_absmax = new_aud[:, :, :_n8].float().abs().flatten(2).max(dim=2).values
                    _ch_std    = new_aud.float().std(dim=(2, 3))
                print(
                    f"[CLSS S1]   audio first-{_n8}f per-ch absmax: "
                    f"[{' '.join(f'{v:.3f}' for v in _ch_absmax[0].tolist())}]  "
                    f"ch_std: [{' '.join(f'{v:.3f}' for v in _ch_std[0].tolist())}]"
                )
                # ── Audio envelope-repetition telemetry (0027) ──────────────
                # The reported "repetitive sound" is a loudness gesture that
                # recurs every chunk (peak_frame 67-88/109 in EVERY chunk of
                # EVERY run).  No existing metric measures it.  This is the
                # Pearson correlation of this chunk's energy envelope against
                # the previous chunk's: >0.7 = the same gesture repeating.
                _env = new_aud.detach().float().pow(2).mean(dim=(0, 1, 3)).cpu()
                if _prev_aud_env is not None and len(_prev_aud_env) > 8:
                    _L = min(len(_env), len(_prev_aud_env))
                    _ea = _env[:_L] - _env[:_L].mean()
                    _eb = _prev_aud_env[:_L] - _prev_aud_env[:_L].mean()
                    _env_corr = float((_ea * _eb).sum() /
                                      (_ea.norm() * _eb.norm() + 1e-8))
                    print(f"[CLSS S1]   audio_env_corr(prev)={_env_corr:.3f}  "
                          f"(>0.7 = same loudness gesture repeating each chunk)")
                    _trend["aud_env"].append(_env_corr)
                _prev_aud_env = _env
                # Chunk-1 onset fix (2026-08-13, v2): the i2v chunk-1 audio is
                # generated from pure noise with no audio context, and the model
                # dumps violent transients into it — measured on the 2026-08-12
                # live run: a −6.185 = 14.4σ sample at frame 4, elevated frames
                # 1-7, and a second cluster mid-chunk (post-fix absmax 2.0924 =
                # 4.9σ = an audible click; Stage 2 measured the seam: aud_bnd
                # 0.194 at 11.6s).  History:
                #   - ±4σ HARD clamp on the first 8 frames + 4σ→5σ tanh knee
                #     beyond: the hard clamp maps every |x|>4σ sample to exactly
                #     4σ — square tops (5.5σ and 14.4σ both → 4.0σ) and a hard
                #     step against untouched neighbours (measured frame 4:
                #     4.0σ next to 2.4σ); the knee's 5σ asymptote lets a 5.7σ
                #     cluster through at 4.94σ (the measured 20.46s CLICK).
                #   - whole-window ±4σ hard clamp (older): shaved every loud
                #     vocal/drum peak (music crest factor 4-5 is normal) = the
                #     measured "flattened voice".
                # Current: ONE smooth limiter over the whole chunk —
                #   x' = sign(x)·(3.5σ + 0.5σ·tanh((|x|−3.5σ)/σ))
                # content ≤3.5σ is bit-exact (dynamics preserved); above 3.5σ the
                # tanh rolls off toward a 4σ asymptote.  Strictly increasing and
                # 1-Lipschitz above the knee (no square tops, no steps: a 5.7σ
                # click → 3.99σ, the 14.4σ onset spike → 4.00σ), while a 4σ music
                # crest loses only ~7% (limiter RMS impact <2% on crest-4 content
                # — the "flattened voice" cannot return).  Fade-in extended to
                # the measured transient span (8 frames).
                # Numeric test: simulations/audio_joint_schedule_sim.py.
                if is_first:
                    _n_fade = min(8, new_aud.shape[2])
                    if _n_fade >= 2:
                        _ramp = torch.linspace(0.125, 1.0, _n_fade, device=device)
                        new_aud = new_aud.clone()
                        new_aud[:, :, :_n_fade] = new_aud[:, :, :_n_fade] * _ramp.view(1, 1, _n_fade, 1)
                        print(f"[CLSS S1]   chunk-1 audio fade-in applied "
                              f"(0.125→1.0 over {_n_fade}f)")
                    _fa = new_aud.float()
                    _sig = _fa.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
                    _c35 = _sig * 3.5
                    _over = (_fa.abs() - _c35).clamp(min=0)
                    _n_over = int((_over > 0).sum())
                    new_aud = (_fa - torch.sign(_fa)
                               * (_over - 0.5 * _sig * torch.tanh(_over / _sig))).to(aud_out.dtype)
                    print(f"[CLSS S1]   chunk-1 audio limiter: smooth 3.5σ→4σ tanh "
                          f"({_n_over} samples)  new_abs_max={new_aud.abs().max().item():.4f}")
                # §item-9: audio within-chunk coherence — detects mid-chunk degradation (§5.4)
                _aud_sims = _aud_within_chunk_sims(new_aud)
                if _aud_sims:
                    print(f"[CLSS S1]   audio_within_chunk_sim: "
                          + " → ".join(f"{s:.3f}" for s in _aud_sims))
                    _trend["aud_wc"].append(_aud_sims[-1])
                # audio boundary_sim — chunk-to-chunk continuity at the sample level
                if _s1_aud_prev_last is not None:
                    _aud_bnd = _aud_cos(_s1_aud_prev_last.to(device), new_aud[:, :, :1])
                    print(f"[CLSS S1]   audio_boundary_sim={_aud_bnd:.4f}")
                    _trend["aud_bnd"].append(_aud_bnd)
                else:
                    print(f"[CLSS S1]   audio_boundary_sim=N/A(first)")
                # RMS envelope — raw RMS + per-segment breakdown + peak location
                with torch.no_grad():
                    _aud_rms = new_aud.float().pow(2).mean().sqrt().item()
                    _aud_peak = int(new_aud.float().abs().mean(dim=(0, 1, 3)).argmax().item())
                    _nseg = 4
                    _seg_t = new_aud.shape[2] // _nseg
                    _seg_rms = [
                        new_aud[:, :, s * _seg_t:(s + 1) * _seg_t].float().pow(2).mean().sqrt().item()
                        for s in range(_nseg)
                    ] if _seg_t > 0 else []
                # Per-freq-bin energy — mean |x| per freq bin ([freq] values).
                # Detects spectral collapse: high-freq decay sounds muffled even when RMS looks OK.
                with torch.no_grad():
                    _freq_e = new_aud.float().abs().mean(dim=(0, 1, 2)).tolist()
                print(
                    f"[CLSS S1]   audio RMS={_aud_rms:.4f}  peak_frame={_aud_peak}/{new_aud.shape[2]}"
                    + (f"  seg_rms=[{' '.join(f'{r:.3f}' for r in _seg_rms)}]" if _seg_rms else "")
                )
                _aud_peak_track.append(_aud_peak)
                if _s1_audio_freq_ref is None:
                    _s1_audio_freq_ref = _freq_e
                    print(f"[CLSS S1]   audio freq_energy(ref)=[{' '.join(f'{e:.3f}' for e in _freq_e)}]")
                else:
                    _freq_ratio = [e / r if r > 1e-6 else 0.0 for e, r in zip(_freq_e, _s1_audio_freq_ref)]
                    print(
                        f"[CLSS S1]   audio freq_energy=[{' '.join(f'{e:.3f}' for e in _freq_e)}]"
                        f"  ratio=[{' '.join(f'{r:.2f}' for r in _freq_ratio)}]"
                    )
                    # high-freq drift = mean ratio of the top 4 freq bins (spectral flattening)
                    if len(_freq_ratio) >= 4:
                        _trend["aud_hf"].append(sum(_freq_ratio[-4:]) / 4.0)
                # Scalar RMS anchor — upward-only (2026-08-14, port-vs-reference
                # audit): a QUIETER chunk is boosted toward the capped EMA target,
                # a louder chunk is kept RAW — the reference law (pipeline.py:741-757,
                # "don't attenuate genuinely louder chunks").  The previous hard
                # bidirectional gain crushed every brighter chunk back to chunk-1's
                # level (measured cuts g=0.93/0.97 in the 2026-08-13/14 logs — a
                # dullness-lock the reference does not have).  Divergence note: the
                # runaway the hardening was built for (RMS ×4 over 15 chunks) is
                # unsuppressed by the upward-only law — gate long runs on the
                # aud_rms trend line.
                #
                # The previous per-(ch×bin) std gain with clamp(min=1.0) was a
                # boost-only ratchet: it pumped energy into bins whose temporal std
                # decayed even when their |x| energy already sat 1.4-1.7× ABOVE the
                # chunk-0 reference (the high-freq overshoot at audio_cfg=7).  A per-bin
                # correction that can only add energy amplifies exactly the bins the
                # rescaled guidance should be taming.  Replaced by a hard per-chunk
                # anchor to chunk-0 (scalar RMS) applied before the audio
                # feeds the SLB/ref — see below.
                if _s1_audio_rms_ref is None:
                    _skip = min(16, new_aud.shape[2])
                    _ref_aud = new_aud[:, :, _skip:] if new_aud.shape[2] > _skip else new_aud
                    _s1_audio_rms_ref  = _ref_aud.float().pow(2).mean().sqrt().item()
                    _s1_audio_ema_rms  = _s1_audio_rms_ref
                    print(f"[CLSS S1]   audio rms_ref={_s1_audio_rms_ref:.4f} (onset-excluded)")
                else:
                    # EMA anchor with capped drift — mirrors the pattern already proven
                    # for video style (ema_lambda / ema_sigma_max_drift), applied here to
                    # audio energy instead of a HARD fixed-forever target.
                    #
                    # The 15-chunk run proved the RAW (uncorrected) audio path is a
                    # DIVERGENT autoregressive loop: RMS quadrupled, DC marched to −3.5.
                    # A fixed-forever target (previous fix) stopped the divergence but
                    # ties every chunk to one fixed instant for the ENTIRE run — combined
                    # with the also-strong SLB/ref conditioning, this was confirmed (via
                    # the per-chunk trend log) to cause repetition: within-chunk audio
                    # similarity collapsed the moment full-strength conditioning kicked
                    # in, meaning the model stopped advancing content and re-rendered a
                    # stabilised loop.
                    #
                    # Fix: let the TARGET slowly track the content's own natural
                    # trajectory (EMA of the raw, pre-correction chunk), capped so it can
                    # never wander more than sigma_max_drift from the true chunk-0 origin
                    # — bounded room for the audio's character to evolve over a long
                    # video, without reopening the divergent-loop failure mode.
                    _lam  = clss_config.ema_lambda
                    _drift = clss_config.ema_sigma_max_drift
                    _rms0 = _s1_audio_rms_ref
                    if audio_anchor == "off":
                        # No anchor — keep the raw model output (open_loop baseline).
                        # Only safe on short runs: the raw path is a divergent AR
                        # loop over many chunks (see the history above).
                        print(f"[CLSS S1]   audio anchor: OFF (raw model output kept)")
                    else:
                        # rms_only (the only arm): reference-exact scalar RMS gain
                        # toward the capped EMA target, NO per-channel DC surgery
                        # (the reference streaming pipeline does only this — a
                        # capped gain).  Upward-only: boost quiet chunks, never
                        # attenuate louder ones (see the block comment above).
                        with torch.no_grad():
                            # RMS EMA, capped to [rms0*(1-drift), rms0*(1+drift)]
                            _ema_rms_raw = (1 - _lam) * _s1_audio_ema_rms + _lam * _aud_rms
                            _s1_audio_ema_rms = min(max(_ema_rms_raw, _rms0 * (1 - _drift)),
                                                     _rms0 * (1 + _drift))
                            _g = _s1_audio_ema_rms / max(_aud_rms, 1e-6)
                            if _g > 1.0:
                                new_aud = (new_aud.float() * _g).to(aud_out.dtype)
                            print(f"[CLSS S1]   audio anchor→RMS-only(upward-only, "
                                  f"±{_drift:.0%}): rms {_aud_rms:.4f}→"
                                  f"{new_aud.float().pow(2).mean().sqrt().item():.4f} "
                                  f"(g={_g if _g > 1.0 else 1.0:.4f}{'' if _g > 1.0 else ' (louder chunk kept raw)'}, "
                                  f"ema_rms={_s1_audio_ema_rms:.4f}, "
                                  f"rms0={_rms0:.4f})")
                    # (A per-bin spectral anchor and an audio noise envelope dither
                    # were live-tested here 2026-07-21 and REMOVED: the dither's
                    # local noise-std shaping is off-distribution for the flow and
                    # the anchor then pinned the run to the corrupted chunk-1
                    # reference.  See git history + simulations/audio_corrections_sim.py.)
                # ── Chunk-1 boundary correction (2026-08-11) ────────────────
                # The code reason chunk 2 sounds better than chunk 1: chunk 2 is
                # generated FROM chunk 1's audio AFTER it is corrected (DC removal
                # + per-bin spectral match + RMS match at SLB-save).  Chunk 1's own
                # raw output never receives that treatment.  Apply the SAME
                # correction to chunk 1's output so the first chunk gets exactly
                # what the model continues chunk 2 from.  (The refs are chunk 1's
                # own stats, so steps 2-3 are near-identity; DC removal is the real
                # effect on chunk 1.  Chunk 1 only — later chunks already generate
                # from corrected context.  The env-flatten step was removed
                # 2026-08-14 — see the function docstring.)
                if is_first and _audio_corr and _s1_audio_freq_ref is not None \
                        and _s1_audio_rms_ref is not None:
                    new_aud, _g_lo, _g_hi = _chunk1_boundary_correction(
                        new_aud, _s1_audio_freq_ref, _s1_audio_rms_ref)
                    print(f"[CLSS S1]   chunk-1 boundary correction applied "
                          f"(DC+spectral+RMS — env-flatten removed 2026-08-14, "
                          f"std {new_aud.float().std():.4f})")
                # Trend: RMS and boundary measured on the FINAL kept audio (post-anchor,
                # post-onset-fix) — this is what actually reaches the SLB/output, so it
                # is the honest stability signal.  The boundary print above is pre-anchor;
                # this one tells us whether the SLB is truly carrying continuity forward.
                _trend["aud_rms"].append(new_aud.float().pow(2).mean().sqrt().item())
                if _s1_aud_prev_last is not None:
                    _bnd_final = _aud_cos(_s1_aud_prev_last.to(device), new_aud[:, :, :1])
                    if abs(_bnd_final - (_trend["aud_bnd"][-1] if _trend["aud_bnd"] else _bnd_final)) > 0.02:
                        print(f"[CLSS S1]   audio_boundary_sim(post-anchor)={_bnd_final:.4f}")
                if audio_overlap_af > 0:
                    ov = audio_overlap_af
                    # Audio SLB for next chunk: last ov frames of new_aud = the temporal
                    # period that will be the next chunk's video SLB time.
                    if new_aud.shape[2] >= ov:
                        audio_slb_latent = new_aud[:, :, -ov:].cpu()
                    else:
                        audio_slb_latent = new_aud.cpu()   # short chunk — use all
                    # Normalize SLB to chunk-1 reference statistics before saving.
                    # Without this, the SLB carries the current chunk's (potentially
                    # drifted) spectral shape into the next chunk — a positive feedback
                    # loop that causes HF decay and LF rise across chunks (the drone).
                    # Fix: remove DC, match per-bin energy and RMS to chunk-1 reference.
                    # The model sees the SLB through tau_c re-noising, so exact sample
                    # values don't matter — only the statistical character.
                    if _audio_corr and _s1_audio_freq_ref is not None \
                            and _s1_audio_rms_ref is not None:
                        _slb = audio_slb_latent.float()
                        _slb = _slb - _slb.mean(dim=2, keepdim=True)  # DC removal
                        _slb_freq = _slb.abs().mean(dim=(0, 1, 2))
                        _slb_gains = torch.tensor([
                            min(max(r / max(c, 1e-8), 0.5), 2.0)
                            for r, c in zip(_s1_audio_freq_ref, _slb_freq.tolist())
                        ])
                        _slb = _slb * _slb_gains.view(1, 1, 1, -1)
                        _slb_rms = _slb.pow(2).mean().sqrt()
                        if _slb_rms > 0:
                            _slb = _slb * (_s1_audio_rms_ref / _slb_rms)
                        audio_slb_latent = _slb.to(audio_slb_latent.dtype)
                    print(f"[CLSS S1]   audio SLB saved: {audio_slb_latent.shape[2]}f  "
                          f"mean={audio_slb_latent.float().mean():.4f}"
                          + (f"  (normalized to chunk-1 ref)"
                             if (_audio_corr and _s1_audio_freq_ref is not None) else ""))
                    # ref_audio for next chunk: frames BEFORE the overlap period,
                    # taken from a rolling tail of accumulated output (reference
                    # pipeline.py:771-806).  The tail keeps the last ov+_ref_len_af
                    # frames across chunk boundaries, so the reference window is
                    # always a FULL _ref_len_af frames ending immediately before the
                    # overlap — even when the within-chunk pre-overlap region is
                    # shorter than the window.  (One-overlap window — the
                    # ref_audio_seconds extension was removed 2026-08-19.)
                    _tail_cur = new_aud.cpu()
                    _s1_audio_tail = (
                        _tail_cur if _s1_audio_tail is None
                        else torch.cat([_s1_audio_tail, _tail_cur], dim=2)
                    )
                    _ref_keep = ov + _ref_len_af   # ov for the next SLB + the ref window
                    if _s1_audio_tail.shape[2] > _ref_keep:
                        _s1_audio_tail = _s1_audio_tail[:, :, -_ref_keep:]
                    _s1_audio_tail = _s1_audio_tail.clone()
                    _tail_lf = _s1_audio_tail.shape[2]
                    pre_ov_end = max(0, _tail_lf - ov)   # tail's last ov frames = next overlap
                    if pre_ov_end > 0:
                        _ref_start = max(0, pre_ov_end - _ref_len_af)
                        audio_overlap_latent = _s1_audio_tail[:, :, _ref_start:pre_ov_end].clone()
                        print(f"[CLSS S1]   audio ref saved: {audio_overlap_latent.shape[2]}f "
                              f"(tail[{_ref_start}:{pre_ov_end}], tail_len={_tail_lf}, "
                              f"target={_ref_len_af}f)  "
                              f"mean={audio_overlap_latent.float().mean():.4f}")
                    else:
                        audio_overlap_latent = None
                        print(f"[CLSS S1]   audio ref NOT saved: tail too short "
                              f"({_tail_lf}f ≤ {ov}f)")
                acc_audio.append(new_aud.cpu())
                audio_chunk_ends.append(sum(a.shape[2] for a in acc_audio))
                _s1_aud_prev_last = new_aud[:, :, -1:].cpu()

            _vid_pos += _cur_new_lf

        if _base_pts is not None and _base_mod is not None:
            _base_mod.process_timestep = _base_pts   # restore (Stage 2 shares the base model)

        # Assemble full output latent (all tensors already on CPU)
        full_vid = torch.cat(acc_video, dim=2)
        if acc_audio:
            full_aud = torch.cat(acc_audio, dim=2)
            print(f"[CLSS] Stage 1 full_aud assembled: shape={list(full_aud.shape)} "
                  f"mean={full_aud.float().mean():.4f} std={full_aud.float().std():.4f} "
                  f"nan={full_aud.isnan().any().item()} inf={full_aud.isinf().any().item()}")
            # energy_beta=0: per-chunk RMS is already anchored to chunk-0 inside the
            # loop (see "audio anchor→chunk0").  The old median-RMS renorm here would
            # re-normalize toward the median of already-matched chunks — pure noise at
            # best, and on the pre-fix runaway it dragged quiet early chunks UP toward
            # the blown-up late ones, baking in a distance→loud ramp.  Keep only the
            # boundary smoothing.
            full_aud = _post_process_audio_latent(full_aud, audio_chunk_ends,
                                                  energy_beta=0.0, label=" S1")
            # Post-hoc spectral normalization (2026-08-11): when the connected
            # scheduler starves audio of low-σ refinement steps, a broadband
            # spectral drift accumulates across chunks (HF decays, LF rises →
            # drone).  The per-chunk RMS anchor can't fix spectral tilt (proven
            # in simulations/audio_onset_and_drift_sim.py: 2A — RMS anchor is
            # scale-invariant, ratio unchanged).  Apply a single global per-bin
            # gain matched to chunk-1's reference spectrum on the FINAL assembled
            # audio — post-hoc, no feedback loop, preserves marginals, introduces
            # no periodicity (sim 2B-i/ii/iii all PASS).
            # UPWARD-ONLY (2026-08-14, port-vs-reference audit): bins that GAINED
            # energy vs chunk 1 keep it (gains floored at 1.0) — the measured
            # 2026-08-13/14 runs had chunks boosting mid bins 1.27-1.54× and this
            # norm cut them back to chunk-1's duller spectrum (gains 0.82-0.86);
            # the RMS pin likewise only boosts a quieter output, never pulls a
            # louder one down.  The anti-drone direction (HF decay → boost back)
            # is preserved; the trade-off is a genuine LF rise now passes through
            # — gate on aud_hf and the drone trend on long runs.
            if (_audio_corr
                    and _s1_audio_freq_ref is not None
                    and _s1_audio_rms_ref is not None):
                _cur_freq = full_aud.float().abs().mean(dim=(0, 1, 2))
                _gains = torch.tensor([
                    r / max(c, 1e-8) for r, c in zip(_s1_audio_freq_ref, _cur_freq.tolist())
                ])
                _gains = _gains.clamp(1.0, 2.0)   # upward-only: never cut a brighter bin
                # Smooth: average neighboring bins (kernel size 3) to avoid comb
                _gains_smooth = _gains.clone()
                for i in range(1, len(_gains) - 1):
                    _gains_smooth[i] = (_gains[i-1] + _gains[i] + _gains[i+1]) / 3.0
                full_aud = full_aud.float() * _gains_smooth.view(1, 1, 1, -1)
                _new_rms = full_aud.pow(2).mean().sqrt().item()
                if 0 < _new_rms < _s1_audio_rms_ref:
                    full_aud = full_aud * (_s1_audio_rms_ref / _new_rms)
                full_aud = full_aud.to(torch.bfloat16)
                print(f"[CLSS] Stage 1 audio spectral norm (upward-only): "
                      f"gains=[{', '.join(f'{g:.3f}' for g in _gains_smooth.tolist())}] "
                      f"(chunk-1 ref → final output, clamped [1.0,2.0], 3-bin smooth)")
            print(f"[CLSS] Stage 1 full_aud post: shape={list(full_aud.shape)} "
                  f"mean={full_aud.float().mean():.4f} std={full_aud.float().std():.4f} "
                  f"min={full_aud.float().min():.4f} max={full_aud.float().max():.4f}")
            output_samples = comfy.nested_tensor.NestedTensor((full_vid, full_aud))
        else:
            output_samples = full_vid

        # ── End-of-run Stage 1 trend summary ────────────────────────────────
        # One block that says WHERE we failed, without scraping N chunks by hand.
        # Each line: first→last value, drift %, and a heuristic PASS/WARN verdict.
        # `start` = chunk number of the series' FIRST entry (metrics needing a
        # previous chunk start at 2) so the @chN worst-value tag is exact —
        # WARN lines localize to a chunk (cross-reference the t=[…] chunk
        # headers for the timestamp) instead of needing a manual scrape.
        def _trend_line(name, vals, want, tol, hi_good=True, start=1):
            if not vals:
                return f"    {name:14s}: (no data)"
            v0, vN = vals[0], vals[-1]
            drift = (vN - v0)
            mn, mx = min(vals), max(vals)
            # verdict: for "hi_good" metrics warn if any value falls below want-tol;
            # for stability metrics (want=None) warn if range exceeds tol.
            if want is None:
                bad = (mx - mn) > tol
                tag = "WARN drift" if bad else "ok"
                extra = f"range={mx - mn:+.3f}"
            elif hi_good:
                bad = mn < want - tol
                tag = "WARN" if bad else "ok"
                extra = f"min={mn:.3f}@ch{vals.index(mn) + start}"
            else:
                bad = mx > want + tol
                tag = "WARN" if bad else "ok"
                extra = f"max={mx:.3f}@ch{vals.index(mx) + start}"
            return (f"    {name:14s}: {v0:.3f}→{vN:.3f} (Δ{drift:+.3f}) {extra:18s} [{tag}]")

        print("[CLSS] ═══ Stage 1 trend summary (first→last / drift / verdict) ═══")
        print(_trend_line("vid_std",   _trend["vid_std"],   want=None, tol=0.04))          # creep
        print(_trend_line("vid_ident",  _trend["vid_ident"], want=0.85, tol=0.0, start=2))  # content drift
        print(_trend_line("vid_intra", _trend["vid_intra"], want=0.90, tol=0.0, hi_good=False))  # repetition (ceiling, not floor)
        print(_trend_line("vid_bnd",    _trend["vid_bnd"],   want=0.95, tol=0.0, start=2))  # seam
        if _trend["vid_obj"]:
            print(_trend_line("vid_obj", _trend["vid_obj"], want=0.85, tol=0.0, start=2))  # decoded seam grid-cell identity (object_identity)
        print(_trend_line("vid_hf",     _trend["vid_hf"],    want=None, tol=0.03))          # detail (HF energy share)
        print(_trend_line("vid_origin", _trend["vid_origin"], want=None, tol=0.10))         # drift vs scene-first
        if _trend["aud_env"]:
            print(_trend_line("aud_env", _trend["aud_env"],  want=None, tol=0.30, start=2))  # loudness-gesture repetition
        print(_trend_line("aud_rms",    _trend["aud_rms"],   want=None, tol=0.10))          # energy stability
        print(_trend_line("aud_bnd",    _trend["aud_bnd"],   want=0.80, tol=0.0, start=2))  # audio seam
        print(_trend_line("aud_slb",    _trend["aud_slb"],   want=0.60, tol=0.0, start=2))  # continuity mech (re-noised by design; want = mask-applied floor)
        print(_trend_line("aud_wc",     _trend["aud_wc"],    want=0.80, tol=0.0))           # intra-chunk audio
        print(_trend_line("aud_hf",     _trend["aud_hf"],    want=None, tol=0.50, start=2))  # spectral drift
        if _trend["g_slb"]:
            # Stability requires rho_closed = g * rho_loop < 1, i.e. g < 1/rho_loop.
            # hi_good=False: this is a CEILING check (warn if g approaches/exceeds
            # the point where the correction stack can no longer bound drift).
            _g_ceiling = 1.0 / _rho_loop if _rho_loop > 0 else None
            print(_trend_line("g_slb", _trend["g_slb"], want=_g_ceiling, tol=0.0,
                               hi_good=False, start=2))
            print(f"[CLSS]   g_SLB stability ceiling: rho_loop={_rho_loop:.4f} "
                  f"→ g must stay below {_g_ceiling:.3f} for rho_closed < 1 "
                  f"(first empirical measurement of this quantity — no prior "
                  f"run to compare against)")
        print("[CLSS]   verdicts: vid_std/aud_rms/aud_hf check STABILITY (range); "
              "others check a floor. WARN = the likely failure locus.")
        if len(_origin_track) > 3:
            _drops = sorted(
                ((i, _origin_track[i] - _origin_track[i - 1])
                 for i in range(1, len(_origin_track))),
                key=lambda x: x[1])[:6]
            print("[CLSS] ═══ origin-drift events (largest per-frame drops vs "
                  "scene-first; scene changes appear as events) ═══")
            for _di, _dd in _drops:
                print(f"[CLSS]   t={_di * 8 / fps:6.1f}s  lf={_di:3d}  "
                      f"chunk≈{_frame_chunk_of(_di, _chunk_plan):2d}  Δorigin_sim={_dd:+.4f}  "
                      f"(now {_origin_track[_di]:.3f})")
        # layout_sim (metric v2, 2026-08-17) is a sign-free coarse POWER-map
        # cosine (channel-mean → square → hard 3×3 low-pass → cosine vs the
        # scene-first frame): it isolates coarse scene LAYOUT from texture and
        # cannot sign-flip.  The pre-v2 pooled-cosine version false-positived —
        # single-frame dips (e.g. −0.55 at t≈35.7 s) that are absent from the
        # decoded video, while origin_sim at the same frames shows no
        # excursion — so pre-v2 layout-drift tables OVER-REPORT; entries there
        # are candidate events, not confirmed ones.  Same top-6-drops
        # mechanism, independent metric: a real hard content re-interpretation
        # mid-chunk is the failure class it exists to catch.
        if len(_layout_track) > 3:
            _ldrops = sorted(
                ((i, _layout_track[i] - _layout_track[i - 1])
                 for i in range(1, len(_layout_track))),
                key=lambda x: x[1])[:6]
            print("[CLSS] ═══ layout-drift events (largest per-frame drops in "
                  "coarse scene layout vs scene-first) ═══")
            for _di, _dd in _ldrops:
                print(f"[CLSS]   t={_di * 8 / fps:6.1f}s  lf={_di:3d}  "
                      f"chunk≈{_frame_chunk_of(_di, _chunk_plan):2d}  Δlayout_sim={_dd:+.4f}  "
                      f"(now {_layout_track[_di]:.3f})")

        # ── Phase-lock check: METRONOMIC repetition detector ────────────────
        # Measured failure mode (both 10-chunk runs): the layout minimum lands
        # at the SAME new-frame index in every chunk (frame 5-7, 10/10 chunks)
        # and the audio energy peak at the SAME frame (102/109, 8 straight
        # chunks) — the model replays one motion/loudness arc per chunk,
        # phase-locked to the chunk grid.  Clustered values here = metronome;
        # scattered = organic variety.  This is the metric that judges
        # anti-repetition experiments at a glance.
        def _lock_line(name, vals):
            if len(vals) < 4:
                return f"    {name}: {vals} (too few chunks to judge)"
            _mode = max(set(vals), key=vals.count)
            _near = sum(1 for v in vals if abs(v - _mode) <= 1)
            _locked = _near >= max(4, int(0.7 * len(vals)))
            _tag = "WARN metronome" if _locked else "ok"
            return (f"    {name}: {vals}  → mode={_mode} "
                    f"(±1 covers {_near}/{len(vals)}) [{_tag}]")

        if len(_layout_argmin_track) >= 2 or len(_aud_peak_track) >= 2:
            print("[CLSS] ═══ phase-lock check (clustered = same arc replayed "
                  "every chunk) ═══")
            print(_lock_line("layout argmin frame/chunk", _layout_argmin_track))
            if _aud_peak_track:
                print(_lock_line("audio peak frame/chunk   ", _aud_peak_track))

        # Output is on CPU — unload models and flush CUDA allocator so the upscale
        # model loads into as much VRAM as possible instead of offloading to CPU.
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"[CLSS] VRAM after sampler cleanup: {free/1024**3:.2f} GB free / {total/1024**3:.2f} GB total")

        return ({"samples": output_samples},)


# ---------------------------------------------------------------------------
# Node 4: CLSSUpscaler
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-modality audio sigma schedule (v3, 2026-08-12)
# ---------------------------------------------------------------------------
def _shifted_shared_schedule(sigmas, sigma_shift, terminal=None):
    """Rebuild the WHOLE sigma schedule with the given sigma_shift, keeping the
    same step count and the same terminal/stretch.  Used to put an over-shifted
    template onto the known-good 1.844 curve (the 121-px regime) for BOTH
    modalities — the model sees one shared sigma per step either way, but the
    audio gets the low-sigma tail it needs.
    """
    import math as _m
    sig = sigmas.double()  # float64 internally — exact schedule reconstruction
    n = sig.shape[0] - 1
    lin = torch.linspace(1.0, 0.0, n + 1, dtype=sig.dtype, device=sig.device)
    if terminal is None:
        terminal = float(sig[-2]) if n >= 2 else 0.1
    terminal = max(float(terminal), 1e-3)
    e = _m.exp(float(sigma_shift))
    raw = torch.where(lin != 0.0, e / (e + (1.0 / lin - 1.0)),
                      torch.zeros_like(lin))
    nz = raw != 0.0
    one_minus = 1.0 - raw[nz]
    k = float(one_minus[-1] / (1.0 - terminal))
    cand = raw.clone()
    cand[nz] = 1.0 - one_minus / k
    return cand.to(sigmas.dtype)


def _estimate_video_sigma_shift(sigmas, terminal=None, iters=48):
    """Recover the LTXVScheduler sigma_shift s that produced the given sigma
    schedule, by bisection on the exact scheduler math
    (comfy_extras/nodes_lt.py LTXVScheduler.execute):
        raw_i  = e^s / (e^s + (1/lin_i − 1))      lin = linspace(1, 0, n+1)
        stretch: the last non-zero raw is mapped to `terminal`
    Returns (s, k) where k is the stretch scale (raw = 1 − (1 − σ)/k).
    The schedule is monotonic in s (higher shift ⇒ more top-heavy sigmas
    everywhere), so bisecting on one interior point converges to machine
    precision.  Used by the per-modality audio shift: the audio gets its own
    schedule generated with s_a = audio_shift_mult × s_v.
    """
    import math as _m
    sig = sigmas.double()  # float64 internally — exact schedule reconstruction
    n = sig.shape[0] - 1
    lin = torch.linspace(1.0, 0.0, n + 1, dtype=sig.dtype, device=sig.device)
    if terminal is None:
        terminal = float(sig[-2]) if n >= 2 else 0.1
    terminal = max(float(terminal), 1e-3)

    def _sched(s_shift):
        e = _m.exp(s_shift)
        raw = torch.where(lin != 0.0, e / (e + (1.0 / lin - 1.0)),
                          torch.zeros_like(lin))
        nz = raw != 0.0
        one_minus = 1.0 - raw[nz]
        k = float(one_minus[-1] / (1.0 - terminal))
        cand = raw.clone()
        cand[nz] = 1.0 - one_minus / k
        return cand, k

    lo, hi = 0.0, 14.0
    mid_idx = max(1, min(n - 2, n // 2))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cand, _ = _sched(mid)
        if float(cand[mid_idx]) > float(sig[mid_idx]):
            hi = mid
        else:
            lo = mid
    s_v = 0.5 * (lo + hi)
    _, k_v = _sched(s_v)
    return s_v, k_v


def _audio_shift_schedule(sigmas, audio_shift_mult, terminal=None):
    """Per-modality audio sigma schedule (v3): the audio follows the
    LTXVScheduler schedule generated with sigma_shift = audio_shift_mult × (the
    video's sigma_shift, recovered from the given sigmas), stretched to the same
    terminal sigma.  σ0 stays 1.0 — the audio starts at FULL noise (v2's re-mix
    start that deflated the audio is gone); only the SCHEDULE SHAPE changes.
    mult=1.0 = the video schedule (identity).  Returns
    (audio_sigmas, remap_fn, s_v) where remap_fn maps a video sigma VALUE to the
    audio sigma (used by the a_timestep patch so the model is told the remapped
    sigma, never 'cleaner than reality').

    DEAD END (2026-08-13): a LEAD CAP (audio_i = max(audio_raw_i, σ_i − lead))
    was added to bound how far the audio runs ahead of the video, then removed
    after its first live validation: lead=0.08 DEFLATED the audio (latent std
    0.42 → 0.24, 'worse than before') — capping the lead compresses exactly the
    low-σ phase where this joint model builds the audio's structure, the same
    mechanism as the v2 re-mix deflation.  Do NOT re-add a lead cap; the audio
    needs its full low-σ descent."""
    import math as _m
    sig = sigmas.double()  # float64 internally — exact schedule reconstruction
    n = sig.shape[0] - 1
    lin = torch.linspace(1.0, 0.0, n + 1, dtype=sig.dtype, device=sig.device)
    if terminal is None:
        terminal = float(sig[-2]) if n >= 2 else 0.1
    terminal = max(float(terminal), 1e-3)

    s_v, k_v = _estimate_video_sigma_shift(sig, terminal=terminal)
    s_a = audio_shift_mult * s_v
    e_a = _m.exp(s_a)
    e_v = _m.exp(s_v)

    def _gen(s_shift):
        e = _m.exp(s_shift)
        raw = torch.where(lin != 0.0, e / (e + (1.0 / lin - 1.0)),
                          torch.zeros_like(lin))
        nz = raw != 0.0
        one_minus = 1.0 - raw[nz]
        k = float(one_minus[-1] / (1.0 - terminal))
        cand = raw.clone()
        cand[nz] = 1.0 - one_minus / k
        return cand, k

    audio, k_a = _gen(s_a)

    def remap(t):
        # float64 internally so the told sigma matches the sampler's audio sigma
        # to full precision; result cast back to the input dtype (the patch
        # multiplies a float32 a_timestep).
        tv = t.double().clamp(min=1e-6, max=1.0)
        raw_v = (1.0 - (1.0 - tv) * k_v).clamp(min=1e-6, max=1.0)
        linv = 1.0 / (1.0 + e_v * (1.0 / raw_v - 1.0))
        raw_a = e_a / (e_a + (1.0 / linv - 1.0))
        out = 1.0 - (1.0 - raw_a) / k_a
        out = torch.where(t <= 0.0, torch.zeros_like(out), out)
        return out.to(t.dtype)

    return audio, remap, s_v


def _make_av_permodality_sampler(audio_sigmas, ancestral=True):
    """Build a k_diffusion sampler function that denoises the AUDIO portion of
    the AV latent on its OWN sigma schedule — audio_sigmas (per-modality shift:
    same length as the video sigmas, σ0 = 1.0 so the audio starts at FULL noise,
    but less top-heavy when the audio shift is lower) — while the video keeps
    the original sigmas: a per-modality shift.

    Why this exists (2026-08-12): the user measured that LOWERING the shared
    scheduler shift helps audio but destroys video — the two modalities want
    DIFFERENT schedules.  v2 (audio_sigma = mult×sigma + re-mixed audio start)
    deflated the audio live: re-mixing the start to mult·σ0 < 1 skipped the
    σ∈[mult,1] phase where the model CREATES the audio structure, so the output
    came out quiet/under-developed (full_aud std 0.48 → 0.24 at mult=0.6, 'no
    audio').  v3 keeps the audio at full noise (a0 = s0 = 1.0, NO re-mix) and
    changes only the SCHEDULE SHAPE; the process_timestep patch reports the
    remapped sigma so the model is never told 'cleaner than reality'.

    Replicates comfy.k_diffusion.sampling.sample_euler_ancestral_RF with the
    audio step computed from audio_sigmas (video step is the exact stock update
    on the original sigmas).  Requires the packed AV latent (video tokens then
    audio tokens) and the guider's _av_latent_shapes — both hold in this
    pipeline.  Falls back to the stock RF-ancestral function when the AV
    shapes aren't available.

    ancestral (2026-08-22): follow the SELECTED sampler — previously the wrapper
    always replicated euler_ancestral_RF, silently overriding a user's 'euler'
    choice.  ancestral=False produces the deterministic RF-Euler update
    (eta=0: identical to sample_euler), matching the reference pipeline's
    EulerDiffusionStep (ltx_core diffusion_steps.py:25-40 — the reference never
    injects per-step noise into the audio latent).
    """
    import comfy.utils as _cu
    import comfy.k_diffusion.sampling as _kds

    _a_sig = [float(s) for s in audio_sigmas]

    _eta_default = 1.0 if ancestral else 0.0

    def _sample(model, x, sigmas, extra_args=None, callback=None, disable=None,
                eta=None, s_noise=1.0, noise_sampler=None):
        if eta is None:
            eta = _eta_default
        extra_args = {} if extra_args is None else extra_args
        guider = getattr(model, "inner_model", None)
        shapes = getattr(guider, "_av_latent_shapes", None)
        if shapes is None or len(shapes) != 2:
            # Not the packed AV path (e.g. video-only) — stock behaviour.
            if ancestral:
                return _kds.sample_euler_ancestral_RF(
                    model, x, sigmas, extra_args=extra_args, callback=callback,
                    disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler)
            return _kds.sample_euler(
                model, x, sigmas, extra_args=extra_args, callback=callback,
                disable=disable, s_noise=s_noise, noise_sampler=noise_sampler)

        def _split(t):
            v, a = _cu.unpack_latents(t, shapes)
            return v, a

        def _join(v, a):
            return _cu.pack_latents([v, a])[0]

        seed = extra_args.get("seed", None)
        noise_sampler = _kds.default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
        # Packed audio denoise mask (for the mask-aware sigma re-conversion).
        _denoise_mask = extra_args.get("denoise_mask", None)
        # SEPARATE RNG for the audio's ancestral noise: the stock loop draws
        # noise_sampler(sigma, sigma_next) once per step for the whole latent;
        # if the audio drew from the same sampler, the video's RNG stream would
        # be consumed twice per step and the VIDEO's ancestral noise at later
        # steps would diverge from stock (caught by simulation 2026-08-12).  A
        # dedicated audio stream keeps the video byte-identical.
        audio_noise_sampler = _kds.default_noise_sampler(
            x, seed=(None if seed is None else (seed + 7919) % (2 ** 63)))
        s_noise = s_noise * getattr(
            guider.model_patcher.get_model_object('model_sampling'), "noise_scale", 1.0)
        s_in = x.new_ones([x.shape[0]])

        # NB (v3, 2026-08-12): NO re-mix — audio_sigmas[0] == sigmas[0] == 1.0,
        # so the audio starts at full noise exactly like the video (the
        # coarse-structure phase is preserved).  v2's re-mix of the audio start
        # to mult·σ0 < 1 is what deflated the audio live (std 0.48 → 0.24).
        for i in range(len(sigmas) - 1):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i],
                          'denoised': denoised})
            si = float(sigmas[i]);    sip = float(sigmas[i + 1])
            ai = _a_sig[i] if i < len(_a_sig) else si
            aip = _a_sig[i + 1] if i + 1 < len(_a_sig) else sip
            xv, xa = _split(x)
            dv, da = _split(denoised)

            # Audio sigma re-conversion: the model's calculate_denoised converts
            # its velocity prediction with the SHARED (video) sigma — denoised_a
            # = x_a - v_a·s_i.  But the audio latent is actually at a_i, so the
            # correct x0 is x_a - v_a·a_i.  Re-derive it (identity when a_i==s_i).
            # MASK-AWARE: only fully-denoiséd (mask==1) audio tokens sit at a_i;
            # the mask-pinned SLB/frozen tokens sit at mask·a_i and are already
            # handled by the mask itself — do NOT convert them (caught by
            # simulation 2026-08-12: converting them corrupted their leakage).
            _conv = ai / si
            if _denoise_mask is not None and len(shapes) == 2:
                _, _ma = _cu.unpack_latents(_denoise_mask, shapes)
                _conv = torch.where(_ma >= 0.999, ai / si, 1.0)
            da = xa * (1.0 - _conv) + da * _conv

            # Video: exact stock RF-ancestral update on the ORIGINAL sigmas.
            if sip == 0.0:
                xv = dv
            else:
                _ratio = 1.0 + (sip / si - 1.0) * eta
                _sd = sip * _ratio
                _aip1 = 1.0 - sip
                _ad = 1.0 - _sd
                _rc = (sip ** 2 - _sd ** 2 * _aip1 ** 2 / _ad ** 2) ** 0.5
                _r = _sd / si
                xv = _r * xv + (1.0 - _r) * dv
                if eta > 0:
                    nv = _split(noise_sampler(si, sip))[0]
                    xv = (_aip1 / _ad) * xv + nv * s_noise * _rc

            # Audio: the same update but on the audio-scaled sigma curve.
            if aip == 0.0:
                xa = da
            else:
                _ratio = 1.0 + (aip / ai - 1.0) * eta
                _sd = aip * _ratio
                _aip1 = 1.0 - aip
                _ad = 1.0 - _sd
                _rc = (aip ** 2 - _sd ** 2 * _aip1 ** 2 / _ad ** 2) ** 0.5
                _r = _sd / ai
                xa = _r * xa + (1.0 - _r) * da
                if eta > 0:
                    na = _split(audio_noise_sampler(ai, aip))[1]
                    xa = (_aip1 / _ad) * xa + na * s_noise * _rc

            x = _join(xv, xa)
        return x
    return _sample


class _SlicedNoise:
    """Per-chunk noise wrapper that draws new-frame noise from a pre-generated full-video tensor.

    All Stage 2 chunks share one global noise field (generated once before the loop).
    Each chunk slices the portion matching its new frames, so every frame's starting
    noise is drawn from the same spatially-coherent realisation — no grain discontinuity
    at chunk boundaries.  SLB overlap frames use independent random noise; their mask
    (tau_c ≈ 0.05) reduces the noise contribution to ~4.5 % anyway.
    """

    def __init__(self, full_noise_vid: torch.Tensor, pos: int, chunk_overlap: int, seed: int = 0,
                 full_noise_aud: torch.Tensor | None = None, a_pos: int = 0, a_overlap: int = 0):
        self._full        = full_noise_vid  # [B, C, T_full, H, W] pre-generated
        self._pos         = pos
        self._chunk_overlap = chunk_overlap
        self._full_aud    = full_noise_aud  # [B, C_a, T_a_full, freq] pre-generated (or None)
        self._a_pos       = a_pos
        self._a_overlap   = a_overlap
        self.seed         = seed  # ComfyUI noise interface

    def generate_noise(self, input_latent: dict) -> "torch.Tensor | comfy.nested_tensor.NestedTensor":
        samples = input_latent["samples"]
        is_av   = isinstance(samples, comfy.nested_tensor.NestedTensor)
        vid     = samples.unbind()[0] if is_av else samples

        # Deterministic baseline for the regions the pre-generated fields don't
        # cover (overlap frames; audio when no field was passed).  Was
        # torch.randn_like → global RNG → runs with identical seeds diverged.
        # Seeded per (seed, pos, a_pos) so every chunk of every pass is distinct
        # yet reproducible.  Seed is masked to 31 bits before mixing so the
        # product stays inside manual_seed's uint64 range at any ComfyUI seed.
        _g = torch.Generator(device="cpu").manual_seed(
            (int(self.seed) % (2 ** 31)) * 1_000_003
            + self._pos * 7_919 + self._a_pos * 104_729)
        noise_vid = torch.randn(vid.shape, generator=_g, dtype=vid.dtype).to(vid.device)

        n_new   = vid.shape[2] - self._chunk_overlap                  # frames to fill
        src_end = min(self._pos + n_new, self._full.shape[2])
        src_n   = src_end - self._pos
        if src_n > 0:                                                  # slice consistent noise
            noise_vid[:, :, self._chunk_overlap:self._chunk_overlap + src_n] = \
                self._full[:, :, self._pos:src_end].to(vid.device)

        if is_av:
            aud = samples.unbind()[1]
            noise_aud = torch.randn(aud.shape, generator=_g, dtype=aud.dtype).to(aud.device)
            if self._full_aud is not None:
                a_new   = aud.shape[2] - self._a_overlap
                a_end   = min(self._a_pos + a_new, self._full_aud.shape[2])
                a_n     = a_end - self._a_pos
                if a_n > 0:
                    noise_aud[:, :, self._a_overlap:self._a_overlap + a_n] = \
                        self._full_aud[:, :, self._a_pos:a_end].to(aud.device)
            return comfy.nested_tensor.NestedTensor((noise_vid, noise_aud))
        return noise_vid


class CLSSStage2:
    """Stage 2 of the CLSS two-stage pipeline — chunked distilled-LoRA refinement.

    The full upscaled AV latent (H×W doubled by LTXVLatentUpsampler) is refined
    in temporal chunks using the same CLSS SLB continuity mechanism as Stage 1.

    Why chunking is necessary: the 2× spatially-upscaled latent has 4× more
    tokens per frame (H and W both doubled). Processing all T frames at once
    would require far more VRAM than Stage 1, making it infeasible on 16 GB.

    How new-frame refinement works:
      Each chunk's new frames are seeded from the clean upscaled latent slice.
      ComfyUI's flow-matching noise_scaling then blends it with noise at sigma_0:
          x_start = sigma_0 × noise + (1 − sigma_0) × upscaled_slice
      The 3-step distilled-LoRA schedule [0.909375→0.725→0.421875→0] denoises
      each chunk from this starting point, guided by the upscaled structure.

    Continuity is maintained exactly as in Stage 1:
      SLB overlap frames are seeded from the previous chunk's refined output
      with calibrated tau_c re-noising (noise_mask = tau_c).  The CLSS AdaIN
      correction is NOT applied in Stage 2 (closed-loop refinement, not
      open-loop generation); a per-chunk detail anchor referenced to the
      Stage-1 upscaled slice counters high-frequency loss instead.

    Noise consistency:
      Full-video noise is generated ONCE before the chunk loop and sliced per
      chunk via _SlicedNoise.  This ensures all new frames share the same noise
      realisation — eliminating grain/texture seams at chunk boundaries.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider":           ("GUIDER",      {}),
                "sampler":          ("SAMPLER",     {}),
                "sigmas":           ("SIGMAS",      {}),
                "noise":            ("NOISE",       {}),
                "latent":           ("LATENT",      {
                    "tooltip": "Full upscaled AV latent from LTXVLatentUpsampler → LTXVConcatAVLatent.",
                }),
                "clss_config":      ("CLSS_CONFIG", {}),
                "frames_per_chunk": ("INT",         {
                    "default": 0, "min": 0, "max": 128,
                    "tooltip": "0 = AUTO (recommended): single chunk when the token budget "
                               "allows (no boundary seam), else fewest evenly-sized chunks.\n\n"
                               "Manual override — number of NEW latent frames per Stage 2 chunk.\n\n"
                               "Higher values = fewer chunks = faster overall (fewer model loads), "
                               "but more VRAM per chunk. Lower values fit tighter VRAM budgets.\n\n"
                               "Stage 2 is closed-loop refinement anchored to the Stage 1 upscaled "
                               "latent — chunk boundaries do NOT cause scene changes (unlike Stage 1).\n\n"
                               "Timing reference (93 latent frames, H=22 W=40 on 16 GB):\n"
                               "  fpc=31 → 3 chunks ≈27 min\n"
                               "  fpc=21 → 5 chunks ≈30 min\n"
                               "  fpc=9  → 11 chunks ≈42 min",
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Same guide image connected to CLSSStreamingSampler. "
                                               "Re-encoded at Stage 2 full resolution to anchor chunk 1."}),
                "vae":   ("VAE",   {"tooltip": "VAE for encoding the Stage 2 i2v guide. "
                                               "Required when image is connected."}),
                "s2_overlap": ("INT", {
                    "default": 0, "min": 0, "max": 32,
                    "tooltip": "Stage 2 SLB overlap in latent frames. 0 = use clss_config's "
                               "overlap (default). Raise (e.g. 12-16) to strengthen chunk-boundary "
                               "continuity in Stage 2 without touching Stage 1 — more frozen "
                               "context per chunk at the cost of more tokens per chunk."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Frame rate used to convert latent frames to seconds in "
                               "logging only (chunk timestamps).  Must match the fps used "
                               "on CLSSStreamingSampler — does not affect generation."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "LTX-CLSS"

    @torch.inference_mode()
    def sample(self, guider, sampler, sigmas, noise, latent,
               clss_config: CLSSConfig, frames_per_chunk: int,
               image=None, vae=None, s2_overlap: int = 0,
               fps: float = 24.0):
        import dataclasses

        # ── Full settings dump (raw inputs, unconditional) ──────────────────
        # Same rationale as CLSSStreamingSampler's block: printed before any
        # auto-derivation (frames_per_chunk=0 → auto, s2_overlap=0 → auto) so
        # two runs with identical widget values produce byte-identical text.
        print("[CLSS] ══════════ SETTINGS: CLSSStage2 ══════════")
        print(f"[CLSS]   frames_per_chunk={frames_per_chunk} (0=auto)  "
              f"s2_overlap={s2_overlap} (0=auto)  fps={fps}")
        print(f"[CLSS]   image={'connected' if image is not None else 'none'}  "
              f"vae={'connected' if vae is not None else 'none'}")
        print(f"[CLSS]   audio: frozen passthrough (Stage-1 audio, mask=0)")
        print(f"[CLSS]   clss_config={dataclasses.asdict(clss_config)}")
        print(f"[CLSS]   noise.seed={getattr(noise, 'seed', 'unknown')}  "
              f"guider.cfg={getattr(guider, 'cfg', getattr(guider, 'cfg_scale', 'unknown'))}  "
              f"guider.audio_cfg={getattr(guider, 'audio_cfg', 'unknown')}")
        print("[CLSS] ═══════════════════════════════════════════")

        samples = latent["samples"]
        is_av = isinstance(samples, comfy.nested_tensor.NestedTensor)
        if is_av:
            full_vid, full_aud = samples.unbind()
        else:
            full_vid = samples
            full_aud = None

        B, C_v, T, H, W = full_vid.shape
        device = full_vid.device
        overlap_lf = s2_overlap if s2_overlap > 0 else clss_config.overlap_latent_frames

        # Pre-encode i2v guide at Stage 2 (full) resolution.
        # Stage 1 chunk 1 anchors frame 0 to the guide image; without the same
        # anchor in Stage 2 chunk 1, Stage 2 regenerates the opening segment from
        # ~91% noise unconstrained → content drifts from the guide ("skip to future").
        s2_guide_latent: torch.Tensor | None = None
        s2_i2v_scale_factors = None
        if image is not None and vae is not None:
            s2_i2v_scale_factors = vae.downscale_index_formula
            _, s2_guide_latent = LTXVAddGuide.encode(vae, W, H, image[:1], s2_i2v_scale_factors)
            print(f"[CLSS S2] i2v: guide encoded at Stage 2 H={H} W={W}, "
                  f"latent={list(s2_guide_latent.shape)}")

        if full_aud is not None:
            B_a, C_a, T_a, freq = full_aud.shape
        else:
            B_a = C_a = T_a = freq = 0

        import math as _math
        # ── Auto-derived Stage 2 chunking (length/resolution-dependent) ─────
        # frames_per_chunk=0 → auto: fit the whole video in ONE chunk when the
        # token budget allows (no chunk boundary = no morphing seam — official
        # unchunked-stage-2 parity); otherwise pick the fewest, evenly-sized
        # chunks under the budget.  ~42k video tokens/chunk validated on 16 GB
        # (41.5k ran with offload).
        if frames_per_chunk <= 0:
            # Probe free VRAM instead of assuming a 16 GB card.  42k video tokens
            # validated at ~15.6 GB total; scale linearly with total VRAM, floor at
            # 24k (below that, chunking overhead dominates anyway).
            _budget_tokens = 42000
            if torch.cuda.is_available():
                _total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                _budget_tokens = max(24000, int(42000 * _total_gb / 15.6))
                print(f"[CLSS] auto: token budget={_budget_tokens} (VRAM={_total_gb:.1f} GB)")
            # 20 s RoPE wall (same as Stage 1): a single window
            # (frames_per_chunk + overlap) must stay inside ~20 s or the video
            # goes near-static.  The VRAM budget alone can exceed the wall on
            # big cards at low resolution, so cap the chunk by it (16 = the max
            # auto s2_overlap).
            _s2_wall_fpc = max(12, int((_ROPE_WALL_S - _ROPE_WALL_MARGIN_S) * fps / 8))
            _fpc_cap = max(12, min(_budget_tokens // max(1, H * W), _s2_wall_fpc - 16))
            if T <= _fpc_cap:
                frames_per_chunk = T
            else:
                _n = _math.ceil(T / _fpc_cap)
                frames_per_chunk = _math.ceil(T / _n)
            print(f"[CLSS] auto: frames_per_chunk={frames_per_chunk} "
                  f"(T={T}, tokens/frame={H * W}, budget≈{_budget_tokens}, "
                  f"RoPE-cap={_s2_wall_fpc - 16})")
        # s2_overlap=0 → auto: ~a third of the chunk, clamped [8, 16]; irrelevant
        # for single-chunk runs.
        if s2_overlap <= 0 and frames_per_chunk < T:
            s2_overlap = min(16, max(8, frames_per_chunk // 3))
            overlap_lf = s2_overlap
            print(f"[CLSS] auto: s2_overlap={s2_overlap}")

        # ── 20 s RoPE-wall check (Stage 2, 2026-08-22) ─────────────────────
        # A single Stage-2 window (frames_per_chunk + overlap) must stay inside
        # ~20 s or the refined video goes near-static (same wall as Stage 1).
        # The AUTO path already caps the chunk by it; this catches manual
        # frames_per_chunk values too.
        _s2_win_lf = frames_per_chunk + (s2_overlap if s2_overlap > 0 and frames_per_chunk < T else 0)
        _s2_win_s = _s2_win_lf * 8 / fps
        if _s2_win_s > _ROPE_WALL_S - _ROPE_WALL_MARGIN_S:
            print(f"[CLSS S2] ⚠ Stage-2 window {_s2_win_s:.1f}s (fpc={frames_per_chunk} + "
                  f"overlap={s2_overlap}) past the {_ROPE_WALL_S - _ROPE_WALL_MARGIN_S:.1f}s "
                  f"RoPE wall — reduce frames_per_chunk (or s2_overlap) or the video "
                  f"goes near-static.")

        num_chunks = max(1, (T + frames_per_chunk - 1) // frames_per_chunk)
        # Evenly distribute T so no runt last chunk (e.g. avoids [21,21,21,21,9]).
        # Each chunk gets either ceil(T/N) or floor(T/N) frames, differing by at most 1.
        _base, _extra = T // num_chunks, T % num_chunks
        chunk_boundaries, _cur = [], 0
        for i in range(num_chunks):
            _cur += _base + (1 if i < _extra else 0)
            chunk_boundaries.append(_cur)
        print(f"[CLSS] Stage 2: T={T} H={H} W={W} tokens/frame={H * W} "
              f"frames_per_chunk={frames_per_chunk} overlap={overlap_lf} "
              f"~{num_chunks} chunks  sigma_0={sigmas[0].item():.6f} steps={len(sigmas) - 1}")
        print(f"[CLSS] Stage 2 chunk boundaries (latent frames): {chunk_boundaries}")

        # Pre-generate full-video noise once so every chunk's new frames draw from
        # the same spatially-coherent noise field (no grain seams at boundaries).
        noise_seed = getattr(noise, "seed", 0)
        full_noise_vid: torch.Tensor = noise.generate_noise({"samples": full_vid})

        has_aud = full_aud is not None
        full_noise_aud: torch.Tensor | None = None  # per-chunk seeded noise (no shared field)
        if has_aud:
            print(f"[CLSS] Stage 2: audio frozen (mask=0) — Stage 1 audio passed through unchanged.")

        print(f"[CLSS] Stage 2: CLSS AdaIN correction DISABLED — "
              f"Stage 2 is closed-loop refinement, not open-loop generation.")

        # Stage 2 SLB state (video) — no CLSSState, no AdaIN corrections.
        overlap_latent: torch.Tensor | None = None

        acc_video: list[torch.Tensor] = []
        acc_audio: list[torch.Tensor] = []

        # §item-1,2,6: coherence tracking for Stage 2
        _s2_prev_last:     torch.Tensor | None = None  # [B, C_v, H, W] last new frame of prev S2 chunk
        _s2_id_ref:        torch.Tensor | None = None  # [B, C_v] identity ref from S2 chunk-1
        _s2_aud_prev_last: torch.Tensor | None = None  # [B, C_a, 1, freq] last new audio frame of prev S2 chunk
        _s2_trend = {"fid_first": [], "fid_last": [], "aud_bnd": []}

        def _astats(t: torch.Tensor, label: str) -> str:
            t = t.float()
            return (f"{label}: shape={list(t.shape)} "
                    f"mean={t.mean():.4f} std={t.std():.4f} "
                    f"min={t.min():.4f} max={t.max():.4f} "
                    f"nan={t.isnan().any().item()} inf={t.isinf().any().item()}")

        lf_to_sec = 8 / fps  # latent frames → seconds, logging only

        pos    = 0
        a_pos  = 0  # running audio accumulation position (avoids rounding drift)
        for chunk_idx in range(num_chunks):
            if pos >= T:
                break

            is_first      = (chunk_idx == 0)
            chunk_overlap = 0 if is_first else overlap_lf
            end_pos       = chunk_boundaries[chunk_idx]  # pre-balanced, no runt
            actual_new    = end_pos - pos
            total_lf      = chunk_overlap + actual_new

            t_start = pos * lf_to_sec
            t_end   = end_pos * lf_to_sec
            print(f"[CLSS S2] ── chunk {chunk_idx + 1}/{num_chunks} ──────────────────────────────")
            print(f"[CLSS S2]   video lf=[{pos}:{end_pos}] t=[{t_start:.2f}s:{t_end:.2f}s] "
                  f"({actual_new} new + {chunk_overlap} SLB)  tokens={total_lf * H * W}")

            # ── Video chunk ──────────────────────────────────────────────────
            lat_vid  = torch.zeros(B, C_v, total_lf, H, W, device=device)
            mask_vid = torch.ones(B, 1, total_lf, 1, 1, device=device)

            if not is_first and overlap_latent is not None:
                lat_vid, mask_vid = LTXVAddGuide.replace_latent_frames(
                    lat_vid, mask_vid,
                    guiding_latent=overlap_latent.to(device),
                    latent_idx=0,
                    strength=1.0 - clss_config.tau_c,
                )

            lat_vid[:, :, chunk_overlap:] = full_vid[:, :, pos:end_pos].to(device)
            print(f"[CLSS S2]   {_astats(lat_vid, 'vid_in')}")

            # ── Stage 2 i2v: anchor chunk 1 frame 0 to guide image ──────────
            # In-place replacement, same rationale as Stage 1: append_keyframe adds an
            # out-of-place token block that the joint AV attention sees all chunk long.
            # Frame 0 of the S2 latent already holds the upscaled S1 frame (which adhered
            # to the guide); replacing it with the guide encoded at S2 resolution and
            # freezing it (mask=0) pins the opening frame exactly.
            active_guider = guider
            if is_first and s2_guide_latent is not None:
                lat_vid, mask_vid = LTXVAddGuide.replace_latent_frames(
                    lat_vid, mask_vid,
                    guiding_latent=s2_guide_latent.to(device),
                    latent_idx=0,
                    strength=1.0,
                )
                print(f"[CLSS S2] i2v: guide placed in-place at frame 0, "
                      f"lat_vid={list(lat_vid.shape)} (no appended tokens)")

            # ── Audio chunk (Stage 2) ───────────────────────────────────────
            # Audio FROZEN at the clean Stage-1 latent (mask=0) — Stage 2
            # refines video only, and the Stage-1 audio passes through
            # unchanged.  The video pass still sees the clean Stage-1 audio as
            # cross-modal context, which keeps chunked Stage 2 video continuous.
            if has_aud:
                a_new_start = a_pos
                a_new_end   = min(round(end_pos * T_a / T), T_a)
                chunk_af    = a_new_end - a_new_start
                a_ov     = 0
                lat_aud  = full_aud[:, :, a_new_start:a_new_end].to(device)
                mask_aud = torch.zeros(B_a, C_a, chunk_af, freq, device=device)
                print(f"[CLSS S2]   {_astats(lat_aud, f's1_aud[{a_new_start}:{a_new_end}]')}")
                print(f"[CLSS S2]   aud_in: af=[{a_new_start}:{a_new_end}] chunk_af={chunk_af} "
                      f"acc_a_pos={a_pos}  mask=0 (frozen)")
                chunk_latent = {
                    "samples":    comfy.nested_tensor.NestedTensor((lat_vid, lat_aud)),
                    "noise_mask": comfy.nested_tensor.NestedTensor((mask_vid, mask_aud)),
                }
            else:
                a_ov = 0
                chunk_latent = {"samples": lat_vid, "noise_mask": mask_vid}

            # ── Denoise with consistent per-chunk noise slice ────────────────
            chunk_noise = _SlicedNoise(full_noise_vid, pos, chunk_overlap, seed=noise_seed,
                                       full_noise_aud=full_noise_aud,
                                       a_pos=(a_pos if has_aud else 0),
                                       a_overlap=(a_ov if has_aud else 0))

            _, denoised = SamplerCustomAdvanced().sample(
                noise=chunk_noise,
                guider=active_guider,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=chunk_latent,
            )

            # ── Unpack and accumulate (no CLSS corrections in Stage 2) ───────
            d_samples = denoised["samples"]
            if is_av:
                vid_out, aud_out = d_samples.unbind()
            else:
                vid_out  = d_samples
                aud_out  = None

            new_vid = vid_out[:, :, chunk_overlap:]

            # ── S2 detail anchor (ported from Stage 1) ───────────────────────
            # Stage 2 runs the CLSS AdaIN correction OFF, so nothing
            # counters progressive high-frequency loss across its windows: the
            # refined video softens toward the end (fid_last drifts down) and the
            # soft overlap feeds that forward into the next window.  Anchor each
            # refined chunk's spatial low/high band energy to ITS OWN Stage-1
            # upscaled slice (exactly the detail S2 is meant to preserve), with
            # symmetric gains sqrt(E_ref/E) hard-capped per chunk — same math as
            # the S1 detail_anchor.  Per-chunk reference = the S1 slice, so there
            # is no cross-window EMA to drift; applied BEFORE the overlap seed so
            # softening cannot compound down the chunk chain.
            _s1_ref_slice = full_vid[:, :, pos:end_pos].to(device)   # matches new_vid frames
            _da_x = new_vid.float()
            _rf_x = _s1_ref_slice.float()
            _b2, _c2, _t2, _h2, _w2 = _da_x.shape
            _da_flat = _da_x.permute(0, 2, 1, 3, 4).contiguous().reshape(_b2 * _t2, _c2, _h2, _w2)
            _rf_flat = _rf_x.permute(0, 2, 1, 3, 4).contiguous().reshape(_b2 * _t2, _c2, _h2, _w2)
            _da_low  = torch.nn.functional.avg_pool2d(_da_flat, 3, stride=1, padding=1)
            _da_high = _da_flat - _da_low
            _rf_low  = torch.nn.functional.avg_pool2d(_rf_flat, 3, stride=1, padding=1)
            _rf_high = _rf_flat - _rf_low
            _e_lo = float(_da_low.pow(2).mean());  _e_hi = float(_da_high.pow(2).mean())
            _r_lo = float(_rf_low.pow(2).mean());  _r_hi = float(_rf_high.pow(2).mean())
            _g_lo = min(1.10, max(0.90, (_r_lo / max(_e_lo, 1e-12)) ** 0.5))
            _g_hi = min(1.12, max(0.90, (_r_hi / max(_e_hi, 1e-12)) ** 0.5))
            if abs(_g_lo - 1.0) > 0.005 or abs(_g_hi - 1.0) > 0.005:
                new_vid = (_da_low * _g_lo + _da_high * _g_hi).reshape(
                    _b2, _t2, _c2, _h2, _w2).permute(0, 2, 1, 3, 4).contiguous().to(vid_out.dtype)
                _hf0 = _e_hi / max(_e_lo + _e_hi, 1e-12)
                _hf1 = (_e_hi * _g_hi ** 2) / max(_e_lo * _g_lo ** 2 + _e_hi * _g_hi ** 2, 1e-12)
                print(f"[CLSS S2]   detail anchor: E_low g={_g_lo:.4f}  E_high g={_g_hi:.4f}  "
                      f"hf_share {_hf0:.4f}→{_hf1:.4f} "
                      f"(ref={_r_hi / max(_r_lo + _r_hi, 1e-12):.4f})")
            else:
                print(f"[CLSS S2]   detail anchor: within ±0.5% "
                      f"(g_lo={_g_lo:.4f} g_hi={_g_hi:.4f}) — no-op")

            n_slb   = min(overlap_lf, actual_new)
            overlap_latent = new_vid[:, :, -n_slb:].clone().cpu()
            acc_video.append(new_vid.cpu())
            print(f"[CLSS S2]   {_astats(new_vid, 'vid_out(new)')}")

            # §item-1: intra-chunk cosine — first vs last new frame
            _s2_intra = _frame_cos(new_vid[:, :, 0], new_vid[:, :, -1])
            # §item-2: boundary cosine — last frame of previous S2 chunk vs first new frame
            if _s2_prev_last is not None:
                _s2_bnd = _frame_cos(_s2_prev_last.to(device), new_vid[:, :, 0])
                print(f"[CLSS S2]   boundary_sim={_s2_bnd:.4f}  intra_chunk_sim={_s2_intra:.4f}")
            else:
                print(f"[CLSS S2]   boundary_sim=N/A(first)  intra_chunk_sim={_s2_intra:.4f}")
            # §item-6: identity-retention vs S2-chunk-1 first frame.
            # Stage 2 has no anchor bank; comparing to S2 chunk-1 measures within-Stage-2
            # content consistency but conflates intended scene changes with drift (same
            # ambiguity as Stage 1 vs-chunk-1 when prompts have multiple scenes).
            _s2_cur_feat = F.normalize(new_vid[:, :, 0].float().reshape(B, C_v, -1).mean(-1), dim=1)
            if _s2_id_ref is None:
                _s2_id_ref = _s2_cur_feat.cpu()
                print(f"[CLSS S2]   identity_sim=1.0000 (reference, ambiguous in multi-scene)")
            else:
                _s2_id_sim = (_s2_cur_feat * _s2_id_ref.to(device)).sum(dim=1).mean().item()
                print(f"[CLSS S2]   identity_sim={_s2_id_sim:.4f} (ambiguous in multi-scene)")
            _s2_prev_last = new_vid[:, :, -1].cpu()
            # Per-frame adjacent sims every chunk — choppiness may be present in all.
            if new_vid.shape[2] > 1:
                _s2_adj = [_frame_cos(new_vid[:, :, i], new_vid[:, :, i + 1])
                           for i in range(new_vid.shape[2] - 1)]
                print(f"[CLSS S2]   per-frame adj sims: "
                      f"[{', '.join(f'{s:.3f}' for s in _s2_adj)}]")
            # S2 fidelity to S1 upscaled input — how much S2 changed the content.
            # Target: high fidelity (>0.95) at first frame, loosening toward the end.
            # Very low values (< 0.85) mean S2 is regenerating content, not refining.
            _s1_slice = full_vid[:, :, pos:end_pos].to(device)
            for _fi, _lbl in [(0, "first"), (actual_new // 2, "mid"), (actual_new - 1, "last")]:
                _fid = _frame_cos(new_vid[:, :, _fi], _s1_slice[:, :, _fi])
                print(f"[CLSS S2]   S1_fidelity[{_lbl}]={_fid:.4f}", end="  ")
                if _lbl == "last":
                    _s2_trend["fid_last"].append(_fid)
                if _lbl == "first":
                    _s2_trend["fid_first"].append(_fid)
            print()

            if aud_out is not None:
                # Audio is frozen (mask=0): aud_out is the Stage-1 audio unchanged.
                new_aud = aud_out
                s1_chunk_ref = full_aud[:, :, a_new_start:a_new_end].to(device)
                _s1_sim = _aud_cos(s1_chunk_ref, new_aud)
                print(f"[CLSS S2]   {_astats(new_aud, 'aud_out(frozen)')}"
                      f"  frozen_verify_sim={_s1_sim:.4f}")

                if _s2_aud_prev_last is not None:
                    _aud_bnd = _aud_cos(_s2_aud_prev_last.to(device), new_aud[:, :, :1])
                    print(f"[CLSS S2]   audio_boundary_sim={_aud_bnd:.4f} "
                          f"(seam @ t={t_start:.2f}s)")
                    _s2_trend["aud_bnd"].append(_aud_bnd)
                else:
                    print(f"[CLSS S2]   audio_boundary_sim=N/A(first)")
                _s2_aud_prev_last = new_aud[:, :, -1:].cpu()

                acc_audio.append(new_aud.cpu())
                a_pos += new_aud.shape[2]
                print(f"[CLSS S2]   acc_audio total={a_pos}/{T_a} frames "
                      f"({a_pos / T_a * 100:.1f}%  "
                      f"≈{a_pos / T_a * T * lf_to_sec:.2f}s/{T * lf_to_sec:.2f}s)")

            pos = end_pos

        # ── Assemble full refined output ─────────────────────────────────────
        full_refined_vid = torch.cat(acc_video, dim=2)
        print(f"[CLSS S2] {_astats(full_refined_vid, 'full_refined_vid')}")
        if acc_audio:
            full_refined_aud = torch.cat(acc_audio, dim=2)
            # Frozen passthrough: Stage 2 audio = Stage 1 audio (already normalized).
            # No _post_process_audio_latent — the audio was never re-rolled, so there
            # are no chunk seams to smooth and re-normalizing would double up.
            print(f"[CLSS S2] {_astats(full_refined_aud, 'full_refined_aud(frozen=S1)')}")
            output = comfy.nested_tensor.NestedTensor((full_refined_vid, full_refined_aud))
        elif full_aud is not None:
            print(f"[CLSS S2] no acc_audio — falling back to Stage 1 audio passthrough")
            output = comfy.nested_tensor.NestedTensor((full_refined_vid, full_aud.cpu()))
        else:
            output = full_refined_vid

        # ── End-of-run Stage 2 trend summary ────────────────────────────────
        def _s2line(name, vals, want, tol, stability=False):
            if not vals:
                return f"    {name:16s}: (no data)"
            v0, vN, mn, mx = vals[0], vals[-1], min(vals), max(vals)
            if stability:
                bad = (mx - mn) > tol; tag = "WARN drift" if bad else "ok"
                return f"    {name:16s}: {v0:.3f}→{vN:.3f} (range={mx - mn:+.3f}) [{tag}]"
            bad = mn < want - tol; tag = "WARN" if bad else "ok"
            return f"    {name:16s}: {v0:.3f}→{vN:.3f} (min={mn:.3f}) [{tag}]"

        if _s2_trend["fid_last"] or _s2_trend["aud_bnd"]:
            print("[CLSS] ═══ Stage 2 trend summary ═══")
            print(_s2line("fid_first",  _s2_trend["fid_first"], want=0.95, tol=0.0))
            print(_s2line("fid_last",   _s2_trend["fid_last"],  want=0.95, tol=0.02))   # end-fade signal
            print(_s2line("aud_bnd",    _s2_trend["aud_bnd"],   want=0.80, tol=0.0))    # S2 audio seams
            if len(chunk_boundaries) > 1:
                print("[CLSS]   S2 window seams at: "
                      + ", ".join(f"{b * lf_to_sec:.1f}s" for b in chunk_boundaries[:-1])
                      + "  (aud_bnd entries map to these, in order)")
            print("[CLSS]   fid_last dropping across chunks = video softening toward the end; "
                  "aud_bnd reflects the frozen Stage-1 audio's own continuity at these "
                  "times (Stage 2 never re-rolls audio).")

        return ({"samples": output},)



# ---------------------------------------------------------------------------
# Split AV Guider
# ---------------------------------------------------------------------------

class CLSSAVGuider:
    """Per-modality CFG for joint audio-video models.

    The standard CFGGuider applies one scale to the entire NestedTensor prediction.
    LTX-AV audio needs cfg≈7 for structured content; video is well-behaved at cfg≈4.
    Under-guided audio (cfg=4) drifts toward unstructured noise across chunks, which
    sounds like spectral flattening and loss of tonal content even when RMS looks OK.

    Implementation: injects sampler_cfg_function into model_options so the sampler
    calls our hook instead of the default scalar multiplication.  The hook unbinds
    the NestedTensor cond/uncond predictions, applies per-modality scales, and returns
    the combined noise estimate that ComfyUI's cfg_function expects.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider":    ("GUIDER", {}),
                "audio_cfg": ("FLOAT",  {
                    "default": 7.0, "min": 1.0, "max": 30.0, "step": 0.5,
                    "tooltip": (
                        "CFG scale applied to the audio modality only.\n"
                        "Video CFG comes from the upstream guider (typically 4–5).\n"
                        "LTX-AV reference pipeline: audio_cfg=7.0 + modality guidance 3.0."
                    ),
                }),
                "rescale":   ("FLOAT",  {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Per-modality CFG rescale (reference MultiModalGuider, "
                        "rescale_scale=0.7).  Pulls the guided prediction's std back "
                        "toward the conditional prediction's std:\n"
                        "  factor = rescale·(cond.std/pred.std) + (1−rescale)\n"
                        "Tames guidance overshoot — without it, audio at cfg=7 "
                        "accumulates excess energy (high-freq bins overshoot, "
                        "end-of-chunk RMS surge).  0.0 = off."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "patch"
    CATEGORY = "LTX-CLSS"

    def patch(self, guider, audio_cfg: float, rescale: float = 0.7):
        import copy
        new_guider = copy.copy(guider)
        new_guider.model_options = comfy.model_patcher.create_model_options_clone(
            guider.model_options
        )

        vid_cfg    = getattr(guider, "cfg", 1.0)
        _audio_cfg = audio_cfg
        _rescale   = rescale
        _log_done  = [False]   # mutable cell so the closure can flip it once

        def _av_cfg_fn(args):
            cond_d   = args["cond_denoised"]
            uncond_d = args["uncond_denoised"]
            scale    = args["cond_scale"]    # video CFG from the guider
            x        = args["input"]

            if isinstance(cond_d, comfy.nested_tensor.NestedTensor):
                vid_c, aud_c = cond_d.unbind()
                vid_u, aud_u = uncond_d.unbind()
                x_vid, x_aud = x.unbind()

                vid_denoised = vid_u + scale    * (vid_c - vid_u)
                aud_denoised = aud_u + _audio_cfg * (aud_c - aud_u)

                # Per-modality CFG rescale — the piece of the reference guider port
                # that was missing.  cond.std is the "natural" scale of the model's
                # prediction; guidance at cfg=7 inflates pred.std well beyond it, and
                # that excess energy compounds over the denoising trajectory
                # (observed: audio high-freq bins at 1.4-1.7× reference by chunk 3,
                # end-of-chunk RMS surge).
                _v_factor = _a_factor = 1.0
                if _rescale > 0.0:
                    with torch.no_grad():
                        _v_factor = (_rescale * (vid_c.float().std()
                                     / vid_denoised.float().std().clamp(min=1e-8))
                                     + (1.0 - _rescale)).item()
                        _a_factor = (_rescale * (aud_c.float().std()
                                     / aud_denoised.float().std().clamp(min=1e-8))
                                     + (1.0 - _rescale)).item()
                    vid_denoised = vid_denoised * _v_factor
                    aud_denoised = aud_denoised * _a_factor

                # Log prediction norms + rescale factors once to confirm the split is active
                if not _log_done[0]:
                    _log_done[0] = True
                    with torch.no_grad():
                        v_norm = (vid_c - vid_u).float().norm().item()
                        a_norm = (aud_c - aud_u).float().norm().item()
                    print(
                        f"[CLSS AVGuider] step-1 cfg_diff_norm: "
                        f"vid({scale:.1f})={v_norm:.4f}  aud({_audio_cfg:.1f})={a_norm:.4f}  "
                        f"rescale={_rescale:.2f} factors: vid={_v_factor:.4f} aud={_a_factor:.4f}"
                    )

                # sampler_cfg_function must return noise (x − denoised) — the caller does:
                #   cfg_result = x - fn(args)  →  cfg_result = denoised  ✓
                return comfy.nested_tensor.NestedTensor((
                    x_vid - vid_denoised,
                    x_aud - aud_denoised,
                ))
            else:
                # Non-AV fallback: standard CFG (identity, same as default).
                # On current ComfyUI this branch is ALWAYS taken for AV latents
                # too: CFGGuider.sample() packs the nested AV latent into one
                # flat tensor before sampling, so cond_denoised is never a
                # NestedTensor here and the split-CFG hook is inert.  Warn
                # loudly instead of silently degrading; V2 handles packing.
                if not _log_done[0]:
                    _log_done[0] = True
                    print("[CLSS AVGuider] WARNING: denoised latents arrive "
                          "PACKED (not nested) on this ComfyUI — the v1 split-"
                          "CFG hook cannot split them and is INERT (plain CFG "
                          "only).  Use 'CLSS AV Guider V2' instead.")
                return x - (uncond_d + scale * (cond_d - uncond_d))

        new_guider.model_options["sampler_cfg_function"] = _av_cfg_fn
        new_guider.audio_cfg = _audio_cfg   # readable by downstream nodes for logging
        print(f"[CLSS] AVGuider patched: video_cfg={vid_cfg:.2f}  audio_cfg={_audio_cfg:.2f}  "
              f"rescale={_rescale:.2f}")
        return (new_guider,)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _stg_value_passthrough(attn, x, context=None, mask=None, pe=None, k_pe=None,
                           transformer_options={}):
    """CrossAttention.forward with the attention itself SKIPPED — exact port of
    the reference STG perturbation (ltx-core attention.py: when a block's
    self-attn is perturbed, ``out = to_v(x)`` — the raw value projection —
    then per-head gating and to_out run as normal).  NOT a zero-out: STG's
    perturbed pass degrades the prediction in a specific, trained-adjacent
    way, and the guider amplifies (cond − ptb).
    """
    context = x if context is None else context
    out = attn.to_v(context)
    if attn.to_gate_logits is not None:
        gate_logits = attn.to_gate_logits(x)
        b, t, _ = out.shape
        out = out.view(b, t, attn.heads, attn.dim_head)
        out = out * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
        out = out.view(b, t, attn.heads * attn.dim_head)
    return attn.to_out(out)


class _SkipSelfAttn:
    """Context manager: skip video+audio self-attention in the given blocks of a
    ComfyUI LTXAVModel for the duration of ONE forward (the STG 'ptb' pass).

    Implemented by shadowing the instance ``forward`` of the target blocks'
    attn1/audio_attn1 modules — restored unconditionally on exit.  Equivalent
    to the reference's PerturbationConfig([SKIP_VIDEO_SELF_ATTN,
    SKIP_AUDIO_SELF_ATTN] @ stg_blocks) applied to the whole batch.
    """

    def __init__(self, diffusion_model, blocks):
        self._mods = []
        tb = getattr(diffusion_model, "transformer_blocks", None)
        if tb is not None:
            for bi in blocks:
                if 0 <= bi < len(tb):
                    for name in ("attn1", "audio_attn1"):
                        m = getattr(tb[bi], name, None)
                        if m is not None:
                            self._mods.append(m)

    def __enter__(self):
        import functools
        for m in self._mods:
            m.forward = functools.partial(_stg_value_passthrough, m)
        return self

    def __exit__(self, *exc):
        for m in self._mods:
            try:
                del m.forward          # unshadow the class method
            except AttributeError:
                pass
        return False


class _GuiderCLSSAV(comfy.samplers.CFGGuider):
    """CFGGuider subclass implementing the reference MultiModalGuider for joint AV.

    Per denoising step (reference denoisers.py::_guided_denoise + guiders.py):
      1. cond + uncond passes (standard, batched by calc_cond_batch)
      2. "mod" pass: positive context with BOTH cross-modal attentions skipped —
         ComfyUI's BasicAVTransformerBlock natively honours
         transformer_options["a2v_cross_attn"/"v2a_cross_attn"] (av_model.py:267-268),
         which is exactly the reference's SKIP_A2V_CROSS_ATTN + SKIP_V2A_CROSS_ATTN
         perturbation.
      3. "ptb" pass (STG): positive context with video+audio SELF-attention
         skipped in stg_blocks (value-passthrough, see _stg_value_passthrough) —
         the reference's SKIP_VIDEO_SELF_ATTN + SKIP_AUDIO_SELF_ATTN in ONE
         extra pass (denoisers.py builds exactly one shared ptb pass).
      4. Per-modality combine (guiders.py::MultiModalGuider.calculate):
           pred = cond + (cfg−1)·(cond−uncond) + stg·(cond−ptb) + (modality−1)·(cond−mod)
      5. Per-modality CFG rescale: pred *= r·(cond.std/pred.std) + (1−r)

    The modality term amplifies the component of each modality's prediction
    that comes from the OTHER modality.  The STG term was MISSING from this
    port until 2026-07-11 while the known-good standalone (generate_clss.py)
    defaults it ON for both modalities (_DEFAULT_VIDEO_STG=1.0,
    _DEFAULT_AUDIO_STG=1.0, blocks=[28]) — identified during the bad-audio
    audit as the largest guidance difference between the ComfyUI port (audio
    consistently bad) and the standalone (audio user-validated).

    Cost: 2 base passes, +1 when modality_scale≠1, +1 when stg≠0 (4 total,
    matching the reference's dynamic batch of up to B=4).
    """

    _video_cfg      = 4.0
    _audio_cfg      = 7.0
    _modality_scale = 3.0
    _rescale        = 0.7
    _video_stg      = 1.0
    _audio_stg      = 1.0
    _stg_blocks     = (28,)
    _logged         = False

    def set_av_params(self, video_cfg, audio_cfg, modality_scale, rescale,
                      video_stg=1.0, audio_stg=1.0, stg_block=28):
        self._video_cfg      = video_cfg
        self._audio_cfg      = audio_cfg
        self._modality_scale = modality_scale
        self._rescale        = rescale
        self._video_stg      = video_stg
        self._audio_stg      = audio_stg
        self._stg_blocks     = (int(stg_block),)
        self._logged         = False
        self.set_cfg(video_cfg)          # used by fallback path + downstream logging
        self.audio_cfg = audio_cfg       # readable by CLSSStreamingSampler logging

    @staticmethod
    def _rescale_pred(pred: torch.Tensor, cond: torch.Tensor, r: float) -> torch.Tensor:
        if r <= 0.0:
            return pred
        # BOUNDED (2026-08-23): the reference math factor = r·(cond.std/pred.std)
        # + (1−r) is UNBOUNDED — when the guidance terms nearly cancel (cfg +
        # modality + stg together can drive pred.std → 0), the factor explodes
        # and the prediction overshoots.  Measured live with rescale=0.7 +
        # modality=3: video latent std 3.22 vs healthy ~1.09 and every frame
        # collapsing to the guide (the "static video" signature), while
        # rescale=0 or modality=1 never showed it.  Clamp the final factor to
        # [0.5, 2.0]: the tame-overshoot intent survives, the explosion does
        # not.
        ratio = cond.float().std() / pred.float().std().clamp(min=1e-8)
        factor = r * ratio + (1.0 - r)
        factor = factor.clamp(0.5, 2.0)
        return pred * factor.to(pred.dtype)

    # Per-modality shapes of the CURRENT sampling run's AV latent, captured in
    # sample() before ComfyUI packs the NestedTensor away (see below).
    _av_latent_shapes = None

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None,
               callback=None, disable_pbar=False, seed=None):
        # ── THE bug that silently disabled this whole guider ────────────────
        # ComfyUI's CFGGuider.sample() PACKS nested AV latents into ONE flat
        # [B, 1, N_vid+N_aud] tensor before sampling (comfy.utils.pack_latents,
        # samplers.py ~1272) and only unpacks at the very end.  So by the time
        # predict_noise runs, x is a plain tensor — the old
        # `isinstance(x, NestedTensor)` check was ALWAYS False and every run
        # fell back to plain shared CFG: no split audio cfg, no modality
        # guidance, no STG, ever (verified: the one-time diagnostic lines
        # appear in no ComfyUI log, and per-step timings never changed across
        # guidance configs).  Capture the per-modality shapes here so
        # predict_noise can unpack/repack the packed representation itself.
        if getattr(latent_image, "is_nested", False):
            self._av_latent_shapes = [t.shape for t in latent_image.unbind()]
        else:
            self._av_latent_shapes = None
        return super().sample(noise, latent_image, sampler, sigmas,
                              denoise_mask=denoise_mask, callback=callback,
                              disable_pbar=disable_pbar, seed=seed)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        positive = self.conds.get("positive", None)
        negative = self.conds.get("negative", None)

        is_nested = isinstance(x, comfy.nested_tensor.NestedTensor)
        shapes = self._av_latent_shapes
        is_packed_av = (not is_nested and shapes is not None
                        and len(shapes) == 2 and getattr(x, "ndim", 0) == 3)

        # Neither nested nor packed-AV (or no negative) → standard CFG path.
        # LOUD: silent fallback is exactly what hid the packing bug.
        if (not is_nested and not is_packed_av) or negative is None:
            if not self._logged:
                self._logged = True
                print(f"[CLSS AVGuiderV2] WARNING: falling back to PLAIN shared "
                      f"CFG (cfg={self._video_cfg}) — latent is neither nested "
                      f"nor packed-AV (x shape={getattr(x, 'shape', '?')}, "
                      f"captured shapes={shapes}, negative={'set' if negative is not None else 'MISSING'}). "
                      f"Split/modality/STG are INACTIVE.")
            return super().predict_noise(x, timestep, model_options, seed)

        def _split(t):
            if isinstance(t, comfy.nested_tensor.NestedTensor):
                return t.unbind()
            return comfy.utils.unpack_latents(t, shapes)

        def _join(v, a):
            if is_nested:
                return comfy.nested_tensor.NestedTensor((v, a))
            packed, _ = comfy.utils.pack_latents([v, a])
            return packed

        out_cond, out_uncond = comfy.samplers.calc_cond_batch(
            self.inner_model, [positive, negative], x, timestep, model_options
        )

        out_mod = None
        if self._modality_scale != 1.0 and self._modality_scale != 0.0:
            mo = model_options.copy()
            to = dict(mo.get("transformer_options", {}))
            to["a2v_cross_attn"] = False   # audio→video cross-attn OFF
            to["v2a_cross_attn"] = False   # video→audio cross-attn OFF
            mo["transformer_options"] = to
            (out_mod,) = comfy.samplers.calc_cond_batch(
                self.inner_model, [positive], x, timestep, mo
            )

        # STG "ptb" pass: positive context, video+audio self-attn skipped in
        # stg_blocks (reference: ONE shared pass for both modalities).
        out_ptb = None
        if self._video_stg != 0.0 or self._audio_stg != 0.0:
            _dm = getattr(self.inner_model, "diffusion_model", None)
            _skipper = _SkipSelfAttn(_dm, self._stg_blocks) if _dm is not None else None
            if _skipper is not None and _skipper._mods:
                with _skipper:
                    (out_ptb,) = comfy.samplers.calc_cond_batch(
                        self.inner_model, [positive], x, timestep, model_options
                    )
            elif not self._logged:
                print(f"[CLSS AVGuiderV2] STG requested but transformer_blocks"
                      f"{list(self._stg_blocks)} not reachable on this model — "
                      f"STG term skipped.")

        vid_c, aud_c = _split(out_cond)
        vid_u, aud_u = _split(out_uncond)

        pred_v = vid_c + (self._video_cfg - 1.0) * (vid_c - vid_u)
        pred_a = aud_c + (self._audio_cfg - 1.0) * (aud_c - aud_u)

        if out_ptb is not None:
            vid_p, aud_p = _split(out_ptb)
            pred_v = pred_v + self._video_stg * (vid_c - vid_p)
            pred_a = pred_a + self._audio_stg * (aud_c - aud_p)
            if not self._logged:
                with torch.no_grad():
                    vp = (vid_c - vid_p).float().norm().item()
                    ap = (aud_c - aud_p).float().norm().item()
                print(f"[CLSS AVGuiderV2] step-1 stg_diff_norm: vid={vp:.4f}  aud={ap:.4f}  "
                      f"(0 would mean the self-attn skip is inert)")

        if out_mod is not None:
            vid_m, aud_m = _split(out_mod)
            pred_v = pred_v + (self._modality_scale - 1.0) * (vid_c - vid_m)
            pred_a = pred_a + (self._modality_scale - 1.0) * (aud_c - aud_m)
            if not self._logged:
                with torch.no_grad():
                    vm = (vid_c - vid_m).float().norm().item()
                    am = (aud_c - aud_m).float().norm().item()
                print(f"[CLSS AVGuiderV2] step-1 mod_diff_norm: vid={vm:.4f}  aud={am:.4f}  "
                      f"(0 would mean cross-attn skip is inert)")

        pred_v = self._rescale_pred(pred_v, vid_c, self._rescale)
        pred_a = self._rescale_pred(pred_a, aud_c, self._rescale)

        if not self._logged:
            self._logged = True
            _n_passes = 2 + (0 if out_mod is None else 1) + (0 if out_ptb is None else 1)
            print(f"[CLSS AVGuiderV2] active: video_cfg={self._video_cfg:.1f}  "
                  f"audio_cfg={self._audio_cfg:.1f}  modality={self._modality_scale:.1f}  "
                  f"stg=v{self._video_stg:.1f}/a{self._audio_stg:.1f}@blk{list(self._stg_blocks)}  "
                  f"rescale={self._rescale:.2f}  passes/step={_n_passes}")

        return _join(pred_v, pred_a)


class CLSSAVGuiderV2:
    """Reference-parity AV guider: split CFG + modality guidance + rescale.

    Replaces the CFGGuider → CLSSAVGuider chain.  Connect model + positive +
    negative directly (positive from LTXVConditioning, exactly as you would
    wire a CFGGuider).  Use for Stage 1 only; Stage 2 (distilled LoRA, cfg=1)
    keeps its plain CFGGuider.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":    ("MODEL",        {}),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
                "video_cfg": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "audio_cfg": ("FLOAT", {
                    "default": 7.0, "min": 1.0, "max": 30.0, "step": 0.5,
                    "tooltip": ("CFG scale for the AUDIO modality.  The LTX-AV reference "
                                "pipeline uses 7.0 (video uses 4.0) — under-guided audio "
                                "(≈4) drifts toward unstructured LF-heavy drone and loses "
                                "tonal content even when RMS looks OK.  The old default of "
                                "4.0 produced exactly that (2026-08-22 A/B: same seed / same "
                                "noise / same schedule, cfg 7 → raw chunk-1 mean −0.057, flat "
                                "spectrum; cfg 4 → mean +0.236, bin-0 energy 2.4×).")
                }),
                "modality_scale": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 10.0, "step": 0.5,
                    "tooltip": (
                        "Cross-modal guidance (reference default 3.0).  Runs one extra\n"
                        "transformer pass per step with audio↔video cross-attention\n"
                        "disabled, then amplifies (cond − mod): the part of each\n"
                        "modality's prediction that comes from the OTHER modality.\n"
                        "This is the audio-quality lever — without it, 4-bit audio\n"
                        "decouples from the video and drifts to generic drone.\n"
                        "1.0 = off (no extra pass, no effect)."
                    ),
                }),
                "rescale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Per-modality CFG rescale (reference 0.7)."}),
            },
            "optional": {
                "video_stg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1,
                    "tooltip": "Video STG scale (reference generate_clss.py default: 1.0, "
                               "block 28).  Runs one extra transformer pass with video+audio "
                               "self-attention SKIPPED in stg_block (value-passthrough, the "
                               "reference perturbation), then amplifies (cond − ptb) per "
                               "modality.  This term was MISSING from the ComfyUI port while "
                               "the standalone that produces good audio has it ON — found in "
                               "the 2026-07-11 bad-audio audit.  0 = off (pre-audit "
                               "behaviour, no extra pass)."}),
                "audio_stg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1,
                    "tooltip": "Audio STG scale (reference default 1.0).  Shares the single "
                               "extra ptb pass with video_stg — enabling either costs the "
                               "same one pass (~+33% step time on top of modality guidance)."}),
                "stg_block": ("INT", {
                    "default": 28, "min": 0, "max": 63,
                    "tooltip": "Transformer block whose self-attention is skipped in the STG "
                               "pass.  Reference: [28] for LTX-2.3, [29] for LTX-2."}),
            },
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = "LTX-CLSS"

    def get_guider(self, model, positive, negative, video_cfg, audio_cfg,
                   modality_scale, rescale,
                   video_stg: float = 1.0, audio_stg: float = 1.0, stg_block: int = 28):
        guider = _GuiderCLSSAV(model)
        guider.set_conds(positive, negative)
        guider.set_av_params(video_cfg, audio_cfg, modality_scale, rescale,
                             video_stg=video_stg, audio_stg=audio_stg,
                             stg_block=stg_block)
        print(f"[CLSS] AVGuiderV2 built: video_cfg={video_cfg:.2f}  audio_cfg={audio_cfg:.2f}  "
              f"modality={modality_scale:.2f}  rescale={rescale:.2f}  "
              f"stg=v{video_stg:.1f}/a{audio_stg:.1f}@blk{stg_block}")
        # LOUD guard against the empirically-destructive audio-guidance combos
        # (2026-08-22, same-seed A/B): the ear-validated raw chunk-1 audio
        # (mean −0.06, flat spectrum) comes from audio_cfg=7 + rescale=0.7;
        # audio_cfg≈4 produced mean +0.24 with 2.4× LF-bin energy (drone), and
        # rescale=0 with modality/stg active leaves the guided prediction
        # un-tamed (overshoot → transients + DC).  These are reference defaults,
        # not aesthetics.
        if audio_cfg < 5.0:
            print(f"[CLSS] ⚠ audio_cfg={audio_cfg:.2f} < 5 — UNDER-GUIDED AUDIO: "
                  f"the LTX-AV reference uses 7.0 (video 4.0).  Measured at cfg 4: "
                  f"raw chunk-1 mean +0.24 (DC) and 2.4× low-bin energy vs cfg 7 — "
                  f"a drone, not the model.")
        if rescale <= 0.0 and (modality_scale != 1.0 or audio_stg > 0.0):
            print(f"[CLSS] ⚠ rescale={rescale:.2f} with modality={modality_scale:.1f}/"
                  f"audio_stg={audio_stg:.1f} — the guided audio prediction is "
                  f"NOT pulled back to the conditional std (reference rescale 0.7).  "
                  f"Expect overshoot: transients, DC offset, hotter output.")
        return (guider,)


# ---------------------------------------------------------------------------
# Node 7: CLSSVideoDecodeSave — streaming temporal-slice VAE decode → disk
# ---------------------------------------------------------------------------

class CLSSVideoDecodeSave:
    """Decode a long video latent in temporal slices, saving PNG frames to disk
    incrementally — the whole decoded video is never materialized in RAM.

    Why: the stock decode (VAEDecodeTiled) pre-allocates the FULL pixel output
    tensor — ~18 GB fp32 for a 4×52-lf run at 704×1280, the single largest
    allocation of the pipeline and the OOM point on long runs.  The latents
    themselves are tens of MB; only the decoded frames are huge.  Here peak
    pixel memory is ONE slice (~(frames_per_slice + context) × 8 px frames).

    Slice alignment (LTXV causal 3D VAE: latent frame 0 → 1 px frame, every
    later latent frame → 8 px frames): slices tile the latent with NO pixel
    overlap or gap by construction — slice 0 decodes [0, end) and keeps
    everything; each later slice decodes [pos-1-ctx, end) and drops the
    1 + 8×ctx leading px frames (the shared boundary latent frame plus the
    context), landing exactly on the next unwritten pixel.  The ctx extra LEFT
    context frames (default 2) absorb the decoder's left-boundary transient —
    its temporal convs replicate-pad the first frame when no cache is carried.

    File naming is deterministic (name_00000.png …), so re-running the same
    workflow overwrites the same files instead of piling up copies.

    Audio is untouched: wire the audio branch of LTXVSeparateAVLatent to
    LTXVAudioVAEDecode as usual (the whole audio latent is <1 MB).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae":    ("VAE",    {"tooltip": "The LTXV video VAE (from LTXVideo Loader)."}),
                "latent": ("LATENT", {"tooltip": "Video latent — the video output of "
                                                 "LTXVSeparateAVLatent after CLSSStage2 (a full "
                                                 "AV latent also works; only the video part is "
                                                 "decoded)."}),
                "filename_prefix": ("STRING", {"default": "clss/CLSS_frame_",
                                               "tooltip": "Output prefix (subfolder/name), same "
                                                          "convention as SaveImage.  Frames are "
                                                          "numbered deterministically "
                                                          "(name_00000.png …) — a re-run "
                                                          "overwrites the same files."}),
                "frames_per_slice": ("INT", {"default": 32, "min": 8, "max": 256,
                                             "tooltip": "NEW latent frames decoded per slice.  "
                                                        "Peak pixel memory ≈ (this + context + 1) "
                                                        "× 8 px frames (32 lf ≈ 3 GB fp32 at "
                                                        "704×1280)."}),
                "context_frames": ("INT", {"default": 2, "min": 0, "max": 16,
                                           "tooltip": "Left context latent frames decoded but "
                                                      "discarded per slice — absorbs the VAE "
                                                      "decoder's left-boundary transient "
                                                      "(replicate-pad when no temporal cache is "
                                                      "carried).  0 = exact tile edges (slight "
                                                      "seam risk), 2 = safe default."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "decode_save"
    OUTPUT_NODE = True
    CATEGORY = "LTX-CLSS"

    @torch.inference_mode()
    def decode_save(self, vae, latent, filename_prefix, frames_per_slice=32,
                    context_frames=2):
        import folder_paths
        import numpy as np
        from PIL import Image

        samples = latent["samples"]
        if isinstance(samples, comfy.nested_tensor.NestedTensor):
            samples = samples.unbind()[0]   # AV latent → keep the video part
        T = samples.shape[2]

        output_dir = folder_paths.get_output_directory()
        full_folder, filename, _, _, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir)
        # Never rely on get_save_image_path's folder-creation side effect —
        # a missing subfolder must not sink a multi-hour run at the final node.
        os.makedirs(full_folder, exist_ok=True)
        print(f"[CLSS] DecodeSave: {T} latent frames → {full_folder}/"
              f"{filename}_00000.png…  (slices of {frames_per_slice} lf + "
              f"{context_frames} ctx)")

        frame_idx = 0
        pos = 0
        while pos < T:
            end = min(pos + frames_per_slice, T)
            ctx = 0 if pos == 0 else max(0, min(context_frames, pos - 1))
            start = pos if pos == 0 else pos - 1 - ctx
            px = vae.decode(samples[:, :, start:end])   # [B, T_px, H, W, C] in [0,1]
            drop = 0 if pos == 0 else 1 + ctx * 8
            arr = (px[0, drop:].float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            for f in range(arr.shape[0]):
                Image.fromarray(arr[f]).save(
                    os.path.join(full_folder, f"{filename}_{frame_idx:05d}.png"),
                    compress_level=4)
                frame_idx += 1
            print(f"[CLSS] DecodeSave: latent [{start}:{end}] → {arr.shape[0]} px frames "
                  f"(total {frame_idx})")
            del px, arr
            pos = end

        print(f"[CLSS] DecodeSave: done — {frame_idx} frames written")
        return ()


NODE_CLASS_MAPPINGS = {
    "CLSSConfig":           CLSSConfigNode,
    "CLSSScenePrompts":     CLSSScenePrompts,
    "CLSSStreamingSampler": CLSSStreamingSampler,
    "CLSSStage2":           CLSSStage2,
    "CLSSAVGuider":         CLSSAVGuider,
    "CLSSAVGuiderV2":       CLSSAVGuiderV2,
    "CLSSVideoDecodeSave":  CLSSVideoDecodeSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLSSConfig":           "CLSS Config",
    "CLSSScenePrompts":     "CLSS Scene Prompts",
    "CLSSStreamingSampler": "CLSS Streaming Sampler",
    "CLSSStage2":           "CLSS Stage 2",
    "CLSSAVGuider":         "CLSS AV Guider (Split CFG)",
    "CLSSAVGuiderV2":       "CLSS AV Guider V2 (Split CFG + Modality)",
    "CLSSVideoDecodeSave":  "CLSS Video Decode+Save (streaming)",
}
