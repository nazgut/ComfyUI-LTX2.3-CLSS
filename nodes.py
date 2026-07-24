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

    Why: measured metronomic repetition (10-chunk runs).  Every chunk's audio
    peaks at the SAME frame (102/109 for 8 straight chunks), the SLB tail fed
    to the next chunk always CONTAINS that crescendo peak, and the ref_audio
    carries the build-up before it — so every chunk sees the same loudness
    geometry at the same context positions and replays the same dip-then-
    crescendo arc (env_corr 0.77-0.90; chunk starts vs their own middles
    collapse 0.95→0.31).  Flattening only the CONTEXT (the output audio is
    untouched) removes the loudness template while keeping content
    continuity.  Validated (overlap-8 run): layout/audio phase-lock broke,
    chunk-start collapse gone (0.90-0.92 all run).

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
                "noise_temporal_corr": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.8, "step": 0.05,
                                      "tooltip": "EXPERIMENTAL (unvalidated): mix a run-constant shared "
                                                 "frame into every S1 video noise frame — targets the ~4s "
                                                 "layout oscillation by making the initial noise "
                                                 "temporally correlated (marginals stay exactly N(0,1); "
                                                 "frame-to-frame noise correlation = this value). "
                                                 "0 = off (bit-exact baseline). Changes the generated "
                                                 "video like a seed change would; too high → static "
                                                 "content. Suggested first trial: 0.3."}),
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

    def build(self, tau_c, beta, overlap, noise_temporal_corr=0.0):
        return (CLSSConfig(
            tau_c=tau_c,
            beta=beta,
            ema_lambda=0.10,                 # fixed: validated EMA rate
            ema_sigma_max_drift=0.05,        # fixed: prevents late-chunk amplification
            anchor_force_every=0,            # sentinel: auto-derived in the sampler
            overlap_latent_frames=overlap,
            adain_max_amplification=1.2,     # fixed: caps AdaIN grain boost
            measure_g=False,                 # fixed: diagnostic-only, disabled
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
    decay = 0.5 ** (chunk_idx / half_life)
    return ceiling - (ceiling - base) * decay


_VIDEO_TAU_C_CEILING = 0.10   # conservative: half the empirically-unstable 0.20
_AUDIO_TAU_C_CEILING = 0.15   # audio has the EMA anchor as a backstop


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
                "length_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.5,
                    "tooltip": "Requested video duration in seconds.  When > 0, num_chunks is "
                               "DERIVED from this (reference build_chunk_schedule behaviour): "
                               "total latent frames = ceil from duration at `fps`, chunks = "
                               "ceil(total_lf / new_lf).  Actual duration (rounded up to whole "
                               "chunks) is logged.  0 = use the num_chunks input directly."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Frame rate used to convert length_seconds to frames.  Must match "
                               "the frame_rate set on LTXVConditioning."}),
                "audio_slb": (["auto", "on", "off"], {
                    "default": "off",
                    "tooltip": "on: freeze previous chunk's overlap-time audio at tau_c (current "
                               "design).  off: reference-pipeline design — overlap audio is "
                               "regenerated at full noise and dropped; continuity via ref_audio "
                               "only.  The SLB is a feedback path: a drifting audio tail gets "
                               "frozen into the next chunk's context and compounds (observed: "
                               "RMS +58% and high-freq +280% over 7 chunks).  Use 'off' to A/B "
                               "whether the SLB loop drives the drift."}),
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
                "audio_ref_af": ("INT", {
                    "default": 67, "min": 0, "max": 200,
                    "tooltip": "Length (audio frames) of the S1 ref_audio conditioning, "
                               "DECOUPLED from video overlap.  ref_audio tokens are "
                               "appended conditioning (entry['ref_audio']), not window "
                               "content — their length is architecturally free.  It was "
                               "implicitly tied to the audio overlap: at overlap_lf=3 that "
                               "starved it to 25af (0.77s) — measured aud_wc collapse "
                               "0.91→0.27 and RMS inflation 0.66→0.89 across the run.  "
                               "67 restores the overlap-8-era anchor at ANY video overlap.  "
                               "0 disables ref_audio entirely."}),
                "audio_ctx_flatten": (["on", "off"], {
                    "default": "off",
                    "tooltip": "Flatten the loudness ENVELOPE of the audio context (SLB + "
                               "ref_audio) fed to each next chunk — content/timbre kept, "
                               "loudness arc removed; output audio untouched.  Targets the "
                               "measured metronome: audio peak locked at frame 102/109 for "
                               "8 straight chunks, env_corr 0.77-0.90, chunk-start vs "
                               "mid-chunk sim collapsing 0.95→0.31 — the crescendo tail in "
                               "the SLB re-seeds the same dip-then-crescendo arc every "
                               "chunk.  'off' = raw context (previous behaviour).  Watch "
                               "aud_env / the phase-lock check to judge; risk to watch: "
                               "aud_bnd (context loudness no longer matches the kept "
                               "previous tail exactly)."}),
                "audio_anchor": (["rms_dc", "rms_only", "off"], {
                    "default": "rms_only",
                    "tooltip": "Per-chunk audio energy anchor applied to the kept audio "
                               "before it feeds the SLB/ref (stops the raw autoregressive "
                               "loop from diverging: uncorrected RMS quadrupled, DC "
                               "marched to -3.5 over 15 chunks).\n"
                               "  rms_dc = scalar RMS gain + per-channel DC removal, both "
                               "drift-capped ±5% vs chunk-0 (previous behaviour).\n"
                               "  rms_only = scalar RMS gain ONLY, no DC surgery — matches "
                               "the reference pipeline exactly (streaming/pipeline.py "
                               "does only a capped RMS gain).  Try this first if audio has "
                               "residual artefacts: the per-chunk DC removal (measured "
                               "±0.2-0.3/ch) is our largest un-reference-justified audio "
                               "edit.\n"
                               "  off = keep raw model output (no anchor) — most faithful "
                               "to a single generation, but watch aud_rms for divergence "
                               "on long runs."}),
                "overlap_jitter": (["on", "off"], {
                    "default": "on",
                    "tooltip": "Per-chunk overlap phase-jitter (anti-loop).  Cycles the "
                               "chunk overlap through [full, half, 3/4] of the configured "
                               "value so no two consecutive chunks have the same window "
                               "shape.  WHY: the measured audio/video repetition is a "
                               "fixed point of the chunk map — every chunk ≥2 has an "
                               "IDENTICAL window shape, so the kept region is always the "
                               "same phase of the model's window arc and the arc replays "
                               "phase-locked to the chunk grid (audio_env_corr → 0.84, "
                               "video boundary_sim → 0.97, peak_frame locked).  Jittering "
                               "the overlap changes each chunk's task (window length, "
                               "drop phase, ref content) so no single arc can be a fixed "
                               "point.  Video and audio overlaps stay proportional — no "
                               "A/V offset.  'off' = constant overlap (previous behaviour)."}),
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
        audio_slb: str = "off",
        length_seconds: float = 0.0,
        fps: float = 24.0,
        detail_anchor: str = "on",
        audio_ref_af: int = 67,
        audio_ctx_flatten: str = "off",
        audio_anchor: str = "rms_only",
        overlap_jitter: str = "on",
    ):
        import dataclasses
        import math

        # ── Full settings dump (raw inputs, unconditional) ──────────────────
        # Printed BEFORE any auto-derivation so two runs with identical widget
        # values produce byte-identical text here — diff this block first when
        # two "same settings" runs disagree.  Auto-derived values (num_chunks
        # from length_seconds, anchor_force_every, audio_slb) are logged
        # separately below, at the point they're computed.
        print("[CLSS] ══════════ SETTINGS: CLSSStreamingSampler (Stage 1) ══════════")
        print(f"[CLSS]   num_chunks={num_chunks}  length_seconds={length_seconds}  fps={fps}")
        print(f"[CLSS]   image={'connected' if image is not None else 'none'}  "
              f"vae={'connected' if vae is not None else 'none'}")
        print(f"[CLSS]   audio_slb={audio_slb!r}  detail_anchor={detail_anchor!r}")
        print(f"[CLSS]   audio_ref_af={audio_ref_af}  "
              f"audio_ctx_flatten={audio_ctx_flatten!r}  audio_anchor={audio_anchor!r}")
        print(f"[CLSS]   overlap_jitter={overlap_jitter!r}")
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
        print("[CLSS] ═════════════════════════════════════════════════════════════")

        # ── Length-derived chunk count (reference build_chunk_schedule parity) ──
        # The chunk count is a function of the requested duration, not a knob:
        # total px = duration·fps → total lf via the causal mapping lf=(px−1)/8+1
        # → chunks = ceil(total_lf / new_lf).  Rounded UP to whole chunks; the
        # actual delivered duration is logged.
        if length_seconds > 0.0:
            _tmpl = latent["samples"]
            _vid_t = (_tmpl.unbind()[0] if isinstance(_tmpl, comfy.nested_tensor.NestedTensor)
                      else _tmpl)
            _chunk_lf = _vid_t.shape[2]
            _total_px = max(1, round(length_seconds * fps))
            _total_lf = (_total_px - 1) // 8 + 1
            num_chunks = max(1, math.ceil(_total_lf / _chunk_lf))
            _actual_px = (num_chunks * _chunk_lf - 1) * 8 + 1
            print(f"[CLSS] auto: num_chunks={num_chunks} from length={length_seconds:.1f}s "
                  f"@{fps:.0f}fps ({_total_px}px → {_total_lf}lf, chunk={_chunk_lf}lf) — "
                  f"actual={_actual_px / fps:.2f}s")

        # ── Auto-derived settings (length-dependent — not user knobs) ──────
        # anchor_force_every: force a bank entry roughly every quarter of the run
        # so the anchor bank actually grows on long videos (7 chunks previously
        # produced bank_size=2 with the fixed default of 5) while never anchoring
        # more often than every 2 chunks.
        if clss_config.anchor_force_every <= 0:
            _auto_anchor = max(2, min(5, math.ceil(num_chunks / 4)))
            clss_config = dataclasses.replace(clss_config, anchor_force_every=_auto_anchor)
            print(f"[CLSS] auto: anchor_force_every={_auto_anchor} (num_chunks={num_chunks})")
        # audio_slb: short runs benefit from the frozen-tail boundary continuity;
        # long runs showed SLB feedback runaway (RMS 0.52→0.82, high-freq 3.8×
        # over 7 chunks).  Length decides; explicit on/off remains for A/B only.
        if audio_slb == "auto":
            # Previously auto-disabled beyond 4 chunks because the frozen SLB fed
            # audio energy/DC drift forward and it compounded.  That reason is now
            # removed: the per-chunk hard anchor to chunk-0 (RMS+DC) is applied
            # BEFORE new_aud feeds the SLB/ref, so the fed-forward context can no
            # longer drift.  Without the SLB, ref_audio is the only cross-chunk
            # continuity and it is too weak (measured audio boundary cos ~0.14 over
            # 15 chunks — audible timbral seams every chunk).  Keep the SLB ON at
            # any length; the anchor keeps it stable.
            audio_slb = "on"
            print(f"[CLSS] auto: audio_slb=on (energy anchor makes the SLB safe at any "
                  f"length; provides cross-chunk audio content continuity)")

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
                f"[CLSS] chunk length: new_lf={new_lf} (~{new_lf * 8 / fps:.1f}s per chunk at {fps:.0f} fps). "
                f"Long chunks are supported — the base model natively handles long single "
                f"windows, and per-chunk generation runs in the model's native regime at "
                f"any chunk length. Two practical notes: (1) VRAM and per-step time grow "
                f"with chunk length; (2) the CLSS correction constants (tau_c schedule, "
                f"AdaIN caps, anchors) were tuned at 13 lf and have not been re-tuned for "
                f"longer chunks. (An early pre-correction test saw incoherence above 21 lf; "
                f"that predates the current correction stack and is kept only as history.)"
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
            print(f"[CLSS] audio accounting: af_per_px={_af_per_px:.4f}  "
                  f"chunk1={new_af}af  chunks2+={new_af_cont}af  overlap={audio_overlap_af}af  "
                  f"total={new_af + (num_chunks - 1) * new_af_cont}af for "
                  f"{(num_chunks * new_lf - 1) * 8 + 1}px")
        else:
            B_a = C_a = new_af = freq = audio_overlap_af = new_af_cont = 0

        # ── Per-chunk overlap phase-jitter (anti-loop, chunk-native) ────────
        # The measured audio/video repetition is a FIXED POINT of the chunk
        # map: with a constant overlap, every chunk ≥2 has an identical window
        # shape, so the kept region is always the SAME PHASE of the model's
        # window arc and the arc replays phase-locked to the chunk grid
        # (audio_env_corr → 0.84, video boundary_sim → 0.97, peak_frame locked
        # at 102/109 on the 10-chunk runs; the lock engages the chunk ref_audio
        # reaches full length — hence "audio repeats after chunk 1").  Cycling
        # the overlap through [full, half, ¾] makes consecutive chunks solve
        # DIFFERENT tasks (window length, drop phase, and ref content all
        # change), so no single arc can be the fixed point.  Video and audio
        # overlaps stay proportional (derived from the same per-chunk value) —
        # zero A/V offset, and the video timeline is untouched (each chunk
        # still keeps exactly new_lf frames; only the regenerated/dropped
        # lead-in length varies).  overlap_lf stays the configured MAXIMUM:
        # the SLB buffer (update_buffer stores min(overlap_lf, F) frames) and
        # the audio tail always hold enough context; each chunk slices the
        # LAST chunk_overlap frames of it.
        def _overlap_for_chunk(i: int) -> int:
            if overlap_jitter == "off" or overlap_lf <= 2:
                return overlap_lf
            return (overlap_lf,
                    max(1, overlap_lf // 2),
                    max(1, (3 * overlap_lf) // 4))[i % 3]

        if overlap_jitter == "on" and overlap_lf > 2:
            print(f"[CLSS] overlap jitter: per-chunk overlap cycle "
                  f"[{_overlap_for_chunk(3)}, {_overlap_for_chunk(1)}, {_overlap_for_chunk(2)}] lf "
                  f"(full/half/¾ of {overlap_lf}) — no two consecutive chunks share a "
                  f"window phase (anti-loop; video+audio stay proportional)")

        # Read scene conditionings already stored inside the guider.
        # original_conds["positive"] is a list of converted cond dicts (one per scene
        # after convert_cond ran inside CFGGuider.set_conds). N > 1 means scene prompts.
        pos_conds = guider.original_conds.get("positive", [])
        num_scenes = len(pos_conds)

        _cfg_val = getattr(guider, "cfg", getattr(guider, "cfg_scale", "unknown"))
        _aud_cfg_val = getattr(guider, "audio_cfg", None)
        _cfg_str = (f"video_cfg={_cfg_val} audio_cfg={_aud_cfg_val} (split)"
                    if _aud_cfg_val is not None else f"guider_cfg={_cfg_val} (shared, no split)")
        print(f"[CLSS] Starting — chunks={num_chunks}, new_lf={new_lf}, overlap_lf={overlap_lf}, "
              f"scenes={num_scenes}, tau_c={clss_config.tau_c}, beta={clss_config.beta}, "
              f"mode={'AV' if is_av else 'video-only'}, {_cfg_str}"
              + (f", new_af={new_af}, audio_overlap_af={audio_overlap_af}" if is_av else ""))

        # §item-7/8: corrections active + reproducibility metadata
        _corrections = {
            "renoise": clss_config.tau_c > 0,
            "adain":   clss_config.beta > 0,
            "shrink":  any(g > 0 for g in clss_config.freq_gamma),
            "anchor":  clss_config.anchor_max_size > 0,
        }
        _rho_loop = (1.0 - clss_config.beta) * (1.0 - clss_config.tau_c)
        _seed = getattr(noise, "seed", "unknown")
        print(
            f"[CLSS] Config: corrections={_corrections}"
            f"  rho_loop={_rho_loop:.4f}  seed={_seed}"
            f"  gamma={clss_config.freq_gamma}  beta={clss_config.beta}"
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
        _s1_audio_ref_mean:  torch.Tensor | None = None  # [B_a, C_a, 1, freq] chunk-0 per-(ch×bin) mean (fixed origin, for drift cap)
        _s1_audio_ema_rms:   float | None = None          # slow-drifting RMS anchor target (capped vs origin)
        _s1_audio_ema_dc:    torch.Tensor | None = None   # slow-drifting per-channel DC anchor target (capped vs origin)
        _s1_audio_rms_ref:   float | None = None         # chunk-0 scalar RMS (onset-excluded) — correction target
        _s1_audio_freq_ref:  list[float]  | None = None  # chunk-0 per-bin energy reference (fixed, diagnostic only)
        # Rolling audio tail (reference pipeline.py:771-806): last 2×overlap frames of
        # accumulated output, kept across chunks.  Lets ref_audio be a FULL overlap-length
        # window ending immediately before the next overlap, even when that window spans
        # a chunk boundary (with new_af=102, ov=60 the within-chunk pre-overlap region is
        # only 42f — the tail restores the missing frames from the previous chunk).
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

        for chunk_idx in range(num_chunks):
            is_first = chunk_idx == 0
            chunk_overlap = 0 if is_first else _overlap_for_chunk(chunk_idx)
            total_lf = chunk_overlap + new_lf
            # Per-chunk audio overlap, proportional to the video overlap (same
            # real-time span).  Varies with the jitter cycle; audio_overlap_af
            # stays the configured MAX (buffer/tail sizing).
            _ov_a_k = round(chunk_overlap * 8 * _af_per_px) if aud_tmpl is not None else 0

            scene_idx = 0
            if num_scenes > 1:
                scene_idx = min(int(chunk_idx * num_scenes / num_chunks), num_scenes - 1)

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
                _s1_audio_ref_mean = None
                _s1_audio_ema_rms  = None
                _s1_audio_ema_dc   = None
                _s1_band_ref       = None
                _origin_ref        = None
                _origin_layout     = None
                print(f"[CLSS S1]   scene {(_prev_scene_idx or 0) + 1}→{scene_idx + 1}: "
                      f"statistics anchors re-baselined (video std, audio RMS/DC now "
                      f"anchor to this scene's first chunk; content continuity "
                      f"mechanisms unchanged)")
            _prev_scene_idx = scene_idx

            has_slb     = not is_first and clss_state._overlap_latent is not None
            has_aud_slb = not is_first and audio_slb_latent is not None
            has_aud_ref = not is_first and audio_overlap_latent is not None
            print(f"[CLSS S1] ── Chunk {chunk_idx + 1}/{num_chunks} ── "
                  f"t=[{chunk_idx * new_lf * 8 / fps:.2f}s:{(chunk_idx + 1) * new_lf * 8 / fps:.2f}s] "
                  f"──────────────────")
            print(f"[CLSS S1]   video lf total={total_lf} (overlap={chunk_overlap}+new={new_lf}) "
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
                _tau_c_v = _tau_c_eff(clss_config.tau_c, _VIDEO_TAU_C_CEILING, chunk_idx - 1)
                # Slice the LAST chunk_overlap frames of the SLB buffer (it holds
                # up to overlap_lf = the configured max; the jittered per-chunk
                # overlap is ≤ that).  Immediately-preceding frames, as required.
                lat_vid, mask_vid = LTXVAddGuide.replace_latent_frames(
                    lat_vid, mask_vid,
                    guiding_latent=clss_state._overlap_latent.to(device)[:, :, -chunk_overlap:],
                    latent_idx=0,
                    strength=1.0 - _tau_c_v,
                )
                print(f"[CLSS S1]   video tau_c_eff={_tau_c_v:.4f} (base={clss_config.tau_c}, "
                      f"ceiling={_VIDEO_TAU_C_CEILING})")

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
            # An in-place anchor NUDGE was tried (retrieve the best non-redundant
            # anchor, replace_latent_frames at the first new frame, strength 0.35)
            # and REVERTED 2026-07-21: it injected an old keyframe into every
            # chunk's first new frame and produced visible content morphing /
            # "jump in time" — including chunks pulled BACK to an earlier anchor
            # than a previous chunk had already moved past (log: chunk 7 nudged to
            # anchor@frame35 after chunk 5 used anchor@frame47).  Retrieval picks
            # the LEAST-similar non-redundant anchor, so the nudge actively drags
            # content toward whatever the scene has moved away from — the opposite
            # of continuity.  The reference library's append-style
            # VideoConditionByKeyframeIndex would corrupt AV audio (see the i2v
            # guide note above); the in-place replace variant morphs video.  No
            # safe anchor-conditioning path is currently known, so the bank stays
            # diagnostic-only until one is designed and validated.

            if aud_tmpl is not None:
                # Audio latent covers same temporal span as video (overlap + new frames).
                # cur_new_af: chunk-1 covers (new_lf−1)·8+1 px; later chunks new_lf·8 px.
                cur_new_af = new_af if is_first else new_af_cont
                chunk_af = _ov_a_k + cur_new_af
                lat_aud  = torch.zeros(B_a, C_a, chunk_af, freq, device=device)
                # [B, 1, T, 1] broadcasts correctly through reshape_mask → [B, C, T, freq]
                mask_aud = torch.ones(B_a, 1, chunk_af, 1, device=device)

                # Audio SLB: place previous chunk's overlap-time audio at mask=tau_c.
                # Required: model_base.process_timestep multiplies audio_denoise_mask×sigma
                # → per-token a_timestep.  Without tau_c here, overlap audio tokens get
                # full-sigma a_timestep → a2v cross-attention treats them as maximally
                # noisy even though video SLB is near-clean → video discontinuity.
                _slb_ctx_used: torch.Tensor | None = None   # what was actually PLACED (for the honored check)
                if has_aud_slb and audio_slb == "on":
                    slb = audio_slb_latent.to(device)
                    n   = min(_ov_a_k, slb.shape[2], chunk_af)
                    _tau_c_a = _tau_c_eff(clss_config.tau_c, _AUDIO_TAU_C_CEILING, chunk_idx - 1)
                    # LAST n frames of the saved tail (it is stored at the
                    # configured-max length; the jittered per-chunk overlap n
                    # is ≤ that) — the audio immediately preceding this window.
                    _slb_ctx = slb[:, :, -n:]
                    if audio_ctx_flatten == "on":
                        _slb_ctx, _fg_lo, _fg_hi = _flatten_audio_env(_slb_ctx)
                        print(f"[CLSS S1]   audio SLB env-flattened: gain=[{_fg_lo:.3f}, "
                              f"{_fg_hi:.3f}] (loudness arc removed from context; "
                              f"content kept, output audio untouched)")
                    lat_aud[:, :, :n]  = _slb_ctx
                    mask_aud[:, :, :n] = _tau_c_a
                    _slb_ctx_used = _slb_ctx.detach().cpu()
                    print(f"[CLSS S1]   audio SLB: {n}f  tau_c_eff={_tau_c_a:.4f} "
                          f"(base={clss_config.tau_c}, ceiling={_AUDIO_TAU_C_CEILING})  "
                          f"mean={_slb_ctx.float().mean():.4f}")
                elif has_aud_slb:
                    print(f"[CLSS S1]   audio SLB: OFF (reference design — overlap audio "
                          f"regenerated at mask=1 and dropped; continuity via ref_audio only)")

                # ref_audio at negative RoPE positions: temporal context for what
                # preceded this chunk (av_model.py line 708 prepends ref tokens).
                if has_aud_ref:
                    ref_slb   = audio_overlap_latent.to(device)   # [B, C, T_ov, freq]
                    # Faithful to reference pipeline.py: the previous chunk's
                    # pre-overlap audio tail is the negative-RoPE conditioning,
                    # passed clean and full-length.
                    #
                    # DEAD-END LOG (2026-07-23, do NOT re-add): every attempt to
                    # break the audio metronome by PERTURBING this reference was
                    # tested live on the user's ears and FAILED identically —
                    #   • ref-length decay (0.85^chunk, floored)      → loop unchanged
                    #   • white-noise blend (ramp 0.05/chunk to a cap) → loop unchanged
                    #                                                    + injected HF hiss
                    #   • env-flatten                                  → loop unchanged
                    # The env_corr metronome still locks (→0.9) the chunk the ref
                    # reaches full window, independent of these.  The loop is a
                    # structural property of chunked autoregression over a static
                    # prompt; perturbing the reference only adds noise energy the
                    # model renders as drone/hiss.  A full-video single-pass
                    # re-render was implemented and REJECTED by design (CLSS is
                    # for arbitrary length — a whole-video window defeats it).
                    # The chunk-native attack is the overlap phase-jitter (see
                    # _overlap_for_chunk): it changes each chunk's window shape,
                    # so no single arc can be the fixed point — the ref stays
                    # clean.
                    if audio_ctx_flatten == "on":
                        # Legacy env-flatten lever (off by default; another de-lock
                        # experiment — see dead-end log above).
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

                _n_slb = min(_ov_a_k, audio_slb_latent.shape[2]) if has_aud_slb else 0
                print(f"[CLSS S1]   audio in: chunk_af={chunk_af} "
                      f"(slb={_n_slb}f tau_c + overlap_rest={_ov_a_k - _n_slb}f + new={cur_new_af}f) "
                      f"mask_mean={mask_aud.mean():.3f}")
                av_samples = comfy.nested_tensor.NestedTensor((lat_vid, lat_aud))
                av_mask    = comfy.nested_tensor.NestedTensor((mask_vid, mask_aud))
                chunk_latent = {"samples": av_samples, "noise_mask": av_mask}
            else:
                chunk_latent = {"samples": lat_vid, "noise_mask": mask_vid}

            # Denoise — slice consistent noise per chunk so chunks 2+ get distinct noise
            # (not a repeated realisation caused by same seed + same tensor shape).
            _s1_noise_pos = chunk_idx * new_lf
            _s1_chunk_noise = _SlicedNoise(
                _full_noise_vid_s1, _s1_noise_pos, chunk_overlap, seed=_noise_seed_s1,
                full_noise_aud=_full_noise_aud_s1,
                a_pos=_s1_a_noise_pos,
                a_overlap=_ov_a_k,
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
            # new_vid (the RAW, pre-AdaIN/shrink output) on both sides, so the
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

            corrected = clss_state.post_process(new_vid)

            # ── Detail-band anchor (scene-first-referenced, symmetric, capped) ──
            # Root cause of the reported progressive detail loss, grounded in the
            # §2.4 implementation (clss.py): band gains are (E_ref/E)^γ
            # .clamp(max=1.0) — SHRINK-ONLY — band 0 (γ=0, coarse structure) is
            # a pure pass-through, and _BandEMARef is an UNCAPPED EMA that
            # tracks the drift.  Measured on every long run: band-0 energy
            # inflates (+19.6% over 7 chunks, +15% over 15 chunks; the EMA ref
            # itself drifted 1081→1127 legitimizing it) while high bands are
            # only ever attenuated → the high-frequency SHARE of energy falls
            # ~20%+ → progressively smoother, less detailed output.  Nothing in
            # the loop could RESTORE band energy.  This anchor closes exactly
            # that gap: a two-band (spatial low/high) equalizer with SYMMETRIC
            # gains sqrt(E_ref/E), hard-capped per chunk, referenced to the
            # scene's first chunk (re-baselined on scene change, 0019
            # philosophy).  Complements — does not replace — the §2.4
            # shrinkage; with detail_anchor="off" behaviour is exactly as
            # before (the hf metric is still logged).
            _da_x = corrected.float()
            _da_b, _da_c, _da_t, _da_h, _da_w = _da_x.shape
            _da_flat = _da_x.permute(0, 2, 1, 3, 4).contiguous().reshape(
                _da_b * _da_t, _da_c, _da_h, _da_w)
            _da_low = torch.nn.functional.avg_pool2d(_da_flat, 3, stride=1, padding=1)
            _da_high = _da_flat - _da_low
            _e_low = float(_da_low.pow(2).mean())
            _e_high = float(_da_high.pow(2).mean())
            _hf_share = _e_high / max(_e_low + _e_high, 1e-12)
            if detail_anchor == "on":
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
            # layout (low-band spatial map) from texture.
            if _origin_ref is None:
                _origin_ref = corrected[:, :, -1:].detach().float().cpu()
                _origin_layout = torch.nn.functional.avg_pool2d(
                    _origin_ref[0].mean(0), 3, stride=1, padding=1).flatten()
            _oc = corrected.detach().float().cpu()
            _o_flat = _origin_ref.flatten()
            _osims, _lsims = [], []
            for _fi in range(_oc.shape[2]):
                _fr = _oc[:, :, _fi:_fi + 1]
                _osims.append(float(torch.nn.functional.cosine_similarity(
                    _fr.flatten(), _o_flat, dim=0)))
                _fl = torch.nn.functional.avg_pool2d(
                    _fr[0].mean(0), 3, stride=1, padding=1).flatten()
                _lsims.append(float(torch.nn.functional.cosine_similarity(
                    _fl, _origin_layout, dim=0)))
                _origin_track.append(_osims[-1])
                _layout_track.append(_lsims[-1])
            print(f"[CLSS S1]   origin_sim/frame: {[round(s, 3) for s in _osims]}")
            print(f"[CLSS S1]   layout_sim/frame: {[round(s, 3) for s in _lsims]}"
                  f"  (coarse layout vs scene-first)")
            _layout_argmin_track.append(int(_lsims.index(min(_lsims))))
            _trend["vid_origin"].append(min(_osims))

            # Gentle global-std anchor to chunk-0.  The EMA AdaIN corrects per-channel
            # stats but its sigma cap still permits ~+5% cumulative std growth over a
            # long run (measured 1.008→1.054 over 15 chunks); late chunks run hot, and
            # Stage 2 inherits progressively hotter latents, dropping fidelity toward
            # the end.  Here we only correct the GLOBAL std when it drifts beyond ±4%
            # of chunk-0, and only partially (blend 0.5) — enough to stop the monotonic
            # creep without flattening legitimate per-scene contrast changes.  Mean is
            # left untouched (it carries scene evolution).
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
            # Per-frame adjacent sim for the last chunk — locates visual breaks precisely.
            if chunk_idx == num_chunks - 1 and corrected.shape[2] > 1:
                _adj = [_frame_cos(corrected[:, :, i], corrected[:, :, i + 1])
                        for i in range(corrected.shape[2] - 1)]
                print(f"[CLSS S1]   per-frame adj sims (last chunk): "
                      f"[{', '.join(f'{s:.3f}' for s in _adj)}]")

            if aud_out is not None:
                # Drop the audio overlap-time region (covers the same time as the video SLB).
                # Non-first chunks generate chunk_af = _ov_a_k + cur_new_af frames;
                # we keep only the cur_new_af portion.  First chunk: no drop.
                aud_drop = _ov_a_k
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
                # SLB-honored check: with tau_c=0.05, the SLB frames should survive nearly
                # unchanged → cosine ≥ 0.97.  Low value → noise_mask not applied → wrong diag.
                if not is_first and audio_slb_latent is not None and _ov_a_k > 0:
                    # Compare against what was PLACED (env-flattened when
                    # audio_ctx_flatten is on), not the raw saved tail —
                    # otherwise the flatten reads as a false SLB violation.
                    _slb_expect = (_slb_ctx_used if _slb_ctx_used is not None
                                   else audio_slb_latent[:, :, -_ov_a_k:])
                    _slb_sim = _aud_cos(_slb_expect.to(device),
                                        aud_out[:, :, :_ov_a_k])
                    print(f"[CLSS S1]   audio SLB honored: {_slb_sim:.4f} (expect ≥0.97)")
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
                # Chunk-1 onset fix: linear fade-in on first 4 latent frames to suppress
                # the audio-VAE transient from generating unconditioned from pure noise.
                # Soft per-channel clamp to ±4σ suppresses any remaining outliers.
                if is_first:
                    if new_aud.shape[2] >= 4:
                        _ramp = torch.linspace(0.25, 1.0, 4, device=device)
                        new_aud = new_aud.clone()
                        new_aud[:, :, :4] = new_aud[:, :, :4] * _ramp.view(1, 1, 4, 1)
                        print(f"[CLSS S1]   chunk-1 audio fade-in applied (0.25→1.0 over 4f)")
                    with torch.no_grad():
                        _clip = new_aud.float().std(dim=(2, 3), keepdim=True).clamp(min=1e-6) * 4.0
                    _fa = new_aud.float()
                    new_aud = torch.max(torch.min(_fa, _clip), -_clip).to(aud_out.dtype)
                    print(f"[CLSS S1]   chunk-1 audio soft-clamp ±4σ applied  "
                          f"new_abs_max={new_aud.abs().max().item():.4f}")
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
                # Scalar upward-only RMS gain — exact port of the reference pipeline
                # (pipeline.py:741-757).  One global gain per chunk, applied ONLY when
                # the chunk is quieter than chunk-0 ("don't attenuate genuinely louder
                # chunks"), blended with β=0.3 and capped at 1.15.
                #
                # The previous per-(ch×bin) std gain with clamp(min=1.0) was a
                # boost-only ratchet: it pumped energy into bins whose temporal std
                # decayed even when their |x| energy already sat 1.4-1.7× ABOVE the
                # chunk-0 reference (the high-freq overshoot at audio_cfg=7).  A per-bin
                # correction that can only add energy amplifies exactly the bins the
                # rescaled guidance should be taming.  Replaced by a hard per-chunk
                # anchor to chunk-0 (RMS + per-channel DC) applied before the audio
                # feeds the SLB/ref — see below.
                if _s1_audio_rms_ref is None:
                    _skip = min(16, new_aud.shape[2])
                    _ref_aud = new_aud[:, :, _skip:] if new_aud.shape[2] > _skip else new_aud
                    _s1_audio_rms_ref  = _ref_aud.float().pow(2).mean().sqrt().item()
                    # Per-channel DC reference (onset-excluded) — the FIXED origin used
                    # only to cap how far the EMA target may drift (never the target
                    # itself — see else-branch).
                    _s1_audio_ref_mean = _ref_aud.float().mean(dim=2, keepdim=True).cpu()
                    _s1_audio_ema_rms  = _s1_audio_rms_ref
                    _s1_audio_ema_dc   = _s1_audio_ref_mean.clone()
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
                        # No anchor — keep the raw model output.  Most faithful to a
                        # single generation; only safe on short runs (the raw path is a
                        # divergent AR loop over many chunks — see the rms_dc history).
                        print(f"[CLSS S1]   audio anchor: OFF (raw model output kept)")
                    else:
                        with torch.no_grad():
                            _cur_m = new_aud.float().mean(dim=2, keepdim=True)   # raw, pre-correction
                            # RMS EMA, capped to [rms0*(1-drift), rms0*(1+drift)]
                            _ema_rms_raw = (1 - _lam) * _s1_audio_ema_rms + _lam * _aud_rms
                            _s1_audio_ema_rms = min(max(_ema_rms_raw, _rms0 * (1 - _drift)),
                                                     _rms0 * (1 + _drift))
                            if audio_anchor == "rms_only":
                                # Reference-exact: scalar RMS gain toward the capped EMA
                                # target, NO per-channel DC surgery (the reference
                                # streaming pipeline does only this — a capped gain).
                                _g = _s1_audio_ema_rms / max(_aud_rms, 1e-6)
                                new_aud = (new_aud.float() * _g).to(aud_out.dtype)
                                print(f"[CLSS S1]   audio anchor→RMS-only(ref-exact, "
                                      f"±{_drift:.0%}): rms {_aud_rms:.4f}→"
                                      f"{new_aud.float().pow(2).mean().sqrt().item():.4f} "
                                      f"(g={_g:.4f}, ema_rms={_s1_audio_ema_rms:.4f}, "
                                      f"rms0={_rms0:.4f})")
                            else:  # "rms_dc" — previous behaviour
                                # DC EMA, capped to origin ± (drift × rms0) per (channel×bin) —
                                # rms0 is the natural amplitude scale for "how big a DC shift matters".
                                _dc0 = _s1_audio_ref_mean.to(device)
                                _ema_dc_raw = (1 - _lam) * _s1_audio_ema_dc.to(device) + _lam * _cur_m
                                _delta = (_ema_dc_raw - _dc0).clamp(min=-_drift * _rms0, max=_drift * _rms0)
                                _s1_audio_ema_dc = (_dc0 + _delta).cpu()

                                _ref_m = _s1_audio_ema_dc.to(device)                   # slow-drifting target
                                _tmp   = new_aud.float() - _cur_m + _ref_m             # DC → EMA target
                                _rms_t = _tmp.pow(2).mean().sqrt().clamp(min=1e-6)
                                _g     = _s1_audio_ema_rms / _rms_t
                                new_aud = (_tmp * _g).to(aud_out.dtype)
                                _dc_removed = (_cur_m - _ref_m).mean(dim=-1).squeeze().tolist()
                                print(f"[CLSS S1]   audio anchor→EMA(drift-capped ±{_drift:.0%}): "
                                      f"rms {_aud_rms:.4f}→{new_aud.float().pow(2).mean().sqrt().item():.4f} "
                                      f"(g={_g.item():.4f}, ema_rms={_s1_audio_ema_rms:.4f}, rms0={_rms0:.4f})  "
                                      f"DC removed/ch=[{' '.join(f'{v:+.3f}' for v in _dc_removed)}]")
                    # (A per-bin spectral anchor and an audio noise envelope dither
                    # were live-tested here 2026-07-21 and REMOVED: the dither's
                    # local noise-std shaping is off-distribution for the flow and
                    # the anchor then pinned the run to the corrupted chunk-1
                    # reference.  See git history + simulations/audio_corrections_sim.py.)
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
                    ov = audio_overlap_af   # configured MAX (sizing only)
                    # Next chunk's jittered overlap — the ref window and the SLB
                    # slice placed next chunk are derived from THIS, not the max.
                    _ov_next   = _overlap_for_chunk(chunk_idx + 1) if chunk_idx + 1 < num_chunks else 0
                    _ov_a_next = round(_ov_next * 8 * _af_per_px)
                    # Audio SLB for next chunk: last ov (max) frames of new_aud =
                    # the temporal period that will be the next chunk's video SLB
                    # time.  Stored at max length; the placement slices the LAST
                    # n = min(_ov_a_k, ...) frames (see the audio SLB block).
                    if new_aud.shape[2] >= ov:
                        audio_slb_latent = new_aud[:, :, -ov:].cpu()
                    else:
                        audio_slb_latent = new_aud.cpu()   # short chunk — use all
                    print(f"[CLSS S1]   audio SLB saved: {audio_slb_latent.shape[2]}f  "
                          f"mean={audio_slb_latent.float().mean():.4f}")
                    # ref_audio for next chunk: frames BEFORE the overlap period,
                    # taken from a rolling tail of accumulated output (reference
                    # pipeline.py:771-806).  The tail keeps the last ov+ref_af
                    # frames across chunk boundaries, so the reference window is
                    # always a FULL ref_af frames ending immediately before the
                    # NEXT chunk's (jittered) overlap — even when the
                    # within-chunk pre-overlap region is shorter than that.
                    _tail_cur = new_aud.cpu()
                    _s1_audio_tail = (
                        _tail_cur if _s1_audio_tail is None
                        else torch.cat([_s1_audio_tail, _tail_cur], dim=2)
                    )
                    _ref_keep = ov + max(int(audio_ref_af), 0)
                    if _s1_audio_tail.shape[2] > _ref_keep:
                        _s1_audio_tail = _s1_audio_tail[:, :, -_ref_keep:]
                    _s1_audio_tail = _s1_audio_tail.clone()
                    _tail_lf = _s1_audio_tail.shape[2]
                    pre_ov_end = max(0, _tail_lf - _ov_a_next)   # tail's last _ov_a_next = next overlap
                    if pre_ov_end > 0 and int(audio_ref_af) > 0:
                        _ref_start = max(0, pre_ov_end - int(audio_ref_af))
                        audio_overlap_latent = _s1_audio_tail[:, :, _ref_start:pre_ov_end].clone()
                        print(f"[CLSS S1]   audio ref saved: {audio_overlap_latent.shape[2]}f "
                              f"(tail[{_ref_start}:{pre_ov_end}], tail_len={_tail_lf}, "
                              f"next_ov={_ov_a_next}f)  "
                              f"mean={audio_overlap_latent.float().mean():.4f}")
                    else:
                        audio_overlap_latent = None
                        print(f"[CLSS S1]   audio ref NOT saved: tail too short "
                              f"({_tail_lf}f ≤ {_ov_a_next}f)")
                acc_audio.append(new_aud.cpu())
                audio_chunk_ends.append(sum(a.shape[2] for a in acc_audio))
                _s1_aud_prev_last = new_aud[:, :, -1:].cpu()

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
        print(_trend_line("vid_hf",     _trend["vid_hf"],    want=None, tol=0.03))          # detail (HF energy share)
        print(_trend_line("vid_origin", _trend["vid_origin"], want=None, tol=0.10))         # drift vs scene-first
        if _trend["aud_env"]:
            print(_trend_line("aud_env", _trend["aud_env"],  want=None, tol=0.30, start=2))  # loudness-gesture repetition
        print(_trend_line("aud_rms",    _trend["aud_rms"],   want=None, tol=0.10))          # energy stability
        print(_trend_line("aud_bnd",    _trend["aud_bnd"],   want=0.80, tol=0.0, start=2))  # audio seam
        print(_trend_line("aud_slb",    _trend["aud_slb"],   want=0.97, tol=0.0, start=2))  # continuity mech
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
                      f"chunk≈{_di // new_lf + 1:2d}  Δorigin_sim={_dd:+.4f}  "
                      f"(now {_origin_track[_di]:.3f})")
        # layout_sim isolates coarse scene LAYOUT from texture and reacts to a
        # different failure class than origin_sim: a hard content re-interpretation
        # mid-chunk can crash layout_sim toward 0 while origin_sim (whole-frame
        # feature cosine) barely moves — origin_sim alone missed exactly this case
        # on a run where the reported visual "jump" landed inside a chunk whose
        # layout_sim dropped to ~0 for several frames.  Same top-6-drops mechanism,
        # independent metric, so it catches what origin-drift events don't.
        if len(_layout_track) > 3:
            _ldrops = sorted(
                ((i, _layout_track[i] - _layout_track[i - 1])
                 for i in range(1, len(_layout_track))),
                key=lambda x: x[1])[:6]
            print("[CLSS] ═══ layout-drift events (largest per-frame drops in "
                  "coarse scene layout vs scene-first) ═══")
            for _di, _dd in _ldrops:
                print(f"[CLSS]   t={_di * 8 / fps:6.1f}s  lf={_di:3d}  "
                      f"chunk≈{_di // new_lf + 1:2d}  Δlayout_sim={_dd:+.4f}  "
                      f"(now {_layout_track[_di]:.3f})")

        # ── Phase-lock check: METRONOMIC repetition detector ────────────────
        # Measured failure mode (both 10-chunk runs): the layout minimum lands
        # at the SAME new-frame index in every chunk (frame 5-7, 10/10 chunks)
        # and the audio energy peak at the SAME frame (102/109, 8 straight
        # chunks) — the model replays one motion/loudness arc per chunk,
        # phase-locked to the chunk grid.  Clustered values here = metronome;
        # scattered = organic variety.  This is the metric that judges
        # anti-repetition experiments (audio_ctx_flatten etc.) at a glance.
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
      with calibrated tau_c re-noising (noise_mask = tau_c). CLSS AdaIN and
      spectral-shrinkage corrections are applied to each chunk's new frames.

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
        print(f"[CLSS]   audio=frozen passthrough (Stage-1 audio, mask=0)")
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
            a_ov_af = round(overlap_lf * T_a / T) if T > 0 else 0
        else:
            B_a = C_a = T_a = freq = a_ov_af = 0

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
            _fpc_cap = max(12, _budget_tokens // max(1, H * W))
            if T <= _fpc_cap:
                frames_per_chunk = T
            else:
                _n = _math.ceil(T / _fpc_cap)
                frames_per_chunk = _math.ceil(T / _n)
            print(f"[CLSS] auto: frames_per_chunk={frames_per_chunk} "
                  f"(T={T}, tokens/frame={H * W}, budget≈{_budget_tokens})")
        # s2_overlap=0 → auto: ~a third of the chunk, clamped [8, 16]; irrelevant
        # for single-chunk runs.
        if s2_overlap <= 0 and frames_per_chunk < T:
            s2_overlap = min(16, max(8, frames_per_chunk // 3))
            overlap_lf = s2_overlap
            print(f"[CLSS] auto: s2_overlap={s2_overlap}")

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
        full_noise_aud: torch.Tensor | None = None  # audio is frozen; no audio noise field
        if has_aud:
            print(f"[CLSS] Stage 2: audio frozen (mask=0) — Stage 1 audio passed through unchanged.")
        print(f"[CLSS] Stage 2: CLSS AdaIN/shrinkage corrections DISABLED — "
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

            # ── Audio chunk (Stage 2) ─────────────────────────────────────────
            # Audio is FROZEN at the clean Stage-1 latent (mask=0): Stage 2 refines
            # video only, and the Stage-1 audio passes through unchanged.  The video
            # pass still sees the clean Stage-1 audio as cross-modal context, which
            # keeps chunked Stage 2 video continuous (no boundary morph).
            if has_aud:
                a_ov        = 0
                a_new_start = a_pos
                a_new_end   = min(round(end_pos * T_a / T), T_a)
                chunk_af    = a_new_end - a_new_start
                lat_aud     = full_aud[:, :, a_new_start:a_new_end].to(device)
                mask_aud    = torch.zeros(B_a, C_a, chunk_af, freq, device=device)
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
            # Stage 2 runs the CLSS AdaIN/shrinkage corrections OFF, so nothing
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
            # Frozen passthrough: Stage 2 audio = Stage 1 audio (already normalized).
            # No _post_process_audio_latent — the audio was never re-rolled, so there
            # are no chunk seams to smooth and re-normalizing would double up.
            full_refined_aud = torch.cat(acc_audio, dim=2)
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
        factor = cond.float().std() / pred.float().std().clamp(min=1e-8)
        factor = r * factor + (1.0 - r)
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
                "audio_cfg": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "modality_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.5,
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
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "CLSSConfig":           CLSSConfigNode,
    "CLSSScenePrompts":     CLSSScenePrompts,
    "CLSSStreamingSampler": CLSSStreamingSampler,
    "CLSSStage2":           CLSSStage2,
    "CLSSAVGuider":         CLSSAVGuider,
    "CLSSAVGuiderV2":       CLSSAVGuiderV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLSSConfig":           "CLSS Config",
    "CLSSScenePrompts":     "CLSS Scene Prompts",
    "CLSSStreamingSampler": "CLSS Streaming Sampler",
    "CLSSStage2":           "CLSS Stage 2",
    "CLSSAVGuider":         "CLSS AV Guider (Split CFG)",
    "CLSSAVGuiderV2":       "CLSS AV Guider V2 (Split CFG + Modality)",
}
