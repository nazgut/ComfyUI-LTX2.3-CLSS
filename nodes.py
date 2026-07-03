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
            }
        }
        # Everything else is fixed or derived automatically:
        #   ema_lambda=0.10, sigma_max_drift=0.05, adain_max_amplification=1.2
        #   (validated internals — wrong values silently corrupt the video);
        #   anchor_force_every is derived from num_chunks inside the sampler.

    RETURN_TYPES = ("CLSS_CONFIG",)
    RETURN_NAMES = ("clss_config",)
    FUNCTION = "build"
    CATEGORY = "LTX-CLSS"

    def build(self, tau_c, beta, overlap):
        return (CLSSConfig(
            tau_c=tau_c,
            beta=beta,
            ema_lambda=0.10,                 # fixed: validated EMA rate
            ema_sigma_max_drift=0.05,        # fixed: prevents late-chunk amplification
            anchor_force_every=0,            # sentinel: auto-derived in the sampler
            overlap_latent_frames=overlap,
            adain_max_amplification=1.2,     # fixed: caps AdaIN grain boost
            measure_g=False,
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
                    "default": 25.0, "min": 1.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Frame rate used to convert length_seconds to frames.  Must match "
                               "the frame_rate set on LTXVConditioning."}),
                "audio_slb": (["auto", "on", "off"], {
                    "default": "auto",
                    "tooltip": "on: freeze previous chunk's overlap-time audio at tau_c (current "
                               "design).  off: reference-pipeline design — overlap audio is "
                               "regenerated at full noise and dropped; continuity via ref_audio "
                               "only.  The SLB is a feedback path: a drifting audio tail gets "
                               "frozen into the next chunk's context and compounds (observed: "
                               "RMS +58% and high-freq +280% over 7 chunks).  Use 'off' to A/B "
                               "whether the SLB loop drives the drift."}),
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
        audio_slb: str = "auto",
        length_seconds: float = 0.0,
        fps: float = 25.0,
    ):
        import dataclasses
        import math

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
            audio_slb = "on" if num_chunks <= 4 else "off"
            print(f"[CLSS] auto: audio_slb={audio_slb} (num_chunks={num_chunks}; "
                  f"SLB feedback compounds on long runs)")

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
                f"[CLSS] WARNING: new_lf={new_lf} exceeds recommended maximum of 21. "
                f"Early testing (before CLSS corrections) showed 21 latent frames caused near-random "
                f"intra-chunk content (intra_chunk_sim≈0.03-0.11). With CLSS corrections active "
                f"(tau_c + AdaIN + shrinkage), values up to ~31 may still be acceptable, but "
                f"quality degrades with lower step counts or weaker guidance. "
                f"Reduce to ≤13 frames (≈4 s @ 24 fps) for safest results. "
                f"Current setting produces ~{new_lf * 8 / 25:.0f}s per chunk at 25fps."
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
            _af_per_px = new_af / _first_px if _first_px > 0 else 0.0
            audio_overlap_af = round(overlap_lf * 8 * _af_per_px)
            new_af_cont      = round(new_lf * 8 * _af_per_px)
            print(f"[CLSS] audio accounting: af_per_px={_af_per_px:.4f}  "
                  f"chunk1={new_af}af  chunks2+={new_af_cont}af  overlap={audio_overlap_af}af  "
                  f"total={new_af + (num_chunks - 1) * new_af_cont}af for "
                  f"{(num_chunks * new_lf - 1) * 8 + 1}px")
        else:
            B_a = C_a = new_af = freq = audio_overlap_af = new_af_cont = 0

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
        _s1_aud_prev_last:   torch.Tensor | None = None  # [B, C_a, 1, freq] last audio frame
        _s1_audio_ref_mean:  torch.Tensor | None = None  # [B_a, C_a, 1, freq] chunk-0 per-(ch×bin) mean (diagnostics)
        _s1_audio_ref_std:   torch.Tensor | None = None  # [B_a, C_a, 1, freq] chunk-0 per-(ch×bin) std (diagnostics)
        _s1_audio_rms_ref:   float | None = None         # chunk-0 scalar RMS (onset-excluded) — correction target
        _s1_audio_freq_ref:  list[float]  | None = None  # chunk-0 per-bin energy reference
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

        for chunk_idx in range(num_chunks):
            is_first = chunk_idx == 0
            chunk_overlap = 0 if is_first else overlap_lf
            total_lf = chunk_overlap + new_lf

            scene_idx = 0
            if num_scenes > 1:
                scene_idx = min(int(chunk_idx * num_scenes / num_chunks), num_scenes - 1)

            has_slb     = not is_first and clss_state._overlap_latent is not None
            has_aud_slb = not is_first and audio_slb_latent is not None
            has_aud_ref = not is_first and audio_overlap_latent is not None
            print(f"[CLSS S1] ── Chunk {chunk_idx + 1}/{num_chunks} ──────────────────────────────")
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

            # §2.1 Place SLB at overlap frames with noise_mask = tau_c
            if has_slb:
                lat_vid, mask_vid = LTXVAddGuide.replace_latent_frames(
                    lat_vid, mask_vid,
                    guiding_latent=clss_state._overlap_latent.to(device),
                    latent_idx=0,
                    strength=1.0 - clss_config.tau_c,
                )

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

            if aud_tmpl is not None:
                # Audio latent covers same temporal span as video (overlap + new frames).
                # cur_new_af: chunk-1 covers (new_lf−1)·8+1 px; later chunks new_lf·8 px.
                cur_new_af = new_af if is_first else new_af_cont
                chunk_af = (audio_overlap_af if not is_first else 0) + cur_new_af
                lat_aud  = torch.zeros(B_a, C_a, chunk_af, freq, device=device)
                # [B, 1, T, 1] broadcasts correctly through reshape_mask → [B, C, T, freq]
                mask_aud = torch.ones(B_a, 1, chunk_af, 1, device=device)

                # Audio SLB: place previous chunk's overlap-time audio at mask=tau_c.
                # Required: model_base.process_timestep multiplies audio_denoise_mask×sigma
                # → per-token a_timestep.  Without tau_c here, overlap audio tokens get
                # full-sigma a_timestep → a2v cross-attention treats them as maximally
                # noisy even though video SLB is near-clean → video discontinuity.
                if has_aud_slb and audio_slb == "on":
                    slb = audio_slb_latent.to(device)
                    n   = min(audio_overlap_af, slb.shape[2], chunk_af)
                    lat_aud[:, :, :n]  = slb[:, :, :n]
                    mask_aud[:, :, :n] = clss_config.tau_c
                    print(f"[CLSS S1]   audio SLB: {n}f  tau_c={clss_config.tau_c}  "
                          f"mean={slb[:, :, :n].float().mean():.4f}")
                elif has_aud_slb:
                    print(f"[CLSS S1]   audio SLB: OFF (reference design — overlap audio "
                          f"regenerated at mask=1 and dropped; continuity via ref_audio only)")

                # ref_audio at negative RoPE positions: temporal context for what
                # preceded this chunk (av_model.py line 708 prepends ref tokens).
                if has_aud_ref:
                    ref_slb   = audio_overlap_latent.to(device)   # [B, C, T_ov, freq]
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
                print(f"[CLSS S1]   audio in: chunk_af={chunk_af} "
                      f"(slb={_n_slb}f tau_c + overlap_rest={audio_overlap_af - _n_slb}f + new={new_af}f) "
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
                _full_noise_vid_s1, _s1_noise_pos, chunk_overlap, seed=_noise_seed_s1
            )
            print(
                f"[CLSS S1]   noise pos={_s1_noise_pos} "
                f"fingerprint={_full_noise_vid_s1[:, :, _s1_noise_pos:_s1_noise_pos+1].flatten()[:4].tolist()}"
            )
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
            corrected = clss_state.post_process(new_vid)
            mu_post   = corrected.mean().item()
            std_post  = corrected.std().item()
            clss_state.update_buffer(corrected)
            acc_video.append(corrected.cpu())

            print(f"[CLSS S1]   video done: pre_AdaIN mean={mu_pre:.4f} std={std_pre:.4f} | "
                  f"post_AdaIN mean={mu_post:.4f} std={std_post:.4f} | "
                  f"video_SLB updated shape={clss_state._overlap_latent.shape if clss_state._overlap_latent is not None else 'None'}")

            # §item-1: intra-chunk cosine — first vs last new frame (corrected latent)
            _intra = _frame_cos(corrected[:, :, 0], corrected[:, :, -1])
            # §item-2: boundary cosine — last frame of previous chunk vs first new frame
            if _s1_prev_last is not None:
                _bnd = _frame_cos(_s1_prev_last.to(device), corrected[:, :, 0])
                print(f"[CLSS S1]   boundary_sim={_bnd:.4f}  intra_chunk_sim={_intra:.4f}")
            else:
                print(f"[CLSS S1]   boundary_sim=N/A(first)  intra_chunk_sim={_intra:.4f}")
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
                # SLB-honored check: with tau_c=0.05, the SLB frames should survive nearly
                # unchanged → cosine ≥ 0.97.  Low value → noise_mask not applied → wrong diag.
                if not is_first and audio_slb_latent is not None and audio_overlap_af > 0:
                    _slb_sim = _aud_cos(audio_slb_latent.to(device),
                                        aud_out[:, :, :audio_overlap_af])
                    print(f"[CLSS S1]   audio SLB honored: {_slb_sim:.4f} (expect ≥0.97)")
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
                # audio boundary_sim — chunk-to-chunk continuity at the sample level
                if _s1_aud_prev_last is not None:
                    _aud_bnd = _aud_cos(_s1_aud_prev_last.to(device), new_aud[:, :, :1])
                    print(f"[CLSS S1]   audio_boundary_sim={_aud_bnd:.4f}")
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
                if _s1_audio_freq_ref is None:
                    _s1_audio_freq_ref = _freq_e
                    print(f"[CLSS S1]   audio freq_energy(ref)=[{' '.join(f'{e:.3f}' for e in _freq_e)}]")
                else:
                    _freq_ratio = [e / r if r > 1e-6 else 0.0 for e, r in zip(_freq_e, _s1_audio_freq_ref)]
                    print(
                        f"[CLSS S1]   audio freq_energy=[{' '.join(f'{e:.3f}' for e in _freq_e)}]"
                        f"  ratio=[{' '.join(f'{r:.2f}' for r in _freq_ratio)}]"
                    )
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
                # rescaled guidance should be taming.  Per-bin std ratios and DC drift
                # are kept below as log-only diagnostics.
                with torch.no_grad():
                    _cur_mean = new_aud.float().mean(dim=2, keepdim=True)  # [B, C_a, 1, freq]
                    _cur_std  = new_aud.float().std(dim=2, keepdim=True).clamp(min=1e-6)
                if _s1_audio_rms_ref is None:
                    _skip = min(16, new_aud.shape[2])
                    _ref_aud = new_aud[:, :, _skip:] if new_aud.shape[2] > _skip else new_aud
                    _s1_audio_rms_ref  = _ref_aud.float().pow(2).mean().sqrt().item()
                    _s1_audio_ref_mean = _cur_mean.cpu()   # DC reference for drift logging
                    _s1_audio_ref_std  = _ref_aud.float().std(dim=2, keepdim=True).clamp(min=1e-6).cpu()
                    print(f"[CLSS S1]   audio rms_ref={_s1_audio_rms_ref:.4f} (onset-excluded)")
                else:
                    _beta = 0.3
                    _raw_gain = _s1_audio_rms_ref / max(_aud_rms, 1e-6)
                    # SYMMETRIC gain (deviation from the reference's upward-only rule,
                    # deliberate): the reference has no audio SLB, so its only failure
                    # mode is decay.  Our SLB feeds the chunk tail forward, and the
                    # 7-chunk run showed runaway GROWTH (RMS 0.52→0.82, raw gains down
                    # to 0.76) that upward-only correction structurally cannot touch —
                    # the hot tail freezes into the next SLB and compounds.  β-blended
                    # and clamped to [0.87, 1.15] so it damps the loop without hard
                    # normalization.
                    _soft_gain = 1.0 + _beta * (_raw_gain - 1.0)
                    _soft_gain = min(max(_soft_gain, 0.87), 1.15)
                    if abs(_soft_gain - 1.0) > 0.005:
                        new_aud = (new_aud.float() * _soft_gain).to(aud_out.dtype)
                        print(f"[CLSS S1]   audio gain (scalar, symmetric): raw={_raw_gain:.4f} "
                              f"applied={_soft_gain:.4f}  "
                              f"rms {_aud_rms:.4f}→{new_aud.float().pow(2).mean().sqrt().item():.4f}")
                    else:
                        print(f"[CLSS S1]   audio gain: none (raw={_raw_gain:.4f} ≈ 1)")
                    # Diagnostics only — no per-bin correction is applied.
                    with torch.no_grad():
                        _ref_m = _s1_audio_ref_mean.to(device)
                        _ref_s = _s1_audio_ref_std.to(device)
                        _std_ratio = (_ref_s / _cur_std).mean(dim=-1).squeeze().tolist()      # [C_a]
                        _dc_drift  = (_cur_mean - _ref_m).mean(dim=-1).squeeze().tolist()     # [C_a]
                    print(
                        f"[CLSS S1]   audio diag: ch_std_ratio(ref/cur)="
                        f"[{' '.join(f'{v:.3f}' for v in _std_ratio)}]  "
                        f"ch_dc_drift=[{' '.join(f'{v:+.4f}' for v in _dc_drift)}]"
                    )
                if audio_overlap_af > 0:
                    ov = audio_overlap_af
                    # Audio SLB for next chunk: last ov frames of new_aud = the temporal
                    # period that will be the next chunk's video SLB time.
                    if new_aud.shape[2] >= ov:
                        audio_slb_latent = new_aud[:, :, -ov:].cpu()
                    else:
                        audio_slb_latent = new_aud.cpu()   # short chunk — use all
                    print(f"[CLSS S1]   audio SLB saved: {audio_slb_latent.shape[2]}f  "
                          f"mean={audio_slb_latent.float().mean():.4f}")
                    # ref_audio for next chunk: frames BEFORE the overlap period,
                    # taken from a rolling tail of accumulated output (reference
                    # pipeline.py:771-806).  The tail keeps the last 2×ov frames across
                    # chunk boundaries, so the reference window is always a FULL ov
                    # frames ending immediately before the overlap — even when the
                    # within-chunk pre-overlap region is shorter than ov.
                    _tail_cur = new_aud.cpu()
                    _s1_audio_tail = (
                        _tail_cur if _s1_audio_tail is None
                        else torch.cat([_s1_audio_tail, _tail_cur], dim=2)
                    )
                    if _s1_audio_tail.shape[2] > 2 * ov:
                        _s1_audio_tail = _s1_audio_tail[:, :, -2 * ov:]
                    _s1_audio_tail = _s1_audio_tail.clone()
                    _tail_lf = _s1_audio_tail.shape[2]
                    pre_ov_end = max(0, _tail_lf - ov)   # tail's last ov frames = next overlap
                    if pre_ov_end > 0:
                        _ref_start = max(0, pre_ov_end - ov)
                        audio_overlap_latent = _s1_audio_tail[:, :, _ref_start:pre_ov_end].clone()
                        print(f"[CLSS S1]   audio ref saved: {audio_overlap_latent.shape[2]}f "
                              f"(tail[{_ref_start}:{pre_ov_end}], tail_len={_tail_lf})  "
                              f"mean={audio_overlap_latent.float().mean():.4f}")
                    else:
                        audio_overlap_latent = None
                        print(f"[CLSS S1]   audio ref NOT saved: tail too short "
                              f"({_tail_lf}f ≤ {ov}f)")
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
            full_aud = _post_process_audio_latent(full_aud, audio_chunk_ends, label=" S1")
            print(f"[CLSS] Stage 1 full_aud post: shape={list(full_aud.shape)} "
                  f"mean={full_aud.float().mean():.4f} std={full_aud.float().std():.4f} "
                  f"min={full_aud.float().min():.4f} max={full_aud.float().max():.4f}")
            output_samples = comfy.nested_tensor.NestedTensor((full_vid, full_aud))
        else:
            output_samples = full_vid

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

        noise_vid = torch.randn_like(vid)                              # random baseline

        n_new   = vid.shape[2] - self._chunk_overlap                  # frames to fill
        src_end = min(self._pos + n_new, self._full.shape[2])
        src_n   = src_end - self._pos
        if src_n > 0:                                                  # slice consistent noise
            noise_vid[:, :, self._chunk_overlap:self._chunk_overlap + src_n] = \
                self._full[:, :, self._pos:src_end].to(vid.device)

        if is_av:
            aud = samples.unbind()[1]
            noise_aud = torch.randn_like(aud)
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
                "audio_mode": (["refine", "freeze"], {
                    "default": "refine",
                    "tooltip": "refine (default): re-noise Stage-1 audio to sigma_0 and refine it "
                               "through the distilled 3-step schedule alongside the video — matches "
                               "the official ti2vid_two_stages pipeline "
                               "(audio initial_latent = stage-1 audio, full denoise mask). "
                               "Continuity: previous chunk's refined audio tail as SLB (tau_c) + "
                               "one shared audio noise field across chunks.\n"
                               "freeze: previous behaviour — Stage-1 audio passed through unchanged "
                               "(mask=0).  Keep for A/B comparison."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "LTX-CLSS"

    @torch.inference_mode()
    def sample(self, guider, sampler, sigmas, noise, latent,
               clss_config: CLSSConfig, frames_per_chunk: int,
               image=None, vae=None, audio_mode: str = "refine", s2_overlap: int = 0):
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

        # Hard guard against a needless chunk boundary (the #1 source of the
        # ~6s morph): if the WHOLE video fits the token budget, refine it in one
        # chunk regardless of a manually-set frames_per_chunk.  ComfyUI keeps old
        # widget values across node-default changes, so a stale fpc (e.g. 31 from
        # an earlier experiment) would otherwise re-introduce the seam even though
        # the auto default is 0.  A boundary is only ever created when the video
        # genuinely does not fit — never as a leftover setting.
        _guard_budget = 42000
        if torch.cuda.is_available():
            _guard_budget = max(24000, int(42000 *
                (torch.cuda.get_device_properties(0).total_memory / 1024**3) / 15.6))
        if frames_per_chunk < T and T * H * W <= _guard_budget:
            print(f"[CLSS] guard: overriding frames_per_chunk={frames_per_chunk}→{T} "
                  f"— whole video ({T * H * W} tokens) fits budget ({_guard_budget}); "
                  f"a single chunk avoids the Stage-2 boundary seam/morph.")
            frames_per_chunk = T

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
        full_noise_aud: torch.Tensor | None = None
        if has_aud:
            if audio_mode == "refine":
                # One coherent audio noise field for all chunks (same rationale as video).
                _g = torch.Generator(device="cpu").manual_seed(noise_seed + 1)
                full_noise_aud = torch.randn(full_aud.shape, generator=_g,
                                             dtype=full_aud.dtype)
                print(f"[CLSS] Stage 2: audio REFINED through distilled schedule "
                      f"(official ti2vid parity: S1 audio as initial latent, mask=1, "
                      f"audio SLB tau_c={clss_config.tau_c}).")
            else:
                print(f"[CLSS] Stage 2: audio frozen (mask=0) — Stage 1 audio passed through unchanged.")
        print(f"[CLSS] Stage 2: CLSS AdaIN/shrinkage corrections DISABLED — "
              f"Stage 2 is closed-loop refinement, not open-loop generation.")

        # Stage 2 SLB state (video) — no CLSSState, no AdaIN corrections.
        overlap_latent: torch.Tensor | None = None
        # Stage 2 SLB state (audio, refine mode): previous chunk's REFINED audio tail.
        s2_audio_overlap: torch.Tensor | None = None

        acc_video: list[torch.Tensor] = []
        acc_audio: list[torch.Tensor] = []
        audio_chunk_ends_s2: list[int] = []

        # §item-1,2,6: coherence tracking for Stage 2
        _s2_prev_last:     torch.Tensor | None = None  # [B, C_v, H, W] last new frame of prev S2 chunk
        _s2_id_ref:        torch.Tensor | None = None  # [B, C_v] identity ref from S2 chunk-1
        _s2_aud_prev_last: torch.Tensor | None = None  # [B, C_a, 1, freq] last new audio frame of prev S2 chunk

        def _astats(t: torch.Tensor, label: str) -> str:
            t = t.float()
            return (f"{label}: shape={list(t.shape)} "
                    f"mean={t.mean():.4f} std={t.std():.4f} "
                    f"min={t.min():.4f} max={t.max():.4f} "
                    f"nan={t.isnan().any().item()} inf={t.isinf().any().item()}")

        lf_to_sec = 8 / 25  # latent frames → seconds @ 25 fps

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
            # Audio SLB frames proportional to video SLB (0 for first chunk)
            a_ov          = 0 if is_first else a_ov_af

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
            # audio_mode="refine" (default, official parity): the reference
            # ti2vid_two_stages passes stage-1 audio as initial_latent into stage 2
            # and refines it through the SAME distilled 3-step schedule at cfg=1
            # (ModalitySpec(..., initial_latent=audio_state.latent)).  Stage-1 audio
            # is a DRAFT the model expects to finalize — freezing it skips the pass
            # that polishes audio and re-aligns it with the refined video.
            #   New-region seeding mirrors video: lat_aud = S1 audio, mask=1 →
            #   flow-matching renoise to σ0, denoised alongside video.
            #   Boundary continuity mirrors video: previous chunk's REFINED audio
            #   tail is placed at the overlap region with mask=tau_c (audio SLB),
            #   and new-region noise is sliced from one pre-generated full-audio
            #   field (see _SlicedNoise) — fixes the historical mask=1.0 failure,
            #   which used zero seeding + independent per-chunk noise.
            #   The historical mask=0.3 failure was a per-token sigma MISMATCH on
            #   the whole chunk; tau_c on only the overlap region matches how video
            #   SLB has worked all along.
            # audio_mode="freeze": previous behaviour (mask=0 passthrough) for A/B.
            if has_aud:
                a_new_start = a_pos
                a_new_end   = min(round(end_pos * T_a / T), T_a)
                a_ov        = 0 if (is_first or audio_mode == "freeze") else min(a_ov_af, a_new_start)
                if audio_mode == "freeze":
                    chunk_af = a_new_end - a_new_start
                    lat_aud  = full_aud[:, :, a_new_start:a_new_end].to(device)
                    mask_aud = torch.zeros(B_a, C_a, chunk_af, freq, device=device)
                    print(f"[CLSS S2]   {_astats(lat_aud, f's1_aud[{a_new_start}:{a_new_end}]')}")
                    print(f"[CLSS S2]   aud_in: af=[{a_new_start}:{a_new_end}] chunk_af={chunk_af} "
                          f"acc_a_pos={a_pos}  mask=0 (frozen passthrough)")
                else:
                    chunk_af = a_ov + (a_new_end - a_new_start)
                    lat_aud  = torch.zeros(B_a, C_a, chunk_af, freq, device=device)
                    mask_aud = torch.ones(B_a, C_a, chunk_af, freq, device=device)
                    # New region: S1 audio seed at mask=1 (renoised to σ0, like video)
                    lat_aud[:, :, a_ov:] = full_aud[:, :, a_new_start:a_new_end].to(device)
                    # Overlap region: previous REFINED audio tail at mask=tau_c
                    if a_ov > 0 and s2_audio_overlap is not None:
                        slb_n = min(a_ov, s2_audio_overlap.shape[2])
                        lat_aud[:, :, a_ov - slb_n:a_ov]  = s2_audio_overlap[:, :, -slb_n:].to(device)
                        mask_aud[:, :, a_ov - slb_n:a_ov] = clss_config.tau_c
                    print(f"[CLSS S2]   aud_in(refine): af=[{a_new_start}:{a_new_end}] "
                          f"chunk_af={chunk_af} (slb={a_ov}f tau_c + new={a_new_end - a_new_start}f) "
                          f"seed_mean={lat_aud[:, :, a_ov:].float().mean():.4f} "
                          f"mask_mean={mask_aud.mean():.3f}")

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
            print()

            if aud_out is not None:
                # Drop the audio SLB region (refine mode; a_ov=0 in freeze/first chunk)
                new_aud = aud_out[:, :, a_ov:]
                s1_chunk_ref = full_aud[:, :, a_new_start:a_new_end].to(device)
                _s1_sim = _aud_cos(s1_chunk_ref, new_aud)
                if audio_mode == "freeze":
                    print(f"[CLSS S2]   {_astats(new_aud, 'aud_out(frozen)')}"
                          f"  frozen_verify_sim={_s1_sim:.4f}")
                else:
                    # refine: sim vs S1 measures how much the distilled pass changed
                    # the audio.  ~0.6-0.9 expected (real refinement); ~1.0 means the
                    # renoise did nothing; ~0 means S1 seeding isn't reaching the model.
                    # Chunk-1 onset treatment (mirrors Stage 1): refined chunk-1 audio
                    # regenerates from near-scratch at sigma_0 and shows the same t≈0
                    # transient (observed min=-6.7 vs ±3 elsewhere).  Fade-in + per-
                    # channel soft-clamp before the tail is saved or accumulated.
                    if is_first:
                        _n_fade = min(4, new_aud.shape[2])
                        for _i in range(_n_fade):
                            _a = 0.25 + 0.75 * (_i / max(1, _n_fade - 1))
                            new_aud[:, :, _i] = new_aud[:, :, _i] * _a
                        with torch.no_grad():
                            _lim = (4.0 * new_aud.float().std(dim=(0, 2, 3), keepdim=False)
                                    ).view(1, -1, 1, 1).to(new_aud.dtype)
                        new_aud = torch.tanh(new_aud / _lim) * _lim
                        print(f"[CLSS S2]   chunk-1 refined-audio fade+clamp applied  "
                              f"new_abs_max={new_aud.float().abs().max():.4f}")
                    print(f"[CLSS S2]   {_astats(new_aud, 'aud_out(refined)')}"
                          f"  s1_seed_sim={_s1_sim:.4f}")
                    if a_ov > 0 and s2_audio_overlap is not None:
                        _slb_n  = min(a_ov, s2_audio_overlap.shape[2])
                        _slb_ok = _aud_cos(s2_audio_overlap[:, :, -_slb_n:].to(device),
                                           aud_out[:, :, a_ov - _slb_n:a_ov])
                        print(f"[CLSS S2]   audio SLB honored: {_slb_ok:.4f} (expect ≥0.97)")
                    # Save refined tail as next chunk's audio SLB
                    _tail_n = min(a_ov_af, new_aud.shape[2])
                    s2_audio_overlap = new_aud[:, :, -_tail_n:].clone().cpu()

                if _s2_aud_prev_last is not None:
                    _aud_bnd = _aud_cos(_s2_aud_prev_last.to(device), new_aud[:, :, :1])
                    print(f"[CLSS S2]   audio_boundary_sim={_aud_bnd:.4f}")
                else:
                    print(f"[CLSS S2]   audio_boundary_sim=N/A(first)")
                _s2_aud_prev_last = new_aud[:, :, -1:].cpu()

                acc_audio.append(new_aud.cpu())
                a_pos += new_aud.shape[2]
                audio_chunk_ends_s2.append(a_pos)
                print(f"[CLSS S2]   acc_audio total={a_pos}/{T_a} frames "
                      f"({a_pos / T_a * 100:.1f}%  "
                      f"≈{a_pos / T_a * T * lf_to_sec:.2f}s/{T * lf_to_sec:.2f}s)")

            pos = end_pos

        # ── Assemble full refined output ─────────────────────────────────────
        full_refined_vid = torch.cat(acc_video, dim=2)
        print(f"[CLSS S2] {_astats(full_refined_vid, 'full_refined_vid')}")
        if acc_audio:
            full_refined_aud = torch.cat(acc_audio, dim=2)
            if audio_mode == "refine":
                # Independently-refined chunks → smooth boundaries (energy already
                # matched via S1 seeding + shared noise field; smoothing only).
                full_refined_aud = _post_process_audio_latent(
                    full_refined_aud, audio_chunk_ends_s2,
                    energy_beta=0.0, label=" S2",
                )
                print(f"[CLSS S2] {_astats(full_refined_aud, 'full_refined_aud(refined)')}")
            else:
                # Frozen passthrough: Stage 2 audio = Stage 1 audio (already normalized).
                # Skip _post_process_audio_latent to avoid double-normalizing.
                print(f"[CLSS S2] {_astats(full_refined_aud, 'full_refined_aud(frozen=S1)')}")
            output = comfy.nested_tensor.NestedTensor((full_refined_vid, full_refined_aud))
        elif full_aud is not None:
            print(f"[CLSS S2] no acc_audio — falling back to Stage 1 audio passthrough")
            output = comfy.nested_tensor.NestedTensor((full_refined_vid, full_aud.cpu()))
        else:
            output = full_refined_vid

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
                # Non-AV fallback: standard CFG (identity, same as default)
                return x - (uncond_d + scale * (cond_d - uncond_d))

        new_guider.model_options["sampler_cfg_function"] = _av_cfg_fn
        new_guider.audio_cfg = _audio_cfg   # readable by downstream nodes for logging
        print(f"[CLSS] AVGuider patched: video_cfg={vid_cfg:.2f}  audio_cfg={_audio_cfg:.2f}  "
              f"rescale={_rescale:.2f}")
        return (new_guider,)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class _GuiderCLSSAV(comfy.samplers.CFGGuider):
    """CFGGuider subclass implementing the reference MultiModalGuider for joint AV.

    Per denoising step (reference denoisers.py::_guided_denoise + guiders.py):
      1. cond + uncond passes (standard, batched by calc_cond_batch)
      2. "mod" pass: positive context with BOTH cross-modal attentions skipped —
         ComfyUI's BasicAVTransformerBlock natively honours
         transformer_options["a2v_cross_attn"/"v2a_cross_attn"] (av_model.py:267-268),
         which is exactly the reference's SKIP_A2V_CROSS_ATTN + SKIP_V2A_CROSS_ATTN
         perturbation.
      3. Per-modality combine:
           pred = cond + (cfg−1)·(cond−uncond) + (modality−1)·(cond−mod)
      4. Per-modality CFG rescale: pred *= r·(cond.std/pred.std) + (1−r)

    The modality term is the audio-critical piece: it amplifies the component of
    the audio prediction that COMES FROM THE VIDEO (and vice versa).  Without it,
    audio↔video coupling relies solely on joint attention, which in the 4-bit
    model degrades to generic text-conditioned ambience (drone).  The working
    standalone generate_clss.py runs with modality_scale=3.0 by default.

    Note: STG (skip self-attn in block 28) is NOT implemented — ComfyUI's
    av_model.py has no per-block self-attn skip plumbing; adding it would require
    patching attention internals.  Modality guidance alone is the documented
    audio-quality lever.

    Cost: +1 transformer pass per step when modality_scale != 1 (~+50%% step time).
    """

    _video_cfg      = 4.0
    _audio_cfg      = 7.0
    _modality_scale = 3.0
    _rescale        = 0.7
    _logged         = False

    def set_av_params(self, video_cfg, audio_cfg, modality_scale, rescale):
        self._video_cfg      = video_cfg
        self._audio_cfg      = audio_cfg
        self._modality_scale = modality_scale
        self._rescale        = rescale
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

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        positive = self.conds.get("positive", None)
        negative = self.conds.get("negative", None)

        # Non-AV latents or missing negative → standard CFG path.
        if not isinstance(x, comfy.nested_tensor.NestedTensor) or negative is None:
            return super().predict_noise(x, timestep, model_options, seed)

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

        vid_c, aud_c = out_cond.unbind()
        vid_u, aud_u = out_uncond.unbind()

        pred_v = vid_c + (self._video_cfg - 1.0) * (vid_c - vid_u)
        pred_a = aud_c + (self._audio_cfg - 1.0) * (aud_c - aud_u)

        if out_mod is not None:
            vid_m, aud_m = out_mod.unbind()
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
            print(f"[CLSS AVGuiderV2] active: video_cfg={self._video_cfg:.1f}  "
                  f"audio_cfg={self._audio_cfg:.1f}  modality={self._modality_scale:.1f}  "
                  f"rescale={self._rescale:.2f}  passes/step={2 if out_mod is None else 3}")

        return comfy.nested_tensor.NestedTensor((pred_v, pred_a))


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
                "audio_cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 30.0, "step": 0.5}),
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
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = "LTX-CLSS"

    def get_guider(self, model, positive, negative, video_cfg, audio_cfg,
                   modality_scale, rescale):
        guider = _GuiderCLSSAV(model)
        guider.set_conds(positive, negative)
        guider.set_av_params(video_cfg, audio_cfg, modality_scale, rescale)
        print(f"[CLSS] AVGuiderV2 built: video_cfg={video_cfg:.2f}  audio_cfg={audio_cfg:.2f}  "
              f"modality={modality_scale:.2f}  rescale={rescale:.2f}")
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
