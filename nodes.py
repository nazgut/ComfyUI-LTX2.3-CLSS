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
    raw = []
    for c in converted:
        tensor = c.get("cross_attn", None)
        d = {k: v for k, v in c.items() if k not in ("cross_attn", "uuid")}
        raw.append([tensor, d])
    return raw
_SCENE_BLEND_W = 0.5
def _blend_scene_cond(prev: dict, new: dict, w: float = _SCENE_BLEND_W) -> dict:
    """Weighted cross-attn embedding blend for scene-transition chunks.

    The scene hand-off swaps the whole window's text conditioning at a chunk
    boundary, which reads as a hard cut: the frozen SLB overlap is the only
    visual bridge, and every new frame is denoised under the incoming scene
    alone at full CFG.  Blending the outgoing scene's enhanced embedding into
    the boundary chunks (25%-incoming on the outgoing block's last chunk,
    75%-incoming on the incoming block's first) lets the tail action finish
    on screen while the new scene's content takes over.  Enhanced scenes
    tokenize to different lengths, so the shorter sequence is edge-padded
    (EOS-repeat) before blending; structurally incompatible entries fall back
    to the new scene unblended.  ``w`` is the incoming-scene weight.
    """
    pe, ne = prev.get("cross_attn"), new.get("cross_attn")
    if pe is None or ne is None or pe.shape[-1] != ne.shape[-1]:
        return new
    t = max(pe.shape[1], ne.shape[1])
    if pe.shape[1] < t:
        pe = torch.cat([pe, pe[:, -1:].expand(-1, t - pe.shape[1], -1)], dim=1)
    if ne.shape[1] < t:
        ne = torch.cat([ne, ne[:, -1:].expand(-1, t - ne.shape[1], -1)], dim=1)
    blended = dict(new)
    blended["cross_attn"] = ((1.0 - w) * pe.float() + w * ne.float()).to(ne.dtype)
    return blended
def _frame_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    with torch.no_grad():
        fa = F.normalize(a.float().reshape(a.shape[0], a.shape[1], -1).mean(-1), dim=1)
        fb = F.normalize(b.float().reshape(b.shape[0], b.shape[1], -1).mean(-1), dim=1)
        return (fa * fb).sum(dim=1).mean().item()
def _aud_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    with torch.no_grad():
        min_t = min(a.shape[2], b.shape[2])
        fa = F.normalize(a[:, :, :min_t].float().reshape(a.shape[0], -1), dim=1)
        fb = F.normalize(b[:, :, :min_t].float().reshape(b.shape[0], -1), dim=1)
        return (fa * fb).sum(dim=1).mean().item()
def _aud_within_chunk_sims(new_aud: torch.Tensor, n_seg: int = 3) -> list[float]:
    T = new_aud.shape[2]
    if T < n_seg * 2:
        return []
    seg_len = T // n_seg
    sims: list[float] = []
    with torch.no_grad():
        for i in range(n_seg - 1):
            s1 = new_aud[:, :, i * seg_len:(i + 1) * seg_len].float().mean(dim=2)
            s2 = new_aud[:, :, (i + 1) * seg_len:(i + 2) * seg_len].float().mean(dim=2)
            f1 = F.normalize(s1.reshape(new_aud.shape[0], -1), dim=1)
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
    if not chunk_ends:
        return audio_lat
    audio_lat = audio_lat.clone()
    T = audio_lat.shape[2]
    boundaries = [0] + list(chunk_ends)
    n = len(chunk_ends)
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
class CLSSConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tau_c":   ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5,  "step": 0.01,
                                      "tooltip": "§2.1 calibrated context re-noising: noise level applied to the SLB overlap fed as context to each chunk. 0 = fully frozen overlap (maximal continuity, maximal drift accumulation); higher = more distributional repair at the cost of softer motion lock. The per-chunk schedule rises from this base toward a 0.10 ceiling with a 5-chunk half-life.",
                                      }),
                "beta":    ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0,  "step": 0.05,
                                      "tooltip": "§2.3 drift correction: blend factor of the EMA-tracked per-channel AdaIN renormalisation applied to every new chunk. 0 = no correction, 1 = full replacement with the EMA reference statistics. The EMA reference resets at every scene change — the first chunk of a new scene is uncorrected and re-anchors it.",
                                      }),
                "overlap": ("INT",   {"default": 8,    "min": 1,   "max": 32,
                                      "tooltip": "SLB size in latent frames shared between consecutive chunks (8 latent frames ≈ 57 pixel frames ≈ 2.4 s at 24 fps) — the hard temporal context the model sees from the previous chunk. Auto-clamped down at runtime so overlap+new stays under the 19.5 s RoPE wall.",
                                      }),
            },
            "optional": {
                "noise_temporal_corr": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.8, "step": 0.05,
                                      "tooltip": "Temporally-correlated video noise: mixes a run-constant shared frame into every noise frame, n_t = sqrt(1-a)·eps_t + sqrt(a)·eps_shared, keeping each frame's marginal exactly N(0,1) while raising frame-to-frame noise correlation (FreeNoise/PYoCo family). Targets the measured ~4 s layout oscillation. 0 = off.",
                                      }),
            },
        }
    RETURN_TYPES = ("CLSS_CONFIG",)
    RETURN_NAMES = ("clss_config",)
    FUNCTION = "build"
    CATEGORY = "LTX-CLSS"
    def build(self, tau_c, beta, overlap, noise_temporal_corr=0.3):
        return (CLSSConfig(
            tau_c=tau_c,
            beta=beta,
            ema_lambda=0.10,
            ema_sigma_max_drift=0.05,
            anchor_force_every=0,
            overlap_latent_frames=overlap,
            adain_max_amplification=1.2,
            noise_temporal_corr=noise_temporal_corr,
        ),)
class CLSSScenePrompts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip":       ("CLIP",   {"tooltip": "CLIP (Gemma) text encoder, used twice per scene: first to LLM-enhance the raw scene text under the LTX2 system-prompt template, then to encode the enhanced text into conditioning."}),
                "prompts":    ("STRING", {"multiline": True, "dynamicPrompts": False,
                                          "default": "Scene 1 description\n---\nScene 2 description",
                                          "tooltip": "One scene per block, separated by a line containing only '---'. Each scene is LLM-enhanced, then encoded as its own CONDITIONING entry; with N entries the sampler assigns one scene per chunk proportionally across num_chunks.",
                                          }),
                "max_length": ("INT",    {"default": 512, "min": 1, "max": 32768,
                                          "tooltip": "Maximum number of tokens the LLM may generate when enhancing each scene prompt.",
                                          }),
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
            formatted = (
                f"<start_of_turn>system\n{LTX2_T2V_SYSTEM_PROMPT.strip()}<end_of_turn>\n"
                f"<start_of_turn>user\nUser Raw Input Prompt: {scene}.<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            tokens = clip.tokenize(formatted, skip_template=True, min_length=1)
            generated_ids = clip.generate(tokens, do_sample=False, max_length=max_length)
            enhanced = clip.decode(generated_ids)
            scene_cond = clip.encode_from_tokens_scheduled(clip.tokenize(enhanced))
            flat_conditioning.extend(scene_cond)
        return (flat_conditioning,)
def _tau_c_eff(base: float, ceiling: float, chunk_idx: int, half_life: float = 5.0) -> float:
    if base <= 0.0:
        return 0.0
    decay = 0.5 ** (chunk_idx / half_life)
    return ceiling - (ceiling - base) * decay
_VIDEO_TAU_C_CEILING = 0.10
_AUDIO_TAU_C_BASE_MULT = 3.0
_AUDIO_TAU_C_CEILING  = 0.35
_KNOWN_GOOD_SHIFT = 1.844
_ROPE_WALL_S = 20.0
_ROPE_WALL_MARGIN_S = 0.5
_MIN_OVERLAP_LF = 2
class CLSSStreamingSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider":      ("GUIDER",      {"tooltip": "GUIDER from CLSSAVGuiderV2 (or a guider patched by CLSSAVGuider). When its positive conditioning holds N scene entries, one scene is unpacked per chunk proportionally across num_chunks."}),
                "sampler":     ("SAMPLER",     {"tooltip": "SAMPLER for the per-chunk denoise. When audio_shift_mult puts audio on its own sigma schedule, the sampler is wrapped so video and audio step on separate shifts."}),
                "sigmas":      ("SIGMAS",      {"tooltip": "SIGMAS schedule (e.g. LTXVScheduler — its shift scales with the connected latent's token count). Video follows it directly; the audio schedule may be re-derived from it (see audio_shift_mult)."}),
                "noise":       ("NOISE",       {"tooltip": "NOISE source. Its seed drives the run-constant full-length noise tensors that each chunk's initial noise is sliced from."}),
                "latent":      ("LATENT",      {"tooltip": "Per-chunk AV latent template (EmptyLTXVLatentVideo + audio latent via LTXVConcatAVLatent). Its frame count sets the per-chunk length; total length = num_chunks × chunk."}),
                "clss_config": ("CLSS_CONFIG", {"tooltip": "CLSS_CONFIG from the CLSS Config node (tau_c, beta, overlap, noise_temporal_corr)."}),
                "num_chunks":  ("INT",         {"default": 10, "min": 1, "max": 500,
                                                "tooltip": "Number of streaming chunks; total video length = num_chunks × chunk length. A chunk whose window would cross the 19.5 s RoPE wall is auto-split into uniform sub-chunks. With scene_handoff=transition_chunk every scene block needs ≥ 2 chunks, i.e. num_chunks ≥ 2×scenes.",
                                                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional i2v guide image; VAE-encoded and pinned as the first frame of chunk 0. Requires vae."}),
                "vae":   ("VAE",   {"tooltip": "Video VAE, only needed together with image for the i2v guide encode."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Frames per second of the output. Sets the pixel↔latent↔audio time mapping used for window sizing, the RoPE-wall check, and the audio tail pin (audio_slb_tau_mult < 0).",
                    }),
                "detail_anchor": (["on", "off"], {
                    "default": "on",
                    "tooltip": "Two-band spatial detail anchor: each chunk's low/high-frequency band energies are re-scaled toward the scene's first-chunk reference (gains clamped to [0.90, 1.10] low / [0.90, 1.12] high) to fight the long-run detail fade. Off = uncorrected.",
                    }),
                "video_slb_tau_mult": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.25,
                    "tooltip": "Scales the video overlap re-noise: effective tau_c × this, still rising toward the 0.10 ceiling with the 5-chunk half-life. 0 = frozen clean seam (no re-noising of the video overlap).",
                    }),
                # < 0: the audio overlap regenerates freely (best musical continuity),
                # but the last |value| SECONDS of the previous tail stay frozen at the
                # end of the overlap — without that pin the vocal phrase restarts at
                # every chunk boundary and words are cut mid-phoneme.
                # 0 = SLB placed fully frozen, > 0 = SLB re-noised with tau_c*mult.
                "audio_slb_tau_mult": ("FLOAT", {
                    "default": 0.0, "min": -4.0, "max": 6.0, "step": 0.5,
                    "tooltip": "Audio SLB handling. 0 = SLB placed fully frozen (no tau_c on audio). > 0 = SLB re-noised with tau_c×mult (ceiling 0.35). < 0 = overlap regenerates freely (best musical continuity) but the last |value| SECONDS of the previous tail stay pinned frozen at the end of the overlap, keeping the vocal phrase glued across the seam instead of restarting mid-phoneme.",
                    }),
                "audio_shift_mult": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Audio sigma-shift control. 0.0 = AUTO: audio follows a schedule re-shifted to min(connected video shift, 1.844) — the ear-validated target. Manual values are raw multipliers on the video shift; 1.0 = off (audio shares the video schedule). At long chunks the video wants a high shift while the audio wants 1.844 — no shared schedule serves both.",
                    }),
                "auto_seam_pin": (["off", "on"], {
                    "default": "off",
                    "tooltip": "Event-triggered frozen seam: when the coarse-layout similarity curve jumps within a chunk (frame-to-frame drop > 0.25 or minimum < 0.05), the next chunk's video overlap is placed fully frozen (tau_c = 0 for that chunk).",
                    }),
                # How the text conditioning changes at a scene boundary:
                # "transition_chunk" — two-step crossfade straddling the boundary: the
                #   outgoing scene block's last chunk is guided by a 25%-incoming blend,
                #   the incoming scene block's first chunk by 75%-incoming; needs every
                #   scene block >= 2 chunks, i.e. num_chunks >= 2*scenes (3 scenes -> 6).
                # "blend" — the first chunk of each new scene gets a single 50/50 blend.
                # "hard"  — plain text swap (pre-crossfade baseline).
                "scene_handoff": (["transition_chunk", "blend", "hard"], {
                    "default": "transition_chunk",
                    "tooltip": "How text conditioning changes at a scene boundary. transition_chunk: two-step crossfade straddling the boundary — the outgoing scene's last chunk is guided by a 25%-incoming embedding blend, the incoming scene's first chunk by 75%-incoming (a single 50/50 chunk between far-apart scenes measured off-manifold and poisoned the next scene's SLB); needs every scene block ≥ 2 chunks. blend: single 50/50 blend on the first chunk of the new scene. hard: plain text swap.",
                    }),
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
        audio_slb_tau_mult: float = 0.0,
        audio_shift_mult: float = 0.0,
        auto_seam_pin: str = "off",
        scene_handoff: str = "transition_chunk",
    ):
        import dataclasses
        import math
        _n_steps = max(1, len(sigmas) - 1)
        _n_hi = int(sum(1 for s in sigmas[:-1] if float(s) >= 0.9))
        _auto_pin = (auto_seam_pin == "on")
        _pin_next = False
        if clss_config.anchor_force_every <= 0:
            _auto_anchor = max(2, min(5, math.ceil(num_chunks / 4)))
            clss_config = dataclasses.replace(clss_config, anchor_force_every=_auto_anchor)
        samples = latent["samples"]
        is_av = isinstance(samples, comfy.nested_tensor.NestedTensor)
        if is_av:
            vid_tmpl, aud_tmpl = samples.unbind()
        else:
            vid_tmpl = samples
            aud_tmpl = None
        B, C_v, new_lf, H, W = vid_tmpl.shape
        overlap_lf = clss_config.overlap_latent_frames
        device = vid_tmpl.device
        img_guide_latent: torch.Tensor | None = None
        if image is not None and vae is not None:
            i2v_scale_factors = vae.downscale_index_formula
            _, img_guide_latent = LTXVAddGuide.encode(vae, W, H, image[:1], i2v_scale_factors)
        if aud_tmpl is not None:
            B_a, C_a, new_af, freq = aud_tmpl.shape
            _first_px  = (new_lf - 1) * 8 + 1
            if overlap_lf > new_lf:
                overlap_lf = new_lf
            _af_per_px = new_af / _first_px if _first_px > 0 else 0.0
            audio_overlap_af = round(overlap_lf * 8 * _af_per_px)
            new_af_cont      = round(new_lf * 8 * _af_per_px)
            _ref_len_af = audio_overlap_af
            _win_af = (new_af if num_chunks == 1
                       else audio_overlap_af + max(new_af, new_af_cont))
            _win_s = _win_af / (fps * _af_per_px) if _af_per_px > 0 else 0.0
        else:
            B_a = C_a = new_af = freq = audio_overlap_af = new_af_cont = _ref_len_af = 0
            _win_af = 0
            _win_s = 0.0
        _max_win_lf = max(1, int((_ROPE_WALL_S - _ROPE_WALL_MARGIN_S) * fps / 8))
        _eff_overlap = overlap_lf
        _eff_num_chunks = num_chunks
        _split_into = 1
        _win_v_s = (new_lf if num_chunks == 1 else overlap_lf + new_lf) * 8 / fps
        _target_s = _ROPE_WALL_S - _ROPE_WALL_MARGIN_S
        _ov_orig = overlap_lf
        if max(_win_v_s, _win_s) > _target_s:
            _win_orig_s = max(_win_v_s, _win_s)
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
                _eff_overlap = min(overlap_lf, max(_MIN_OVERLAP_LF, _max_win_lf // 5))
                _max_new = max(1, _max_win_lf - _eff_overlap)
                _split_into = max(2, math.ceil(new_lf / _max_new))
                _eff_num_chunks = num_chunks * _split_into
                if aud_tmpl is not None:
                    audio_overlap_af = round(_eff_overlap * 8 * _af_per_px)
                    _ref_len_af = audio_overlap_af
            if _eff_num_chunks != num_chunks:
                _auto_anchor = max(2, min(5, math.ceil(_eff_num_chunks / 4)))
                clss_config = dataclasses.replace(clss_config, anchor_force_every=_auto_anchor)
        if _eff_overlap != overlap_lf:
            clss_config = dataclasses.replace(clss_config, overlap_latent_frames=_eff_overlap)
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
        pos_conds = guider.original_conds.get("positive", [])
        num_scenes = len(pos_conds)
        # Scene hand-off plan (_cond_plan), one entry per chunk:
        #   int               → chunk guided by that scene's prompt alone
        #   (int, int, float) → crossfade chunk guided by the embedding blend of
        #                       (outgoing, incoming) scene with incoming weight w
        #                       (_blend_scene_cond)
        # "transition_chunk": TWO-STEP crossfade straddling each boundary — the
        # outgoing scene's last chunk gets w=0.25 (mostly outgoing), the incoming
        # scene's first chunk gets w=0.75 (mostly incoming).  A single 50/50 chunk
        # between far-apart scenes is off-manifold guidance: measured live, the
        # 50/50 transition chunk drifted to anchor-sim 0.24 and poisoned the next
        # scene's SLB.  A scene block needs >=2 chunks to host its half of the
        # crossfade; 1-chunk blocks fall through to hard swaps.
        _scene_of = [min(int(_i * num_scenes / _eff_num_chunks), num_scenes - 1)
                     if num_scenes > 1 else 0
                     for _i in range(_eff_num_chunks)]
        _cond_plan: list = list(_scene_of)
        if num_scenes > 1 and scene_handoff != "hard":
            for _i in range(_eff_num_chunks):
                _s = _scene_of[_i]
                _prv = _scene_of[_i - 1] if _i > 0 else None
                _nxt = _scene_of[_i + 1] if _i + 1 < _eff_num_chunks else None
                if scene_handoff == "blend":
                    if _prv is not None and _s != _prv:
                        _cond_plan[_i] = (_prv, _s, 0.5)
                elif _nxt is not None and _nxt != _s and _scene_of.count(_s) >= 2:
                    _cond_plan[_i] = (_s, _nxt, 0.25)
                elif _prv is not None and _prv != _s and _scene_of.count(_s) >= 2:
                    _cond_plan[_i] = (_prv, _s, 0.75)
            if (scene_handoff == "transition_chunk"
                    and not any(isinstance(_e, tuple) for _e in _cond_plan)):
                print(f"[CLSS] scene_handoff=transition_chunk but every scene has a "
                      f"single chunk ({num_scenes} scenes / {_eff_num_chunks} chunks) — "
                      f"no crossfade inserted; use num_chunks >= 2*scenes "
                      f"(e.g. 6 for 3 scenes).")
        _cfg_val = getattr(guider, "cfg", getattr(guider, "cfg_scale", "unknown"))
        _aud_cfg_val = getattr(guider, "audio_cfg", None)
        _cfg_str = (f"video_cfg={_cfg_val} audio_cfg={_aud_cfg_val} (split)"
                    if _aud_cfg_val is not None else f"guider_cfg={_cfg_val} (shared, no split)")
        _corrections = {
            "renoise": clss_config.tau_c > 0,
            "adain":   clss_config.beta > 0,
            "anchor":  clss_config.anchor_max_size > 0,
        }
        _seed = getattr(noise, "seed", "unknown")
        clss_state = CLSSState(clss_config)
        acc_video: list[torch.Tensor] = []
        acc_audio: list[torch.Tensor] = []
        audio_chunk_ends: list[int] = []
        audio_slb_latent:     torch.Tensor | None = None
        audio_overlap_latent: torch.Tensor | None = None
        _s1_prev_last:       torch.Tensor | None = None
        _s1_vid_std_ref:     float | None = None
        _prev_scene_idx:     int | None = None
        _s1_band_ref:        tuple[float, float] | None = None
        _origin_ref:         torch.Tensor | None = None
        _origin_layout:      torch.Tensor | None = None
        _origin_track:       list = []
        _layout_track:       list = []
        _layout_argmin_track: list = []
        _aud_peak_track:     list = []
        _prev_aud_env:       torch.Tensor | None = None
        _trend = {
            "vid_std":   [],
            "vid_ident": [],
            "vid_intra": [],
            "vid_bnd":   [],
            "vid_hf":    [],
            "vid_origin": [],
            "aud_env":   [],
            "aud_rms":   [],
            "aud_bnd":   [],
            "aud_slb":   [],
            "aud_wc":    [],
            "aud_hf":    [],
        }
        _s1_aud_prev_last:   torch.Tensor | None = None
        _s1_audio_freq_ref:  list[float]  | None = None
        _s1_audio_tail:      torch.Tensor | None = None
        _noise_seed_s1 = getattr(noise, "seed", 0)
        _noise_tmpl_s1 = torch.zeros(B, C_v, num_chunks * new_lf, H, W, device=device)
        _full_noise_vid_s1: torch.Tensor = noise.generate_noise({"samples": _noise_tmpl_s1})
        del _noise_tmpl_s1
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
        _full_noise_aud_s1: torch.Tensor | None = None
        _s1_a_noise_pos = 0
        if aud_tmpl is not None:
            _aud_seed_s1 = (int(_noise_seed_s1) + 1) % (2 ** 63)
            _g_aud_s1 = torch.Generator(device="cpu").manual_seed(_aud_seed_s1)
            _total_af_s1 = new_af + (num_chunks - 1) * new_af_cont
            _full_noise_aud_s1 = torch.randn(
                B_a, C_a, _total_af_s1, freq, generator=_g_aud_s1, dtype=aud_tmpl.dtype)
        _base_pts = None
        _base_mod = getattr(getattr(guider, "model_patcher", None), "model", None)
        _a_sig = None
        _remap_fn = None
        _mult = float(audio_shift_mult)
        if (_mult <= 0.0 and aud_tmpl is not None
                and sigmas is not None and len(sigmas) >= 3):
            _term_auto = max(float(sigmas[-2]), 1e-3)
            _s_v_auto, _ = _estimate_video_sigma_shift(sigmas.double(), terminal=_term_auto)
            _mult = 1.0
            if _s_v_auto > _KNOWN_GOOD_SHIFT + 1e-6:
                sigmas = _shifted_shared_schedule(sigmas, _KNOWN_GOOD_SHIFT, terminal=_term_auto)
                _n_lo = int(sum(1 for s in sigmas[:-1] if float(s) < 0.9))
            else:
                _n_lo = int(sum(1 for s in sigmas[:-1] if float(s) < 0.9))
        _s_a_eff = _mult
        if (_mult != 1.0 and aud_tmpl is not None
                and sigmas is not None and len(sigmas) >= 3):
            _a_sig, _remap_fn, _s_v_est = _audio_shift_schedule(sigmas, _mult)
            _s_a_eff = _mult * _s_v_est
        if _base_mod is not None and hasattr(_base_mod, "process_timestep"):
            _base_pts = _base_mod.process_timestep
            if _remap_fn is not None:
                def _pts(timestep, x, denoise_mask=None, audio_denoise_mask=None, **kw):
                    _ret = _base_pts(timestep, x, denoise_mask=denoise_mask,
                                     audio_denoise_mask=audio_denoise_mask, **kw)
                    if isinstance(_ret, tuple) and len(_ret) >= 2 and _ret[1] is not None:
                        _ts = timestep.float()
                        _scale = torch.where(
                            _ts > 1e-6, _remap_fn(_ts) / _ts.clamp(min=1e-6),
                            torch.ones_like(_ts))
                        return _ret[0], _ret[1] * _scale
                    return _ret
                _base_mod.process_timestep = _pts
        if _a_sig is not None and _mult != 1.0 and aud_tmpl is not None:
            sampler = copy.copy(sampler)
            _fname = getattr(getattr(sampler, "sampler_function", None), "__name__", "")
            _ancestral = "ancestral" in _fname
            sampler.sampler_function = _make_av_permodality_sampler(_a_sig, ancestral=_ancestral)
        _vid_pos = 0
        for chunk_idx in range(_eff_num_chunks):
            is_first = chunk_idx == 0
            _cur_new_lf = _chunk_plan[chunk_idx][0]
            chunk_overlap = 0 if is_first else _eff_overlap
            total_lf = chunk_overlap + _cur_new_lf
            _plan_entry = _cond_plan[chunk_idx]
            _is_transition = isinstance(_plan_entry, tuple)
            # A crossfade chunk statistically belongs to the scene its text leans
            # toward (w < 0.5 → outgoing, w >= 0.5 → incoming): the per-scene ref
            # resets (incl. the §2.3 EMA) fire on the first incoming-leaning chunk.
            scene_idx = (_plan_entry[1] if _plan_entry[2] >= 0.5 else _plan_entry[0]) \
                if _is_transition else _plan_entry
            _scene_switch = (num_scenes > 1 and chunk_idx > 0
                             and _prev_scene_idx is not None
                             and scene_idx != _prev_scene_idx)
            if _scene_switch:
                _s1_vid_std_ref    = None
                _s1_band_ref       = None
                _origin_ref        = None
                _origin_layout     = None
                # §2.3: drop the old scene's EMA reference — the first chunk of the
                # new scene is uncorrected and re-anchors the EMA (incl. _init_std).
                clss_state.reset_drift_refs()
            _prev_scene_idx = scene_idx
            has_slb     = not is_first and clss_state._overlap_latent is not None
            has_aud_slb = not is_first and audio_slb_latent is not None
            has_aud_ref = not is_first and audio_overlap_latent is not None
            guider_chunk = copy.copy(guider)
            if num_scenes > 1:
                _pos_entry = (_blend_scene_cond(pos_conds[_plan_entry[0]],
                                                pos_conds[_plan_entry[1]],
                                                _plan_entry[2])
                              if _is_transition else pos_conds[scene_idx])
                guider_chunk.original_conds = {
                    **guider.original_conds,
                    "positive": [_pos_entry],
                }
            lat_vid = torch.zeros(B, C_v, total_lf, H, W, device=device)
            mask_vid = torch.ones(B, 1, total_lf, 1, 1, device=device)
            if has_slb:
                _tau_c_v = _tau_c_eff(clss_config.tau_c * video_slb_tau_mult,
                                      _VIDEO_TAU_C_CEILING, chunk_idx - 1)
                if _pin_next:
                    _tau_c_v = 0.0
                _pin_next = False
                lat_vid, mask_vid = LTXVAddGuide.replace_latent_frames(
                    lat_vid, mask_vid,
                    guiding_latent=clss_state._overlap_latent.to(device),
                    latent_idx=0,
                    strength=1.0 - _tau_c_v,
                )
            if is_first and img_guide_latent is not None:
                lat_vid, mask_vid = LTXVAddGuide.replace_latent_frames(
                    lat_vid, mask_vid,
                    guiding_latent=img_guide_latent.to(device),
                    latent_idx=0,
                    strength=1.0,
                )
            if aud_tmpl is not None:
                cur_new_af = _chunk_plan[chunk_idx][1]
                chunk_af = (audio_overlap_af if not is_first else 0) + cur_new_af
                lat_aud  = torch.zeros(B_a, C_a, chunk_af, freq, device=device)
                mask_aud = torch.ones(B_a, 1, chunk_af, 1, device=device)
                _slb_ctx_used: torch.Tensor | None = None
                _slb_ctx_pos = 0
                if has_aud_slb and audio_slb_tau_mult >= 0.0:
                    slb = audio_slb_latent.to(device)
                    n   = min(audio_overlap_af, slb.shape[2], chunk_af)
                    _slb_ctx = slb[:, :, :n]
                    lat_aud[:, :, :n]  = _slb_ctx
                    if audio_slb_tau_mult > 0.0:
                        _tau_c_a = _tau_c_eff(clss_config.tau_c * audio_slb_tau_mult,
                                              _AUDIO_TAU_C_CEILING, chunk_idx - 1)
                    else:
                        _tau_c_a = 0.0
                    mask_aud[:, :, :n] = _tau_c_a
                    _slb_ctx_used = _slb_ctx.detach().cpu()
                elif has_aud_slb:
                    # audio_slb_tau_mult < 0: regenerate the overlap freely (best
                    # musical continuity — a fully frozen SLB drags the whole window
                    # toward an exact repeat of the tail), but pin the last |mult|
                    # SECONDS of the previous tail frozen at the END of the overlap.
                    # Without the pin the vocal phrase restarts at every chunk
                    # boundary: words are cut mid-phoneme and the singing goes dead.
                    # The pinned frames sit immediately before the kept region, so
                    # they glue the seam without constraining the rest of the window.
                    slb = audio_slb_latent.to(device)
                    n   = min(audio_overlap_af, slb.shape[2], chunk_af)
                    _pin_af = min(round(-audio_slb_tau_mult * fps * _af_per_px), n)
                    if _pin_af > 0:
                        _slb_ctx_pos = n - _pin_af
                        _slb_ctx = slb[:, :, _slb_ctx_pos:n]
                        lat_aud[:, :, _slb_ctx_pos:n]  = _slb_ctx
                        mask_aud[:, :, _slb_ctx_pos:n] = 0.0
                        _slb_ctx_used = _slb_ctx.detach().cpu()
                if has_aud_ref:
                    ref_slb   = audio_overlap_latent.to(device)
                    b_r, c_r, t_r, f_r = ref_slb.shape
                    ref_tokens = ref_slb.permute(0, 2, 1, 3).reshape(b_r, t_r, c_r * f_r)
                    ref_audio_dict = {"tokens": ref_tokens}
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
                _n_slb = min(audio_overlap_af, audio_slb_latent.shape[2]) \
                    if (has_aud_slb and audio_slb_tau_mult >= 0.0) else 0
                _ov_gen = (audio_overlap_af - _n_slb) if not is_first else 0
                av_samples = comfy.nested_tensor.NestedTensor((lat_vid, lat_aud))
                av_mask    = comfy.nested_tensor.NestedTensor((mask_vid, mask_aud))
                chunk_latent = {"samples": av_samples, "noise_mask": av_mask}
            else:
                chunk_latent = {"samples": lat_vid, "noise_mask": mask_vid}
            _s1_noise_pos = _vid_pos
            _s1_chunk_noise = _SlicedNoise(
                _full_noise_vid_s1, _s1_noise_pos, chunk_overlap, seed=_noise_seed_s1,
                full_noise_aud=_full_noise_aud_s1,
                a_pos=_s1_a_noise_pos,
                a_overlap=(audio_overlap_af if not is_first else 0),
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
            denoised_samples = denoised["samples"]
            if is_av:
                vid_out, aud_out = denoised_samples.unbind()
            else:
                vid_out = denoised_samples
                aud_out = None
            if is_first and img_guide_latent is not None:
                _guide_sim = _frame_cos(vid_out[:, :, 0], img_guide_latent.to(device)[:, :, 0])
            new_vid   = vid_out[:, :, chunk_overlap:]
            mu_pre    = new_vid.mean().item()
            std_pre   = new_vid.std().item()
            corrected = clss_state.post_process(new_vid)
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
                else:
                    _g_lo = min(1.10, max(0.90, (_s1_band_ref[0] / max(_e_low, 1e-12)) ** 0.5))
                    _g_hi = min(1.12, max(0.90, (_s1_band_ref[1] / max(_e_high, 1e-12)) ** 0.5))
                    if abs(_g_lo - 1.0) > 0.005 or abs(_g_hi - 1.0) > 0.005:
                        corrected = (_da_low * _g_lo + _da_high * _g_hi).reshape(
                            _da_b, _da_t, _da_c, _da_h, _da_w
                        ).permute(0, 2, 1, 3, 4).contiguous().to(corrected.dtype)
                        _e_low_p, _e_high_p = _e_low * _g_lo ** 2, _e_high * _g_hi ** 2
                        _hf_p = _e_high_p / max(_e_low_p + _e_high_p, 1e-12)
                        _hf_share = _hf_p
            _trend["vid_hf"].append(_hf_share)
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
            _layout_argmin_track.append(int(_lsims.index(min(_lsims))))
            _trend["vid_origin"].append(min(_osims))
            if _auto_pin:
                _dd_max = (max(_lsims[i] - _lsims[i + 1]
                               for i in range(len(_lsims) - 1))
                           if len(_lsims) > 1 else 0.0)
                _hit = chunk_idx >= 1 and (
                    (_dd_max > 0.25) or (min(_lsims) < 0.05))
                if _hit and chunk_idx + 1 < num_chunks:
                    _pin_next = True
            if _s1_vid_std_ref is None:
                _s1_vid_std_ref = corrected.float().std().item()
            else:
                _cur_vstd = corrected.float().std().item()
                _ratio = _s1_vid_std_ref / max(_cur_vstd, 1e-6)
                if _ratio < 0.96 or _ratio > 1.04:
                    _g_v = 1.0 + 0.5 * (_ratio - 1.0)
                    _m = corrected.float().mean()
                    corrected = ((corrected.float() - _m) * _g_v + _m).to(corrected.dtype)
            mu_post   = corrected.mean().item()
            std_post  = corrected.std().item()
            clss_state.update_buffer(corrected)
            acc_video.append(corrected.cpu())
            _intra = _frame_cos(corrected[:, :, 0], corrected[:, :, -1])
            _trend["vid_intra"].append(_intra)
            if _s1_prev_last is not None:
                _bnd = _frame_cos(_s1_prev_last.to(device), corrected[:, :, 0])
                _trend["vid_bnd"].append(_bnd)
            _trend["vid_std"].append(std_post)
            if not is_first:
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
                    _trend["vid_ident"].append(_best_sim)
            _s1_prev_last = corrected[:, :, -1].cpu()
            if chunk_idx == num_chunks - 1 and corrected.shape[2] > 1:
                _adj = [_frame_cos(corrected[:, :, i], corrected[:, :, i + 1])
                        for i in range(corrected.shape[2] - 1)]
            if aud_out is not None:
                aud_drop = audio_overlap_af if not is_first else 0
                if aud_drop > 0 and aud_out.shape[2] < aud_drop:
                    aud_drop = 0
                new_aud = aud_out[:, :, aud_drop:]
                aud_acc_start = sum(a.shape[2] for a in acc_audio)
                aud_acc_end   = aud_acc_start + new_aud.shape[2]
                if not is_first and _slb_ctx_used is not None and audio_overlap_af > 0:
                    _slb_sim = _aud_cos(
                        _slb_ctx_used.to(device),
                        aud_out[:, :, _slb_ctx_pos:_slb_ctx_pos + _slb_ctx_used.shape[2]])
                    _trend["aud_slb"].append(_slb_sim)
                with torch.no_grad():
                    _n8 = min(8, new_aud.shape[2])
                    _ch_absmax = new_aud[:, :, :_n8].float().abs().flatten(2).max(dim=2).values
                    _ch_std    = new_aud.float().std(dim=(2, 3))
                _env = new_aud.detach().float().pow(2).mean(dim=(0, 1, 3)).cpu()
                if _prev_aud_env is not None and len(_prev_aud_env) > 8:
                    _L = min(len(_env), len(_prev_aud_env))
                    _ea = _env[:_L] - _env[:_L].mean()
                    _eb = _prev_aud_env[:_L] - _prev_aud_env[:_L].mean()
                    _env_corr = float((_ea * _eb).sum() /
                                      (_ea.norm() * _eb.norm() + 1e-8))
                    _trend["aud_env"].append(_env_corr)
                _prev_aud_env = _env
                if is_first:
                    _n_fade = min(8, new_aud.shape[2])
                    if _n_fade >= 2:
                        _ramp = torch.linspace(0.125, 1.0, _n_fade, device=device)
                        new_aud = new_aud.clone()
                        new_aud[:, :, :_n_fade] = new_aud[:, :, :_n_fade] * _ramp.view(1, 1, _n_fade, 1)
                    _fa = new_aud.float()
                    _sig = _fa.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
                    _c35 = _sig * 3.5
                    _over = (_fa.abs() - _c35).clamp(min=0)
                    _n_over = int((_over > 0).sum())
                    new_aud = (_fa - torch.sign(_fa)
                               * (_over - 0.5 * _sig * torch.tanh(_over / _sig))).to(aud_out.dtype)
                _aud_sims = _aud_within_chunk_sims(new_aud)
                if _aud_sims:
                    _trend["aud_wc"].append(_aud_sims[-1])
                if _s1_aud_prev_last is not None:
                    _aud_bnd = _aud_cos(_s1_aud_prev_last.to(device), new_aud[:, :, :1])
                    _trend["aud_bnd"].append(_aud_bnd)
                with torch.no_grad():
                    _aud_rms = new_aud.float().pow(2).mean().sqrt().item()
                    _aud_peak = int(new_aud.float().abs().mean(dim=(0, 1, 3)).argmax().item())
                    _nseg = 4
                    _seg_t = new_aud.shape[2] // _nseg
                    _seg_rms = [
                        new_aud[:, :, s * _seg_t:(s + 1) * _seg_t].float().pow(2).mean().sqrt().item()
                        for s in range(_nseg)
                    ] if _seg_t > 0 else []
                with torch.no_grad():
                    _freq_e = new_aud.float().abs().mean(dim=(0, 1, 2)).tolist()
                _aud_peak_track.append(_aud_peak)
                if _s1_audio_freq_ref is None:
                    _s1_audio_freq_ref = _freq_e
                else:
                    _freq_ratio = [e / r if r > 1e-6 else 0.0 for e, r in zip(_freq_e, _s1_audio_freq_ref)]
                    if len(_freq_ratio) >= 4:
                        _trend["aud_hf"].append(sum(_freq_ratio[-4:]) / 4.0)
                _trend["aud_rms"].append(new_aud.float().pow(2).mean().sqrt().item())
                if _s1_aud_prev_last is not None:
                    _bnd_final = _aud_cos(_s1_aud_prev_last.to(device), new_aud[:, :, :1])
                if audio_overlap_af > 0:
                    ov = audio_overlap_af
                    if new_aud.shape[2] >= ov:
                        audio_slb_latent = new_aud[:, :, -ov:].cpu()
                    else:
                        audio_slb_latent = new_aud.cpu()
                    _tail_cur = new_aud.cpu()
                    _s1_audio_tail = (
                        _tail_cur if _s1_audio_tail is None
                        else torch.cat([_s1_audio_tail, _tail_cur], dim=2)
                    )
                    _ref_keep = ov + _ref_len_af
                    if _s1_audio_tail.shape[2] > _ref_keep:
                        _s1_audio_tail = _s1_audio_tail[:, :, -_ref_keep:]
                    _s1_audio_tail = _s1_audio_tail.clone()
                    _tail_lf = _s1_audio_tail.shape[2]
                    pre_ov_end = max(0, _tail_lf - ov)
                    if pre_ov_end > 0:
                        _ref_start = max(0, pre_ov_end - _ref_len_af)
                        audio_overlap_latent = _s1_audio_tail[:, :, _ref_start:pre_ov_end].clone()
                    else:
                        audio_overlap_latent = None
                acc_audio.append(new_aud.cpu())
                audio_chunk_ends.append(sum(a.shape[2] for a in acc_audio))
                _s1_aud_prev_last = new_aud[:, :, -1:].cpu()
            _vid_pos += _cur_new_lf
        if _base_pts is not None and _base_mod is not None:
            _base_mod.process_timestep = _base_pts
        full_vid = torch.cat(acc_video, dim=2)
        # End-of-run trend dump (structure metrics — they localize failures, they
        # never prove a quality win).  vid_hf is the detail-decay curve, vid_bnd
        # the seam-continuity curve; both were accumulated but never printed,
        # which made the long-run fade invisible in logs.
        for _k, _v in _trend.items():
            if _v:
                print(f"[CLSS] trend {_k}: " + " ".join(f"{_x:.3f}" for _x in _v))
        if acc_audio:
            full_aud = torch.cat(acc_audio, dim=2)
            full_aud = _post_process_audio_latent(full_aud, audio_chunk_ends,
                                                  energy_beta=0.0, label=" S1")
            output_samples = comfy.nested_tensor.NestedTensor((full_vid, full_aud))
        else:
            output_samples = full_vid
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        return ({"samples": output_samples},)
def _shifted_shared_schedule(sigmas, sigma_shift, terminal=None):
    import math as _m
    sig = sigmas.double()
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
    import math as _m
    sig = sigmas.double()
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
    import math as _m
    sig = sigmas.double()
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
        tv = t.double().clamp(min=1e-6, max=1.0)
        raw_v = (1.0 - (1.0 - tv) * k_v).clamp(min=1e-6, max=1.0)
        linv = 1.0 / (1.0 + e_v * (1.0 / raw_v - 1.0))
        raw_a = e_a / (e_a + (1.0 / linv - 1.0))
        out = 1.0 - (1.0 - raw_a) / k_a
        out = torch.where(t <= 0.0, torch.zeros_like(out), out)
        return out.to(t.dtype)
    return audio, remap, s_v
def _make_av_permodality_sampler(audio_sigmas, ancestral=True):
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
        _denoise_mask = extra_args.get("denoise_mask", None)
        audio_noise_sampler = _kds.default_noise_sampler(
            x, seed=(None if seed is None else (seed + 7919) % (2 ** 63)))
        s_noise = s_noise * getattr(
            guider.model_patcher.get_model_object('model_sampling'), "noise_scale", 1.0)
        s_in = x.new_ones([x.shape[0]])
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
            _conv = ai / si
            if _denoise_mask is not None and len(shapes) == 2:
                _, _ma = _cu.unpack_latents(_denoise_mask, shapes)
                _conv = torch.where(_ma >= 0.999, ai / si, 1.0)
            da = xa * (1.0 - _conv) + da * _conv
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
    def __init__(self, full_noise_vid: torch.Tensor, pos: int, chunk_overlap: int, seed: int = 0,
                 full_noise_aud: torch.Tensor | None = None, a_pos: int = 0, a_overlap: int = 0):
        self._full        = full_noise_vid
        self._pos         = pos
        self._chunk_overlap = chunk_overlap
        self._full_aud    = full_noise_aud
        self._a_pos       = a_pos
        self._a_overlap   = a_overlap
        self.seed         = seed
    def generate_noise(self, input_latent: dict) -> "torch.Tensor | comfy.nested_tensor.NestedTensor":
        samples = input_latent["samples"]
        is_av   = isinstance(samples, comfy.nested_tensor.NestedTensor)
        vid     = samples.unbind()[0] if is_av else samples
        _g = torch.Generator(device="cpu").manual_seed(
            (int(self.seed) % (2 ** 31)) * 1_000_003
            + self._pos * 7_919 + self._a_pos * 104_729)
        noise_vid = torch.randn(vid.shape, generator=_g, dtype=vid.dtype).to(vid.device)
        n_new   = vid.shape[2] - self._chunk_overlap
        src_end = min(self._pos + n_new, self._full.shape[2])
        src_n   = src_end - self._pos
        if src_n > 0:
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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider":           ("GUIDER",      {"tooltip": "GUIDER for the refinement pass (typically low-CFG with a distilled LoRA). Applied per chunk with the same SLB continuity mechanism as Stage 1."}),
                "sampler":          ("SAMPLER",     {"tooltip": "SAMPLER for the per-chunk refinement denoise."}),
                "sigmas":           ("SIGMAS",      {"tooltip": "SIGMAS schedule for the refinement pass (the short, low-noise schedule of the distilled refinement LoRA)."}),
                "noise":            ("NOISE",       {"tooltip": "NOISE source; its seed drives the full-length noise tensor the per-chunk slices are cut from."}),
                "latent":           ("LATENT",      {
                    "tooltip": "Full Stage-1 latent (video or AV nested tensor) to refine. Audio is frozen passthrough — fed as context with a zero denoise mask.",
                }),
                "clss_config":      ("CLSS_CONFIG", {"tooltip": "CLSS_CONFIG; supplies the default SLB overlap and the tau_c used to re-noise the Stage-2 overlap."}),
                "frames_per_chunk": ("INT",         {
                    "default": 0, "min": 0, "max": 128,
                    "tooltip": "Latent frames per Stage-2 chunk. 0 = auto: sized from a VRAM token budget (~42000 tokens at 15.6 GB, scaled by total VRAM) and capped so a window stays under the 19.5 s RoPE wall.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional i2v guide image, re-pinned to the first frame of the first Stage-2 chunk. Requires vae."}),
                "vae":   ("VAE",   {"tooltip": "Video VAE, only needed together with image for the i2v guide encode."}),
                "s2_overlap": ("INT", {
                    "default": 0, "min": 0, "max": 32,
                    "tooltip": "Stage-2 SLB size in latent frames. 0 = auto: roughly frames_per_chunk/3 clamped to [8, 16] when chunking is active, otherwise the CLSSConfig overlap.",
                    }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Frames per second, used to keep auto-chunked windows under the 19.5 s RoPE wall.",
                    }),
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
        s2_guide_latent: torch.Tensor | None = None
        s2_i2v_scale_factors = None
        if image is not None and vae is not None:
            s2_i2v_scale_factors = vae.downscale_index_formula
            _, s2_guide_latent = LTXVAddGuide.encode(vae, W, H, image[:1], s2_i2v_scale_factors)
        if full_aud is not None:
            B_a, C_a, T_a, freq = full_aud.shape
        else:
            B_a = C_a = T_a = freq = 0
        import math as _math
        if frames_per_chunk <= 0:
            _budget_tokens = 42000
            if torch.cuda.is_available():
                _total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                _budget_tokens = max(24000, int(42000 * _total_gb / 15.6))
            _s2_wall_fpc = max(12, int((_ROPE_WALL_S - _ROPE_WALL_MARGIN_S) * fps / 8))
            _fpc_cap = max(12, min(_budget_tokens // max(1, H * W), _s2_wall_fpc - 16))
            if T <= _fpc_cap:
                frames_per_chunk = T
            else:
                _n = _math.ceil(T / _fpc_cap)
                frames_per_chunk = _math.ceil(T / _n)
        if s2_overlap <= 0 and frames_per_chunk < T:
            s2_overlap = min(16, max(8, frames_per_chunk // 3))
            overlap_lf = s2_overlap
        _s2_win_lf = frames_per_chunk + (s2_overlap if s2_overlap > 0 and frames_per_chunk < T else 0)
        _s2_win_s = _s2_win_lf * 8 / fps
        num_chunks = max(1, (T + frames_per_chunk - 1) // frames_per_chunk)
        _base, _extra = T // num_chunks, T % num_chunks
        chunk_boundaries, _cur = [], 0
        for i in range(num_chunks):
            _cur += _base + (1 if i < _extra else 0)
            chunk_boundaries.append(_cur)
        noise_seed = getattr(noise, "seed", 0)
        full_noise_vid: torch.Tensor = noise.generate_noise({"samples": full_vid})
        has_aud = full_aud is not None
        full_noise_aud: torch.Tensor | None = None
        overlap_latent: torch.Tensor | None = None
        acc_video: list[torch.Tensor] = []
        acc_audio: list[torch.Tensor] = []
        _s2_prev_last:     torch.Tensor | None = None
        _s2_id_ref:        torch.Tensor | None = None
        _s2_aud_prev_last: torch.Tensor | None = None
        _s2_trend = {"fid_first": [], "fid_last": [], "aud_bnd": []}
        lf_to_sec = 8 / fps
        pos    = 0
        a_pos  = 0
        for chunk_idx in range(num_chunks):
            if pos >= T:
                break
            is_first      = (chunk_idx == 0)
            chunk_overlap = 0 if is_first else overlap_lf
            end_pos       = chunk_boundaries[chunk_idx]
            actual_new    = end_pos - pos
            total_lf      = chunk_overlap + actual_new
            t_start = pos * lf_to_sec
            t_end   = end_pos * lf_to_sec
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
            active_guider = guider
            if is_first and s2_guide_latent is not None:
                lat_vid, mask_vid = LTXVAddGuide.replace_latent_frames(
                    lat_vid, mask_vid,
                    guiding_latent=s2_guide_latent.to(device),
                    latent_idx=0,
                    strength=1.0,
                )
            if has_aud:
                a_new_start = a_pos
                a_new_end   = min(round(end_pos * T_a / T), T_a)
                chunk_af    = a_new_end - a_new_start
                a_ov     = 0
                lat_aud  = full_aud[:, :, a_new_start:a_new_end].to(device)
                mask_aud = torch.zeros(B_a, C_a, chunk_af, freq, device=device)
                chunk_latent = {
                    "samples":    comfy.nested_tensor.NestedTensor((lat_vid, lat_aud)),
                    "noise_mask": comfy.nested_tensor.NestedTensor((mask_vid, mask_aud)),
                }
            else:
                a_ov = 0
                chunk_latent = {"samples": lat_vid, "noise_mask": mask_vid}
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
            d_samples = denoised["samples"]
            if is_av:
                vid_out, aud_out = d_samples.unbind()
            else:
                vid_out  = d_samples
                aud_out  = None
            new_vid = vid_out[:, :, chunk_overlap:]
            _s1_ref_slice = full_vid[:, :, pos:end_pos].to(device)
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
            n_slb   = min(overlap_lf, actual_new)
            overlap_latent = new_vid[:, :, -n_slb:].clone().cpu()
            acc_video.append(new_vid.cpu())
            _s2_intra = _frame_cos(new_vid[:, :, 0], new_vid[:, :, -1])
            if _s2_prev_last is not None:
                _s2_bnd = _frame_cos(_s2_prev_last.to(device), new_vid[:, :, 0])
            _s2_cur_feat = F.normalize(new_vid[:, :, 0].float().reshape(B, C_v, -1).mean(-1), dim=1)
            if _s2_id_ref is None:
                _s2_id_ref = _s2_cur_feat.cpu()
            else:
                _s2_id_sim = (_s2_cur_feat * _s2_id_ref.to(device)).sum(dim=1).mean().item()
            _s2_prev_last = new_vid[:, :, -1].cpu()
            if new_vid.shape[2] > 1:
                _s2_adj = [_frame_cos(new_vid[:, :, i], new_vid[:, :, i + 1])
                           for i in range(new_vid.shape[2] - 1)]
            _s1_slice = full_vid[:, :, pos:end_pos].to(device)
            for _fi, _lbl in [(0, "first"), (actual_new // 2, "mid"), (actual_new - 1, "last")]:
                _fid = _frame_cos(new_vid[:, :, _fi], _s1_slice[:, :, _fi])
                if _lbl == "last":
                    _s2_trend["fid_last"].append(_fid)
                if _lbl == "first":
                    _s2_trend["fid_first"].append(_fid)
            if aud_out is not None:
                new_aud = aud_out
                s1_chunk_ref = full_aud[:, :, a_new_start:a_new_end].to(device)
                _s1_sim = _aud_cos(s1_chunk_ref, new_aud)
                if _s2_aud_prev_last is not None:
                    _aud_bnd = _aud_cos(_s2_aud_prev_last.to(device), new_aud[:, :, :1])
                    _s2_trend["aud_bnd"].append(_aud_bnd)
                _s2_aud_prev_last = new_aud[:, :, -1:].cpu()
                acc_audio.append(new_aud.cpu())
                a_pos += new_aud.shape[2]
            pos = end_pos
        full_refined_vid = torch.cat(acc_video, dim=2)
        if acc_audio:
            full_refined_aud = torch.cat(acc_audio, dim=2)
            output = comfy.nested_tensor.NestedTensor((full_refined_vid, full_refined_aud))
        elif full_aud is not None:
            output = comfy.nested_tensor.NestedTensor((full_refined_vid, full_aud.cpu()))
        else:
            output = full_refined_vid
        return ({"samples": output},)
class CLSSAVGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider":    ("GUIDER", {"tooltip": "Existing GUIDER to patch with split video/audio CFG; the video CFG stays whatever the source guider carries."}),
                "audio_cfg": ("FLOAT",  {
                    "default": 7.0, "min": 1.0, "max": 30.0, "step": 0.5,
                    "tooltip": "CFG scale for the audio modality only. Audio wants higher CFG (~7) than video (~4) — audio drift is a recurring failure mode.",
                }),
                "rescale":   ("FLOAT",  {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Per-modality CFG rescale: pulls the guided prediction's std back toward the conditional prediction's std (factor = r·std_ratio + 1−r). 0 = off. Counteracts CFG oversaturation.",
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
        _log_done  = [False]
        def _av_cfg_fn(args):
            cond_d   = args["cond_denoised"]
            uncond_d = args["uncond_denoised"]
            scale    = args["cond_scale"]
            x        = args["input"]
            if isinstance(cond_d, comfy.nested_tensor.NestedTensor):
                vid_c, aud_c = cond_d.unbind()
                vid_u, aud_u = uncond_d.unbind()
                x_vid, x_aud = x.unbind()
                vid_denoised = vid_u + scale    * (vid_c - vid_u)
                aud_denoised = aud_u + _audio_cfg * (aud_c - aud_u)
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
                if not _log_done[0]:
                    _log_done[0] = True
                    with torch.no_grad():
                        v_norm = (vid_c - vid_u).float().norm().item()
                        a_norm = (aud_c - aud_u).float().norm().item()
                return comfy.nested_tensor.NestedTensor((
                    x_vid - vid_denoised,
                    x_aud - aud_denoised,
                ))
            else:
                if not _log_done[0]:
                    _log_done[0] = True
                return x - (uncond_d + scale * (cond_d - uncond_d))
        new_guider.model_options["sampler_cfg_function"] = _av_cfg_fn
        new_guider.audio_cfg = _audio_cfg
        return (new_guider,)
def _stg_value_passthrough(attn, x, context=None, mask=None, pe=None, k_pe=None,
                           transformer_options={}):
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
                del m.forward
            except AttributeError:
                pass
        return False
class _GuiderCLSSAV(comfy.samplers.CFGGuider):
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
        self.set_cfg(video_cfg)
        self.audio_cfg = audio_cfg
    @staticmethod
    def _rescale_pred(pred: torch.Tensor, cond: torch.Tensor, r: float) -> torch.Tensor:
        if r <= 0.0:
            return pred
        ratio = cond.float().std() / pred.float().std().clamp(min=1e-8)
        factor = r * ratio + (1.0 - r)
        factor = factor.clamp(0.5, 2.0)
        return pred * factor.to(pred.dtype)
    _av_latent_shapes = None
    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None,
               callback=None, disable_pbar=False, seed=None):
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
        if (not is_nested and not is_packed_av) or negative is None:
            if not self._logged:
                self._logged = True
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
            to["a2v_cross_attn"] = False
            to["v2a_cross_attn"] = False
            mo["transformer_options"] = to
            (out_mod,) = comfy.samplers.calc_cond_batch(
                self.inner_model, [positive], x, timestep, mo
            )
        out_ptb = None
        if self._video_stg != 0.0 or self._audio_stg != 0.0:
            _dm = getattr(self.inner_model, "diffusion_model", None)
            _skipper = _SkipSelfAttn(_dm, self._stg_blocks) if _dm is not None else None
            if _skipper is not None and _skipper._mods:
                with _skipper:
                    (out_ptb,) = comfy.samplers.calc_cond_batch(
                        self.inner_model, [positive], x, timestep, model_options
                    )
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
        if out_mod is not None:
            vid_m, aud_m = _split(out_mod)
            pred_v = pred_v + (self._modality_scale - 1.0) * (vid_c - vid_m)
            pred_a = pred_a + (self._modality_scale - 1.0) * (aud_c - aud_m)
            if not self._logged:
                with torch.no_grad():
                    vm = (vid_c - vid_m).float().norm().item()
                    am = (aud_c - aud_m).float().norm().item()
        pred_v = self._rescale_pred(pred_v, vid_c, self._rescale)
        pred_a = self._rescale_pred(pred_a, aud_c, self._rescale)
        if not self._logged:
            self._logged = True
            _n_passes = 2 + (0 if out_mod is None else 1) + (0 if out_ptb is None else 1)
        return _join(pred_v, pred_a)
class CLSSAVGuiderV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":    ("MODEL",        {"tooltip": "MODEL (LTX-2.3) the guider is built on."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive CONDITIONING. One entry per scene (from CLSSScenePrompts) enables per-scene chunk guidance in the sampler."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative CONDITIONING, required for split CFG; without it the guider falls back to a plain conditional pass."}),
                "video_cfg": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 30.0, "step": 0.5,
                                        "tooltip": "Video CFG scale. Validated default 4.0."}),
                "audio_cfg": ("FLOAT", {
                    "default": 7.0, "min": 1.0, "max": 30.0, "step": 0.5,
                    "tooltip": "Audio CFG scale, applied to the audio modality independently of video_cfg. Validated default 7.0 — audio wants higher CFG than video; audio drift is a recurring failure mode.",
                }),
                "modality_scale": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Modality guidance: an extra forward pass with a2v/v2a cross-attention disabled, pushed away from with weight (scale − 1). 0 or 1 = off (skips the extra pass). Validated default 3.0.",
                }),
                "rescale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Per-modality CFG rescale toward the conditional prediction's std (factor = r·std_ratio + 1−r, clamped to [0.5, 2.0]). 0 = off. Validated default 0.7.",
                                      }),
            },
            "optional": {
                "video_stg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1,
                    "tooltip": "STG (spatio-temporal guidance) strength for video: an extra forward pass with self-attention skipped in stg_block, added with this weight. 0 = off.",
                    }),
                "audio_stg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1,
                    "tooltip": "STG strength for audio (shares the same skipped-self-attention pass as video_stg). 0 = off.",
                    }),
                "stg_block": ("INT", {
                    "default": 28, "min": 0, "max": 63,
                    "tooltip": "Transformer block index whose self-attention (attn1 / audio_attn1) is replaced by a value passthrough during the STG pass. 28 validated for LTX-2.3.",
                    }),
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
        return (guider,)
class CLSSVideoDecodeSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae":    ("VAE",    {"tooltip": "Video VAE used to decode each temporal slice."}),
                "latent": ("LATENT", {"tooltip": "Full video latent to decode (an AV nested tensor works too — the audio part is ignored by this node)."}),
                "filename_prefix": ("STRING", {"default": "clss/CLSS_frame_",
                                               "tooltip": "Output filename prefix; frames are written as <prefix>_NNNNN.png under the ComfyUI output directory.",
                                               }),
                "frames_per_slice": ("INT", {"default": 32, "min": 8, "max": 256,
                                             "tooltip": "Latent frames decoded per VAE call. This is a streaming temporal-slice decode: the whole decoded video never materializes in RAM — frames are saved to disk as each slice decodes.",
                                             }),
                "context_frames": ("INT", {"default": 2, "min": 0, "max": 16,
                                           "tooltip": "Extra latent frames of temporal context prepended to each non-first slice and dropped after decode, hiding VAE slice-boundary seams. 0 = no context.",
                                           }),
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
            samples = samples.unbind()[0]
        T = samples.shape[2]
        output_dir = folder_paths.get_output_directory()
        full_folder, filename, _, _, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir)
        os.makedirs(full_folder, exist_ok=True)
        frame_idx = 0
        pos = 0
        while pos < T:
            end = min(pos + frames_per_slice, T)
            ctx = 0 if pos == 0 else max(0, min(context_frames, pos - 1))
            start = pos if pos == 0 else pos - 1 - ctx
            px = vae.decode(samples[:, :, start:end])
            drop = 0 if pos == 0 else 1 + ctx * 8
            arr = (px[0, drop:].float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            for f in range(arr.shape[0]):
                Image.fromarray(arr[f]).save(
                    os.path.join(full_folder, f"{filename}_{frame_idx:05d}.png"),
                    compress_level=4)
                frame_idx += 1
            del px, arr
            pos = end
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
