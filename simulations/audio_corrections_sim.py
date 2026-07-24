"""Pure-math simulation of the two 2026-07-21 audio corrections in nodes.py.

No model, no ComfyUI — this replays the MEASURED failure trajectories from the
live runs through the exact correction math, and verifies the noise-envelope
dither's statistics.  It validates the control laws, not perception: the live
run remains the ground truth for how the model responds.

1. Per-bin spectral anchor (audio_freq_anchor):
   drift-capped per-bin EMA target (lambda=0.25, cap ±15% vs scene ref),
   per-chunk gain clamp [0.75, 1.25], RMS-preserving renormalisation.
   Calibrated against the three measured trajectories:
     - 31-lf T2V run: top bins compounded to 2.1-2.65x reference by ch7-9
     - 12-lf T2V run: top bins reached 1.29-1.45x by ch9-10
     - I2V run:       flat 0.87x (benign — must NOT be fought)

2. Audio noise envelope dither (audio_env_dither):
   smooth seeded loudness envelope (reflect-padded Gaussian smoothing) on the
   S1 audio noise field — verifies marginals, determinism, seam smoothness,
   absence of systematic cross-chunk repetition (the metronome criterion),
   and that the injected suggestion sits far above the i.i.d. envelope floor.

An earlier revision of this sim caught two real production defects (zero-pad
edge attenuation; EMA lambda=0.10 fighting benign spectra for ~15 chunks) —
both fixed in nodes.py before any live run.

Run:  python simulations/audio_corrections_sim.py
"""

import torch

torch.manual_seed(0)

FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ───────────────────────────────────────────────────────────────────────────
# Part 1 — per-bin spectral anchor control law
# ───────────────────────────────────────────────────────────────────────────

# chunk-0 reference spectrum from the 2026-07-21 12-lf run log (16 bins)
REF_E = torch.tensor([1.757, 0.912, 0.532, 0.455, 0.525, 0.513, 0.441, 0.369,
                      0.293, 0.237, 0.202, 0.210, 0.223, 0.219, 0.283, 0.333])
FLAM, FDRIFT, GCLAMP = 0.25, 0.15, (0.75, 1.25)   # = production values


def anchor_step(cur_e, ref_e, ema):
    """Exact per-bin anchor math from nodes.py (incl. the RMS renorm scalar)."""
    ema_prev = ref_e if ema is None else ema
    ema_raw = (1 - FLAM) * ema_prev + FLAM * cur_e
    ema_new = torch.minimum(torch.maximum(ema_raw, ref_e * (1 - FDRIFT)),
                            ref_e * (1 + FDRIFT))
    gains = (ema_new / cur_e.clamp(min=1e-6)).clamp(*GCLAMP)
    shaped = cur_e * gains
    # RMS preservation (proxy: total squared bin energy) — a common scalar on
    # all bins, exactly as in production.
    r = (cur_e.pow(2).sum() / shaped.pow(2).sum().clamp(min=1e-12)).sqrt()
    return shaped * r, ema_new, gains


def run_loop(g_open, chunks=10, cur0=None, alpha=1.0, intrinsic=None):
    """Autoregressive spectrum loop.

    cur[k] = alpha * kept[k-1] * g_open + (1 - alpha) * intrinsic
    alpha=1: fully SLB-inherited compounding drift (worst case; what the two
    T2V logs show).  alpha<1 with an intrinsic target models a scene whose
    natural spectrum simply differs from chunk-1 (the benign I2V case).
    """
    start = cur0.clone() if cur0 is not None else REF_E.clone()
    cor, unc, ema = start.clone(), start.clone(), None
    cor_hist, unc_hist, gain_hist = [], [], []
    for _ in range(chunks):
        cor = alpha * cor * g_open + (1 - alpha) * (intrinsic if intrinsic is not None else REF_E)
        unc = alpha * unc * g_open + (1 - alpha) * (intrinsic if intrinsic is not None else REF_E)
        kept, ema, gains = anchor_step(cor, REF_E, ema)
        cor = kept
        cor_hist.append((cor[-4:] / REF_E[-4:]).mean().item())
        unc_hist.append((unc[-4:] / REF_E[-4:]).mean().item())
        gain_hist.append(gains)
    return cor_hist, unc_hist, gain_hist


print("═══ Part 1: per-bin spectral anchor (replaying measured drift) ═══")

# Scenario A — 31-lf T2V drift: +12%/chunk on top-4 bins, +6% on bins 11-12.
g_a = torch.ones(16)
g_a[-4:] = 1.12
g_a[11:13] = 1.06
cor_a, unc_a, gains_a = run_loop(g_a)
check("A: uncorrected replicates the 31-lf failure (2.0-3.0x by ch10)",
      2.0 <= unc_a[-1] <= 3.0, f"uncorrected top4 ratio ch10 = {unc_a[-1]:.2f}")
check("A: corrected stays bounded (<=1.25x every chunk)",
      max(cor_a) <= 1.25, f"max corrected ratio = {max(cor_a):.3f}")
check("A: bounded limit cycle only (late chunk-to-chunk step < 0.05)",
      max(abs(cor_a[i] - cor_a[i - 1]) for i in range(5, len(cor_a))) < 0.05,
      f"late steps max = {max(abs(cor_a[i] - cor_a[i-1]) for i in range(5, len(cor_a))):.4f}")

# Scenario B — 12-lf T2V drift: +3.5%/chunk on top-4 bins.
g_b = torch.ones(16)
g_b[-4:] = 1.035
cor_b, unc_b, _ = run_loop(g_b)
check("B: uncorrected replicates the 12-lf drift (1.25-1.55x by ch10)",
      1.25 <= unc_b[-1] <= 1.55, f"uncorrected = {unc_b[-1]:.2f}")
check("B: corrected bounded (<=1.20x every chunk)",
      max(cor_b) <= 1.20, f"max = {max(cor_b):.3f}")

# Scenario C — I2V benign: the scene's NATURAL spectrum sits at 0.87x ref
# (chunk-1 ran hot; later chunks are intrinsically quieter in those bins, no
# compounding).  The anchor must converge toward not correcting.
cor_c, _, gains_c = run_loop(torch.ones(16), cur0=REF_E * 0.87,
                             alpha=0.3, intrinsic=REF_E * 0.87)
late_c = torch.stack(gains_c[5:])
g5 = (gains_c[4] - 1.0).abs().max().item()
g10 = (gains_c[-1] - 1.0).abs().max().item()
check("C: benign below-ref spectrum not fought (late gains within 6% of 1.0)",
      ((late_c - 1.0).abs() < 0.06).all().item(),
      f"late gain range [{late_c.min():.3f}, {late_c.max():.3f}]")
check("C: correction is converging away, not persisting (|g-1| ch10 < ch5)",
      g10 < g5, f"|g-1|: ch5 {g5:.3f} -> ch10 {g10:.3f}")

# Scenario D — extreme +30%/chunk stress (2.5x beyond anything measured).
# The hard <=cap guarantee holds while g_open * clamp_min * renorm < 1
# (g_open up to ~1.3); beyond that the anchor degrades GRACEFULLY (slow
# residual growth) instead of exploding.
g_d = torch.ones(16)
g_d[-4:] = 1.30
cor_d, unc_d, _ = run_loop(g_d)
check("D: extreme +30%/chunk degrades gracefully (<=2.1x while uncorrected >10x)",
      max(cor_d) <= 2.1 and unc_d[-1] > 10.0,
      f"corrected max = {max(cor_d):.3f}, uncorrected ch10 = {unc_d[-1]:.1f}")

# Scenario E — INTENTIONAL in-band timbre change (+10% on top bins, within
# the ±15% band, fully carried by the SLB): must be permitted.
cur0_e = REF_E.clone()
cur0_e[-4:] *= 1.10
cor_e, _, gains_e = run_loop(torch.ones(16), cur0=cur0_e)
late_e = torch.stack(gains_e[5:])
check("E: intentional in-band (+10%) change is permitted (late gains ~1.0)",
      ((late_e - 1.0).abs() < 0.03).all().item(),
      f"late gain range [{late_e.min():.3f}, {late_e.max():.3f}]")

# RMS preservation — exact tensor-level check of the production operation.
x = torch.randn(1, 8, 102, 16)
gains = torch.empty(16).uniform_(0.75, 1.25)
rms_pre = x.pow(2).mean().sqrt()
shaped = x * gains.view(1, 1, 1, -1)
y = shaped * (rms_pre / shaped.pow(2).mean().sqrt())
check("RMS preserved exactly by the renorm (rel err < 1e-6)",
      abs(y.pow(2).mean().sqrt().item() - rms_pre.item()) / rms_pre.item() < 1e-6)

# ───────────────────────────────────────────────────────────────────────────
# Part 2 — audio noise envelope dither
# ───────────────────────────────────────────────────────────────────────────

print("═══ Part 2: envelope dither statistics ═══")

TOTAL_AF = 1013                    # from the 12-lf run: 95 + 9x102
BOUNDS = [0, 95, 197, 299, 401, 503, 605, 707, 809, 911, 1013]
SEED = 1125899906842624
DITHER = 0.15


def smooth_reflect(x: torch.Tensor, sig: float = 24.0) -> torch.Tensor:
    """Reflect-padded Gaussian smoothing — exact production code."""
    ks = (int(6 * sig)) | 1
    t = torch.arange(ks, dtype=torch.float32) - ks // 2
    k = torch.exp(-0.5 * (t / sig) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    xp = torch.nn.functional.pad(x.view(1, 1, -1), (ks // 2, ks // 2), mode="reflect")
    return torch.nn.functional.conv1d(xp, k).flatten()


def build_env(seed: int, total_af: int, dither: float) -> torch.Tensor:
    """Exact production code from nodes.py (envelope dither block)."""
    g = torch.Generator(device="cpu").manual_seed((seed + 3) % (2 ** 63))
    white = torch.randn(total_af, generator=g)
    s = smooth_reflect(white)
    s = (s - s.mean()) / s.std().clamp(min=1e-6)
    env = (1.0 + dither * s).clamp(0.6, 1.4)
    return env / env.pow(2).mean().sqrt()


env = build_env(SEED, TOTAL_AF, DITHER)

check("global noise RMS preserved (mean(env^2) = 1)",
      abs(env.pow(2).mean().item() - 1.0) < 1e-6)
check("determinism: same seed -> bit-identical envelope",
      torch.equal(env, build_env(SEED, TOTAL_AF, DITHER)))
check("bounded: env within [0.55, 1.45] after renorm",
      0.55 <= env.min().item() and env.max().item() <= 1.45,
      f"range [{env.min():.3f}, {env.max():.3f}]")
max_step = (env[1:] - env[:-1]).abs().max().item()
check("smooth: max per-frame step < 0.02 (no discontinuity at any seam)",
      max_step < 0.02, f"max step = {max_step:.4f}")

# No SYSTEMATIC cross-chunk repetition.  A smooth envelope has few degrees of
# freedom per 102-af window, so individual adjacent-pair correlations scatter
# widely (that IS organic variety); the metronome signature would be
# consistently POSITIVE correlations (same gesture every chunk, like the
# measured env_corr 0.84-0.98).  Test the signed mean, not the magnitudes.
profiles = []
common = 90
for i in range(len(BOUNDS) - 1):
    seg = env[BOUNDS[i]:BOUNDS[i + 1]]
    idx = torch.linspace(0, len(seg) - 1, common).long()
    p = seg[idx]
    profiles.append((p - p.mean()) / p.std().clamp(min=1e-6))
corrs = [float((profiles[i] * profiles[i + 1]).mean()) for i in range(len(profiles) - 1)]
mean_corr = sum(corrs) / len(corrs)
check("no systematic gesture repetition (signed mean corr in [-0.4, 0.4])",
      -0.4 <= mean_corr <= 0.4,
      f"signed mean = {mean_corr:+.3f}, pairs = {[f'{c:+.2f}' for c in corrs]}")

# Peak-position scatter — the actual production metronome criterion
# (phase-lock check: mode ±1 covering >=70% of chunks = WARN).  The dither's
# per-chunk energy-suggestion peak must NOT be window-locked.
peaks = []
for i in range(len(BOUNDS) - 1):
    seg = env[BOUNDS[i]:BOUNDS[i + 1]]
    peaks.append(round(100 * seg.argmax().item() / len(seg)))   # % of window
mode = max(set(peaks), key=peaks.count)
near = sum(1 for p in peaks if abs(p - mode) <= 10)             # ±10% of window
check("suggestion peak position scatters across chunks (mode ±10% covers <70%)",
      near < 0.7 * len(peaks),
      f"peak %-positions = {peaks}, mode={mode} covers {near}/{len(peaks)}")

# Suggestion strength: the dither envelope must sit far above the i.i.d.
# noise field's own smoothed-envelope fluctuation (8 ch x 16 bins = 128
# samples/frame), or the model receives no usable per-chunk difference.
iid = torch.randn(8, TOTAL_AF, 16)
e_s = smooth_reflect(iid.abs().mean(dim=(0, 2)))
floor_rel = (e_s.std() / e_s.mean()).item()
env_rel = (env.std() / env.mean()).item()
check("dither suggestion >> i.i.d. envelope floor (ratio > 5)",
      env_rel / max(floor_rel, 1e-9) > 5,
      f"dither rel-amp {env_rel:.3f} vs iid floor {floor_rel:.4f} "
      f"-> {env_rel / max(floor_rel, 1e-9):.1f}x")

# Marginals: enveloped noise stays Gaussian per frame with std = env[t].
# 256 channels x 16 bins = 4096 samples -> sampling error ~1.1%.
field = torch.randn(256, TOTAL_AF, 16) * env.view(1, -1, 1)
hi, lo = env.argmax().item(), env.argmin().item()
for name, fr in (("env-max", hi), ("env-min", lo)):
    fs = field[:, fr, :].std().item()
    check(f"per-frame std tracks the envelope at {name} (within 5%)",
          abs(fs - env[fr].item()) / env[fr].item() < 0.05,
          f"frame std {fs:.3f} vs env {env[fr].item():.3f}")

print()
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
    raise SystemExit(1)
print("RESULT: ALL CHECKS PASSED")
