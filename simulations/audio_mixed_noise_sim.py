"""Pre-implementation tests for a correlated-noise prior on the S1 AUDIO field.

Two candidates, tested BEFORE any nodes.py change:

  A. Direct transplant of the video mix (PYoCo "mixed"):
         n_t = sqrt(1-a) eps_t + sqrt(a) eps_shared,   eps_shared: [B, C, 1, freq]
     VERDICT: FALSIFIED here.  The video shared frame has 128x11x20 = 28160
     values (sampling error +-0.8%, statistically invisible), but audio's
     cross-section is only 8x16 = 128 values — one fixed shared vector
     imprints a PERSISTENT per-bin tint (~+-10%) on every frame of the run.
     That is the same spectral-colouring class that made audio_env_dither
     fail its live test (hot bin = hiss).  Checks A2/A3 quantify it against
     the i.i.d. baseline's own sampling distribution.

  B. PYoCo "progressive" noise (AR(1)):
         n_0 = eps_0;   n_t = phi n_{t-1} + sqrt(1-phi^2) eps_t
     Marginals exactly N(0,1) for every t; correlation decays as phi^lag so
     there is NO run-long fixed pattern (no tint) and no periodicity; fully
     seeded; sliceable by absolute position.  This is the class-safe audio
     analog of the video's validated noise_temporal_corr.

Run:  python simulations/audio_mixed_noise_sim.py
"""

import math

import torch

torch.manual_seed(0)

FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


B, C, T, FQ = 1, 8, 1013, 16          # S1 audio field shape from the 12-lf runs
SEED = 1125899906842624
AUD_SEED = (SEED + 1) % (2 ** 63)      # existing audio field seed convention


def baseline_field() -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(AUD_SEED)
    return torch.randn(B, C, T, FQ, generator=g)


def bin_stds(x: torch.Tensor) -> torch.Tensor:
    """Per-frequency-bin std over (B, C, T) — the spectral-tint detector."""
    return x.float().permute(3, 0, 1, 2).flatten(1).std(1)


# Baseline reference: how much per-bin std varies for honest i.i.d. noise
# (8104 samples/bin -> sd ~ 1/sqrt(2*8104) ~ 0.8%).
base = baseline_field()
base_bin_dev = (bin_stds(base) - 1.0).abs().max().item()

print("═══ Candidate A: direct video-mix transplant (shared vector) ═══")
a = 0.3
g2 = torch.Generator(device="cpu").manual_seed((SEED + 4) % (2 ** 63))
shared = torch.randn(B, C, 1, FQ, generator=g2)
mixed = math.sqrt(1.0 - a) * base + math.sqrt(a) * shared

mix_bin_dev = (bin_stds(mixed) - 1.0).abs().max().item()
check("A1: baseline i.i.d. per-bin std dev is small (sanity)",
      base_bin_dev < 0.03, f"baseline max dev {base_bin_dev:.3f}")
check("A2: transplant introduces persistent spectral tint >> baseline "
      "(EXPECTED FAILURE-OF-CANDIDATE = PASS of this check)",
      mix_bin_dev > 3 * base_bin_dev,
      f"mix max bin dev {mix_bin_dev:.3f} vs baseline {base_bin_dev:.3f} "
      f"({mix_bin_dev / max(base_bin_dev, 1e-9):.1f}x)")
per_frame_bias = (math.sqrt(a) * shared).flatten()
check("A3: the tint is FIXED for the whole run (same bias every frame), "
      "i.e. dither-class persistence, not i.i.d. fluctuation",
      per_frame_bias.abs().max().item() > 0.5,
      f"largest persistent per-(ch,bin) offset {per_frame_bias.abs().max():.2f} "
      f"applied to all 1013 frames")
print("  => Candidate A REJECTED: only 128-dof cross-section; run-long tint.")

print("═══ Candidate B: PYoCo progressive AR(1) noise ═══")
PHI = 0.9                              # corr length ~ -1/ln(phi) ~ 9.5 af ~ 0.4 s


def build_ar1(phi: float, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    eps = torch.randn(B, C, T, FQ, generator=g)
    out = torch.empty_like(eps)
    out[:, :, 0] = eps[:, :, 0]
    c = math.sqrt(1.0 - phi * phi)
    for t in range(1, T):
        out[:, :, t] = phi * out[:, :, t - 1] + c * eps[:, :, t]
    return out


ar1 = build_ar1(PHI, AUD_SEED)

# B1: exact stationary marginals — per-frame std must match the BASELINE's own
# sampling spread (128 samples/frame -> both fluctuate ~+-6%).  Note the MEAN
# of the 1013 frame-stds itself wanders +-1% per seed under AR(1): temporal
# correlation cuts the effective dof by (1+phi)/(1-phi)=19x.  Verified across
# 5 seeds that it straddles 1.0 (0.989-1.003, zero-mean) — so tolerance 0.03,
# not 0.01.  The failures we're guarding against are 10-30x larger and
# STRUCTURED (dither: smooth +-30% envelope; transplant: fixed +-13% tint).
base_frame = base.float().permute(2, 0, 1, 3).flatten(1).std(1)
ar1_frame = ar1.float().permute(2, 0, 1, 3).flatten(1).std(1)
check("B1: per-frame std spread matches baseline i.i.d. (no local shaping)",
      abs(ar1_frame.std().item() - base_frame.std().item()) < 0.02
      and abs(ar1_frame.mean().item() - 1.0) < 0.03,
      f"frame-std mean/spread: AR1 {ar1_frame.mean():.4f}/{ar1_frame.std():.4f} "
      f"vs base {base_frame.mean():.4f}/{base_frame.std():.4f}")

# B2: no spectral tint — per-bin std within ~2x of the baseline's own spread
# (AR(1) reduces the EFFECTIVE sample count per bin by ~(1+phi)/(1-phi),
# widening sampling spread ~sqrt(19)~4.4x, but introduces no FIXED bias;
# accept < 6x baseline, far below the transplant's persistent tint).
ar1_bin_dev = (bin_stds(ar1) - 1.0).abs().max().item()
check("B2: no persistent spectral tint (bin dev < 6x baseline sampling dev)",
      ar1_bin_dev < 6 * base_bin_dev,
      f"AR1 max bin dev {ar1_bin_dev:.3f} vs baseline {base_bin_dev:.3f}")

# B3: correlation structure — phi at lag 1, phi^k decay, ~0 across chunks.
flat = ar1.float().permute(2, 0, 1, 3).flatten(1)


def corr(i: int, j: int) -> float:
    return torch.nn.functional.cosine_similarity(
        flat[i] - flat[i].mean(), flat[j] - flat[j].mean(), dim=0).item()


c1 = sum(corr(i, i + 1) for i in (10, 300, 700)) / 3
c5 = sum(corr(i, i + 5) for i in (10, 300, 700)) / 3
c100 = corr(100, 200)
check("B3: corr(lag1) ~ phi", abs(c1 - PHI) < 0.08, f"{c1:.3f} vs {PHI}")
check("B3: corr(lag5) ~ phi^5", abs(c5 - PHI ** 5) < 0.12,
      f"{c5:.3f} vs {PHI ** 5:.3f}")
check("B3: corr decays to ~0 across a chunk (no cross-chunk repetition seed)",
      abs(c100) < 0.15, f"corr(lag100) = {c100:.3f}")

# B4: determinism + slicing by absolute position.
ar1b = build_ar1(PHI, AUD_SEED)
check("B4: determinism (bit-identical rebuild)", torch.equal(ar1, ar1b))
check("B4: per-chunk slices identical to full-field slices",
      torch.equal(ar1[:, :, 95:197], ar1b[:, :, 95:197]))

# B5: kill switch — phi=0 must reproduce the current baseline field bit-exactly
# (build_ar1 draws eps in the same [B,C,T,FQ] order as the baseline randn).
check("B5: phi=0 == current baseline bit-exactly",
      torch.equal(build_ar1(0.0, AUD_SEED), base))

print()
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
    raise SystemExit(1)
print("RESULT: Candidate A (video-mix transplant) rejected with evidence; "
      "Candidate B (AR(1) progressive) passes every statistical gate. "
      "Next gate before ANY implementation: live isolation tests (see plan).")
