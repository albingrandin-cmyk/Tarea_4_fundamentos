from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import io, color
from skimage.restoration import denoise_bilateral

try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


# Set to True if you want interactive windows in addition to saved PNG files
SHOW_PLOTS = False
OUTPUT_DIR = Path("results_p2")


# ===========================================================================
# 1. Image Loading, Global Normalization & Luminance Extraction
# ===========================================================================

def resolve_image_path(preferred_path: str) -> Path:
    """Finds the HDR image either in the preferred folder or current working directory."""
    p = Path(preferred_path)
    if p.exists():
        return p
    fallback = Path(p.name)
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find image '{preferred_path}' or '{fallback}'.")


def load_normalized_rgb(image_path: Path) -> np.ndarray:
    """
    Loads an HDR/LDR RGB image, converts it to float64, replaces any non-finite
    values, clamps negative radiance values to 0, and normalizes globally by the
    maximum value across all pixels and channels:
        I_norm(x, y, c) = I(x, y, c) / max_{x, y, c} I(x, y, c)  in [0, 1].
    Global normalization preserves exact R:G:B chromaticity ratios.
    """
    if HAS_IMAGEIO:
        img = iio.imread(image_path)
    else:
        img = io.imread(image_path)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if np.issubdtype(img.dtype, np.integer):
        rgb = img.astype(np.float64) / np.iinfo(img.dtype).max
    else:
        rgb = img.astype(np.float64)
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
        rgb = np.maximum(rgb, 0.0)
        max_val = float(np.max(rgb))
        if max_val > 0.0:
            rgb /= max_val

    return np.clip(rgb, 0.0, 1.0)


def extract_luminance(rgb: np.ndarray) -> np.ndarray:
    """
    Computes ITU-R BT.709 relative luminance:
        Y = 0.2125 * R + 0.7154 * G + 0.0721 * B  in [0, 1].
    """
    return np.clip(color.rgb2gray(rgb), 0.0, 1.0)


def log_range_and_clipping(
    label: str,
    arr_before_clip: np.ndarray,
    arr_after_clip: np.ndarray,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Reports the pre-clip dynamic range [min, max] and the exact percentage of
    values clipped below 0.0 or above 1.0.
    """
    low_clip_pct = 100.0 * float(np.mean(arr_before_clip < 0.0))
    high_clip_pct = 100.0 * float(np.mean(arr_before_clip > 1.0))
    total_clip_pct = low_clip_pct + high_clip_pct

    if verbose:
        print(
            f"  [{label}] Pre-clip range: [{arr_before_clip.min():.4f}, {arr_before_clip.max():.4f}] | "
            f"Clipped: {total_clip_pct:.3f}% (low: {low_clip_pct:.3f}%, high: {high_clip_pct:.3f}%)"
        )

    return {
        "min_raw": float(arr_before_clip.min()),
        "max_raw": float(arr_before_clip.max()),
        "low_clip_pct": low_clip_pct,
        "high_clip_pct": high_clip_pct,
        "clipped_pct": total_clip_pct,
    }


def compute_color_metrics(rgb_orig: np.ndarray, rgb_rec: np.ndarray) -> dict[str, float]:
    """
    Quantifies color behavior before and after tone mapping:
      (i)  Mean HSV saturation S before (orig_sat) and after (rec_sat) processing.
      (ii) Mean circular hue shift (in degrees, [0, 180]) evaluated on chromatic,
           non-black pixels (S_orig > 0.05 and V_orig > 1e-4) where hue is well-defined.
    """
    hsv_orig = color.rgb2hsv(np.clip(rgb_orig, 0.0, 1.0))
    hsv_rec = color.rgb2hsv(np.clip(rgb_rec, 0.0, 1.0))

    orig_sat = float(np.mean(hsv_orig[:, :, 1]))
    rec_sat = float(np.mean(hsv_rec[:, :, 1]))

    valid = (hsv_orig[:, :, 1] > 0.05) & (hsv_orig[:, :, 2] > 1e-4)
    if np.any(valid):
        dh = np.abs(hsv_rec[:, :, 0][valid] - hsv_orig[:, :, 0][valid])
        dh_circ = np.minimum(dh, 1.0 - dh) * 360.0
        mean_hue_shift = float(np.mean(dh_circ))
    else:
        mean_hue_shift = 0.0

    return {
        "orig_sat": orig_sat,
        "rec_sat": rec_sat,
        "delta_sat": rec_sat - orig_sat,
        "mean_hue_deg": mean_hue_shift,
    }


# ===========================================================================
# 2. HDR Compression Method A: Homomorphic Filtering (Frequency Domain)
# ===========================================================================

def homomorphic_filter_channel(
    channel: np.ndarray,
    gamma_l: float = 0.35,
    gamma_h: float = 1.60,
    c: float = 1.0,
    d0: float = 30.0,
    eps: float = 1e-6,
    p_low: float = 0.5,
    p_high: float = 99.5,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Homomorphic dynamic range compression on a single 2D channel:
      1. Log transform with eps guard: z = ln(max(I, 0) + eps).
      2. Symmetric reflect padding (H//2, W//2) to prevent circular FFT wrap-around.
      3. Gaussian high-pass filter:
             H(u, v) = (gamma_H - gamma_L) * (1 - exp(-c * D^2(u, v) / D_0^2)) + gamma_L.
      4. Inverse FFT, crop, and exponential inversion: out_raw = max(exp(g) - eps, 0).
      5. Robust percentile rescaling using [0.5%, 99.5%] percentiles followed by
         explicit clipping to [0, 1].
    """
    h, w = channel.shape
    z = np.log(np.maximum(channel, 0.0) + eps)

    pad_h, pad_w = h // 2, w // 2
    z_pad = np.pad(z, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    hp, wp = z_pad.shape

    u = np.arange(hp) - hp / 2.0
    v = np.arange(wp) - wp / 2.0
    uu, vv = np.meshgrid(u, v, indexing="ij")
    d_sq = uu ** 2 + vv ** 2

    h_uv = (gamma_h - gamma_l) * (1.0 - np.exp(-c * (d_sq / (d0 ** 2 + eps)))) + gamma_l

    z_fft = np.fft.fftshift(np.fft.fft2(z_pad))
    g_pad = np.real(np.fft.ifft2(np.fft.ifftshift(z_fft * h_uv)))
    g = g_pad[pad_h : pad_h + h, pad_w : pad_w + w]

    out_raw = np.maximum(np.exp(g) - eps, 0.0)

    lo = float(np.percentile(out_raw, p_low))
    hi = float(np.percentile(out_raw, p_high))
    out_norm = (out_raw - lo) / (hi - lo + eps)
    out_clipped = np.clip(out_norm, 0.0, 1.0)

    stats = log_range_and_clipping("Homomorphic channel", out_norm, out_clipped, verbose=verbose)
    return out_clipped, stats


def homomorphic_rgb_independent(rgb: np.ndarray, **homo_params) -> np.ndarray:
    """Applies the exact same homomorphic filter independently to R, G, and B."""
    channels = [
        homomorphic_filter_channel(rgb[:, :, ch], verbose=False, **homo_params)[0]
        for ch in range(3)
    ]
    return np.stack(channels, axis=-1)


# ===========================================================================
# 3. HDR Compression Method B: Durand & Dorsey (2002) Bilateral Tone Mapping
# ===========================================================================

def durand_tone_mapping_channel(
    channel: np.ndarray,
    sigma_spatial: float = 12.0,
    sigma_range_norm: float = 0.12,
    target_contrast: float = 1.8,
    eps: float = 1e-6,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Durand & Dorsey (2002) edge-preserving base/detail tone mapping:
      1. Base-10 log-luminance with eps guard: L = log10(max(I, 0) + eps).
      2. Linear normalization of L to [0, 1] for scale-consistent bilateral filtering.
      3. Edge-preserving bilateral filter yields base layer B; detail layer D = L - B.
      4. Compresses base contrast to target_contrast (1.8 decades):
             gamma_c = target_contrast / (max(B) - min(B) + eps),
             L' = gamma_c * (B - max(B)) + D.
      5. Linear recovery Y' = 10^(L') with explicit clipping to [0, 1].
    """
    log_lum = np.log10(np.maximum(channel, 0.0) + eps)

    l_min = float(np.min(log_lum))
    l_max = float(np.max(log_lum))
    l_span = l_max - l_min + eps
    log_lum_norm = (log_lum - l_min) / l_span

    base_norm = denoise_bilateral(
        log_lum_norm,
        sigma_color=sigma_range_norm,
        sigma_spatial=sigma_spatial,
        channel_axis=None,
    )
    base = base_norm * l_span + l_min
    detail = log_lum - base

    b_max = float(np.max(base))
    b_min = float(np.min(base))
    compression_factor = target_contrast / (b_max - b_min + eps)

    log_compressed = compression_factor * (base - b_max) + detail
    out_raw = np.power(10.0, log_compressed)
    out_clipped = np.clip(out_raw, 0.0, 1.0)

    stats = log_range_and_clipping("Durand channel", out_raw, out_clipped, verbose=verbose)
    return out_clipped, stats


def durand_rgb_independent(rgb: np.ndarray, **durand_params) -> np.ndarray:
    """Applies the exact same Durand bilateral tone mapping independently to R, G, and B."""
    channels = [
        durand_tone_mapping_channel(rgb[:, :, ch], verbose=False, **durand_params)[0]
        for ch in range(3)
    ]
    return np.stack(channels, axis=-1)


# ===========================================================================
# 4. Color Reconstruction Strategy 1: Chromaticity Ratio with Parameter beta
# ===========================================================================

def reconstruct_color_chromaticity_ratio(
    rgb: np.ndarray,
    y_orig: np.ndarray,
    y_comp: np.ndarray,
    beta: float = 1.0,
    eps: float = 1e-6,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Strategy 1 — Luminance / Chromaticity Ratio:
        C' = Y' * ( max(C / (Y + eps), 0) )^beta,   for C in {R, G, B}.
      - Division-by-zero protection via denominator Y + eps (eps = 1e-6).
      - Out-of-range values above 1.0 (when Y' * ratio^beta > 1) are explicitly
        clipped to [0, 1] using np.clip(rgb_raw, 0.0, 1.0).
    """
    denom = np.maximum(y_orig, 0.0) + eps
    ratios = np.maximum(rgb / denom[:, :, np.newaxis], 0.0)
    rgb_raw = y_comp[:, :, np.newaxis] * np.power(ratios, beta)
    rgb_clipped = np.clip(rgb_raw, 0.0, 1.0)

    stats = log_range_and_clipping(
        f"Chromaticity ratio (beta={beta:.2f})", rgb_raw, rgb_clipped, verbose=verbose
    )
    return rgb_clipped, stats


# ===========================================================================
# 5. Color Reconstruction Strategy 2: Explicit Saturation Adjustment in HSV
# ===========================================================================

def reconstruct_color_hsv_saturation(
    rgb: np.ndarray,
    y_orig: np.ndarray,
    y_comp: np.ndarray,
    sat_gain: float = 1.0,
    eps: float = 1e-6,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Strategy 2 — Explicit Saturation Scaling in HSV Color Space:
      1. Base color reconstruction at beta = 1.0:
             I_base = clip(Y' * (I_rgb / (Y + eps)), 0.0, 1.0).
      2. Convert I_base from RGB to HSV in [0, 1]^3.
      3. Scale saturation S linearly by k_S (k_S = 1.0 is the identity mapping)
         and explicitly clip out-of-range saturation values to [0.0, 1.0]:
             S' = clip(k_S * S, 0.0, 1.0).
      4. Convert (H, S', V) back to RGB and explicitly clip to [0.0, 1.0].
    """
    rgb_base, base_stats = reconstruct_color_chromaticity_ratio(
        rgb, y_orig, y_comp, beta=1.0, eps=eps, verbose=False
    )
    hsv = color.rgb2hsv(rgb_base)

    s_raw = sat_gain * hsv[:, :, 1]
    s_clipped = np.clip(s_raw, 0.0, 1.0)
    sat_high_clip_pct = 100.0 * float(np.mean(s_raw > 1.0))

    hsv_mod = hsv.copy()
    hsv_mod[:, :, 1] = s_clipped

    rgb_raw = color.hsv2rgb(hsv_mod)
    rgb_clipped = np.clip(rgb_raw, 0.0, 1.0)

    stats = log_range_and_clipping(
        f"HSV saturation (k_S={sat_gain:.2f})", rgb_raw, rgb_clipped, verbose=verbose
    )
    stats["sat_high_clip_pct"] = sat_high_clip_pct
    stats["base_rgb_clip_pct"] = base_stats["clipped_pct"]
    if verbose:
        print(f"    -> HSV Saturation S > 1.0 clipped in {sat_high_clip_pct:.3f}% of pixels.")
    return rgb_clipped, stats


# ===========================================================================
# 6. Experiment 1: Chromaticity Ratio Across beta Values (Fixed HDR Params)
# ===========================================================================

def run_experiment_1_beta(
    image_name: str,
    rgb: np.ndarray,
    y_orig: np.ndarray,
    y_homo: np.ndarray,
    rgb_homo_indep: np.ndarray,
    y_durand: np.ndarray,
    rgb_durand_indep: np.ndarray,
    homo_params: dict[str, float],
    durand_params: dict[str, float],
    betas: list[float],
    crop_color_rel: tuple[float, float, float, float],
    crop_edge_rel: tuple[float, float, float, float],
) -> tuple[dict[str, dict[float, np.ndarray]], list[dict]]:
    """
    Experiment 1: Evaluates chromaticity ratio reconstruction for beta = 1.0,
    beta < 1.0 (reduced saturation), and beta > 1.0 (increased saturation).
    """
    orig_sat = float(np.mean(color.rgb2hsv(rgb)[:, :, 1]))
    print(f"\n=== Experiment 1 (Chromaticity Ratio beta): {image_name} | Orig Mean Sat S = {orig_sat:.4f} ===")
    print(
        f"{'HDR Method':<12} | {'Variant':<16} | {'Mean Sat S':<10} | "
        f"{'Hue Shift':<10} | {'Pre-clip Max':<12} | {'RGB Clip (%)':<12}"
    )
    print("-" * 86)

    methods = [
        ("Homomorphic", y_homo, rgb_homo_indep),
        ("Durand", y_durand, rgb_durand_indep),
    ]

    reconstructions: dict[str, dict[float, np.ndarray]] = {"Homomorphic": {}, "Durand": {}}
    clip_pcts: dict[str, dict[float, float]] = {"Homomorphic": {}, "Durand": {}}
    records: list[dict] = []

    for method_name, y_comp, rgb_indep in methods:
        indep_m = compute_color_metrics(rgb, rgb_indep)
        print(
            f"{method_name:<12} | {'Indep. RGB':<16} | {indep_m['rec_sat']:>10.4f} | "
            f"{indep_m['mean_hue_deg']:>9.2f}° | {'---':>12} | {'---':>12}"
        )
        records.append({
            "image": image_name,
            "method": method_name,
            "strategy": "Independent RGB",
            "param": "N/A",
            "orig_sat": indep_m["orig_sat"],
            "rec_sat": indep_m["rec_sat"],
            "hue_deg": indep_m["mean_hue_deg"],
            "clip_info": "N/A",
        })

        for beta in betas:
            rgb_rec, stats = reconstruct_color_chromaticity_ratio(
                rgb, y_orig, y_comp, beta=beta, eps=homo_params["eps"], verbose=False
            )
            reconstructions[method_name][beta] = rgb_rec
            clip_pcts[method_name][beta] = stats["clipped_pct"]

            m = compute_color_metrics(rgb, rgb_rec)
            print(
                f"{method_name:<12} | {f'beta = {beta:.1f}':<16} | {m['rec_sat']:>10.4f} | "
                f"{m['mean_hue_deg']:>9.2f}° | {stats['max_raw']:>12.4f} | {stats['clipped_pct']:>11.3f}%"
            )
            records.append({
                "image": image_name,
                "method": method_name,
                "strategy": "Chromaticity Ratio",
                "param": f"beta = {beta:.1f}",
                "orig_sat": m["orig_sat"],
                "rec_sat": m["rec_sat"],
                "hue_deg": m["mean_hue_deg"],
                "clip_info": f"RGB: {stats['clipped_pct']:.3f}%",
            })
        print("-" * 86)

    param_subtitle = (
        f"Homomorphic: $\\gamma_L={homo_params['gamma_l']}, \\gamma_H={homo_params['gamma_h']}, "
        f"c={homo_params['c']}, D_0={homo_params['d0']}$  |  "
        f"Durand: $\\sigma_s={durand_params['sigma_spatial']}, \\sigma_r={durand_params['sigma_range_norm']}, "
        f"\\Delta L_{{target}}={durand_params['target_contrast']}$"
    )

    # Full-image figure
    ncols = 2 + len(betas)
    fig, axes = plt.subplots(2, ncols, figsize=(3.4 * ncols, 7.6))
    for row, (method_name, y_comp, rgb_indep) in enumerate(methods):
        axes[row, 0].imshow(y_comp, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 0].set_title(f"{method_name}\nCompressed $Y'$ [0, 1]", fontsize=9.0)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(np.clip(rgb_indep, 0.0, 1.0))
        axes[row, 1].set_title(f"{method_name}\nIndependent RGB", fontsize=9.0)
        axes[row, 1].axis("off")

        for idx, beta in enumerate(betas):
            col = 2 + idx
            role = "reduced" if beta < 1.0 else ("baseline" if np.isclose(beta, 1.0) else "boosted")
            axes[row, col].imshow(reconstructions[method_name][beta])
            axes[row, col].set_title(
                f"{method_name} ($\\beta = {beta:.1f}$, {role})\nRGB clip: {clip_pcts[method_name][beta]:.2f}%",
                fontsize=9.0,
            )
            axes[row, col].axis("off")

    fig.suptitle(
        f"Experiment 1: Luminance-Ratio Color Reconstruction ($\\beta \\in {betas}$) — {image_name}\n{param_subtitle}",
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.03, wspace=0.06, hspace=0.18)
    fig.savefig(OUTPUT_DIR / f"exp1_beta_{image_name}.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    # Zoomed crops figure
    h, w, _ = rgb.shape
    r0_c, r1_c = int(crop_color_rel[0] * h), int(crop_color_rel[1] * h)
    c0_c, c1_c = int(crop_color_rel[2] * w), int(crop_color_rel[3] * w)
    r0_e, r1_e = int(crop_edge_rel[0] * h), int(crop_edge_rel[1] * h)
    c0_e, c1_e = int(crop_edge_rel[2] * w), int(crop_edge_rel[3] * w)

    fig_z, axes_z = plt.subplots(4, 1 + len(betas), figsize=(3.4 * (1 + len(betas)), 12.5))
    row_specs = [
        ("Homomorphic (Intense Color)", "Homomorphic", rgb_homo_indep, (r0_c, r1_c, c0_c, c1_c)),
        ("Homomorphic (Strong Edge)", "Homomorphic", rgb_homo_indep, (r0_e, r1_e, c0_e, c1_e)),
        ("Durand (Intense Color)", "Durand", rgb_durand_indep, (r0_c, r1_c, c0_c, c1_c)),
        ("Durand (Strong Edge)", "Durand", rgb_durand_indep, (r0_e, r1_e, c0_e, c1_e)),
    ]
    for row_idx, (row_label, method_name, rgb_indep, (r0, r1, c0, c1)) in enumerate(row_specs):
        axes_z[row_idx, 0].imshow(np.clip(rgb_indep[r0:r1, c0:c1], 0.0, 1.0))
        axes_z[row_idx, 0].set_title(f"{row_label}\nIndependent RGB", fontsize=8.5)
        axes_z[row_idx, 0].axis("off")

        for b_idx, beta in enumerate(betas):
            axes_z[row_idx, 1 + b_idx].imshow(reconstructions[method_name][beta][r0:r1, c0:c1])
            axes_z[row_idx, 1 + b_idx].set_title(f"{row_label}\n$\\beta = {beta:.1f}$", fontsize=8.5)
            axes_z[row_idx, 1 + b_idx].axis("off")

    fig_z.suptitle(
        f"Experiment 1 Zooms: Intense Color vs. Illumination Edge — {image_name}\n{param_subtitle}",
        fontsize=10.5,
    )
    fig_z.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02, wspace=0.06, hspace=0.22)
    fig_z.savefig(OUTPUT_DIR / f"exp1_beta_zooms_{image_name}.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig_z)

    return reconstructions, records


# ===========================================================================
# 7. Experiment 2: Explicit HSV Saturation Strategy & Comparison vs. beta
# ===========================================================================

def run_experiment_2_hsv(
    image_name: str,
    rgb: np.ndarray,
    y_orig: np.ndarray,
    y_homo: np.ndarray,
    y_durand: np.ndarray,
    beta_recs: dict[str, dict[float, np.ndarray]],
    homo_params: dict[str, float],
    durand_params: dict[str, float],
    sat_gains: list[float],
    crop_color_rel: tuple[float, float, float, float],
    crop_edge_rel: tuple[float, float, float, float],
    display_gamma: float = 0.35,
) -> tuple[dict[str, dict[float, np.ndarray]], list[dict]]:
    """
    Experiment 2: Tests explicit HSV saturation adjustment S' = clip(k_S * S, 0, 1)
    for k_S in {0.6, 1.0 (identity), 1.4} and compares directly against Strategy 1 (beta).
    """
    orig_sat = float(np.mean(color.rgb2hsv(rgb)[:, :, 1]))
    print(f"\n=== Experiment 2 (HSV Saturation k_S vs. beta): {image_name} | Orig Mean Sat S = {orig_sat:.4f} ===")
    print(
        f"{'HDR Method':<12} | {'Strategy / Param':<20} | {'Mean Sat S':<10} | "
        f"{'Hue Shift':<10} | {'Sat S>1 Clip':<12} | {'Base RGB Clip':<13}"
    )
    print("-" * 91)

    methods = [("Homomorphic", y_homo), ("Durand", y_durand)]
    hsv_recs: dict[str, dict[float, np.ndarray]] = {"Homomorphic": {}, "Durand": {}}
    sat_clips: dict[str, dict[float, float]] = {"Homomorphic": {}, "Durand": {}}
    records: list[dict] = []

    for method_name, y_comp in methods:
        for k_s in sat_gains:
            rgb_rec, stats = reconstruct_color_hsv_saturation(
                rgb, y_orig, y_comp, sat_gain=k_s, eps=homo_params["eps"], verbose=False
            )
            hsv_recs[method_name][k_s] = rgb_rec
            sat_clips[method_name][k_s] = stats["sat_high_clip_pct"]

            m = compute_color_metrics(rgb, rgb_rec)
            label = f"HSV k_S = {k_s:.1f}" + (" (id)" if np.isclose(k_s, 1.0) else "")
            print(
                f"{method_name:<12} | {label:<20} | {m['rec_sat']:>10.4f} | "
                f"{m['mean_hue_deg']:>9.2f}° | {stats['sat_high_clip_pct']:>11.3f}% | "
                f"{stats['base_rgb_clip_pct']:>12.3f}%"
            )
            records.append({
                "image": image_name,
                "method": method_name,
                "strategy": "HSV Saturation",
                "param": label,
                "orig_sat": m["orig_sat"],
                "rec_sat": m["rec_sat"],
                "hue_deg": m["mean_hue_deg"],
                "clip_info": f"S>1: {stats['sat_high_clip_pct']:.2f}% (Base RGB: {stats['base_rgb_clip_pct']:.2f}%)",
            })

        for beta_ref in [0.6, 1.4]:
            bm = compute_color_metrics(rgb, beta_recs[method_name][beta_ref])
            print(
                f"{method_name:<12} | {f'Ratio beta = {beta_ref:.1f}':<20} | {bm['rec_sat']:>10.4f} | "
                f"{bm['mean_hue_deg']:>9.2f}° | {'N/A':>12} | {'(see Exp 1)':>13}"
            )
        print("-" * 91)

    param_subtitle = (
        f"Homomorphic: $\\gamma_L={homo_params['gamma_l']}, \\gamma_H={homo_params['gamma_h']}, "
        f"c={homo_params['c']}, D_0={homo_params['d0']}$  |  "
        f"Durand: $\\sigma_s={durand_params['sigma_spatial']}, \\sigma_r={durand_params['sigma_range_norm']}, "
        f"\\Delta L_{{target}}={durand_params['target_contrast']}$"
    )

    # Full-image HSV figure
    rgb_orig_disp = np.power(np.clip(rgb, 0.0, 1.0), display_gamma)
    ncols = 2 + len(sat_gains)
    fig, axes = plt.subplots(2, ncols, figsize=(3.4 * ncols, 7.6))
    for row, (method_name, y_comp) in enumerate(methods):
        axes[row, 0].imshow(rgb_orig_disp)
        axes[row, 0].set_title(f"Original HDR RGB\n(Display $\\gamma = {display_gamma}$)", fontsize=9.0)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(y_comp, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 1].set_title(f"{method_name}\nCompressed $Y'$ [0, 1]", fontsize=9.0)
        axes[row, 1].axis("off")

        for idx, k_s in enumerate(sat_gains):
            col = 2 + idx
            role = "reduced" if k_s < 1.0 else ("identity" if np.isclose(k_s, 1.0) else "boosted")
            axes[row, col].imshow(hsv_recs[method_name][k_s])
            axes[row, col].set_title(
                f"{method_name} ($k_S = {k_s:.1f}$, {role})\n$S>1$ clip: {sat_clips[method_name][k_s]:.2f}%",
                fontsize=9.0,
            )
            axes[row, col].axis("off")

    fig.suptitle(
        f"Experiment 2: Explicit HSV Saturation Scaling ($k_S \\in {sat_gains}$) — {image_name}\n{param_subtitle}",
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.03, wspace=0.06, hspace=0.18)
    fig.savefig(OUTPUT_DIR / f"exp2_hsv_{image_name}.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    # Side-by-side zoomed comparison: beta vs. HSV k_S
    h, w, _ = rgb.shape
    r0_c, r1_c = int(crop_color_rel[0] * h), int(crop_color_rel[1] * h)
    c0_c, c1_c = int(crop_color_rel[2] * w), int(crop_color_rel[3] * w)
    r0_e, r1_e = int(crop_edge_rel[0] * h), int(crop_edge_rel[1] * h)
    c0_e, c1_e = int(crop_edge_rel[2] * w), int(crop_edge_rel[3] * w)

    fig_z, axes_z = plt.subplots(4, 4, figsize=(13.6, 12.5))
    row_specs = [
        ("Homomorphic (Intense Color)", "Homomorphic", (r0_c, r1_c, c0_c, c1_c)),
        ("Homomorphic (Strong Edge)", "Homomorphic", (r0_e, r1_e, c0_e, c1_e)),
        ("Durand (Intense Color)", "Durand", (r0_c, r1_c, c0_c, c1_c)),
        ("Durand (Strong Edge)", "Durand", (r0_e, r1_e, c0_e, c1_e)),
    ]
    for row_idx, (row_label, method_name, (r0, r1, c0, c1)) in enumerate(row_specs):
        axes_z[row_idx, 0].imshow(beta_recs[method_name][0.6][r0:r1, c0:c1])
        axes_z[row_idx, 0].set_title(f"{row_label}\nRatio $\\beta = 0.6$", fontsize=8.5)
        axes_z[row_idx, 0].axis("off")

        axes_z[row_idx, 1].imshow(hsv_recs[method_name][0.6][r0:r1, c0:c1])
        axes_z[row_idx, 1].set_title(f"{row_label}\nHSV $k_S = 0.6$", fontsize=8.5)
        axes_z[row_idx, 1].axis("off")

        axes_z[row_idx, 2].imshow(beta_recs[method_name][1.4][r0:r1, c0:c1])
        axes_z[row_idx, 2].set_title(f"{row_label}\nRatio $\\beta = 1.4$", fontsize=8.5)
        axes_z[row_idx, 2].axis("off")

        axes_z[row_idx, 3].imshow(hsv_recs[method_name][1.4][r0:r1, c0:c1])
        axes_z[row_idx, 3].set_title(f"{row_label}\nHSV $k_S = 1.4$", fontsize=8.5)
        axes_z[row_idx, 3].axis("off")

    fig_z.suptitle(
        f"Experiment 2 Zooms: Chromaticity Ratio ($\\beta$) vs. HSV Saturation ($k_S$) — {image_name}\n{param_subtitle}",
        fontsize=10.5,
    )
    fig_z.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02, wspace=0.06, hspace=0.22)
    fig_z.savefig(OUTPUT_DIR / f"exp2_hsv_vs_beta_zooms_{image_name}.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig_z)

    return hsv_recs, records


# ===========================================================================
# 8. Experiment 3: Main Cases Overview & Highlighted Zoom Crops
# ===========================================================================

def plot_experiment_3_main_cases(
    image_name: str,
    rgb: np.ndarray,
    y_homo: np.ndarray,
    y_durand: np.ndarray,
    rgb_durand_indep: np.ndarray,
    beta_recs: dict[str, dict[float, np.ndarray]],
    hsv_recs: dict[str, dict[float, np.ndarray]],
    homo_params: dict[str, float],
    durand_params: dict[str, float],
    crop_color_rel: tuple[float, float, float, float],
    crop_edge_rel: tuple[float, float, float, float],
    display_gamma: float = 0.35,
) -> None:
    """
    Experiment 3: Unified figure displaying Original HDR (with crop boxes),
    compressed luminance Y' for both methods, main reconstructions, and zoomed
    crops for (i) intense colors and (ii) a strong illumination transition.
    """
    h, w, _ = rgb.shape
    r0_c, r1_c = int(crop_color_rel[0] * h), int(crop_color_rel[1] * h)
    c0_c, c1_c = int(crop_color_rel[2] * w), int(crop_color_rel[3] * w)
    r0_e, r1_e = int(crop_edge_rel[0] * h), int(crop_edge_rel[1] * h)
    c0_e, c1_e = int(crop_edge_rel[2] * w), int(crop_edge_rel[3] * w)

    rgb_orig_disp = np.power(np.clip(rgb, 0.0, 1.0), display_gamma)
    fig, axes = plt.subplots(3, 4, figsize=(14.0, 10.8))

    # Row 1: Original + Homomorphic main cases
    axes[0, 0].imshow(rgb_orig_disp)
    axes[0, 0].add_patch(
        Rectangle((c0_c, r0_c), c1_c - c0_c, r1_c - r0_c, lw=1.6, ec="cyan", fc="none", ls="--", label="Intense color")
    )
    axes[0, 0].add_patch(
        Rectangle((c0_e, r0_e), c1_e - c0_e, r1_e - r0_e, lw=1.6, ec="yellow", fc="none", ls="--", label="Strong edge")
    )
    axes[0, 0].legend(loc="lower right", fontsize=7.5, framealpha=0.75)
    axes[0, 0].set_title(f"Original HDR ($\\gamma = {display_gamma}$)", fontsize=9.0)

    axes[0, 1].imshow(y_homo, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("Homomorphic: Compressed $Y'$", fontsize=9.0)

    axes[0, 2].imshow(beta_recs["Homomorphic"][1.0])
    axes[0, 2].set_title("Homomorphic: Ratio ($\\beta = 1.0$)", fontsize=9.0)

    axes[0, 3].imshow(hsv_recs["Homomorphic"][1.4])
    axes[0, 3].set_title("Homomorphic: HSV ($k_S = 1.4$)", fontsize=9.0)

    # Row 2: Independent RGB baseline + Durand main cases
    axes[1, 0].imshow(np.clip(rgb_durand_indep, 0.0, 1.0))
    axes[1, 0].set_title("Durand: Independent RGB", fontsize=9.0)

    axes[1, 1].imshow(y_durand, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title("Durand: Compressed $Y'$", fontsize=9.0)

    axes[1, 2].imshow(beta_recs["Durand"][0.6])
    axes[1, 2].set_title("Durand: Ratio ($\\beta = 0.6$)", fontsize=9.0)

    axes[1, 3].imshow(hsv_recs["Durand"][1.4])
    axes[1, 3].set_title("Durand: HSV ($k_S = 1.4$)", fontsize=9.0)

    # Row 3: Zoomed crops (Intense Color vs. Strong Illumination Edge)
    axes[2, 0].imshow(beta_recs["Durand"][0.6][r0_c:r1_c, c0_c:c1_c])
    axes[2, 0].set_title("Intense Color: Durand ($\\beta = 0.6$)", fontsize=9.0)

    axes[2, 1].imshow(hsv_recs["Durand"][1.4][r0_c:r1_c, c0_c:c1_c])
    axes[2, 1].set_title("Intense Color: Durand (HSV $k_S = 1.4$)", fontsize=9.0)

    axes[2, 2].imshow(beta_recs["Homomorphic"][1.0][r0_e:r1_e, c0_e:c1_e])
    axes[2, 2].set_title("Strong Edge: Homomorphic ($\\beta = 1.0$)", fontsize=9.0)

    axes[2, 3].imshow(beta_recs["Durand"][0.6][r0_e:r1_e, c0_e:c1_e])
    axes[2, 3].set_title("Strong Edge: Durand ($\\beta = 0.6$)", fontsize=9.0)

    for ax in axes.ravel():
        ax.axis("off")

    param_subtitle = (
        f"Homomorphic: $\\gamma_L={homo_params['gamma_l']}, \\gamma_H={homo_params['gamma_h']}, "
        f"c={homo_params['c']}, D_0={homo_params['d0']}$  |  "
        f"Durand: $\\sigma_s={durand_params['sigma_spatial']}, \\sigma_r={durand_params['sigma_range_norm']}, "
        f"\\Delta L_{{target}}={durand_params['target_contrast']}$"
    )
    fig.suptitle(f"Experiment 3: Main Cases & Zoomed Crops — {image_name}\n{param_subtitle}", fontsize=10.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02, wspace=0.05, hspace=0.14)
    fig.savefig(OUTPUT_DIR / f"exp3_main_cases_{image_name}.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ===========================================================================
# 9. Main Execution Pipeline
# ===========================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fixed internal parameters for both HDR methods
    homo_params = {"gamma_l": 0.35, "gamma_h": 1.60, "c": 1.0, "d0": 30.0, "eps": 1e-6}
    durand_params = {"sigma_spatial": 12.0, "sigma_range_norm": 0.12, "target_contrast": 1.8, "eps": 1e-6}

    betas = [0.6, 1.0, 1.4]
    sat_gains = [0.6, 1.0, 1.4]

    image_configs = [
        {
            "path": "archivos_t4/memorial.hdr",
            "crop_color": (0.48, 0.74, 0.58, 0.92),  # Stained-glass apse windows (intense colors)
            "crop_edge": (0.14, 0.42, 0.24, 0.56),   # Rose window border vs. dark stone (strong edge)
        },
        {
            "path": "archivos_t4/belgium.hdr",
            "crop_color": (0.45, 0.80, 0.10, 0.45),  # Vivid red wall & exterior foliage (intense colors)
            "crop_edge": (0.10, 0.50, 0.45, 0.85),   # Bright doorway/window transition (strong edge)
        },
    ]

    all_records: list[dict] = []

    for cfg in image_configs:
        img_path = resolve_image_path(cfg["path"])
        img_name = img_path.stem
        print("\n" + "=" * 86)
        print(f"Processing HDR Image: {img_path}")
        print("=" * 86)

        # Load & normalize RGB, then extract Rec. 709 luminance Y
        rgb = load_normalized_rgb(img_path)
        y_orig = extract_luminance(rgb)
        print(f"Shape: {rgb.shape} | Global normalized RGB in [0, 1] | Luminance Y range: [{y_orig.min():.6f}, {y_orig.max():.6f}]")

        # Apply Homomorphic filtering (luminance Y + independent RGB baseline)
        y_homo, _ = homomorphic_filter_channel(y_orig, verbose=True, **homo_params)
        rgb_homo_indep = homomorphic_rgb_independent(rgb, **homo_params)

        # Apply Durand's bilateral method (luminance Y + independent RGB baseline)
        y_durand, _ = durand_tone_mapping_channel(y_orig, verbose=True, **durand_params)
        rgb_durand_indep = durand_rgb_independent(rgb, **durand_params)

        # Experiment 1: Luminance / chromaticity ratio strategy across beta values
        beta_recs, exp1_records = run_experiment_1_beta(
            image_name=img_name,
            rgb=rgb,
            y_orig=y_orig,
            y_homo=y_homo,
            rgb_homo_indep=rgb_homo_indep,
            y_durand=y_durand,
            rgb_durand_indep=rgb_durand_indep,
            homo_params=homo_params,
            durand_params=durand_params,
            betas=betas,
            crop_color_rel=cfg["crop_color"],
            crop_edge_rel=cfg["crop_edge"],
        )
        all_records.extend(exp1_records)

        # Experiment 2: Explicit HSV saturation strategy across k_S values
        hsv_recs, exp2_records = run_experiment_2_hsv(
            image_name=img_name,
            rgb=rgb,
            y_orig=y_orig,
            y_homo=y_homo,
            y_durand=y_durand,
            beta_recs=beta_recs,
            homo_params=homo_params,
            durand_params=durand_params,
            sat_gains=sat_gains,
            crop_color_rel=cfg["crop_color"],
            crop_edge_rel=cfg["crop_edge"],
            display_gamma=0.35,
        )
        all_records.extend(exp2_records)

        # Experiment 3: Main cases overview & highlighted crops
        plot_experiment_3_main_cases(
            image_name=img_name,
            rgb=rgb,
            y_homo=y_homo,
            y_durand=y_durand,
            rgb_durand_indep=rgb_durand_indep,
            beta_recs=beta_recs,
            hsv_recs=hsv_recs,
            homo_params=homo_params,
            durand_params=durand_params,
            crop_color_rel=cfg["crop_color"],
            crop_edge_rel=cfg["crop_edge"],
            display_gamma=0.35,
        )

    # Experiment 4 & 5: Consolidated quantitative summary
    print("\n" + "=" * 96)
    print("EXPERIMENT 4 CONSOLIDATED QUANTITATIVE SUMMARY (SATURATION BEFORE/AFTER & HUE SHIFT)")
    print("=" * 96)
    print(
        f"{'Image':<9} | {'HDR Method':<11} | {'Strategy':<18} | {'Parameter':<16} | "
        f"{'S_orig':<6} | {'S_rec':<6} | {'Hue Shift':<9} | {'Clipping'}"
    )
    print("-" * 96)
    for r in all_records:
        print(
            f"{r['image']:<9} | {r['method']:<11} | {r['strategy']:<18} | {r['param']:<16} | "
            f"{r['orig_sat']:>6.4f} | {r['rec_sat']:>6.4f} | {r['hue_deg']:>8.2f}° | {r['clip_info']}"
        )
    print("=" * 96)
    print(f"All figures successfully saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()