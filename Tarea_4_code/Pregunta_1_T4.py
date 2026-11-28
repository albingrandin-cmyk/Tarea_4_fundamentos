import numpy as np
from scipy.ndimage import convolve1d
from skimage import io, color
import matplotlib.pyplot as plt

# The Explicit 1D-smoothing, 5 elements filter (B_3 spline/Binomial, see report for motivation)
SMOOTHING_KERNEL_1D = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0


def log_compress(img: np.ndarray, k: float = 500.0) -> np.ndarray:
    """
    Global logarithmic compression for visualization of linear HDR luminance [0, 1].
    Maps 0 -> 0 and 1 -> 1 while lifting dark shadow regions without altering raw data.
    """
    clipped = np.clip(img, 0.0, 1.0)
    return np.log1p(k * clipped) / np.log1p(k)


"""Laplacian pyramid construction"""
def load_normalized_luminance(image_path: str) -> np.ndarray:
    """
    Reading in an image, extracting luminance and normalizing to [0,1].
    Safety measurements to avoid errors such as integer under- or overflow as well as
    allow the image to be either RGB or grayscale.
    """
    img = io.imread(image_path)

    if np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.float64) / np.iinfo(img.dtype).max
    else:
        img = img.astype(np.float64)
        if img.max() > 1.0:
            img /= img.max()

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        luminance = color.rgb2gray(img)
    else:
        luminance = img

    return np.clip(luminance, 0.0, 1.0)


def reduce_level(c_prev: np.ndarray, kernel_1d: np.ndarray = SMOOTHING_KERNEL_1D) -> np.ndarray:
    """
    Smoothing with the explicit 2D-separable filter H = w^T * w where w is the binomial kernel,
    followed by downsampling with a factor 2.
    """
    smoothed = convolve1d(c_prev, kernel_1d, axis=0, mode="reflect")
    smoothed = convolve1d(smoothed, kernel_1d, axis=1, mode="reflect")
    return smoothed[::2, ::2]


def expand_level(
    c_next: np.ndarray,
    target_shape: tuple[int, int],
    kernel_1d: np.ndarray = SMOOTHING_KERNEL_1D,
) -> np.ndarray:
    """
    Expansion operator U(c_j):
    1. Upsampling with factor 2 by zero-injection to target_shape.
    2. Interpolation with 4*H (2*w per axis) to preserve light intensity/DC-level.
    """
    upsampled = np.zeros(target_shape, dtype=np.float64)
    upsampled[::2, ::2] = c_next

    interp_kernel_1d = 2.0 * kernel_1d
    interpolated = convolve1d(upsampled, interp_kernel_1d, axis=0, mode="reflect")
    interpolated = convolve1d(interpolated, interp_kernel_1d, axis=1, mode="reflect")
    return interpolated


def laplacian_pyramid_decompose(
    c0: np.ndarray,
    num_levels: int = 5,
    kernel_1d: np.ndarray = SMOOTHING_KERNEL_1D,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Decomposing c0 in 'num_levels' detail layers [d_1, ..., d_J] as well as the residual c_J.
    d_j = c_{j-1} - U(c_j)
    """
    c_current = c0.astype(np.float64)
    detail_layers: list[np.ndarray] = []

    for _ in range(num_levels):
        c_next = reduce_level(c_current, kernel_1d)
        u_c_next = expand_level(c_next, c_current.shape, kernel_1d)
        d_j = c_current - u_c_next
        detail_layers.append(d_j)
        c_current = c_next

    residual = c_current
    return detail_layers, residual


def laplacian_pyramid_reconstruct(
    detail_layers: list[np.ndarray],
    residual: np.ndarray,
    kernel_1d: np.ndarray = SMOOTHING_KERNEL_1D,
) -> np.ndarray:
    """
    Reconstructing original image from the residual and detail layers
    with the same expansion operators U(c_j): c_{j-1} = U(c_j) + d_j.
    """
    c_current = residual.astype(np.float64)

    for d_j in reversed(detail_layers):
        u_c = expand_level(c_current, d_j.shape, kernel_1d)
        c_current = u_c + d_j

    return c_current


"""Implementing Starlet-transform"""
def build_atrous_kernel_1d(
    scale: int,
    base_kernel: np.ndarray = SMOOTHING_KERNEL_1D,
) -> np.ndarray:
    """
    Constructs the 1D dilated B3-spline kernel at scale j (1-indexed)
    by inserting 2^(j-1) - 1 zeros between filter coefficients.
    """
    step = 2 ** (scale - 1)
    dilated = np.zeros((len(base_kernel) - 1) * step + 1, dtype=np.float64)
    dilated[::step] = base_kernel
    return dilated


def starlet_decompose(
    c0: np.ndarray,
    num_scales: int = 5,
    base_kernel: np.ndarray = SMOOTHING_KERNEL_1D,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Decomposes c0 into 'num_scales' detail layers [w_1, ..., w_J] and residual c_J
    using the a trous algorithm: w_j = c_{j-1} - c_j.
    """
    c_current = c0.astype(np.float64)
    detail_layers: list[np.ndarray] = []

    for j in range(1, num_scales + 1):
        h_j = build_atrous_kernel_1d(j, base_kernel)
        c_next = convolve1d(c_current, h_j, axis=0, mode="reflect")
        c_next = convolve1d(c_next, h_j, axis=1, mode="reflect")
        w_j = c_current - c_next
        detail_layers.append(w_j)
        c_current = c_next

    residual = c_current
    return detail_layers, residual


def starlet_reconstruct(
    detail_layers: list[np.ndarray],
    residual: np.ndarray,
) -> np.ndarray:
    """
    Exact reconstruction of the Starlet transform: c_0 = c_J + sum_{j=1}^J w_j.
    """
    reconstructed = residual.astype(np.float64).copy()
    for w_j in detail_layers:
        reconstructed += w_j
    return reconstructed


def compute_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Computes Root Mean Square Error between original and reconstructed images."""
    return float(np.sqrt(np.mean((original - reconstructed) ** 2)))


"""Experimentation and analysis, 1"""
def print_layer_dimensions(
    c0: np.ndarray,
    lap_details: list[np.ndarray],
    lap_res: np.ndarray,
    star_details: list[np.ndarray],
    star_res: np.ndarray,
) -> None:
    """
    Prints a comparison table of layer dimensions and total coefficient counts
    for the Laplacian Pyramid and the Starlet Transform.
    """
    total_pixels = c0.size
    print(f"\nInput luminance shape: {c0.shape[0]}x{c0.shape[1]} ({total_pixels:,} pixels)")
    print("-" * 72)
    print(f"{'Layer':<14} | {'Laplacian Shape':<18} | {'Starlet Shape':<18}")
    print("-" * 72)

    for j, (d_j, w_j) in enumerate(zip(lap_details, star_details), start=1):
        lap_dim = f"{d_j.shape[0]}x{d_j.shape[1]}"
        star_dim = f"{w_j.shape[0]}x{w_j.shape[1]}"
        print(f"Detail j={j:<5} | {lap_dim:<18} | {star_dim:<18}")

    lap_res_dim = f"{lap_res.shape[0]}x{lap_res.shape[1]}"
    star_res_dim = f"{star_res.shape[0]}x{star_res.shape[1]}"
    print(f"Residual c_5   | {lap_res_dim:<18} | {star_res_dim:<18}")
    print("-" * 72)

    lap_coeffs = sum(d.size for d in lap_details) + lap_res.size
    star_coeffs = sum(w.size for w in star_details) + star_res.size
    print(f"Total coefficients — Laplacian: {lap_coeffs:,} ({lap_coeffs / total_pixels:.3f}x input)")
    print(f"Total coefficients — Starlet:   {star_coeffs:,} ({star_coeffs / total_pixels:.3f}x input)\n")


def plot_multiscale_layers(
    lap_details: list[np.ndarray],
    lap_res: np.ndarray,
    star_details: list[np.ndarray],
    star_res: np.ndarray,
    percentile: float = 99.5,
    log_k: float = 500.0,
    save_path: str = "exp1_multiscale_layers.png",
) -> None:
    """
    Visualizes the 5 detail layers and the coarse residual for both representations
    using consistent visualization scales and displaying each layer's dimensions.
    """
    num_levels = len(lap_details)

    # Shared symmetric scale [-v_detail, +v_detail] across all 10 detail layers
    all_detail_abs = np.concatenate([np.abs(layer).ravel() for layer in lap_details + star_details])
    if percentile < 100.0:
        v_detail = float(np.percentile(all_detail_abs, percentile))
    else:
        v_detail = float(np.max(all_detail_abs))

    fig, axes = plt.subplots(2, num_levels + 1, figsize=(18, 6.8), constrained_layout=True)

    for j in range(num_levels):
        # Row 0: Laplacian Pyramid detail layers d_j
        im_det = axes[0, j].imshow(
            lap_details[j], cmap="gray", vmin=-v_detail, vmax=v_detail
        )
        axes[0, j].set_title(
            f"Laplacian $d_{{{j + 1}}}$\n({lap_details[j].shape[0]}×{lap_details[j].shape[1]})",
            fontsize=10,
        )
        axes[0, j].set_xticks([])
        axes[0, j].set_yticks([])

        # Row 1: Starlet Transform detail layers w_j
        axes[1, j].imshow(
            star_details[j], cmap="gray", vmin=-v_detail, vmax=v_detail
        )
        axes[1, j].set_title(
            f"Starlet $w_{{{j + 1}}}$\n({star_details[j].shape[0]}×{star_details[j].shape[1]})",
            fontsize=10,
        )
        axes[1, j].set_xticks([])
        axes[1, j].set_yticks([])

    # Last column: Coarse residuals c_5 on a shared log-compressed [0, 1] scale
    im_res = axes[0, num_levels].imshow(log_compress(lap_res, k=log_k), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, num_levels].set_title(
        f"Laplacian $c_{{{num_levels}}}$ (res)\n({lap_res.shape[0]}×{lap_res.shape[1]})",
        fontsize=10,
    )
    axes[0, num_levels].set_xticks([])
    axes[0, num_levels].set_yticks([])

    axes[1, num_levels].imshow(log_compress(star_res, k=log_k), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, num_levels].set_title(
        f"Starlet $c_{{{num_levels}}}$ (res)\n({star_res.shape[0]}×{star_res.shape[1]})",
        fontsize=10,
    )
    axes[1, num_levels].set_xticks([])
    axes[1, num_levels].set_yticks([])

    # Shared colorbars
    fig.colorbar(
        im_det,
        ax=axes[:, :num_levels],
        orientation="horizontal",
        fraction=0.045,
        pad=0.05,
        label=f"Detail coefficient amplitude (shared symmetric scale: ±{v_detail:.4f}, p={percentile}%)",
    )
    fig.colorbar(
        im_res,
        ax=axes[:, num_levels],
        orientation="horizontal",
        fraction=0.045,
        pad=0.05,
        label=f"Log-compressed residual intensity (k={log_k:g}, [0, 1])",
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


"""Experimentation and analysis, 2"""
def hard_threshold_layers(
    detail_layers: list[np.ndarray],
    alpha: float,
) -> tuple[list[np.ndarray], list[float], float]:
    """
    Applies relative hard thresholding T_j = alpha * max(|d_j|) to each detail layer.
    Returns thresholded layers, percentage of retained coefficients per layer,
    and overall percentage of retained detail coefficients.
    """
    thresholded_layers: list[np.ndarray] = []
    retained_per_layer: list[float] = []
    total_kept = 0
    total_coeffs = 0

    for d_j in detail_layers:
        max_val = float(np.max(np.abs(d_j)))
        t_j = alpha * max_val
        mask = np.abs(d_j) >= t_j
        d_thresh = np.where(mask, d_j, 0.0)

        kept = int(np.sum(mask))
        pct = 100.0 * kept / d_j.size

        thresholded_layers.append(d_thresh)
        retained_per_layer.append(pct)
        total_kept += kept
        total_coeffs += d_j.size

    total_retained_pct = 100.0 * total_kept / total_coeffs
    return thresholded_layers, retained_per_layer, total_retained_pct


def run_hard_thresholding_experiment(
    c0: np.ndarray,
    lap_details: list[np.ndarray],
    lap_res: np.ndarray,
    star_details: list[np.ndarray],
    star_res: np.ndarray,
    alphas: list[float] = [0.02, 0.10, 0.25],
    log_k: float = 500.0,
    save_path: str = "exp2_hard_thresholding.png",
) -> None:
    """
    Runs hard thresholding for common alpha values on both representations,
    prints retained coefficient percentages and RMSE, and saves a visual comparison.
    """
    num_alphas = len(alphas)
    fig, axes = plt.subplots(num_alphas, 4, figsize=(15, 3.8 * num_alphas))
    if num_alphas == 1:
        axes = axes[np.newaxis, :]

    # Crop region covering both a bright stained-glass window and dark interior arches
    h, w = c0.shape
    r0, r1 = int(0.18 * h), int(0.55 * h)
    c_start, c_end = int(0.25 * w), int(0.75 * w)

    for row, alpha in enumerate(alphas):
        # Threshold and reconstruct Laplacian Pyramid (on linear c0)
        lap_thresh, lap_pcts, lap_tot_pct = hard_threshold_layers(lap_details, alpha)
        c0_lap_hat = laplacian_pyramid_reconstruct(lap_thresh, lap_res)
        rmse_lap = compute_rmse(c0, c0_lap_hat)

        # Threshold and reconstruct Starlet Transform (on linear c0)
        star_thresh, star_pcts, star_tot_pct = hard_threshold_layers(star_details, alpha)
        c0_star_hat = starlet_reconstruct(star_thresh, star_res)
        rmse_star = compute_rmse(c0, c0_star_hat)

        # Print quantitative report table
        print(f"\n=== Hard Thresholding Results (alpha = {alpha:.2f}) ===")
        print(f"{'Layer':<12} | {'Laplacian Retained (%)':<22} | {'Starlet Retained (%)':<22}")
        print("-" * 62)
        for j, (lp, sp) in enumerate(zip(lap_pcts, star_pcts), start=1):
            print(f"Detail j={j:<3} | {lp:>20.2f}% | {sp:>20.2f}%")
        print("-" * 62)
        print(f"{'Total Detail':<12} | {lap_tot_pct:>20.2f}% | {star_tot_pct:>20.2f}%")
        print(f"Reconstruction RMSE — Laplacian: {rmse_lap:.6e} | Starlet: {rmse_star:.6e}")

        # Global log compression for display so dark regions are clearly visible
        lap_disp = log_compress(c0_lap_hat, k=log_k)
        star_disp = log_compress(c0_star_hat, k=log_k)

        # Column 0: Full Laplacian reconstruction
        axes[row, 0].imshow(lap_disp, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 0].set_title(
            f"Laplace ($\\alpha={alpha}$)\nKept: {lap_tot_pct:.1f}% | RMSE: {rmse_lap:.4f}",
            fontsize=9.0,
            pad=6,
        )
        axes[row, 0].axis("off")

        # Column 1: Full Starlet reconstruction
        axes[row, 1].imshow(star_disp, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 1].set_title(
            f"Starlet ($\\alpha={alpha}$)\nKept: {star_tot_pct:.1f}% | RMSE: {rmse_star:.4f}",
            fontsize=9.0,
            pad=6,
        )
        axes[row, 1].axis("off")

        # Column 2: Zoomed crop for Laplacian
        axes[row, 2].imshow(lap_disp[r0:r1, c_start:c_end], cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 2].set_title(f"Laplace Zoom ($\\alpha={alpha}, k={log_k:g}$)", fontsize=9.0, pad=6)
        axes[row, 2].axis("off")

        # Column 3: Zoomed crop for Starlet
        axes[row, 3].imshow(star_disp[r0:r1, c_start:c_end], cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 3].set_title(f"Starlet Zoom ($\\alpha={alpha}, k={log_k:g}$)", fontsize=9.0, pad=6)
        axes[row, 3].axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.08, hspace=0.28)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


"""Experimentation and analysis, 3"""
def apply_multiplicative_gains(
    detail_layers: list[np.ndarray],
    gains: list[float],
) -> list[np.ndarray]:
    """Scales each detail layer j by a multiplicative gain factor g_j."""
    if len(detail_layers) != len(gains):
        raise ValueError("Number of gain factors must match number of detail layers.")
    return [g * layer for g, layer in zip(gains, detail_layers)]


def run_gain_enhancement_experiment(
    c0: np.ndarray,
    lap_details: list[np.ndarray],
    lap_res: np.ndarray,
    star_details: list[np.ndarray],
    star_res: np.ndarray,
    log_k: float = 500.0,
    save_path: str = "exp3_gain_enhancement.png",
) -> None:
    """
    Applies two symmetric multiplicative gain configurations (fine-scale vs. coarse-scale)
    to both the Laplacian pyramid and the Starlet transform, reconstructs the images,
    and generates a side-by-side comparison with zoomed crops.
    """
    configs = [
        ("Original (Reference)", [1.0, 1.0, 1.0, 1.0, 1.0]),
        ("Fine-Scale Emphasis", [3.5, 2.5, 1.5, 1.0, 1.0]),
        ("Coarse-Scale Emphasis", [1.0, 1.0, 1.5, 2.5, 3.5]),
    ]

    # Zoom region capturing both the bright rose window and dark textured stone arches
    h, w = c0.shape
    r0, r1 = int(0.18 * h), int(0.58 * h)
    c_start, c_end = int(0.22 * w), int(0.78 * w)

    fig, axes = plt.subplots(len(configs), 4, figsize=(15, 3.8 * len(configs)))

    for row, (label, gains) in enumerate(configs):
        # Modify and reconstruct Laplacian Pyramid (on linear c0)
        lap_mod = apply_multiplicative_gains(lap_details, gains)
        c0_lap_hat = laplacian_pyramid_reconstruct(lap_mod, lap_res)
        rmse_lap = compute_rmse(c0, c0_lap_hat)

        # Modify and reconstruct Starlet Transform (on linear c0)
        star_mod = apply_multiplicative_gains(star_details, gains)
        c0_star_hat = starlet_reconstruct(star_mod, star_res)
        rmse_star = compute_rmse(c0, c0_star_hat)

        # Report quantitative metrics (RMSE and dynamic range before clipping)
        print(f"\n=== {label} | Gains g = {gains} ===")
        print(
            f"Laplacian — RMSE: {rmse_lap:.6e} | "
            f"Range before clip: [{c0_lap_hat.min():.4f}, {c0_lap_hat.max():.4f}]"
        )
        print(
            f"Starlet   — RMSE: {rmse_star:.6e} | "
            f"Range before clip: [{c0_star_hat.min():.4f}, {c0_star_hat.max():.4f}]"
        )

        # Consistent global log compression for display
        lap_disp = log_compress(c0_lap_hat, k=log_k)
        star_disp = log_compress(c0_star_hat, k=log_k)

        gain_str = "[" + ", ".join(f"{g:.1f}" for g in gains) + "]"

        # Column 0: Full Laplacian
        axes[row, 0].imshow(lap_disp, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 0].set_title(
            f"Laplace — {label}\ng = {gain_str} | RMSE: {rmse_lap:.4f}",
            fontsize=9.0,
            pad=6,
        )
        axes[row, 0].axis("off")

        # Column 1: Full Starlet
        axes[row, 1].imshow(star_disp, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 1].set_title(
            f"Starlet — {label}\ng = {gain_str} | RMSE: {rmse_star:.4f}",
            fontsize=9.0,
            pad=6,
        )
        axes[row, 1].axis("off")

        # Column 2: Zoomed Laplacian
        axes[row, 2].imshow(lap_disp[r0:r1, c_start:c_end], cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 2].set_title(f"Laplace Zoom (log k = {log_k:g})", fontsize=9.0, pad=6)
        axes[row, 2].axis("off")

        # Column 3: Zoomed Starlet
        axes[row, 3].imshow(star_disp[r0:r1, c_start:c_end], cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 3].set_title(f"Starlet Zoom (log k = {log_k:g})", fontsize=9.0, pad=6)
        axes[row, 3].axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.08, hspace=0.28)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    c0 = load_normalized_luminance("archivos_t4/memorial.tif")

    # Verifying the Laplace pyramid (5 levels + residual)
    lap_details, lap_res = laplacian_pyramid_decompose(c0, num_levels=5)
    c0_lap_rec = laplacian_pyramid_reconstruct(lap_details, lap_res)
    rmse_lap = compute_rmse(c0, c0_lap_rec)

    # Verifying Starlet-transform (5 scales + residual)
    star_details, star_res = starlet_decompose(c0, num_scales=5)
    c0_star_rec = starlet_reconstruct(star_details, star_res)
    rmse_star = compute_rmse(c0, c0_star_rec)

    print(f"Laplacian Pyramid Reconstruction RMSE: {rmse_lap:.6e}")
    print(f"Starlet Transform Reconstruction RMSE: {rmse_star:.6e}")

    # Experiment 1: Dimensions and visual comparison of all 5 detail layers + residual
    print_layer_dimensions(c0, lap_details, lap_res, star_details, star_res)
    plot_multiscale_layers(lap_details, lap_res, star_details, star_res, percentile=99.5, log_k=500.0)

    # Experiment 2: Relative hard thresholding
    run_hard_thresholding_experiment(
        c0, lap_details, lap_res, star_details, star_res, alphas=[0.02, 0.10, 0.25], log_k=500.0
    )

    # Experiment 3: Multiplicative gain enhancement
    run_gain_enhancement_experiment(
        c0, lap_details, lap_res, star_details, star_res, log_k=500.0
    )