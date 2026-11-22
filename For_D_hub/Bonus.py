import time
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt


# ==============================================================================
# 1. Image Loading & Color Helpers
# ==============================================================================
def load_normalized_rgb(image_path: str) -> np.ndarray:
    img = iio.imread(image_path).astype(np.float64)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] > 3:
        img = img[..., :3]
    img = np.maximum(img, 0.0)
    max_val = np.max(img)
    return img / max_val if max_val > 0 else img


def extract_luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2125 * rgb[..., 0] + 0.7154 * rgb[..., 1] + 0.0721 * rgb[..., 2]


def reconstruct_color_chromaticity_ratio(
    rgb_orig: np.ndarray,
    y_orig: np.ndarray,
    y_comp: np.ndarray,
    beta: float = 0.6,
    eps: float = 1e-6,
) -> np.ndarray:
    denom = y_orig[..., None] + eps
    chroma_ratio = np.maximum(rgb_orig / denom, 0.0)
    rgb_rec = y_comp[..., None] * np.power(chroma_ratio, beta)
    return np.clip(rgb_rec, 0.0, 1.0)


# ==============================================================================
# 2. Consistent Discrete Differential Operators
# ==============================================================================
def grad_forward(u: np.ndarray):
    """Forward differences: gx[i,j] = u[i,j+1] - u[i,j], gy[i,j] = u[i+1,j] - u[i,j]."""
    gx = np.zeros_like(u)
    gy = np.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gx, gy


def div_backward(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    Backward differences on interior nodes (1..H-2, 1..W-2):
    div(g)[i,j] = (gx[i,j] - gx[i,j-1]) + (gy[i,j] - gy[i-1,j]).
    When g = grad_forward(u), div_backward(g) equals the exact 5-point discrete Laplacian.
    """
    div = np.zeros_like(gx)
    div[1:-1, 1:-1] = (
        (gx[1:-1, 1:-1] - gx[1:-1, :-2])
        + (gy[1:-1, 1:-1] - gy[:-2, 1:-1])
    )
    return div


def laplacian_5pt(u: np.ndarray, h: float = 1.0) -> np.ndarray:
    """5-point discrete Laplacian on interior nodes with grid spacing h."""
    inv_h2 = 1.0 / (h * h)
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (
        u[1:-1, 2:] + u[1:-1, :-2] + u[2:, 1:-1] + u[:-2, 1:-1] - 4.0 * u[1:-1, 1:-1]
    ) * inv_h2
    return lap


# ==============================================================================
# 3. Multigrid Poisson Solver (V-Cycle with Red-Black Gauss-Seidel)
# ==============================================================================
def smooth_rbgs(u: np.ndarray, rhs: np.ndarray, h: float = 1.0, iters: int = 3) -> np.ndarray:
    """
    Vectorized Red-Black Gauss-Seidel smoother for Laplace(u) = rhs
    subject to fixed Dirichlet values on the outer border.
    """
    H, W = u.shape
    if H <= 2 or W <= 2:
        return u

    h2 = h * h
    ii, jj = np.ogrid[1 : H - 1, 1 : W - 1]
    red = (ii + jj) % 2 == 0
    black = ~red

    for _ in range(iters):
        nbr_r = u[1:-1, 2:] + u[1:-1, :-2] + u[2:, 1:-1] + u[:-2, 1:-1]
        u[1:-1, 1:-1][red] = 0.25 * (nbr_r[red] - h2 * rhs[1:-1, 1:-1][red])

        nbr_b = u[1:-1, 2:] + u[1:-1, :-2] + u[2:, 1:-1] + u[:-2, 1:-1]
        u[1:-1, 1:-1][black] = 0.25 * (nbr_b[black] - h2 * rhs[1:-1, 1:-1][black])

    return u


def restrict_full_weighting(r: np.ndarray) -> np.ndarray:
    """Full-weighting restriction operator (3x3 stencil [1,2,1; 2,4,2; 1,2,1]/16)."""
    H, W = r.shape
    pad_h = 1 if (H % 2 == 0) else 0
    pad_w = 1 if (W % 2 == 0) else 0
    r_pad = np.pad(r, ((0, pad_h), (0, pad_w)), mode="constant") if (pad_h or pad_w) else r

    Hc = (r_pad.shape[0] + 1) // 2
    Wc = (r_pad.shape[1] + 1) // 2
    rc = np.zeros((Hc, Wc), dtype=r.dtype)

    rc[1:-1, 1:-1] = (
        4.0 * r_pad[2:-2:2, 2:-2:2]
        + 2.0 * (
            r_pad[1:-3:2, 2:-2:2]
            + r_pad[3:-1:2, 2:-2:2]
            + r_pad[2:-2:2, 1:-3:2]
            + r_pad[2:-2:2, 3:-1:2]
        )
        + 1.0 * (
            r_pad[1:-3:2, 1:-3:2]
            + r_pad[1:-3:2, 3:-1:2]
            + r_pad[3:-1:2, 1:-3:2]
            + r_pad[3:-1:2, 3:-1:2]
        )
    ) / 16.0
    return rc


def prolong_bilinear(ec: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Bilinear prolongation operator preserving zero Dirichlet boundary on error."""
    H, W = target_shape
    pad_h = 1 if (H % 2 == 0) else 0
    pad_w = 1 if (W % 2 == 0) else 0
    Hp, Wp = H + pad_h, W + pad_w

    ef_pad = np.zeros((Hp, Wp), dtype=ec.dtype)
    ef_pad[0::2, 0::2] = ec
    ef_pad[0::2, 1::2] = 0.5 * (ec[:, :-1] + ec[:, 1:])
    ef_pad[1::2, 0::2] = 0.5 * (ec[:-1, :] + ec[1:, :])
    ef_pad[1::2, 1::2] = 0.25 * (ec[:-1, :-1] + ec[:-1, 1:] + ec[1:, :-1] + ec[1:, 1:])

    ef = ef_pad[:H, :W]
    ef[0, :] = ef[-1, :] = ef[:, 0] = ef[:, -1] = 0.0
    return ef


def multigrid_v_cycle(
    u: np.ndarray,
    rhs: np.ndarray,
    h: float = 1.0,
    level: int = 0,
    max_levels: int = 6,
    nu1: int = 3,
    nu2: int = 3,
) -> np.ndarray:
    """Recursive Multigrid V-cycle solving Laplace(u) = rhs."""
    H, W = u.shape
    if level >= max_levels - 1 or min(H, W) <= 5:
        return smooth_rbgs(u, rhs, h=h, iters=50)

    # 1. Pre-smoothing
    u = smooth_rbgs(u, rhs, h=h, iters=nu1)

    # 2. Residual r = rhs - Laplace(u)
    r = rhs - laplacian_5pt(u, h=h)
    r[0, :] = r[-1, :] = r[:, 0] = r[:, -1] = 0.0

    # 3. Restrict residual to coarser grid (2h)
    rc = restrict_full_weighting(r)
    ec = np.zeros_like(rc)

    # 4. Coarse-grid correction via recursive V-cycle
    ec = multigrid_v_cycle(
        ec, rc, h=2.0 * h, level=level + 1, max_levels=max_levels, nu1=nu1, nu2=nu2
    )

    # 5. Prolongate correction and update solution
    ef = prolong_bilinear(ec, u.shape)
    u = u + ef

    # 6. Post-smoothing
    u = smooth_rbgs(u, rhs, h=h, iters=nu2)
    return u


def solve_poisson_multigrid(
    rhs: np.ndarray,
    u_boundary: np.ndarray,
    max_levels: int = 6,
    nu1: int = 3,
    nu2: int = 3,
    max_cycles: int = 30,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-10,
):
    """
    Solves Laplace(u_hat) = rhs with Dirichlet boundary conditions taken from u_boundary.
    Returns (u_hat, residual_history).
    """
    u_hat = np.zeros_like(rhs)
    u_hat[0, :] = u_boundary[0, :]
    u_hat[-1, :] = u_boundary[-1, :]
    u_hat[:, 0] = u_boundary[:, 0]
    u_hat[:, -1] = u_boundary[:, -1]

    r0 = rhs - laplacian_5pt(u_hat, h=1.0)
    r0_norm = np.linalg.norm(r0[1:-1, 1:-1])
    res_history = [r0_norm]

    for _ in range(max_cycles):
        u_hat = multigrid_v_cycle(
            u_hat, rhs, h=1.0, level=0, max_levels=max_levels, nu1=nu1, nu2=nu2
        )
        r = rhs - laplacian_5pt(u_hat, h=1.0)
        r_norm = np.linalg.norm(r[1:-1, 1:-1])
        res_history.append(r_norm)

        if r_norm < abs_tol or (r0_norm > 0 and (r_norm / r0_norm) < rel_tol):
            break

    return u_hat, res_history


# ==============================================================================
# 4. Gradient-Domain HDR Compression
# ==============================================================================
def compress_gradient_field(gx: np.ndarray, gy: np.ndarray, gamma: float, alpha_scale: float = 0.1):
    """
    Non-linear gradient magnitude attenuation preserving local direction (Fattal et al.):
        g_mod = g * ((||g|| + eps) / alpha)^(gamma - 1)
    where alpha = alpha_scale * mean(||g||).
    """
    eps = 1e-6
    mag = np.sqrt(gx**2 + gy**2)
    alpha = alpha_scale * np.mean(mag)
    attenuation = np.power((mag + eps) / (alpha + eps), gamma - 1.0)
    return gx * attenuation, gy * attenuation


def compute_local_contrast(y_img: np.ndarray) -> float:
    """Mean local gradient contrast measure."""
    gx, gy = grad_forward(y_img)
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


# ==============================================================================
# 5. Main Execution: Validation & Gradient-Domain HDR Compression
# ==============================================================================
if __name__ == "__main__":
    out_dir = Path("results_bonus")
    out_dir.mkdir(exist_ok=True)

    image_path = "archivos_t4/memorial.hdr"
    rgb_orig = load_normalized_rgb(image_path)
    y_orig = extract_luminance(rgb_orig)

    eps = 1e-6
    u_orig = np.log(y_orig + eps)

    # --------------------------------------------------------------------------
    # PART A: Validation — Exact Reconstruction from Unmodified Gradients
    # --------------------------------------------------------------------------
    print("=" * 72)
    print("BONUS PART 1: POISSON MULTIGRID VALIDATION (UNMODIFIED GRADIENTS)")
    print("=" * 72)

    gx_orig, gy_orig = grad_forward(u_orig)
    rhs_exact = div_backward(gx_orig, gy_orig)

    t0 = time.time()
    u_rec, res_hist = solve_poisson_multigrid(
        rhs_exact,
        u_boundary=u_orig,
        max_levels=6,
        nu1=3,
        nu2=3,
        max_cycles=30,
        rel_tol=1e-10,
        abs_tol=1e-11,
    )
    elapsed = time.time() - t0
    rmse_u = float(np.sqrt(np.mean((u_rec - u_orig) ** 2)))
    y_rec_val = np.maximum(np.exp(u_rec) - eps, 0.0)
    rmse_y = float(np.sqrt(np.mean((y_rec_val - y_orig) ** 2)))

    print(f"Grid shape: {u_orig.shape} | Levels: 6 | Smoother: RBGS (nu1=3, nu2=3)")
    print(f"V-cycles executed: {len(res_hist) - 1} | Runtime: {elapsed:.2f} s")
    print(f"Initial residual ||r_0||_2: {res_hist[0]:.4e}")
    print(f"Final residual   ||r_k||_2: {res_hist[-1]:.4e}")
    print(f"Reconstruction RMSE (log-domain u): {rmse_u:.4e}")
    print(f"Reconstruction RMSE (linear Y):     {rmse_y:.4e}")
    print("\nResidual norm per V-cycle:")
    for k, rnorm in enumerate(res_hist):
        print(f"  Cycle {k:2d}: ||r||_2 = {rnorm:.4e}")

    # Plot validation convergence and reconstruction error
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].semilogy(range(len(res_hist)), res_hist, "o-", linewidth=2, markersize=5)
    axes[0].set_title("Multigrid V-Cycle Convergence")
    axes[0].set_xlabel("V-Cycle Iteration $k$")
    axes[0].set_ylabel(r"Residual Norm $\|r_k\|_2$")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)

    axes[1].imshow(np.clip(y_rec_val ** 0.35, 0, 1), cmap="gray")
    axes[1].set_title(f"Reconstructed Luminance ($\\gamma_{{disp}}=0.35$)\nRMSE(u) = {rmse_u:.2e}")
    axes[1].axis("off")

    err_map = np.abs(u_rec - u_orig)
    im = axes[2].imshow(err_map, cmap="inferno")
    axes[2].set_title(r"Absolute Error $|\hat{u} - u|$")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_dir / "bonus_part1_validation.png", dpi=200)
    plt.close()

    # --------------------------------------------------------------------------
    # PART B: Gradient-Domain HDR Compression (gamma in {0.85, 0.70, 0.55})
    # --------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("BONUS PART 2: GRADIENT-DOMAIN HDR COMPRESSION")
    print("=" * 72)

    gammas = [0.85, 0.70, 0.55]
    results = []

    p1_orig, p99_orig = np.percentile(y_orig[y_orig > 0], [1, 99])
    dr_orig_db = 20.0 * np.log10(p99_orig / (p1_orig + eps))
    lc_orig = compute_local_contrast(y_orig / np.percentile(y_orig, 99.5))

    mag_orig = np.sqrt(gx_orig**2 + gy_orig**2)
    alpha_val = 0.1 * np.mean(mag_orig)
    print(f"Luminance Y range: [{y_orig.min():.6f}, {y_orig.max():.6f}]")
    print(f"Log-luminance u range: [{u_orig.min():.4f}, {u_orig.max():.4f}]")
    print(
        f"Original ||g|| -> mean: {np.mean(mag_orig):.4f}, max: {np.max(mag_orig):.4f}, "
        f"alpha: {alpha_val:.4f}"
    )
    print(
        f"Original HDR Luminance -> Dynamic Range (P99/P1): {dr_orig_db:.2f} dB | "
        f"Local Gradient Contrast: {lc_orig:.4f}\n"
    )

    for gamma in gammas:
        gx_mod, gy_mod = compress_gradient_field(gx_orig, gy_orig, gamma=gamma, alpha_scale=0.1)

        mag_mod = np.sqrt(gx_mod**2 + gy_mod**2)
        mask = mag_orig > 1e-8
        cos_sim = float(
            np.min(
                (gx_orig[mask] * gx_mod[mask] + gy_orig[mask] * gy_mod[mask])
                / (mag_orig[mask] * mag_mod[mask])
            )
        )
        print(
            f"[gamma = {gamma:.2f}] Modified ||g_tilde|| -> mean: {np.mean(mag_mod):.4f}, "
            f"max: {np.max(mag_mod):.4f} | min cos(theta): {cos_sim:.6f}"
        )

        rhs_mod = div_backward(gx_mod, gy_mod)

        # Scale Dirichlet boundary consistently with gamma so boundary matches compressed dynamic range
        u_bound = gamma * u_orig
        u_hdr, res_hdr = solve_poisson_multigrid(
            rhs_mod,
            u_boundary=u_bound,
            max_levels=6,
            nu1=3,
            nu2=3,
            max_cycles=25,
            rel_tol=1e-7,
            abs_tol=1e-8,
        )

        y_hdr = np.maximum(np.exp(u_hdr) - eps, 0.0)
        lo, hi = np.percentile(y_hdr, [0.5, 99.5])
        y_disp = np.clip((y_hdr - lo) / (hi - lo + eps), 0.0, 1.0)

        # Color reconstruction using best strategy from Question 2 (beta = 0.65)
        rgb_hdr = reconstruct_color_chromaticity_ratio(rgb_orig, y_orig, y_disp, beta=0.65)

        p1, p99 = np.percentile(y_disp[y_disp > 1e-5], [1, 99])
        dr_db = 20.0 * np.log10(p99 / (p1 + 1e-5))
        lc = compute_local_contrast(y_disp)

        print(
            f"               Poisson V-cycles: {len(res_hdr)-1:2d} | Final res: {res_hdr[-1]:.2e} | "
            f"DR (P99/P1): {dr_db:.2f} dB | Local Contrast: {lc:.4f}\n"
        )
        results.append((gamma, y_disp, rgb_hdr, dr_db, lc))

    # Plot comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes[0, 0].imshow(np.clip(y_orig ** 0.35, 0, 1), cmap="gray")
    axes[0, 0].set_title(f"Original Luminance ($\\gamma_{{disp}}=0.35$)\nDR: {dr_orig_db:.1f} dB")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(np.clip(rgb_orig ** 0.35, 0, 1))
    axes[1, 0].set_title("Original Color ($\\gamma_{disp}=0.35$)")
    axes[1, 0].axis("off")

    for idx, (gamma, y_disp, rgb_hdr, dr_db, lc) in enumerate(results, start=1):
        axes[0, idx].imshow(y_disp, cmap="gray")
        axes[0, idx].set_title(f"Gradient HDR $\\gamma = {gamma:.2f}$ (Lum.)\nDR: {dr_db:.1f} dB | LC: {lc:.4f}")
        axes[0, idx].axis("off")

        axes[1, idx].imshow(rgb_hdr)
        axes[1, idx].set_title(f"Gradient HDR $\\gamma = {gamma:.2f}$ (Color, $\\beta=0.65$)")
        axes[1, idx].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "bonus_part2_gradient_hdr.png", dpi=200)
    plt.close()
    print(f"Saved figures to '{out_dir}/'.")