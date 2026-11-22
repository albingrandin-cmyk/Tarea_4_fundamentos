# Multigrid Poisson Solver & Gradient-Domain HDR Compression (Bonus)

Implementation and experimental evaluation of the **Bonus Task (Poisson Reconstruction via Multigrid and Gradient-Domain HDR Compression)** for *IEE2714 — Fundamentos de Procesamiento de Imágenes (Tarea 4)*.

The script implements a custom **Multigrid V-Cycle Poisson Solver** from scratch in NumPy (without any external Poisson or multigrid libraries) and applies it to:

1. **Exact Gradient Reconstruction Validation:** Recovers log-luminance $u = \ln(Y + \varepsilon)$ from unmodified gradients $\mathbf{g} = \nabla u$ by solving $\Delta \hat{u} = \nabla \cdot \mathbf{g}$ subject to Dirichlet boundary conditions taken from $u$.
2. **Gradient-Domain HDR Compression:** Attenuates gradient magnitudes non-linearly while preserving local gradient orientation:
   $$\tilde{\mathbf{g}} = \mathbf{g} \cdot \left(\frac{\Vert{}\mathbf{g}\Vert{} + \varepsilon}{\alpha}\right)^{\gamma - 1}, \qquad \alpha = 0.1\,\overline{\Vert{}\mathbf{g}\Vert{}},$$
   and solves $\Delta \hat{u} = \nabla \cdot \tilde{\mathbf{g}}$ across three compression parameters $\gamma \in \{0.85, 0.70, 0.55\}$, followed by luminance recovery $Y' = \exp(\hat{u}) - \varepsilon$ and chromaticity-ratio color reconstruction ($\beta = 0.65$).

---

## Requirements & Installation

- **Python:** 3.9 or higher
- **Packages:** `numpy`, `imageio`, and `matplotlib`

Install all required dependencies with:

```bash
pip install numpy imageio matplotlib
```

---

## Repository Structure & Input Data

Ensure the input HDR image `memorial.hdr` is placed inside `archivos_t4/`:

```text
.
├── bonus_poisson_multigrid.py       # Main script for Bonus Task (Validation & Gradient HDR)
├── archivos_t4/
│   └── memorial.hdr                 # Input HDR RGB image
└── results_bonus/                   # Generated automatically on execution
    ├── bonus_part1_validation.png   # V-cycle convergence curve & reconstruction error map
    └── bonus_part2_gradient_hdr.png # Luminance & color comparison for gamma in {0.85, 0.70, 0.55}
```

---

## How to Run & Reproduce Results

Run the script directly from the terminal:

```bash
python bonus_poisson_multigrid.py
```

The script automatically executes both parts and saves high-resolution figures (`200 DPI`) inside `results_bonus/`.

### Solver & Compression Parameters

- **Discrete Operators:** Forward-difference gradient $\nabla u$ and backward-difference divergence $\nabla \cdot \mathbf{g}$ on interior nodes (consistent with the 5-point discrete Laplacian $\Delta_h u$).
- **Multigrid V-Cycle Configuration:**
  - **Grid Levels:** $L = 6$ dyadic levels.
  - **Smoother:** Vectorized Red-Black Gauss–Seidel (RBGS) with $\nu_1 = 3$ pre-smoothing sweeps, $\nu_2 = 3$ post-smoothing sweeps, and $50$ RBGS iterations on the coarsest grid.
  - **Inter-Grid Transfers:** $3 \times 3$ full-weighting restriction $\frac{1}{16}\begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$ and bilinear prolongation with zero Dirichlet boundary enforcement ($e_h|_{\partial\Omega} = 0$).
  - **Stopping Criterion:** Relative residual $< 10^{-10}$ or absolute residual $< 10^{-11}$ (max $30$ cycles) for validation; relative residual $< 10^{-7}$ or absolute residual $< 10^{-8}$ (max $25$ cycles) for HDR compression.
- **Gradient-Domain HDR Parameters:** $\alpha = 0.1\,\overline{\|\mathbf{g}\|}$, $\gamma \in \{0.85, 0.70, 0.55\}$, scaled Dirichlet boundary $\hat{u}|_{\partial\Omega} = \gamma u|_{\partial\Omega}$, $[0.5\%, 99.5\%]$ percentile rescaling, and chromaticity-ratio color reconstruction with $\beta = 0.65$.

---

## Generated Outputs (`results_bonus/`)

| Filename | Description |
| :--- | :--- |
| `bonus_part1_validation.png` | Three-panel validation figure showing semi-log V-cycle residual decay $\|r_k\|_2$, reconstructed luminance ($\gamma_{\text{disp}} = 0.35$), and spatial absolute error $\|\hat{u} - u\|$. |
| `bonus_part2_gradient_hdr.png` | Side-by-side comparison of global gamma reference ($\gamma_{\text{disp}} = 0.35$) and gradient-domain HDR luminance and color reconstructions for $\gamma \in \{0.85, 0.70, 0.55\}$. |

### Terminal Metrics

Executing the script also prints:

- **Part 1 (Validation):** Execution time, V-cycle count, residual norm $\|r_k\|_2$ at every V-cycle, and final reconstruction RMSE in both log-luminance ($u$) and linear luminance ($Y$).
- **Part 2 (Gradient HDR):** Linear and log-luminance ranges, original vs. modified gradient magnitude statistics ($\overline{\|\mathbf{g}\|}$, $\max\|\mathbf{g}\|$, threshold $\alpha$), minimum cosine similarity $\min\cos\theta$ confirming exact local direction preservation, V-cycle convergence, dynamic range ($P_{99}/P_1$ in dB), and mean local gradient contrast.
