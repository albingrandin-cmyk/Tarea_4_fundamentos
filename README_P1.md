# Multiscale Image Representations: Laplacian Pyramid vs. Starlet Transform (Question 1)

Implementation and experimental comparison of **Question 1 (Laplacian Pyramid and Starlet Transform)** for *IEE2714 — Fundamentos de Procesamiento de Imágenes (Tarea 4)*.

The script constructs two explicit 5-scale multiscale representations from scratch using a separable 1D $B_3$-spline (binomial) filter $w = \frac{1}{16}[1, 4, 6, 4, 1]$ with symmetric `reflect` boundary handling:

1. **Decimated Laplacian Pyramid (5 detail levels + residual):**
   - **Reduction:** Separable 2D smoothing $H = w^T w$ followed by dyadic downsampling by a factor of $2$.
   - **Expansion ($U$):** Zero-injection upsampling by a factor of $2$ followed by separable interpolation with $4H$ ($2w$ per axis).
   - **Decomposition & Reconstruction:**
     $$d_j = c_{j-1} - U(c_j), \qquad c_{j-1} = U(c_j) + d_j, \quad j = 1, \dots, 5$$
2. **Undecimated Starlet Transform (5 detail scales + residual via *à trous*):**
   - **Dilated Kernels:** Inserts $2^{j-1} - 1$ zeros between filter taps of $w$ at each scale $j$ without subsampling.
   - **Decomposition & Reconstruction:**
     $$w_j = c_{j-1} - c_j, \qquad c_0 = c_5 + \sum_{j=1}^{5} w_j$$

For visualization of HDR luminance, outputs are mapped via global logarithmic compression:
$$I_{\text{disp}} = \frac{\ln(1 + k \cdot \operatorname{clip}(I, 0, 1))}{\ln(1 + k)}, \quad k = 500.$$

---

## Requirements & Installation

- **Python:** 3.9 or higher
- **Packages:** `numpy`, `scipy`, `scikit-image`, and `matplotlib`

Install all required dependencies with:

```bash
pip install numpy scipy scikit-image matplotlib
```

---

## Repository Structure & Input Data

Ensure the input HDR luminance image `memorial.tif` is placed inside `archivos_t4/`:

```text
.
├── pregunta1_multiescala.py         # Main script for Question 1 (Verification & Experiments 1–3)
├── archivos_t4/
│   └── memorial.tif                 # Input HDR image (converted to normalized luminance [0, 1])
├── exp1_multiscale_layers.png       # Generated: 5 detail layers + residual comparison
├── exp2_hard_thresholding.png       # Generated: Relative hard thresholding comparison
└── exp3_gain_enhancement.png        # Generated: Fine- vs. coarse-scale gain enhancement
```

---

## How to Run & Reproduce Results

Run the script directly from the terminal:

```bash
python pregunta1_multiescala.py
```

The script executes the following pipeline automatically:

1. **Exact Reconstruction Verification:** Decomposes and reconstructs unmodified normalized luminance $c_0 \in [0, 1]$ and prints the numerical reconstruction RMSE for both transforms.
2. **Experiment 1 (Layer Dimensions & Visualization):** Compares spatial dimensions and coefficient redundancy across all 5 scales and saves `exp1_multiscale_layers.png` using a shared symmetric amplitude scale ($\pm p_{99.5}$) for detail layers.
3. **Experiment 2 (Relative Hard Thresholding):** Applies $T_j = \alpha \max(|d_j|)$ for $\alpha \in \{0.02, 0.10, 0.25\}$, logs retained coefficient percentages per layer and reconstruction RMSE, and saves `exp2_hard_thresholding.png`.
4. **Experiment 3 (Multiplicative Gain Enhancement):** Evaluates symmetric gain profiles across scales and saves `exp3_gain_enhancement.png`:
   - **Original (Reference):** $g = [1.0, 1.0, 1.0, 1.0, 1.0]$
   - **Fine-Scale Emphasis:** $g = [3.5, 2.5, 1.5, 1.0, 1.0]$
   - **Coarse-Scale Emphasis:** $g = [1.0, 1.0, 1.5, 2.5, 3.5]$

---

## Generated Outputs

| Filename | Description |
| :--- | :--- |
| `exp1_multiscale_layers.png` | Side-by-side grid of detail layers $d_1 \dots d_5$ (Laplacian) and $w_1 \dots w_5$ (Starlet) plus coarse residuals $c_5$ with exact layer dimensions. |
| `exp2_hard_thresholding.png` | Full-image and zoomed-crop comparisons after hard thresholding at $\alpha \in \{0.02, 0.10, 0.25\}$, annotated with retained coefficient percentages and RMSE. |
| `exp3_gain_enhancement.png` | Full-image and zoomed-crop comparisons contrasting fine-scale vs. coarse-scale multiplicative enhancement across both representations. |

### Terminal Metrics

Executing the script also prints:

- Exact reconstruction RMSE before coefficient modification.
- Layer-by-layer dimensions and total coefficient redundancy factors relative to the input image.
- Per-layer and total retained coefficient percentages and RMSE across all $\alpha$ thresholds.
- Reconstruction RMSE and pre-clip dynamic ranges for each gain configuration.
