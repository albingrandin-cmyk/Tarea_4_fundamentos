# HDR Dynamic Range Compression & Color Recovery (Question 2)

Implementation and experimental evaluation of **Question 2 (HDR Compression and Color Reconstruction)** for *IEE2714 — Fundamentos de Procesamiento de Imágenes (Tarea 4)*.

The script applies two dynamic range compression methods—**Homomorphic Filtering** and **Durand & Dorsey (2002) Bilateral Tone Mapping**—to high dynamic range (HDR) RGB images, and compares independent RGB channel processing against two luminance-coupled color reconstruction strategies:

1. **Chromaticity Ratio Strategy ($\beta$):**
   $$C' = Y' \cdot \left(\max\left(\frac{C}{Y + \varepsilon},\, 0\right)\right)^\beta, \quad C \in \{R, G, B\}$$
2. **Explicit HSV Saturation Scaling ($k_S$):**
   $$S' = \mathrm{clip}(k_S \cdot S,\, 0,\, 1)$$
   applied on top of the $\beta = 1.0$ base reconstruction.

---

## Requirements & Installation

- **Python:** 3.9 or higher
- **Packages:** `numpy`, `matplotlib`, `scikit-image`, and `imageio` *(optional; falls back to `skimage.io` if unavailable)*

Install all required dependencies with:

```bash
pip install numpy matplotlib scikit-image imageio
```

---

## Repository Structure & Input Data

Place the input `.hdr` images inside `archivos_t4/` (or in the root working directory, as `resolve_image_path()` automatically checks both locations):

```text
.
├── pregunta2_hdr_color.py       # Main script for Question 2 (Experiments 1–4)
├── archivos_t4/
│   ├── memorial.hdr             # HDR test image 1
│   └── belgium.hdr              # HDR test image 2
└── results_p2/                  # Generated automatically on execution
```

---

## How to Run & Reproduce Results

Run the script directly from the terminal:

```bash
python pregunta2_hdr_color.py
```

By default, `SHOW_PLOTS = False` saves all figures directly to `results_p2/` at **300 DPI** without blocking execution. Set `SHOW_PLOTS = True` at the top of the script to open interactive Matplotlib windows.

### Fixed Internal Parameters

All experiments keep the tone-mapping parameters fixed across both images:

- **Homomorphic Filtering:** $\gamma_L = 0.35$, $\gamma_H = 1.60$, $c = 1.0$, $D_0 = 30.0$, $\varepsilon = 10^{-6}$, percentile rescaling $[0.5\%, 99.5\%]$.
- **Durand Bilateral Tone Mapping:** $\sigma_{\text{spatial}} = 12.0$, $\sigma_{\text{range}} = 0.12$, target base contrast $\Delta L_{\text{target}} = 1.8$, $\varepsilon = 10^{-6}$.
- **Color Reconstruction Sweeps:** $\beta \in \{0.6, 1.0, 1.4\}$ and $k_S \in \{0.6, 1.0, 1.4\}$.

---

## Generated Outputs (`results_p2/`)

For each image (`<name>` $\in \{\texttt{memorial}, \texttt{belgium}\}$), the script generates five publication-ready figures:

| Filename | Description |
| :--- | :--- |
| `exp1_beta_<name>.png` | Full-image comparison of compressed luminance $Y'$, independent RGB filtering, and chromaticity-ratio reconstructions for $\beta \in \{0.6, 1.0, 1.4\}$. |
| `exp1_beta_zooms_<name>.png` | Zoomed crops highlighting intense colors and strong illumination transitions across $\beta$. |
| `exp2_hsv_<name>.png` | Full-image comparison of explicit HSV saturation scaling for $k_S \in \{0.6, 1.0, 1.4\}$. |
| `exp2_hsv_vs_beta_zooms_<name>.png` | Side-by-side zoomed crop comparison between chromaticity ratio ($\beta \in \{0.6, 1.4\}$) and HSV saturation scaling ($k_S \in \{0.6, 1.4\}$). |
| `exp3_main_cases_<name>.png` | Unified overview of the original HDR image (with crop overlays), compressed $Y'$, main color reconstructions, and zoomed insets. |

### Terminal Metrics

Executing the script also prints:

- Pre-clip dynamic ranges and exact RGB / HSV clipping percentages.
- Per-experiment tables reporting **mean HSV saturation** ($\bar{S}_{\text{orig}}$ vs. $\bar{S}_{\text{rec}}$) and **mean circular hue shift** ($\overline{\Delta H}$ in degrees).
- A consolidated quantitative summary across all methods, strategies, and parameter settings.
