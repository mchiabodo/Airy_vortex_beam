# -*- coding: utf-8 -*-
# ============================================================================
# MODE "oam": orbital angular momentum <Lz> (validation + conservation).
# For each charge m, propagate the vortex beam (vortex centered on the main
# lobe) and compute <Lz>/hbar at each z plane. Two results:
#   - LEFT : mean measured <Lz> vs charge m (should be a line of slope 1);
#   - RIGHT: <Lz>(z) for each m (should stay constant = OAM conservation).
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy

from airy.core import (build_grid_and_propagator, build_tukey_apod, make_field0,
                       propagate_full, compute_oam, params_caption)


def run(m_values=(0, 1, 2, 3, 4, 5)):
    print("=== Orbital angular momentum <Lz>: validation + conservation ===")

    # ---- Parameters (taken from the OAM computation program) ----
    wavelength = 0.532e-4
    no         = 1.0
    x_scale    = 20e-4
    a_trunc    = 0.1
    Fo         = 1.0
    # WIDENED window (0.5 cm) + fine grid (640): the original program used
    # 0.10 cm, but without the apodization-at-every-step (which masked the
    # problem by absorbing energy), the beam reaches the edges and aliases the
    # spectrum, which corrupts <Lz> (large standard deviation). With this window,
    # <Lz> is well conserved (std dev ~0.01) and ~ m up to m=5.
    Nx       = 640
    x_window = 0.5
    z_max    = 2.0
    Nz       = 120
    alpha_ap = 0.15

    x, X, Y, prop_full, z_phys_mm, x_um, axis_min, axis_max = build_grid_and_propagator(
        wavelength, no, x_scale, x_window, Nx, z_max, Nz
    )
    apod = build_tukey_apod(Nx, alpha_ap)
    dx = x[1] - x[0]

    # Vortex centered on the Airy main lobe (well-defined OAM ~ m).
    Ai_x, _, _, _ = airy(x)
    x0_um = x[np.argmax(Ai_x)] * x_scale * 1e4

    results = {}
    for m in m_values:
        field0 = make_field0(m, x0_um, x0_um, x_scale, a_trunc, Fo, x, X, Y, apod)
        fields = propagate_full(field0, prop_full, Nz, Nx)
        lz = compute_oam(fields, x, dx)
        results[m] = lz
        print(f"  m={m:+d} : <Lz>/hbar = {lz.mean():.4f} +/- {lz.std():.4f}  (theory: {m})")

    palette = ['cyan', 'lime', 'yellow', 'orange', 'magenta', 'red', '#ff4466', '#88aaff']

    fig, (ax_v, ax_c) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0d0d14')
    for ax in (ax_v, ax_c):
        ax.set_facecolor('#1a1a24')
        ax.tick_params(colors='#ccc')
        for sp in ax.spines.values(): sp.set_edgecolor('#555')
        ax.grid(color='#333', linestyle='--', alpha=0.5)

    # --- LEFT: measured <Lz> vs m + regression ---
    m_arr = np.array(m_values, dtype=float)
    lz_mean = np.array([results[m].mean() for m in m_values])
    lz_std = np.array([results[m].std() for m in m_values])
    coeffs = np.polyfit(m_arr, lz_mean, 1)
    mfit = np.linspace(m_arr.min() - 0.5, m_arr.max() + 0.5, 100)
    ss_res = np.sum((lz_mean - np.polyval(coeffs, m_arr))**2)
    ss_tot = np.sum((lz_mean - lz_mean.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    ax_v.plot(mfit, mfit, '--', color='#888', lw=1.5, label="theory (slope 1)")
    ax_v.plot(mfit, np.polyval(coeffs, mfit), '-', color='#e06c35', lw=2,
              label=f"regression (slope {coeffs[0]:.4f})")
    ax_v.errorbar(m_arr, lz_mean, yerr=lz_std, fmt='none', ecolor='#ccc', elinewidth=1.2, capsize=4, alpha=0.6)
    for i, m in enumerate(m_values):
        ax_v.scatter(m, lz_mean[i], color=palette[i % len(palette)], s=80, zorder=5, edgecolors='white', linewidths=0.6)
    ax_v.set_xlabel("Topological charge m", color='#bbb')
    ax_v.set_ylabel("<Lz>/hbar  (measured)", color='#bbb')
    ax_v.set_title(f"Measurement vs theory   (R2 = {r2:.5f})", color='#eee', fontsize=11)
    ax_v.legend(facecolor='#1a1a24', edgecolor='#555', labelcolor='white', fontsize=9, loc='lower right')

    # --- RIGHT: <Lz>(z) for each charge (conservation) ---
    for i, m in enumerate(m_values):
        col = palette[i % len(palette)]
        ax_c.plot(z_phys_mm, results[m], color=col, lw=1.8, label=f"m = {m} (mean {results[m].mean():.2f})")
        ax_c.axhline(m, color=col, lw=0.7, ls=':', alpha=0.4)
    ax_c.set_xlabel("Propagation z (mm)", color='#bbb')
    ax_c.set_ylabel("<Lz>/hbar", color='#bbb')
    ax_c.set_title("OAM conservation in free propagation", color='#eee', fontsize=11)
    ax_c.legend(facecolor='#1a1a24', edgecolor='#555', labelcolor='white', fontsize=8, loc='right')

    cap = params_caption(wavelength, no, x_scale, a_trunc, Fo, Nx, x_window, z_max, Nz, alpha_ap,
                         extra=f"m={list(m_values)}   vortex on the main lobe")
    fig.suptitle("<Lz>/hbar: simulation vs theory  -  " + cap,
                 color='#888', fontsize=7, family='monospace', y=0.99, va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    print("\nClose the window to finish.")
    plt.show()
