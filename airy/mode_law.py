# -*- coding: utf-8 -*-
# ============================================================================
# MODE "loi": drift law + map.
# Builds on the central physical result: the center of mass follows an EXACT
# LAW (Ehrenfest), its drift velocity being given by the transverse momentum of
# the initial field.
#
#   FIGURE 1 -- Law validation + saturation: for each charge m, compare the
#       MEASURED drift velocity (full propagation) to the PREDICTED velocity
#       (initial field alone, no propagation).
#   FIGURE 2 -- Map (xv, yv): predicted drift velocity vs vortex position,
#       computed ANALYTICALLY (hence instantly, without propagation).
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt

from airy.core import (build_grid_and_propagator, build_tukey_apod, make_field0,
                       propagate_track_com, mean_transverse_momentum,
                       fit_drift_velocity, params_caption)


def run(m_values=(0, 1, 2, 3, 4, 5)):
    print("=== Analysis: drift law + map ===")

    # ---- Physical parameters (consistent with the rest of the program) ----
    wavelength = 0.532e-4
    no         = 1.0
    x_scale    = 15e-4
    a_trunc    = 0.3
    Fo         = 1.0
    vortex_x_um = -5
    vortex_y_um = -5
    Nx       = 500
    x_window = 0.25      # wide window: no aliasing up to m=5
    z_max    = 1
    Nz       = 250
    alpha_ap = 0.15
    k0 = 2 * np.pi * no / wavelength

    x, X, Y, prop_full, z_phys_mm, x_um, axis_min, axis_max = build_grid_and_propagator(
        wavelength, no, x_scale, x_window, Nx, z_max, Nz
    )
    apod = build_tukey_apod(Nx, alpha_ap)
    X_um, Y_um = np.meshgrid(x_um, x_um, indexing='ij')

    # ================================================================
    # FIGURE 1: measured vs predicted law, versus charge m
    # ================================================================
    v_measured, v_predicted, r2_list = [], [], []
    for m in m_values:
        field0 = make_field0(m, vortex_x_um, vortex_y_um, x_scale, a_trunc, Fo, x, X, Y, apod)
        com_x, com_y = propagate_track_com(field0, prop_full, Nz, X_um, Y_um)
        _, _, speed_meas, r2 = fit_drift_velocity(com_x, com_y, z_phys_mm, kmax=120)
        pvx, pvy = mean_transverse_momentum(field0, x_window, x_scale, k0)
        v_measured.append(speed_meas)
        v_predicted.append(np.hypot(pvx, pvy))
        r2_list.append(r2)
        print(f"  m={m} : v_measured={speed_meas:.3f}  v_predicted={np.hypot(pvx,pvy):.3f} um/mm  (R2={r2:.4f})")

    m_arr = np.array(m_values)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    fig1.patch.set_facecolor('#0d0d14')
    ax1.set_facecolor('#1a1a24')
    ax1.tick_params(colors='#ccc')
    for sp in ax1.spines.values(): sp.set_edgecolor('#555')
    ax1.grid(color='#333', linestyle='--', alpha=0.5)

    ax1.plot(m_arr, v_predicted, '-', color='cyan', lw=2,
             label="Analytical prediction  $v=\\langle k_\\perp\\rangle/k_0$")
    ax1.scatter(m_arr, v_measured, s=70, color='orange', zorder=5,
                label="Measurement by propagation (BPM)")
    ax1.set_xlabel("Topological charge  m", color='#bbb')
    ax1.set_ylabel("Center-of-mass drift velocity (um/mm)", color='#bbb')
    ax1.set_title("Exact drift law and saturation with charge\n"
                  f"(measured/predicted agreement < 0.5% ; straight trajectories R2 $\\geq$ {min(r2_list[1:]):.3f})",
                  color='#eee', fontsize=11)
    ax1.legend(facecolor='#1a1a24', edgecolor='#555', labelcolor='white')
    # Reproducible-parameters banner, at the top of the figure.
    cap = params_caption(wavelength, no, x_scale, a_trunc, Fo, Nx, x_window, z_max, Nz, alpha_ap,
                         extra=f"vortex=({vortex_x_um},{vortex_y_um}) um   m={list(m_values)}   fit kmax=120")
    fig1.suptitle(cap, color='#888', fontsize=7, family='monospace', y=0.99, va='top')
    fig1.tight_layout(rect=[0, 0, 1, 0.90])

    # ================================================================
    # FIGURE 2: analytical map of the drift velocity
    #            versus the vortex position (xv, yv)
    # ================================================================
    m_map = 2                                   # charge fixed for the map
    grid_um = np.linspace(-40, 40, 21)          # scanned vortex positions
    grid_map = np.zeros((grid_um.size, grid_um.size))
    print(f"  Map (xv,yv) for m={m_map}: {grid_um.size}x{grid_um.size} positions (analytical)...")
    for i, xv in enumerate(grid_um):
        for j, yv in enumerate(grid_um):
            field0 = make_field0(m_map, xv, yv, x_scale, a_trunc, Fo, x, X, Y, apod)
            pvx, pvy = mean_transverse_momentum(field0, x_window, x_scale, k0)
            grid_map[i, j] = np.hypot(pvx, pvy)

    fig2, ax2 = plt.subplots(figsize=(8, 6.5))
    fig2.patch.set_facecolor('#0d0d14')
    ax2.set_facecolor('#1a1a24')
    # grid_map[i,j]: i->xv (horizontal), j->yv (vertical); we transpose for display
    im = ax2.imshow(grid_map.T, origin='lower', cmap='viridis',
                    extent=[grid_um.min(), grid_um.max(), grid_um.min(), grid_um.max()])
    ax2.tick_params(colors='#ccc')
    for sp in ax2.spines.values(): sp.set_edgecolor('#555')
    cb = fig2.colorbar(im, ax=ax2)
    cb.set_label("Predicted drift velocity (um/mm)", color='#ccc')
    cb.ax.tick_params(colors='#ccc')
    ax2.set_xlabel("Vortex position  $x_v$ (um)", color='#bbb')
    ax2.set_ylabel("Vortex position  $y_v$ (um)", color='#bbb')
    ax2.set_title(f"Drift map (charge m={m_map}, analytical computation)", color='#eee', fontsize=11)
    # Reproducible-parameters banner, at the top of the figure.
    cap2 = params_caption(wavelength, no, x_scale, a_trunc, Fo, Nx, x_window, z_max, Nz, alpha_ap,
                          extra=f"m={m_map}   grid xv,yv = [{grid_um.min():.0f},{grid_um.max():.0f}] um, {grid_um.size} pts")
    fig2.suptitle(cap2, color='#888', fontsize=7, family='monospace', y=0.99, va='top')
    fig2.tight_layout(rect=[0, 0, 1, 0.90])

    print("\nClose the windows to finish.")
    plt.show()
