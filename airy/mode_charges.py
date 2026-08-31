# -*- coding: utf-8 -*-
# ============================================================================
# MODE "charges": center-of-mass trajectory for several charges m.
# Simulate the propagation for EACH topological charge in the list, keeping
# everything else identical, then plot how the vortex charge deflects the
# beam's center-of-mass trajectory.
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt

from airy.core import (build_grid_and_propagator, build_tukey_apod, make_field0,
                       propagate_track_com, params_caption)


def run(m_values=(0, 1, 2, 3, 4, 5), limit_frame=150):
    print(f"=== Center-of-mass trajectory for m = {list(m_values)} ===")

    # ---- Physical parameters (identical to the simple simulation) ----
    wavelength = 0.532e-4        # Wavelength in cm (532 nm)
    no         = 1.0             # refractive index of air
    x_scale    = 15e-4           # Characteristic main-lobe width in cm (15 um)
    a_trunc    = 0.3             # Truncation factor of the Airy beam
    Fo         = 1.0             # Normalized initial intensity

    vortex_x_um = -5             # Vortex position (x) in um
    vortex_y_um = -5             # Vortex position (y) in um

    Nx       = 500               # Number of spatial-grid points
    # x_window enlarged to 0.25 cm here: the central "hole" of the vortex grows
    # with the charge m, so the intensity ring moves away from the center. A
    # window too small would make the high-charge beam "spill over" onto the
    # opposite edge (FFT spectral aliasing), which would bias the center-of-mass
    # computation. At 0.25 cm, this aliasing is zero for all charges from m=0 to
    # m=5 (numerically verified).
    x_window = 0.25              # Size of the observation window, in cm
    z_max    = 1                 # Propagation distance in cm
    Nz       = 250               # Number of computation steps
    alpha_ap = 0.15              # Relative width of the smoothed edge zone

    # The grid and the propagation operator do not depend on m:
    # so we compute them only once, before the loop over the charges.
    x, X, Y, prop_full, z_phys_mm, x_um, axis_min, axis_max = build_grid_and_propagator(
        wavelength, no, x_scale, x_window, Nx, z_max, Nz
    )
    apod = build_tukey_apod(Nx, alpha_ap)
    X_um, Y_um = np.meshgrid(x_um, x_um, indexing='ij')

    # A distinct color per charge m (recycled if the list exceeds the palette).
    palette = ['cyan', 'lime', 'yellow', 'orange', 'magenta', 'red']

    # We compute the center-of-mass trajectory for each charge m.
    results = {}
    for m in m_values:
        print(f"  - propagation for m = {m} ...")
        field0 = make_field0(m, vortex_x_um, vortex_y_um, x_scale, a_trunc, Fo, x, X, Y, apod)
        com_x, com_y = propagate_track_com(field0, prop_full, Nz, X_um, Y_um)
        results[m] = (com_x, com_y)

    # ---- Build the figure: two side-by-side plots ----
    # LEFT   : trajectories in ABSOLUTE position (each charge starts from its own
    #          center-of-mass position at z=0).
    # RIGHT  : trajectories RELATIVE to the start (we subtract the initial
    #          position, so all charges start from a common origin). This view
    #          isolates the drift due to PROPAGATION, without the initial-position
    #          offset linked to the charge.
    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor('#0d0d14')

    for ax in (ax_abs, ax_rel):
        ax.set_facecolor('#1a1a24')
        ax.tick_params(colors='#ccc')
        for sp in ax.spines.values(): sp.set_edgecolor('#555')
        ax.grid(color='#333', linestyle='--', alpha=0.5)

    # --- LEFT plot: absolute position ---
    for i, m in enumerate(m_values):
        com_x, com_y = results[m]
        color = palette[i % len(palette)]
        # Convention from the simple simulation: horizontal axis = com_y, vertical axis = com_x.
        ax_abs.plot(com_y[:limit_frame], com_x[:limit_frame], color=color, lw=2, label=f"Charge m = {m}")
        ax_abs.scatter(com_y[limit_frame-1], com_x[limit_frame-1], color=color, zorder=5)   # end point

    # One "Start (z=0)" marker per charge (the start point is not strictly
    # identical for all m). We put the legend label only on the first one, so as
    # not to repeat the same entry several times.
    for i, m in enumerate(m_values):
        com_x, com_y = results[m]
        label = 'Start (z=0)' if i == 0 else None
        ax_abs.scatter(com_y[0], com_x[0], marker='x', s=140, color='white', label=label, zorder=6)

    ax_abs.set_xlabel("Horizontal position X (um)", color='#bbb')
    ax_abs.set_ylabel("Vertical position Y (um)", color='#bbb')
    ax_abs.set_title("Absolute center-of-mass position", color='#eee', fontsize=12)
    ax_abs.legend(facecolor='#1a1a24', edgecolor='#555', labelcolor='white')

    # --- RIGHT plot: displacement relative to the start ---
    for i, m in enumerate(m_values):
        com_x, com_y = results[m]
        color = palette[i % len(palette)]
        # We subtract the initial position: all curves start from (0,0).
        dx = com_x[:limit_frame] - com_x[0]
        dy = com_y[:limit_frame] - com_y[0]
        ax_rel.plot(dy, dx, color=color, lw=2, label=f"Charge m = {m}")
        ax_rel.scatter(dy[-1], dx[-1], color=color, zorder=5)   # end point

    ax_rel.scatter(0, 0, marker='x', s=140, color='white', label='Start (common origin)', zorder=6)
    ax_rel.set_xlabel("Horizontal center-of-mass displacement (um)", color='#bbb')
    ax_rel.set_ylabel("Vertical center-of-mass displacement (um)", color='#bbb')
    ax_rel.set_title("Drift relative to the start (effect of propagation)", color='#eee', fontsize=12)
    ax_rel.legend(facecolor='#1a1a24', edgecolor='#555', labelcolor='white')

    fig.suptitle(
        "Center-of-mass trajectory vs topological charge\n"
        f"(Vortex placed at X={vortex_x_um}um, Y={vortex_y_um}um)",
        color='#eee', fontsize=14
    )
    # Reproducible-parameters banner, below the title.
    cap = params_caption(wavelength, no, x_scale, a_trunc, Fo, Nx, x_window, z_max, Nz, alpha_ap,
                         extra=f"m={list(m_values)}   limit_frame={limit_frame}")
    fig.text(0.5, 0.90, cap, color='#888', fontsize=7, family='monospace', ha='center', va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.85])

    # ================================================================
    # EXTRA FIGURE: longitudinal deviation decomposed per axis
    # We replot the center-of-mass trajectory, but this time SEPARATING its two
    # components: on the left the deviation along X versus the propagation
    # distance z, on the right the deviation along Y versus z. One curve per
    # charge. This shows how each axis deviates along the propagation (and how
    # the vortex breaks the symmetry between X and Y).
    # ================================================================
    fig2, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(15, 7))
    fig2.patch.set_facecolor('#0d0d14')

    for ax, axis_label, axis_color in [(ax_x, "Horizontal axis (X)", 'cyan'),
                                       (ax_y, "Vertical axis (Y)", 'magenta')]:
        ax.set_facecolor('#1a1a24')
        ax.tick_params(colors='#ccc')
        for sp in ax.spines.values(): sp.set_edgecolor('#555')
        ax.grid(color='#333', linestyle='--', alpha=0.5)
        ax.set_title(axis_label, color=axis_color, fontsize=13)
        ax.set_xlabel("Propagation distance Z (mm)", color='#bbb')

    ax_x.set_ylabel("Deviation X (um)", color='#bbb')
    ax_y.set_ylabel("Deviation Y (um)", color='#bbb')

    z = z_phys_mm[:limit_frame]
    for i, m in enumerate(m_values):
        com_x, com_y = results[m]
        color = palette[i % len(palette)]
        ax_x.plot(z, com_x[:limit_frame], color=color, lw=2, label=f"Charge m = {m}")
        ax_x.scatter(z[-1], com_x[limit_frame-1], color=color, zorder=5)   # end point
        ax_y.plot(z, com_y[:limit_frame], color=color, lw=2, label=f"Charge m = {m}")
        ax_y.scatter(z[-1], com_y[limit_frame-1], color=color, zorder=5)

    # Single "Start (z=0)" marker per plot (label only on the 1st m).
    for i, m in enumerate(m_values):
        com_x, com_y = results[m]
        lab = 'Start (z=0)' if i == 0 else None
        ax_x.scatter(z[0], com_x[0], marker='x', s=120, color='white', label=lab, zorder=6)
        ax_y.scatter(z[0], com_y[0], marker='x', s=120, color='white', label=lab, zorder=6)

    ax_x.legend(facecolor='#1a1a24', edgecolor='#555', labelcolor='white')
    ax_y.legend(facecolor='#1a1a24', edgecolor='#555', labelcolor='white')

    fig2.suptitle(
        "Longitudinal center-of-mass deviation, decomposed per axis\n"
        f"(Vortex placed at X={vortex_x_um}um, Y={vortex_y_um}um)",
        color='#eee', fontsize=14
    )
    fig2.text(0.5, 0.90, cap, color='#888', fontsize=7, family='monospace', ha='center', va='top')
    fig2.tight_layout(rect=[0, 0, 1, 0.85])

    print("\nClose the windows to finish.")
    plt.show()
