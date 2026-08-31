# -*- coding: utf-8 -*-
# ============================================================================
# MODE "single": simulate ONE topological charge m and display the result.
# Interactive figure (intensity and/or phase, z slider) + center-of-mass
# analysis + GIF export.
# ============================================================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable

from airy.core import (build_grid_and_propagator, build_tukey_apod, make_field0,
                       propagate_full, params_caption)


def run(display_mode="intensity"):
    """Mode "single": simulate ONE topological charge m and display the result.

    Steps: build the initial field -> split-step propagation (all z planes kept)
    -> compute the center of mass -> interactive figure (intensity and/or phase,
    z slider) + center-of-mass analysis figure, then export a GIF animation. All
    settings (m, vortex position, grid, etc.) are at the top of the function.
    MEMORY WARNING: this mode keeps all fields (Nz x Nx x Nx complex) -> keep Nx
    reasonable (see note below).

    display_mode: "both" (intensity + phase), "intensity" or "phase".
    """
    print("=== Airy 2D + OAM  |  Interactive display option ===")
    start_time = time.time()   # record the start time, to report how long the computation took

    # DISPLAY zoom ONLY (in um, half-width around 0): changes nothing in the
    # simulation grid (x_window), so no risk of aliasing. Set to None to display
    # the whole simulated window.
    zoom_um = 800

    # ================================================================
    # PHYSICAL PARAMETERS
    # ================================================================
    wavelength = 0.532e-4        # Wavelength in cm (i.e. 532 nm, a common laser green)
    no         = 1.0             # refractive index of air
    x_scale    = 15e-4           # Characteristic width of the beam's main lobe in cm (here 15 um)
    a_trunc    = 0.3             # Truncation factor of the Airy beam
    m          = 8              # Topological charge of the optical vortex (OAM)
    Fo         = 1.0             # Initial intensity of the normalized electric field

    # Vortex coordinates (x,y) (in um)
    vortex_x_um = -20
    vortex_y_um = -40

    # ================================================================
    # DYNAMIC GRID
    # ================================================================
    Nx       = 700               # Number of sampling points on the spatial grid (x,y)
    # WARNING (memory): "single" mode keeps ALL propagated fields in memory
    # (Nz x Nx x Nx complex numbers), needed for the slider and the GIF. So keep
    # Nx reasonable: Nx=700, Nz=250 -> ~2 GB; Nx=1500 -> ~13 GB (memory crash on
    # most PCs).
    # The x_window must stay wide enough that the beam does not reach the edges
    # during propagation (otherwise spectral aliasing); check case by case
    # depending on m, z_max and the vortex position.
    x_window = 0.18              # Size of the observation window, in cm
    z_max    = 0.3                 # Propagation distance in cm
    Nz       = 250               # Number of computation steps along the propagation
    alpha_ap = 0.15              # Relative width of the smoothed zone at the grid edges (15%)

    x, X, Y, prop_full, z_phys_mm, x_um, axis_min, axis_max = build_grid_and_propagator(
        wavelength, no, x_scale, x_window, Nx, z_max, Nz
    )
    apod = build_tukey_apod(Nx, alpha_ap)
    field0 = make_field0(m, vortex_x_um, vortex_y_um, x_scale, a_trunc, Fo, x, X, Y, apod)

    # view_min/view_max are used ONLY to frame the display (set_xlim/ylim). The
    # imshow keeps extent=[axis_min, axis_max] (full, non-recut data): zooming
    # here resamples nothing and introduces no aliasing.
    if zoom_um is not None:
        view_min, view_max = -min(zoom_um, axis_max), min(zoom_um, axis_max)
    else:
        view_min, view_max = axis_min, axis_max

    # ================================================================
    # PROPAGATION LOOP: the core of the physical computation
    # ================================================================
    fields = propagate_full(field0, prop_full, Nz, Nx)

    intensities = np.abs(fields)**2
    phases = np.angle(fields)
    vmax = intensities.max()

    # ================================================================
    # CENTER-OF-MASS COMPUTATION
    # ================================================================
    X_um, Y_um = np.meshgrid(x_um, x_um, indexing='ij')
    com_x = np.zeros(Nz)
    com_y = np.zeros(Nz)

    for iz in range(Nz):
        I = intensities[iz]
        total_I = np.sum(I)
        if total_I > 0:
            com_x[iz] = np.sum(X_um * I) / total_I
            com_y[iz] = np.sum(Y_um * I) / total_I
        else:
            com_x[iz] = 0
            com_y[iz] = 0

    # ================================================================
    # STRUCTURAL SETUP OF THE LIVE FIGURE
    # ================================================================
    if display_mode == "both":
        fig_live, (ax_int, ax_phase) = plt.subplots(1, 2, figsize=(12, 5.5))
        plt.subplots_adjust(bottom=0.18, wspace=0.35)
    elif display_mode == "intensity":
        fig_live, ax_int = plt.subplots(1, 1, figsize=(7, 6))
        plt.subplots_adjust(bottom=0.18)
        ax_phase = None
    elif display_mode == "phase":
        fig_live, ax_phase = plt.subplots(1, 1, figsize=(7, 6))
        plt.subplots_adjust(bottom=0.18)
        ax_int = None
    else:
        raise ValueError("display_mode must be 'both', 'intensity' or 'phase'")

    fig_live.patch.set_facecolor('#0d0d14')
    active_axes = [ax for ax in [ax_int, ax_phase] if ax is not None]

    for ax in active_axes:
        ax.set_facecolor('#0d0d14')
        ax.tick_params(colors='#999', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#333')
        ax.set_xlabel("X (um)", color='#bbb', fontsize=9)
        ax.set_ylabel("Y (um)", color='#bbb', fontsize=9)
        ax.set_xlim(view_min, view_max)
        ax.set_ylim(view_min, view_max)

    if ax_int is not None:
        im_int = ax_int.imshow(intensities[0], extent=[axis_min, axis_max, axis_min, axis_max], origin='lower', cmap='inferno', vmin=0, vmax=vmax)
        ax_int.set_title("Intensity", color='#eee', fontsize=10)
        divider_int = make_axes_locatable(ax_int)
        fig_live.colorbar(im_int, cax=divider_int.append_axes("right", size="5%", pad=0.1)).set_label("Norm. intensity", color='#999', fontsize=8)
        com_pt_live, = ax_int.plot([com_y[0]], [com_x[0]], 'w+', markersize=10, markeredgewidth=1.5, alpha=0.8)

    if ax_phase is not None:
        im_phase = ax_phase.imshow(phases[0], extent=[axis_min, axis_max, axis_min, axis_max], origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi)
        ax_phase.set_title("Phase (rad)", color='#eee', fontsize=10)
        divider_phase = make_axes_locatable(ax_phase)
        fig_live.colorbar(im_phase, cax=divider_phase.append_axes("right", size="5%", pad=0.1)).set_label("Phase (rad)", color='#999', fontsize=8)

    ttl_live = fig_live.suptitle(f"Propagation  |  m={m}  |  z=0.000 mm", color='#eee', fontsize=11, y=0.96)

    # ================================================================
    # SLIDER FIX: KEEP A REFERENCE IN MEMORY
    # ================================================================
    ax_sl = fig_live.add_axes([0.15, 0.05, 0.7, 0.025])
    ax_sl.set_facecolor('#1a1a24')
    sl = Slider(ax_sl, 'z-frame', 0, Nz - 1, valinit=0, valstep=1, color='#e06c35', track_color='#333')
    sl.label.set_color('#ccc')
    sl.valtext.set_color('#ccc')

    # CRUCIAL LINE: we attach the slider object to the figure to prevent
    # Python's garbage collector from deleting it.
    fig_live.sl = sl

    def update(val):
        k = int(sl.val)
        if ax_int is not None:
            im_int.set_data(intensities[k])
            com_pt_live.set_data([com_y[k]], [com_x[k]])
        if ax_phase is not None:
            im_phase.set_data(phases[k])

        ttl_live.set_text(f"Free propagation  |  Vortex ({vortex_x_um:.1f},{vortex_y_um:.1f}) um  |  z={z_phys_mm[k]:.3f} mm")
        fig_live.canvas.draw_idle()

    sl.on_changed(update)

    # ================================================================
    # QUANTITATIVE ANALYSIS: CENTER-OF-MASS PLOTS
    # ================================================================
    limit_frame = 150

    fig_com, (ax_com_xy, ax_com_z) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig_com.patch.set_facecolor('#0d0d14')

    for ax in [ax_com_xy, ax_com_z]:
        ax.set_facecolor('#1a1a24')
        ax.tick_params(colors='#ccc')
        for sp in ax.spines.values(): sp.set_edgecolor('#555')
        ax.grid(color='#333', linestyle='--', alpha=0.5)

    ax_com_xy.plot(com_y[:limit_frame], com_x[:limit_frame], color='lime', lw=2)
    ax_com_xy.scatter(com_y[0], com_x[0], color='cyan', label='Start (z=0)', zorder=5)
    ax_com_xy.scatter(com_y[limit_frame-1], com_x[limit_frame-1], color='red', label=f'End (z={z_phys_mm[limit_frame-1]:.1f} mm)', zorder=5)
    ax_com_xy.set_xlabel("Horizontal position (um)", color='#bbb')
    ax_com_xy.set_ylabel("Vertical position (um)", color='#bbb')
    ax_com_xy.set_title("2D trajectory of the center of mass", color='white')
    ax_com_xy.legend(facecolor='#1a1a24', edgecolor='#555', labelcolor='white')

    ax_com_z.plot(z_phys_mm[:limit_frame], com_y[:limit_frame], color='cyan', label='Horizontal axis (X)')
    ax_com_z.plot(z_phys_mm[:limit_frame], com_x[:limit_frame], color='magenta', label='Vertical axis (Y)')
    ax_com_z.set_xlabel("Propagation Z (mm)", color='#bbb')
    ax_com_z.set_ylabel("Position (um)", color='#bbb')
    ax_com_z.set_title(f"Longitudinal deviation (0 to {z_phys_mm[limit_frame-1]:.0f} mm)", color='white')
    ax_com_z.legend(facecolor='#1a1a24', edgecolor='#555', labelcolor='white')

    fig_com.suptitle(f"Motion quantification (Vortex charge {m}) - Limited to useful frames", color='#eee')
    # Reproducible-parameters banner.
    cap = params_caption(wavelength, no, x_scale, a_trunc, Fo, Nx, x_window, z_max, Nz, alpha_ap,
                         extra=f"m={m}   vortex=({vortex_x_um},{vortex_y_um}) um   limit_frame={limit_frame}")
    fig_com.text(0.5, 0.91, cap, color='#888', fontsize=7, family='monospace', ha='center', va='top')

    # TIGHT LAYOUT FIX: apply the re-layout ONLY to the statistics figure.
    # Otherwise plt.tight_layout() would crush the neighboring figure (fig_live) and hide the slider.
    fig_com.tight_layout(rect=[0, 0, 1, 0.86])

    print("\nClose the interactive windows to generate the GIF.")
    plt.show()

    # ================================================================
    # RENDERING SETUP FOR THE GIF
    # ================================================================
    print("\nGenerating the GIF...")

    if display_mode == "both":
        fig_gif, (ax_gif_int, ax_gif_phase) = plt.subplots(1, 2, figsize=(12, 5.5))
        plt.subplots_adjust(wspace=0.35)
    elif display_mode == "intensity":
        fig_gif, ax_gif_int = plt.subplots(1, 1, figsize=(7, 6))
        ax_gif_phase = None
    elif display_mode == "phase":
        fig_gif, ax_gif_phase = plt.subplots(1, 1, figsize=(7, 6))
        ax_gif_int = None

    fig_gif.patch.set_facecolor('#0d0d14')
    active_gif_axes = [ax for ax in [ax_gif_int, ax_gif_phase] if ax is not None]

    for ax in active_gif_axes:
        ax.set_facecolor('#0d0d14')
        ax.tick_params(colors='#999', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#333')
        ax.set_xlabel("X (um)", color='#bbb', fontsize=9)
        ax.set_ylabel("Y (um)", color='#bbb', fontsize=9)
        ax.set_xlim(view_min, view_max)
        ax.set_ylim(view_min, view_max)

    return_elements = []

    if ax_gif_int is not None:
        im_gif_int = ax_gif_int.imshow(intensities[0], extent=[axis_min, axis_max, axis_min, axis_max], origin='lower', cmap='inferno', vmin=0, vmax=vmax)
        ax_gif_int.set_title("Intensity", color='#eee', fontsize=10)
        divider_gif_int = make_axes_locatable(ax_gif_int)
        fig_gif.colorbar(im_gif_int, cax=divider_gif_int.append_axes("right", size="5%", pad=0.1)).set_label("Norm. intensity", color='#999', fontsize=8)
        return_elements.append(im_gif_int)

        com_pt_gif, = ax_gif_int.plot([com_y[0]], [com_x[0]], 'w+', markersize=10, markeredgewidth=1.5, alpha=0.8)
        return_elements.append(com_pt_gif)

    if ax_gif_phase is not None:
        im_gif_phase = ax_gif_phase.imshow(phases[0], extent=[axis_min, axis_max, axis_min, axis_max], origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi)
        ax_gif_phase.set_title("Phase", color='#eee', fontsize=10)
        divider_gif_phase = make_axes_locatable(ax_gif_phase)
        fig_gif.colorbar(im_gif_phase, cax=divider_gif_phase.append_axes("right", size="5%", pad=0.1)).set_label("Phase (rad)", color='#999', fontsize=8)
        return_elements.append(im_gif_phase)

    ttl_gif = fig_gif.suptitle(f"Vortex custom position  |  z=0.000 mm", color='#eee', fontsize=11, y=0.99)
    return_elements.append(ttl_gif)
    # Reproducible-parameters banner, fixed (identical on every GIF frame).
    cap = params_caption(wavelength, no, x_scale, a_trunc, Fo, Nx, x_window, z_max, Nz, alpha_ap,
                         extra=f"m={m}   vortex=({vortex_x_um},{vortex_y_um}) um")
    fig_gif.text(0.5, 0.93, cap, color='#888', fontsize=6.5, family='monospace', ha='center', va='top')
    fig_gif.subplots_adjust(top=0.86)

    def update_gif(k):
        if ax_gif_int is not None:
            im_gif_int.set_data(intensities[k])
            com_pt_gif.set_data([com_y[k]], [com_x[k]])
        if ax_gif_phase is not None: im_gif_phase.set_data(phases[k])
        ttl_gif.set_text(f"Propagation  |  Vortex ({vortex_x_um:.1f},{vortex_y_um:.1f}) um  |  z={z_phys_mm[k]:.3f} mm")
        return return_elements

    # GIF SPEED-UP: saving the GIF is BY FAR the slowest step of the program
    # (encoding each frame to the GIF format is costly). We make it much faster
    # in two ways:
    #   - gif_step: we save only one frame out of "gif_step" (e.g. 2 -> we skip
    #     one frame out of two). The physical computation keeps all its steps: we
    #     lighten ONLY the animation, not the simulation.
    #   - gif_dpi: we slightly lower the GIF resolution (120 -> 80).
    # With gif_step=2 and gif_dpi=80, the time drops from about 60 s to ~23 s and
    # the file size from ~69 MB to ~15 MB, with no visible loss. Set gif_step=1
    # and gif_dpi=120 for maximum quality (but slow).
    gif_step = 1
    gif_dpi  = 120
    gif_frames = range(0, Nz, gif_step)

    ani = animation.FuncAnimation(fig_gif, update_gif, frames=gif_frames, interval=50, blit=True)
    gif_path = os.path.join(os.getcwd(), f"airy_vortex_{display_mode}.gif")
    ani.save(gif_path, writer="pillow", fps=20, dpi=gif_dpi)

    plt.close(fig_gif)
    print(f"GIF ({display_mode}) saved successfully: {gif_path}")
