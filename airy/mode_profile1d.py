# -*- coding: utf-8 -*-
# ============================================================================
# MODE "profile1d": I(x) and I(y) cuts along the main lobe.
# Simulate a single charge, and display the 1D INTENSITY PROFILE along X and Y,
# at the main-lobe position (automatically tracked by the intensity maximum).
# Interactive figure (z slider) + GIF. Highlights the deformation of the profile
# by the vortex, then its reconstruction (self-healing).
# ============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

from airy.core import (build_grid_and_propagator, build_tukey_apod, make_field0,
                       propagate_full)


def run():
    print("=== 1D intensity profile along the main lobe ===")

    # ---- Parameters (taken from the 1D-profile program) ----
    wavelength = 0.532e-4
    no         = 1.0
    x_scale    = 20e-4
    a_trunc    = 0.1
    Fo         = 1.0
    m          = 2
    vortex_x_um = -200
    vortex_y_um = -18
    Nx       = 256
    x_window = 0.10
    z_max    = 2.0
    Nz       = 200
    alpha_ap = 0.15

    x, X, Y, prop_full, z_phys_mm, x_um, axis_min, axis_max = build_grid_and_propagator(
        wavelength, no, x_scale, x_window, Nx, z_max, Nz
    )
    apod = build_tukey_apod(Nx, alpha_ap)
    field0 = make_field0(m, vortex_x_um, vortex_y_um, x_scale, a_trunc, Fo, x, X, Y, apod)

    fields = propagate_full(field0, prop_full, Nz, Nx)
    intensities = np.abs(fields)**2
    vmax = intensities.max()
    intensities = intensities / vmax   # normalized intensity (0..1) for display

    # Automatic main-lobe tracking: 2D intensity maximum at each z.
    lobe_ix = np.zeros(Nz, dtype=int)   # X index of the lobe
    lobe_iy = np.zeros(Nz, dtype=int)   # Y index of the lobe
    for iz in range(Nz):
        ix, iy = np.unravel_index(np.argmax(intensities[iz]), intensities[iz].shape)
        lobe_ix[iz], lobe_iy[iz] = ix, iy
    lobe_x_um = x_um[lobe_ix]   # X position of the lobe in um
    lobe_y_um = x_um[lobe_iy]   # Y position of the lobe in um

    def build_axes(axx, axy):
        for ax in (axx, axy):
            ax.set_facecolor('#1a1a24')
            ax.tick_params(colors='#ccc', labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor('#555')
            ax.grid(color='#333', linestyle='--', alpha=0.5)
            ax.set_xlim(axis_min, axis_max); ax.set_ylim(0, 1.0)
        axx.set_xlabel("X (um)", color='#bbb', fontsize=9)
        axx.set_ylabel("Normalized intensity", color='#bbb', fontsize=9)
        axx.set_title("Profile I(x)  |  y = main lobe", color='#eee', fontsize=10)
        axy.set_xlabel("Y (um)", color='#bbb', fontsize=9)
        axy.set_ylabel("Normalized intensity", color='#bbb', fontsize=9)
        axy.set_title("Profile I(y)  |  x = main lobe", color='#eee', fontsize=10)

    # ---------- Interactive figure (z slider) ----------
    fig_live, (ax_px, ax_py) = plt.subplots(1, 2, figsize=(13, 5.5))
    plt.subplots_adjust(bottom=0.18, wspace=0.35)
    fig_live.patch.set_facecolor('#0d0d14')
    build_axes(ax_px, ax_py)

    line_px, = ax_px.plot(x_um, intensities[0][:, lobe_iy[0]], color='#e06c35', lw=1.5, label='I(x)')
    vline_px = ax_px.axvline(lobe_x_um[0], color='cyan', lw=1, ls='--', label='lobe')
    ax_px.legend(facecolor='#1a1a24', edgecolor='#555', labelcolor='white', fontsize=8)
    line_py, = ax_py.plot(x_um, intensities[0][lobe_ix[0], :], color='#c678dd', lw=1.5, label='I(y)')
    vline_py = ax_py.axvline(lobe_y_um[0], color='lime', lw=1, ls='--', label='lobe')
    ax_py.legend(facecolor='#1a1a24', edgecolor='#555', labelcolor='white', fontsize=8)

    ttl_live = fig_live.suptitle(f"1D profiles  |  m={m}  |  z=0.000 mm", color='#eee', fontsize=11, y=0.96)

    ax_sl = fig_live.add_axes([0.15, 0.05, 0.7, 0.025])
    ax_sl.set_facecolor('#1a1a24')
    sl = Slider(ax_sl, 'z-frame', 0, Nz - 1, valinit=0, valstep=1, color='#e06c35', track_color='#333')
    sl.label.set_color('#ccc'); sl.valtext.set_color('#ccc')
    fig_live.sl = sl   # keep a reference (otherwise the slider is disabled by the GC)

    def update(val):
        k = int(sl.val)
        line_px.set_ydata(intensities[k][:, lobe_iy[k]]); vline_px.set_xdata([lobe_x_um[k]])
        line_py.set_ydata(intensities[k][lobe_ix[k], :]); vline_py.set_xdata([lobe_y_um[k]])
        ttl_live.set_text(f"1D profiles  |  m={m}  |  z={z_phys_mm[k]:.3f} mm  |  "
                          f"lobe @ ({lobe_x_um[k]:.1f}, {lobe_y_um[k]:.1f}) um")
        fig_live.canvas.draw_idle()
    sl.on_changed(update)

    print("\nClose the interactive window to generate the GIF.")
    plt.show()

    # ---------- GIF ----------
    print("\nGenerating the GIF...")
    fig_gif, (axg_x, axg_y) = plt.subplots(1, 2, figsize=(13, 5.5))
    plt.subplots_adjust(wspace=0.35)
    fig_gif.patch.set_facecolor('#0d0d14')
    build_axes(axg_x, axg_y)

    lg_x, = axg_x.plot(x_um, intensities[0][:, lobe_iy[0]], color='#e06c35', lw=1.5, label='I(x)')
    vg_x = axg_x.axvline(lobe_x_um[0], color='cyan', lw=1, ls='--', label='lobe')
    axg_x.legend(facecolor='#1a1a24', edgecolor='#555', labelcolor='white', fontsize=8)
    lg_y, = axg_y.plot(x_um, intensities[0][lobe_ix[0], :], color='#c678dd', lw=1.5, label='I(y)')
    vg_y = axg_y.axvline(lobe_y_um[0], color='lime', lw=1, ls='--', label='lobe')
    axg_y.legend(facecolor='#1a1a24', edgecolor='#555', labelcolor='white', fontsize=8)
    ttl_gif = fig_gif.suptitle(f"1D profiles  |  m={m}  |  z=0.000 mm", color='#eee', fontsize=11, y=0.96)
    elements = [lg_x, vg_x, lg_y, vg_y, ttl_gif]

    def update_gif(k):
        lg_x.set_ydata(intensities[k][:, lobe_iy[k]]); vg_x.set_xdata([lobe_x_um[k]])
        lg_y.set_ydata(intensities[k][lobe_ix[k], :]); vg_y.set_xdata([lobe_y_um[k]])
        ttl_gif.set_text(f"1D profiles  |  m={m}  |  z={z_phys_mm[k]:.3f} mm")
        return elements

    gif_step = 2
    ani = animation.FuncAnimation(fig_gif, update_gif, frames=range(0, Nz, gif_step), interval=50, blit=True)
    gif_path = os.path.join(os.getcwd(), f"airy_profil1d_m{m}.gif")
    ani.save(gif_path, writer="pillow", fps=20, dpi=90)
    plt.close(fig_gif)
    print(f"1D-profile GIF saved: {gif_path}")
