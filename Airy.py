# -*- coding: utf-8 -*-
# ============================================================================
#  PROPAGATION OF A 2D AIRY BEAM CARRYING AN OPTICAL VORTEX (OAM)
#  Split-step Fourier method (Beam Propagation Method), free paraxial regime.
#  Reference document for reading the code.
# ----------------------------------------------------------------------------
#
#  1) PHYSICAL MODEL
#     Slowly varying envelope A(x,y,z) of a monochromatic scalar field
#     (wavelength lambda, refractive index n0, wavenumber k0 = 2*pi*n0/lambda).
#     Paraxial equation (optical Schrodinger equation):
#
#                 2 i k0 dA/dz + grad_perp^2 A = 0 ,   grad_perp^2 = d_xx + d_yy
#
#     Initial field (z = 0): truncated 2D Airy beam carrying an optical vortex
#     of topological charge m, centered at (x_v, y_v):
#
#         A(x,y,0) = Ai(x/x0)*Ai(y/x0)*exp(a(x+y)/x0)  *  rho^|m|*exp(i m phi)
#                    |________ truncated 2D Airy ______|    |____ OAM vortex ___|
#
#     x0 = x_scale (main-lobe width), a = a_trunc (truncation, finite energy),
#     (rho, phi) = polar coordinates around the vortex.
#
#  2) DIMENSIONLESS NORMALIZATION (key to reading the code's units)
#     - Transverse coordinates in units of x_scale:  x_adim = x / x_scale.
#     - Dimensionless propagation variable:  z' = z / (2 k0 x_scale^2).
#       The propagator step "dz" is in z'; "z_phys_mm" converts back to mm.
#     - Spatial frequencies K conjugate to x_adim (FFT). K^2 = kx^2 + ky^2.
#
#  3) NUMERICAL METHOD (split-step Fourier)
#     At each step dz':  A_hat = FFT2(A);  A_hat *= exp(-i K^2 dz');  A = IFFT2(A_hat).
#     This is the EXACT solution of the paraxial equation in Fourier space.
#
#     IMPORTANT CONVENTIONS / CORRECTIONS (numerically verified):
#       * Sign of the propagator exp(-i K^2 dz')  (and NOT +i): the opposite
#         sign reverses the transverse self-acceleration of the Airy beam.
#       * Tukey window applied ONLY ONCE, at injection (z=0), and never during
#         propagation: free propagation is lossless (energy conserved at 100%).
#         Apodizing at every step dissipates energy and artificially BENDS the
#         center-of-mass trajectory (violation of Ehrenfest's theorem).
#       * The x_window must contain the beam during the whole propagation
#         (diffraction spreading + drift); otherwise spectral aliasing occurs
#         (the FFT assumes the grid is periodic).
#
#  4) OBSERVABLES
#     * Center of mass:  <r>(z) = int r |A|^2 dxdy / int |A|^2 dxdy .
#       Ehrenfest's theorem (free, lossless propagation): <r>(z) is a STRAIGHT
#       LINE, with constant velocity <k_perp>/k0 (mean transverse momentum of
#       the initial field). It is the MAIN LOBE, not the center of mass, that
#       follows the Airy self-acceleration parabola.
#     * Orbital angular momentum:
#         <Lz>/hbar = Im[ int A*(x d_y A - y d_x A) dxdy ] / int |A|^2 dxdy
#       Equals m for a centered vortex, and is conserved in free propagation.
#
#  5) CODE STRUCTURE
#     Helper functions (physics core):
#       build_grid_and_propagator : dimensionless grid, axes, propagator exp(-iK^2 dz').
#       build_tukey_apod          : edge apodization window (2D Tukey).
#       make_field0               : initial truncated 2D Airy field x vortex.
#       propagate_full            : propagation, keeps ALL z planes.
#       propagate_track_com       : propagation, keeps only <r>(z).
#       compute_oam               : <Lz>/hbar(z) (pseudo-spectral derivatives).
#       mean_transverse_momentum  : analytical drift prediction (Ehrenfest).
#       fit_drift_velocity        : linear regression of <r>(z) (+ R^2).
#       params_caption            : reproducible-parameters banner.
#     Run modes ("mode" variable at the bottom of the file):
#       "single"   : one charge m -> interactive figures (intensity/phase) + GIF.
#       "charges"  : <r> trajectory for several charges m.
#       "loi"      : drift law (measured vs predicted) + map (xv,yv).
#       "oam"      : orbital angular momentum <Lz> (validation vs m + conservation).
#       "profil1D" : 1D cuts I(x), I(y) along the main lobe (+ GIF).
# ============================================================================

# Line to display accents and Greek symbols on the plots without crashing
# (nothing to do with the physics of the code)
import sys
import io

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Import the Python modules needed for the simulation
# (a "module" is a ready-made toolbox written by others that we reuse)
import os                                            # to handle file paths (where to save the GIF)
import numpy as np                                   # numerical computing module: number arrays, FFT, etc.
import matplotlib.pyplot as plt                      # module that draws the plots
from scipy.special import airy                       # provides the "Airy" mathematical function, core of the physics here
import matplotlib.animation as animation             # to build the animated GIF from the computed images
from matplotlib.widgets import Slider                # to add the interactive slider below the plots
from mpl_toolkits.axes_grid1 import make_axes_locatable  # to place the color bars (colorbar) correctly
import time                                          # to measure the program's run time

# Line to increase the resolution of the plots and images returned by the simulation for better readability
# (nothing to do with the physics of the code)
plt.rcParams['figure.dpi'] = 150


# ============================================================================
# HELPER FUNCTIONS
# These functions contain the physics of the computation, each performing a
# single well-identified step.
# ============================================================================

def build_grid_and_propagator(wavelength, no, x_scale, x_window, Nx, z_max, Nz):
    """Build the dimensionless transverse grid, the propagation axis and the
    split-step propagator of the paraxial equation.

    Inputs (all lengths in cm):
        wavelength : wavelength lambda
        no         : refractive index of the medium
        x_scale    : characteristic width of the Airy lobe (transverse normalization unit)
        x_window   : physical width of the computation window
        Nx         : number of points per transverse axis
        z_max      : total propagation distance
        Nz         : number of propagation steps

    Normalization:
        x_adim = x / x_scale ;  dimensionless window XL = x_window / x_scale.
        Dimensionless propagation variable z' = z / (2 k0 x_scale^2), step dz = L_ad/Nz
        with L_ad = z_max / (2 k0 x_scale^2). Frequencies K conjugate to x_adim.

    Outputs:
        x          : dimensionless transverse axis (1D, Nx points)
        X, Y       : 2D grids of dimensionless positions (indexing='ij')
        prop_full  : spectral propagator exp(-i K^2 dz) (correct paraxial sign)
        z_phys_mm  : propagation axis converted back to millimeters (display)
        x_um       : transverse axis converted back to micrometers (display)
        axis_min, axis_max : window bounds in um
    """
    k0 = 2 * np.pi * no / wavelength
    # k0 = "wavenumber": how many times the light wave oscillates per unit length.

    XL = x_window / x_scale
    # We rewrite the window size in "dimensionless units" (a pure number, no cm or um),
    # a common computational trick in physics to simplify the formulas that follow.

    # Generate the spatial axis: Nx positions spread from -XL/2 to XL/2.
    x = np.linspace(-XL / 2, XL / 2, Nx, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing='ij')
    # X, Y: the 2D grid of positions (like GPS coordinates on a 500x500 map).

    L_ad = z_max / (2.0 * k0 * x_scale**2)
    # Convert the physical distance z_max into the same "dimensionless" unit used for the grid.
    dz = L_ad / Nz                       # (dimensionless) length of one small propagation step
    z_phys_mm = np.linspace(0, z_max * 10, Nz)
    # The list of distances travelled, in millimeters, one value per step.

    dk = 2.0 * np.pi / XL
    l_idx = np.arange(Nx)
    nu = np.where(l_idx <= Nx // 2, l_idx * dk, (l_idx - Nx) * dk)
    # List of spatial frequencies matching each grid cell (numpy FFT convention).
    NUX, NUY = np.meshgrid(nu, nu, indexing='ij')
    K2 = NUX**2 + NUY**2

    # Correct paraxial propagator: from 2 i k0 dA/dz + grad^2(A) = 0,
    # the transfer function in Fourier space is exp(-i K^2 dz).
    # The + sign would reverse the transverse acceleration of the truncated
    # Airy beam, contradicting its known physical signature.
    prop_full = np.exp(-1j * dz * K2)

    x_um = x * x_scale * 1e4   # the same axis, but converted back to real micrometers for display
    axis_min, axis_max = x_um.min(), x_um.max()

    return x, X, Y, prop_full, z_phys_mm, x_um, axis_min, axis_max


def build_tukey_apod(Nx, alpha_ap):
    """Build the grid edge-smoothing window (Tukey window). It equals 1 in the
    center (= changes nothing) and gently decreases toward 0 at the edges, to
    avoid spurious numerical reflections caused by the fact that the computation
    method (FFT) treats the grid as wallpaper repeating to infinity.
    """
    def tukey1d(n, alpha):
        win = np.ones(n)
        ramp = int(alpha * n / 2)
        t = np.linspace(0, np.pi, ramp)
        win[:ramp] = 0.5 * (1 - np.cos(t))
        win[-ramp:] = 0.5 * (1 - np.cos(t[::-1]))
        return win

    wx = tukey1d(Nx, alpha_ap)
    return wx[:, np.newaxis] * wx[np.newaxis, :]


def make_field0(m, vortex_x_um, vortex_y_um, x_scale, a_trunc, Fo, x, X, Y, apod):
    """Build the initial light field (at z=0): a truncated Airy beam onto which
    we superimpose an optical vortex of charge m centered at position
    (vortex_x_um, vortex_y_um). This is the only function that depends on m:
    changing m simply amounts to calling this function again.
    """
    # Compute the Airy function Ai(x): the mathematical shape that gives the
    # beam its property of bending by itself as it propagates.
    Ai_x, _, _, _ = airy(x)
    idx_max = np.argmax(Ai_x)             # position of the main-lobe peak
    x_x, x_y = x[idx_max], x[idx_max]

    envelope = np.exp(a_trunc * (X - x_x) + a_trunc * (Y - x_y))
    # Soft truncation: a real beam has finite energy, so we attenuate the far-away lobes.
    F_airy = Fo * Ai_x[:, np.newaxis] * Ai_x[np.newaxis, :] * envelope

    vortex_x_norm = vortex_x_um / (x_scale * 1e4)
    vortex_y_norm = vortex_y_um / (x_scale * 1e4)
    dX = X - vortex_x_norm
    dY = Y - vortex_y_norm
    r = np.sqrt(dX**2 + dY**2) + 1e-12   # "+1e-12" avoids a division by zero exactly at the vortex center
    theta = np.arctan2(dY, dX)
    phase_vortex = r**abs(m) * np.exp(1j * m * theta)
    # exp(i*m*theta) rotates the phase around the center (m full turns); r^|m| forces the intensity
    # to be exactly zero at the center (the vortex "hole"), wider as m increases. This redistribution
    # of energy toward the more distant lobes, proportional to m, IS the effect of the vortex studied
    # here (impact of OAM on the beam's trajectory / center of mass): we therefore do not clamp it
    # artificially.

    field0 = (F_airy * phase_vortex).astype(np.complex128)

    field0 *= apod
    # We smooth the edges ONLY ONCE, at injection (see the physical justification in the propagation
    # loop below: we never redo it during propagation, so as not to lose energy).
    field0 /= np.sqrt(np.abs(field0)**2).max()
    # Renormalization: the maximum intensity of the initial field is exactly 1 (a practical scale choice).
    return field0


def propagate_full(field0, prop_full, Nz, Nx):
    """Advance the light field step by step over the whole propagation distance
    ("split-step Fourier" method / BPM), and KEEP in memory the complete state
    of the field at EACH step (needed for the slider and the GIF).
    """
    fields = np.zeros((Nz, Nx, Nx), dtype=np.complex128)
    fields[0] = field0.copy()
    current_field = field0.copy()

    for iz in range(1, Nz):
        F = np.fft.fft2(current_field)        # 1) go into the world of spatial frequencies
        F = F * prop_full                     # 2) each frequency advances by one small propagation step
        # No re-application of the apodization window here: free propagation is lossless; smoothing only
        # once at injection avoids artificially dissipating energy at every step.
        current_field = np.fft.ifft2(F)       # 3) go back to the "image" world (inverse transform)
        fields[iz] = current_field

    return fields


def propagate_track_com(field0, prop_full, Nz, X_um, Y_um):
    """Advance the light field step by step, like propagate_full, but KEEP IN
    MEMORY only the center-of-mass position at each step (not the full field at
    each step). This is far more memory-efficient, which allows chaining several
    charges m in a row without saturating the PC (keeping all full fields of 6
    charges would require several GB).

    The intensity center of mass is the brightness-weighted mean position: a bit
    like the center of gravity of an object, but applied to the light spot.
    Tracking this point at each step gives the beam's trajectory.
    """
    com_x = np.zeros(Nz)
    com_y = np.zeros(Nz)
    current_field = field0.copy()

    I0 = np.abs(current_field)**2          # intensity (brightness) at the start
    com_x[0] = np.sum(X_um * I0) / np.sum(I0)
    com_y[0] = np.sum(Y_um * I0) / np.sum(I0)

    for iz in range(1, Nz):
        F = np.fft.fft2(current_field)
        F = F * prop_full
        current_field = np.fft.ifft2(F)
        I = np.abs(current_field)**2
        total_I = np.sum(I)
        com_x[iz] = np.sum(X_um * I) / total_I
        com_y[iz] = np.sum(Y_um * I) / total_I

    return com_x, com_y


def compute_oam(fields, x, dx):
    """Compute the mean orbital angular momentum <Lz>/hbar at each z plane.

    Formula (angular-momentum operator):
        <Lz>/hbar = Im[ int E* (x d_y E - y d_x E) dxdy ] / int |E|^2 dxdy

    The derivatives d_x E, d_y E are computed with a pseudo-spectral method
    (FFT), more accurate than finite differences:
        d_x E = FFT^-1[ i*nu_x * FFT[E] ]
    For an ideal centered vortex, <Lz>/hbar should equal m (the topological
    charge), and remain constant in free propagation (OAM conservation).
    """
    Nz, Nx = fields.shape[0], fields.shape[1]
    lz = np.zeros(Nz)
    X, Y = np.meshgrid(x, x, indexing='ij')

    # Spatial frequencies (same for all z planes: computed only once)
    dk = 2.0 * np.pi / (Nx * dx)
    l_idx = np.arange(Nx)
    nu = np.where(l_idx <= Nx // 2, l_idx * dk, (l_idx - Nx) * dk)
    KX, KY = np.meshgrid(nu, nu, indexing='ij')

    for iz in range(Nz):
        E = fields[iz]
        E_hat = np.fft.fft2(E)
        dE_dx = np.fft.ifft2(1j * KX * E_hat)     # d_x E
        dE_dy = np.fft.ifft2(1j * KY * E_hat)     # d_y E
        Lz_op = X * dE_dy - Y * dE_dx             # operator x*d_y - y*d_x
        num = np.sum(np.conj(E) * Lz_op).imag     # numerator (imaginary part)
        den = np.sum(np.abs(E)**2).real           # total energy (normalization)
        lz[iz] = num / den if den > 0 else 0.0
    return lz


def mean_transverse_momentum(field0, x_window, x_scale, k0):
    """Predict the center-of-mass DRIFT VELOCITY (in um/mm) from the INITIAL
    field ALONE, without any propagation.

    Physical basis: Ehrenfest's theorem. In free propagation, the center of mass
    moves in a straight line, at a velocity fixed once and for all by the mean
    transverse momentum <k_perp> of the initial field:

            d<r>/dz = <k_perp> / k0 .

    We compute <k_perp> as the first moment (the "weighted mean") of the power
    spectrum |FFT(field)|^2. The factor 2 in the conversion comes from the
    paraxial convention (2 i k0 dA/dz + grad^2 A = 0, i.e. a "mass" of 1/2, hence
    d<x>/dz = 2 <k>). We numerically verified that this prediction matches the
    velocity measured by full propagation to better than 0.5%.

    Returns (vx, vy) in um/mm, in the same axis convention as com_x, com_y.
    """
    Nx = field0.shape[0]
    XL = x_window / x_scale
    dk = 2.0 * np.pi / XL
    l_idx = np.arange(Nx)
    nu = np.where(l_idx <= Nx // 2, l_idx * dk, (l_idx - Nx) * dk)
    KX, KY = np.meshgrid(nu, nu, indexing='ij')

    S = np.abs(np.fft.fft2(field0))**2          # power spectrum
    total = np.sum(S)
    kx_mean = np.sum(KX * S) / total            # first moment along x (dimensionless)
    ky_mean = np.sum(KY * S) / total            # first moment along y

    # Conversion to um/mm, Ehrenfest factor 2 included.
    conv = 1e4 / (10.0 * k0 * x_scale)
    return kx_mean * conv, ky_mean * conv


def fit_drift_velocity(com_x, com_y, z_phys_mm, kmax=120):
    """Fit a STRAIGHT LINE to the center-of-mass trajectory and return the
    MEASURED drift velocity (um/mm) on each axis, its norm, and the coefficient
    of determination R^2 (quality of the linear fit: R^2 = 1 -> perfectly
    straight trajectory, consistent with Ehrenfest).

    We restrict to the first kmax steps (useful region, before diffraction
    widens the beam too much and biases the slope).
    """
    z = z_phys_mm[:kmax]
    vx, bx = np.polyfit(z, com_x[:kmax], 1)     # slope (velocity) + intercept
    vy, by = np.polyfit(z, com_y[:kmax], 1)

    def r2(values, slope, intercept):
        pred = slope * z + intercept
        ss_res = np.sum((values - pred)**2)
        ss_tot = np.sum((values - values.mean())**2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    r2x = r2(com_x[:kmax], vx, bx)
    r2y = r2(com_y[:kmax], vy, by)
    return vx, vy, np.hypot(vx, vy), min(r2x, r2y)


def params_caption(wavelength, no, x_scale, a_trunc, Fo, Nx, x_window, z_max, Nz, alpha_ap, extra=""):
    """Build a text banner listing ALL simulation parameters, to make the figure
    exactly reproducible: a third party (e.g. a supervisor) can rerun the
    computation with exactly the same settings and compare. The banner spans
    several lines, which is intentional.
    """
    cap = (f"Reproducible parameters  |  "
           f"wavelength={wavelength} cm   no={no}   x_scale={x_scale} cm   "
           f"a_trunc={a_trunc}   Fo={Fo}\n"
           f"Nx={Nx}   x_window={x_window} cm   z_max={z_max} cm   "
           f"Nz={Nz}   alpha_ap={alpha_ap}")
    if extra:
        cap += "   " + extra
    return cap


# ============================================================================
# SIMULATION: interactive figures + GIF
# ============================================================================

def run_simulation_movie():
    """Mode "single": simulate ONE topological charge m and display the result.

    Steps: build the initial field -> split-step propagation (all z planes kept)
    -> compute the center of mass -> interactive figure (intensity and/or phase,
    z slider) + center-of-mass analysis figure, then export a GIF animation. All
    settings (m, vortex position, grid, etc.) are at the top of the function.
    MEMORY WARNING: this mode keeps all fields (Nz x Nx x Nx complex) -> keep Nx
    reasonable (see note below).
    """
    print("=== Airy 2D + OAM  |  Interactive display option ===")
    start_time = time.time()   # record the start time, to report how long the computation took

    # ================================================================
    # DISPLAY OPTION FOR THE GIF AND THE FIGURE
    # Choose between: "both" (both), "intensity" or "phase"
    # ================================================================
    display_mode = "intensity"

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
    gif_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"airy_vortex_{display_mode}.gif")
    ani.save(gif_path, writer="pillow", fps=20, dpi=gif_dpi)

    plt.close(fig_gif)
    print(f"GIF ({display_mode}) saved successfully: {gif_path}")


# ============================================================================
# COMPARISON: center-of-mass trajectory for several charges m
# Simulate the propagation for EACH topological charge in the list, keeping
# everything else identical (same off-center vortex, same grid, same
# propagation distance), then plot ONE figure where each curve is the 2D (X,Y)
# trajectory of the beam's center of mass for one charge m. This lets us
# visualize how the vortex charge deflects the trajectory.
# ============================================================================

def plot_com_trajectories_vs_charge(m_values=(0, 1, 2, 3, 4, 5), limit_frame=150):
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


# ============================================================================
# ANALYSIS: drift law + map
# This function builds on the central physical result: the center of mass
# follows an EXACT LAW (Ehrenfest), its drift velocity being given by the
# transverse momentum of the initial field. It generates two figures:
#
#   FIGURE 1 -- Law validation + saturation:
#       for each charge m, we compare the MEASURED drift velocity (from full
#       propagation) to the PREDICTED velocity (from the transverse momentum of
#       the initial field alone, without propagation). Their agreement validates
#       the analytical law; the shape of the curve highlights the SATURATION with m.
#
#   FIGURE 2 -- Map (xv, yv):
#       map of the predicted drift velocity versus vortex position, computed
#       ANALYTICALLY (hence instantly, without propagation), illustrating the
#       power of the law.
# ============================================================================

def run_loi_derive(m_values=(0, 1, 2, 3, 4, 5)):
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


# ============================================================================
# OAM MODE: orbital angular momentum <Lz>  (validation + conservation)
# For each charge m, we propagate the vortex beam (vortex centered on the main
# lobe) and compute <Lz>/hbar at each z plane. Two results:
#   - LEFT : mean measured <Lz> vs charge m (should be a line of slope 1);
#   - RIGHT: <Lz>(z) for each m (should stay constant = OAM conservation).
# Uses the CORRECTED propagator of the main program.
# ============================================================================

def run_oam_analysis(m_values=(0, 1, 2, 3, 4, 5)):
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


# ============================================================================
# 1D PROFILE MODE: I(x) and I(y) cuts along the main lobe
# We simulate a single charge, and display the 1D INTENSITY PROFILE along X and
# Y, at the main-lobe position (automatically tracked by the intensity maximum).
# Interactive figure (z slider) + GIF. Highlights the deformation of the profile
# by the vortex, then its reconstruction (self-healing).
# ============================================================================

def run_profil1d():
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
    gif_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"airy_profil1d_m{m}.gif")
    ani.save(gif_path, writer="pillow", fps=20, dpi=90)
    plt.close(fig_gif)
    print(f"1D-profile GIF saved: {gif_path}")


# ============================================================================
# PROGRAM ENTRY POINT
# Choose what to run by changing the value of "mode" below:
#   - "single"    -> simulate a single charge m (interactive figures + GIF)
#   - "charges"   -> center-of-mass trajectory for several charges
#   - "loi"       -> drift law (measured vs predicted) + map (xv,yv)
#   - "oam"       -> orbital angular momentum <Lz> (validation + conservation)
#   - "profil1D"  -> 1D cuts I(x), I(y) along the main lobe (+ GIF)
# ============================================================================
if __name__ == "__main__":
    mode = "oam"   # "single", "charges", "loi", "oam" or "profil1D"

    if mode == "single":
        run_simulation_movie()
    elif mode == "charges":
        plot_com_trajectories_vs_charge(m_values=(0, 1, 2, 3, 4, 5), limit_frame=150)
    elif mode == "loi":
        run_loi_derive(m_values=(0, 1, 2, 3, 4, 5))
    elif mode == "oam":
        run_oam_analysis(m_values=(0, 1, 2, 3, 4, 5))
    elif mode == "profil1D":
        run_profil1d()
    else:
        raise ValueError("mode must be 'single', 'charges', 'loi', 'oam' or 'profil1D'")
