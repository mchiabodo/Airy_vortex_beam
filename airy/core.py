# -*- coding: utf-8 -*-
# ============================================================================
#  PROPAGATION OF A 2D AIRY BEAM CARRYING AN OPTICAL VORTEX (OAM)
#  Split-step Fourier method (Beam Propagation Method), free paraxial regime.
#  Shared physics core: every run mode imports the helper functions below.
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
# ============================================================================

import numpy as np
from scipy.special import airy


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
