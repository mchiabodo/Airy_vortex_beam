# 2D Airy Beam with Optical Vortex — Propagation Simulator

A self-contained Python program that simulates how a **2D Airy beam carrying an
optical vortex** propagates through free space, and measures how the vortex's
topological charge affects the beam's trajectory and its orbital angular
momentum (OAM).

Everything lives in a single file, [`Airy.py`](Airy.py): the physics, the
numerics, and five ready-made analysis modes with interactive figures and
animated GIF export.

## What it computes

The beam starts as a truncated 2D Airy field multiplied by an optical vortex of
integer charge `m`:

```
A(x, y, 0) = Ai(x/x₀)·Ai(y/x₀)·exp(a(x+y)/x₀) · ρ^|m|·exp(i·m·φ)
```

and is propagated with the paraxial wave equation

```
2·i·k₀·∂A/∂z + ∇⊥²A = 0
```

solved with the **split-step Fourier method** (Beam Propagation Method): at each
step the field is taken to Fourier space, multiplied by the exact spectral
propagator `exp(-i·K²·dz)`, and transformed back. Free propagation is treated as
lossless — the edge (Tukey) window is applied only once, at injection, so energy
is preserved and the center-of-mass motion stays physically correct.

From the propagated field the code extracts two main observables:

- the **center of mass** `⟨r⟩(z)` — the brightness-weighted position of the beam,
  which drifts in a straight line whose velocity is set by the initial field's
  mean transverse momentum;
- the **orbital angular momentum** `⟨Lz⟩/ℏ` — computed with pseudo-spectral
  derivatives, expected to be close to the charge `m` and conserved along `z`.

## Run modes

Pick a mode with the `mode` variable at the bottom of `Airy.py`, then run the
file. Each mode is a self-contained function with its own parameters at the top.

| `mode`     | What it does |
|------------|--------------|
| `single`   | Propagates one charge `m`; interactive intensity/phase viewer with a `z` slider, plus a center-of-mass analysis figure and a GIF. |
| `charges`  | Runs several charges `m` and overlays their center-of-mass trajectories to show how the vortex deflects the beam. |
| `loi`      | Compares the measured drift velocity against the value predicted analytically from the initial field alone, and maps the drift versus vortex position. |
| `oam`      | Computes `⟨Lz⟩/ℏ` for each charge: validates it against `m` and checks its conservation during propagation. |
| `profil1D` | Tracks the main lobe and plots the 1D intensity profiles `I(x)`, `I(y)` through it; interactive viewer + GIF. |

## Getting started

```bash
pip install numpy scipy matplotlib pillow
python Airy.py
```

Then edit the `mode` line at the bottom of `Airy.py` to choose what to run.

## Code structure

The physics is factored into small, single-purpose helper functions:

| Function | Role |
|----------|------|
| `build_grid_and_propagator` | Dimensionless transverse grid, propagation axis and spectral propagator. |
| `build_tukey_apod`          | 2D Tukey window that softens the grid edges. |
| `make_field0`               | Initial truncated Airy field × optical vortex (the only `m`-dependent piece). |
| `propagate_full`            | Split-step propagation keeping every `z` plane (for viewers and GIFs). |
| `propagate_track_com`       | Memory-light propagation keeping only `⟨r⟩(z)`. |
| `compute_oam`               | `⟨Lz⟩/ℏ(z)` from pseudo-spectral derivatives. |
| `mean_transverse_momentum`  | Analytical drift-velocity prediction from the initial field. |
| `fit_drift_velocity`        | Linear fit of the trajectory (drift speed + R²). |
| `params_caption`            | Reproducible-parameters banner drawn on every figure. |

The code is fully commented in English, with a reference header at the top of
`Airy.py` explaining the model, the normalization and the numerical conventions.

## Requirements

Python 3 with **NumPy**, **SciPy**, **Matplotlib** and **Pillow**.
