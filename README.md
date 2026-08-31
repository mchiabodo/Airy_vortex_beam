# 2D Airy Beam with Optical Vortex — Propagation Simulator

A Python package that simulates how a **2D Airy beam carrying an optical
vortex** propagates through free space, and measures how the vortex's
topological charge affects the beam's trajectory and its orbital angular
momentum (OAM).

The project is organised as a small Python package: the shared physics lives in
[`airy/core.py`](airy/core.py), each of the five analysis modes is its own
module, and [`main.py`](main.py) is the command-line entry point. Every mode
provides interactive figures and animated GIF export.

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

Choose a mode on the command line (see below). Each mode is a self-contained
module with its own parameters at the top of its `run()` function.

| `mode`     | What it does |
|------------|--------------|
| `single`   | Propagates one charge `m`; interactive intensity/phase viewer with a `z` slider, plus a center-of-mass analysis figure and a GIF. |
| `charges`  | Runs several charges `m` and overlays their center-of-mass trajectories to show how the vortex deflects the beam. |
| `loi`      | Compares the measured drift velocity against the value predicted analytically from the initial field alone, and maps the drift versus vortex position. |
| `oam`      | Computes `⟨Lz⟩/ℏ` for each charge: validates it against `m` and checks its conservation during propagation. |
| `profil1D` | Tracks the main lobe and plots the 1D intensity profiles `I(x)`, `I(y)` through it; interactive viewer + GIF. |

## Getting started

Install the dependencies once:

```bash
pip install -r requirements.txt
```

Then run any mode from the repository root:

```bash
python main.py oam        # or: single, charges, loi, profil1D
```

Running `python main.py` with no argument defaults to the `oam` mode.

### Tuning the parameters

Each mode keeps its physical and numerical settings (wavelength, charge `m`,
vortex position, grid size `Nx`, propagation distance, etc.) at the top of the
`run()` function in its own module — for example edit `airy/mode_single.py` to
change the `single` run. The shared physics in `airy/core.py` does not need to
be touched to explore different cases.

## Project structure

```
main.py                    command-line entry point (dispatches to a mode)
requirements.txt           dependencies
airy/
├── core.py                shared physics (grid, propagator, field, observables)
├── mode_single.py         one charge m: intensity/phase viewer + GIF
├── mode_charges.py        center-of-mass trajectory vs charge
├── mode_law.py            drift law (measured vs predicted) + map
├── mode_oam.py            orbital angular momentum <Lz>
└── mode_profile1d.py      1D intensity profiles along the main lobe
```

The shared physics in `airy/core.py` is factored into small, single-purpose
functions:

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
`airy/core.py` explaining the model, the normalization and the numerical
conventions.

## Requirements

Python 3 with **NumPy**, **SciPy**, **Matplotlib** and **Pillow**.
