# 2D Airy Beam + Optical Vortex — Numerical Simulation

Simulation of the **free propagation of a 2D Airy beam carrying an optical
vortex** (orbital angular momentum, OAM), in the paraxial regime, using the
**split-step Fourier method** (Beam Propagation Method).

Research project: correction and validation of an existing propagation code,
then derivation of an **analytical law for the drift of the beam's center of
mass** and its numerical verification.

## Physical overview

The initial field (at `z = 0`) is a truncated 2D Airy beam multiplied by an
optical vortex of topological charge `m`, centered at `(x_v, y_v)`:

```
A(x,y,0) = Ai(x/x₀)·Ai(y/x₀)·exp(a(x+y)/x₀) · ρ^|m|·exp(i m φ)
           └────────── truncated 2D Airy ─────────┘   └─ OAM vortex ─┘
```

It is propagated with the paraxial (optical Schrödinger) equation

```
2 i k₀ ∂A/∂z + ∇⊥²A = 0
```

solved **exactly in Fourier space** at each step: `Â *= exp(-i K² dz')`.

## Key results

- **Energy conserved at 100%** and a **straight** center-of-mass trajectory
  (R² = 1.0000), consistent with Ehrenfest's theorem — after fixing an
  apodization bug that artificially bent the trajectory.
- **Exact analytical drift law**: `⟨r⟩(z) = ⟨r⟩(0) + (⟨k⊥⟩/k₀)·z`, with the
  slope predicted without propagation and a measured/predicted agreement
  **< 0.5%**.
- **Orbital angular momentum** `⟨Lz⟩/ℏ ≈ m`, conserved in free propagation
  (slope 0.995 ; R² = 0.999).

## Features

The simulation file contains **5 modes**, selected via the `mode` variable at
the bottom of the file:

| Mode        | What it produces                                             |
|-------------|--------------------------------------------------------------|
| `single`    | Intensity / phase for one charge `m` + GIF                  |
| `charges`   | Center-of-mass trajectory for `m = 0..5`                    |
| `loi`       | Drift law (measured vs predicted) + map                    |
| `oam`       | Orbital angular momentum `⟨Lz⟩` (validation + conservation) |
| `profil1D`  | 1D cuts I(x), I(y) along the main lobe + GIF               |

## Run

```bash
pip install numpy scipy matplotlib pillow
python Airy.py
```

Open `Airy.py`, set the `mode` variable at the bottom of the file to the mode
you want, then run it.

## Method & implementation

- Dimensionless grid in units of the Airy lobe width `x₀`, spectral propagator
  `exp(-i K² dz')` (exact solution of the paraxial equation).
- Tukey window applied **only once** at injection (lossless free propagation) —
  a key point to respect energy conservation and Ehrenfest's theorem.
- Observables computed by spectral integration: center of mass `⟨r⟩(z)` and
  orbital angular momentum `⟨Lz⟩(z)` (pseudo-spectral derivatives).

## Technologies

Python 3 · NumPy · SciPy · Matplotlib · Pillow

---

> The full study also includes reports, a bibliographic review, slides and
> high-resolution animations. These are not versioned in this repository and are
> available on request.
