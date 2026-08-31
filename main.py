# -*- coding: utf-8 -*-
# ============================================================================
# ENTRY POINT
# Run one of the five analysis modes from the command line:
#
#     python main.py single      -> one charge m (interactive figures + GIF)
#     python main.py charges     -> center-of-mass trajectory for several charges
#     python main.py law         -> drift law (measured vs predicted) + map
#     python main.py oam         -> orbital angular momentum <Lz> (default)
#     python main.py profile1d   -> 1D cuts I(x), I(y) along the main lobe + GIF
#
# With no argument, the "oam" mode is run.
# ============================================================================

import sys
import io
import argparse

# Make stdout able to print accents / Greek symbols without crashing.
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150   # higher-resolution figures for readability

from airy import mode_single, mode_charges, mode_law, mode_oam, mode_profile1d

MODES = {
    "single":    lambda: mode_single.run(),
    "charges":   lambda: mode_charges.run(m_values=(0, 1, 2, 3, 4, 5), limit_frame=150),
    "law":       lambda: mode_law.run(m_values=(0, 1, 2, 3, 4, 5)),
    "oam":       lambda: mode_oam.run(m_values=(0, 1, 2, 3, 4, 5)),
    "profile1d": lambda: mode_profile1d.run(),
}


def main():
    parser = argparse.ArgumentParser(
        description="2D Airy beam + optical vortex (OAM) propagation simulator."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="oam",
        choices=list(MODES.keys()),
        help="which analysis to run (default: oam)",
    )
    args = parser.parse_args()
    MODES[args.mode]()


if __name__ == "__main__":
    main()
