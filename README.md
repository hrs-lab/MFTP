# Reproducing the numerical checks

Manuscript: **Detailed balance and first-passage-time uncertainty in diffusions with resetting**.

This supplement reproduces the two numerical appendix tables, the supplementary formula and construction checks, and the eight curves in Figure 1. The scripts use floating-point arithmetic. They do not compute certified global optima or implement the full certification procedure in the paper.

## Contents

| File | Purpose |
| --- | --- |
| `verify_landscape_passage.py` | Appendix A: local constants, residuals, admissibility and finite-difference comparisons |
| `landscape_passage_verification.json` | Results of the Appendix A script |
| `verify_landscape_global.py` | Appendix B: Hermite brackets, Legendre bands, positive pairs, finite matrix identity and moment repair |
| `landscape_global_verification.json` | Results of the Appendix B script |
| `plot_landscape_crossover.py` | Figure 1: exact rank-one width formulas evaluated by quadrature and spectral integration |
| `landscape_crossover_plot_data.json` | Figure coordinates, parameters, amplitudes, denominator checks and resolution comparisons |

## Requirements and commands

The scripts require Python 3.9 or later and NumPy. The supplied reports were produced with Python 3.12.14 and NumPy 2.3.5. No SciPy or Python plotting package is required. Each script can be run independently.

