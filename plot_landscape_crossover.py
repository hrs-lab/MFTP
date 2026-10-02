#!/usr/bin/env python3
"""Reproduce the feasible-pair crossover figure in the landscape manuscript.

Requires Python 3.9+ and NumPy; no SciPy, plotting library, or other script is
needed. From this directory run:

    python plot_landscape_crossover.py
    python plot_landscape_crossover.py --manuscript /path/to/manuscript.tex

The first command writes landscape_crossover_plot_data.json beside this script.
The second also compares every generated point with the eight inline PGFPlots
curves in the supplied manuscript. Use --output PATH to choose another JSON
destination. Add --print-pgfplots to print replacement coordinate blocks to
standard output; the script never edits a manuscript.

The two numerical appendix tables have separate verification scripts. This
script concerns the figure at D=0.3, nu=0.7, a=1/2, x=0 and m=32,64,128,256.
It evaluates exact rank-one width formulas using floating-point quadrature;
it does not solve a global optimization problem or provide interval-certified
bounds. The positivity bound comes from the analytical |u_m| <= 10 estimate,
not from sampling a grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re

import numpy as np
from numpy.polynomial.legendre import legint, legval, legvander


D = 0.3
NU = 0.7
A = 0.5
C_U = 10.0
ORDERS = (32, 64, 128, 256)
K = np.sqrt(NU / D)
DEN = 0.5 + np.tanh(K * A) / K
S_ZERO = (1.0 - 1.0 / np.cosh(K * A)) / NU
T_ZERO = S_ZERO / DEN
G_ZERO = np.tanh(K * A) / (D * K) + NU * S_ZERO * T_ZERO
# G is decreasing on [0,a], and its zero extension is zero on [a,1].
V_SUP = max(G_ZERO - T_ZERO, T_ZERO)
FREE_CAP = 1.0 / (2.0 * C_U * V_SUP)
REV_CAP = 1.0 / (2.0 * C_U**2)


def gauss_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights, with Newton-refined roots."""
    z = np.cos(np.pi * (np.arange(1, n + 1) - 0.25) / (n + 0.5))

    def polynomial_and_derivative(nodes):
        p0 = np.ones_like(nodes)
        p1 = nodes.copy()
        for j in range(2, n + 1):
            p0, p1 = p1, ((2 * j - 1) * nodes * p1 - (j - 1) * p0) / j
        return p1, n * (nodes * p1 - p0) / (nodes**2 - 1.0)

    for _ in range(12):
        pn, dpn = polynomial_and_derivative(z)
        correction = pn / dpn
        z -= correction
        if np.max(np.abs(correction)) <= 3.0e-16:
            break
    _, dpn = polynomial_and_derivative(z)
    weights = 2.0 / ((1.0 - z**2) * dpn**2)
    return z, weights


def baseline(x):
    """s, T, G_0, and v=G_0-T(0), evaluated only on [0,a]."""
    s = (1.0 - np.cosh(K * x) / np.cosh(K * A)) / NU
    lifetime = s / DEN
    green = np.sinh(K * (A - x)) / (D * K * np.cosh(K * A))
    green += NU * S_ZERO * lifetime
    return s, lifetime, green, green - T_ZERO


def band_coefficients(m: int) -> tuple[np.ndarray, float]:
    """Return ordinary shifted-Legendre coefficients and exact norm formula."""
    degree = np.arange(2 * m + 1)
    central = legvander(np.array([0.0]), 2 * m)[0]
    orthonormal = -central * np.sqrt(2 * degree + 1) / m
    orthonormal[: m + 1] = 0.0
    coefficients = orthonormal * np.sqrt(2 * degree + 1)
    return coefficients, float(np.dot(orthonormal, orthonormal))


def spectral_local_resolvent(x, coefficient, spectral_degree: int):
    """Evaluate K u and K v using the explicit mixed-boundary Green kernel.

    K f(x) = [sinh(k(a-x)) int_0^x cosh(ky) f(y)dy
              + cosh(kx) int_x^a sinh(k(a-y)) f(y)dy] / (Dk cosh(ka)).

    Weighted right-hand sides are projected onto Legendre polynomials on
    [0,a] at the stated spectral degree and then integrated analytically.
    This treats the Green kernel's diagonal derivative corner by splitting
    the integral at x, rather than quadraturing across that corner.
    """
    z, weights = gauss_rule(spectral_degree + 1)
    y = A * (z + 1.0) / 2.0
    u = legval(2.0 * y - 1.0, coefficient)
    v = baseline(y)[3]
    right_sides = np.column_stack((u, v))
    values = legvander(z, spectral_degree)
    scales = (2 * np.arange(spectral_degree + 1) + 1) / 2.0

    def primitive(multiplier):
        projection = values.T @ (weights[:, None] * multiplier[:, None] * right_sides)
        projection *= scales[:, None]
        # The primitive is zero at the left endpoint; scl changes dz to dy.
        return legint(projection, lbnd=-1.0, scl=A / 2.0, axis=0)

    lower_coefficients = primitive(np.cosh(K * y))
    upper_coefficients = primitive(np.sinh(K * (A - y)))
    evaluation_points = 2.0 * x / A - 1.0
    lower = legval(evaluation_points, lower_coefficients).T
    upper = legval(1.0, upper_coefficients)[None, :] - legval(
        evaluation_points, upper_coefficients
    ).T
    return (
        np.sinh(K * (A - x))[:, None] * lower
        + np.cosh(K * x)[:, None] * upper
    ) / (D * K * np.cosh(K * A))


def evaluate_pairings(m: int, quadrature_nodes: int, spectral_degree: int):
    coefficient, norm_squared = band_coefficients(m)
    z, weights = gauss_rule(quadrature_nodes)
    x = A * (z + 1.0) / 2.0
    weights *= A / 2.0
    u = legval(2.0 * x - 1.0, coefficient)
    s, lifetime, green, v = baseline(x)
    gamma_squared = float(np.dot(weights, v**2) + (1.0 - A) * T_ZERO**2)
    ku_kv = spectral_local_resolvent(x, coefficient, spectral_degree)
    u_s = np.dot(weights, u * s)
    s_v = np.dot(weights, s * v)
    # R=K+(nu/d) s tensor s. All input functions are restricted to A.
    u_r_u = np.dot(weights * u, ku_kv[:, 0]) + NU / DEN * u_s**2
    u_r_v = np.dot(weights * u, ku_kv[:, 1]) + NU / DEN * u_s * s_v
    # u is even about 1/2; twice the half-interval integral is its full norm.
    norm_squared_quadrature = float(2.0 * np.dot(weights, u**2))
    return {
        "quadrature_nodes_per_half_interval": quadrature_nodes,
        "spectral_degree": spectral_degree,
        "norm_u_squared_coefficients": norm_squared,
        "norm_u_squared_quadrature": norm_squared_quadrature,
        "norm_u_squared_relative_discrepancy": abs(norm_squared_quadrature / norm_squared - 1.0),
        "gamma_squared": gamma_squared,
        "u_T": float(np.dot(weights, u * lifetime)),
        "u_G": float(np.dot(weights, u * green)),
        "u_R_u": float(u_r_u),
        "u_R_v": float(u_r_v),
    }


def curve_values(m: int, kernel_class: str, xi, pairings):
    norm_squared = pairings["norm_u_squared_coefficients"]
    if kernel_class == "unrestricted":
        epsilon = xi / np.sqrt(m)
        amplitude = np.minimum(FREE_CAP, epsilon / np.sqrt(norm_squared * pairings["gamma_squared"]))
        denominator = 1.0 - (amplitude * NU * pairings["u_R_v"]) ** 2
        width = 2.0 * amplitude * NU * pairings["u_T"] * pairings["gamma_squared"] / denominator
        used_radius = amplitude * np.sqrt(norm_squared * pairings["gamma_squared"])
        lower_bound = 1.0 - amplitude * C_U * V_SUP
        scaled_width = m**2 * width
    else:
        epsilon = xi / m
        amplitude = np.minimum(REV_CAP, epsilon / norm_squared)
        denominator = 1.0 - (amplitude * NU * pairings["u_R_u"]) ** 2
        width = 2.0 * amplitude * NU * pairings["u_T"] * pairings["u_G"] / denominator
        used_radius = amplitude * norm_squared
        lower_bound = 1.0 - amplitude * C_U**2
        scaled_width = m**4 * width
    return {
        "xi": xi.tolist(),
        "epsilon": epsilon.tolist(),
        "amplitude": amplitude.tolist(),
        "denominator": denominator.tolist(),
        "constructed_width": width.tolist(),
        "scaled_constructed_width": scaled_width.tolist(),
        "used_L2_radius": used_radius.tolist(),
        "analytical_kernel_lower_bound_evaluated_in_floating_point": lower_bound.tolist(),
    }


def compare_manuscript(path: Path, curves):
    content = path.read_bytes()
    source = content.decode("utf-8-sig")
    label_position = source.index(r"\label{fig:crossover-pairs}")
    figure_start = source.rfind(r"\begin{figure}", 0, label_position)
    figure_source = source[figure_start:label_position]
    blocks = re.findall(
        r"coordinates\s*\{(.*?)\};\s*\\addlegendentry\{\$m=(\d+)\$\}",
        figure_source,
        re.S,
    )
    if len(blocks) != len(curves):
        raise ValueError(f"Expected eight inline crossover curves, found {len(blocks)}")
    comparisons = []
    for (block, manuscript_m), curve in zip(blocks, curves):
        points = np.array(
            [(float(x), float(y)) for x, y in re.findall(r"\(([^,]+),([^\)]+)\)", block)]
        )
        generated = np.column_stack((curve["xi"], curve["scaled_constructed_width"]))
        if int(manuscript_m) != curve["moment_order"] or points.shape != generated.shape:
            raise ValueError("Manuscript moment orders or point counts differ from the construction")
        comparisons.append({
            "kernel_class": curve["kernel_class"],
            "moment_order": curve["moment_order"],
            "point_count": len(points),
            "max_relative_x_difference": float(np.max(np.abs(generated[:, 0] / points[:, 0] - 1.0))),
            "max_relative_scaled_width_difference": float(np.max(np.abs(generated[:, 1] / points[:, 1] - 1.0))),
        })
    return {
        "source_path": str(path.resolve()),
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "curves": comparisons,
        "max_relative_coordinate_difference": max(
            max(item["max_relative_x_difference"], item["max_relative_scaled_width_difference"])
            for item in comparisons
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manuscript", type=Path, help="Optional current LaTeX source for inline-coordinate comparison")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("landscape_crossover_plot_data.json"))
    parser.add_argument("--print-pgfplots", action="store_true", help="Print generated PGFPlots coordinate blocks; do not edit files")
    args = parser.parse_args()
    pairings = {}
    for m in ORDERS:
        pairings[m] = (
            evaluate_pairings(m, 2 * m + 64, 2 * m + 48),
            evaluate_pairings(m, 2 * m + 160, 2 * m + 96),
        )
    curves = []
    width_discrepancies = []
    for kernel_class in ("unrestricted", "reversible"):
        for m in ORDERS:
            coarse, fine = pairings[m]
            norm_squared = fine["norm_u_squared_coefficients"]
            if kernel_class == "unrestricted":
                transition = FREE_CAP * np.sqrt(norm_squared * fine["gamma_squared"] * m)
            else:
                transition = REV_CAP * norm_squared * m
            # The manuscript samples 72 logarithmic points and the cap transition.
            xi = np.unique(np.concatenate((np.logspace(-4, 0, 72), [transition])))
            coarse_values = curve_values(m, kernel_class, xi, coarse)
            fine_values = curve_values(m, kernel_class, xi, fine)
            discrepancy = float(np.max(np.abs(
                np.array(coarse_values["constructed_width"]) / fine_values["constructed_width"] - 1.0
            )))
            width_discrepancies.append(discrepancy)
            curves.append({
                "kernel_class": kernel_class,
                "moment_order": m,
                "scaled_radius_transition": float(transition),
                "amplitude_cap": FREE_CAP if kernel_class == "unrestricted" else REV_CAP,
                "max_relative_width_change_between_resolutions": discrepancy,
                **fine_values,
            })
    report = {
        "purpose": "Floating-point reproduction of explicit feasible-pair lower bounds; no global optimization or interval certificate.",
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "parameters": {"D": D, "nu": NU, "target_boundary": A, "initial_state": 0.0, "C_u": C_U, "moment_orders": list(ORDERS)},
        "definitions": {
            "unrestricted_kernel": "1 +/- t u_m(y) (G_0(x)-T(0))",
            "reversible_kernel": "1 +/- t u_m(y) u_m(x)",
            "u_m": "-sum_{n=m+1}^{2m} ell_n(1/2) ell_n(x)/m",
            "unrestricted_axes": ["epsilon sqrt(m)", "m^2 constructed_width"],
            "reversible_axes": ["epsilon m", "m^4 constructed_width"],
            "positivity_justification": "The proved sup bound |u_m| <= C_u and the stated amplitude caps guarantee p >= 1/2 everywhere.",
            "budget_justification": "The amplitude minimum enforces t ||u_m||_2 ||v||_2 <= epsilon or t ||u_m||_2^2 <= epsilon.",
        },
        "baseline": {"k": float(K), "d": float(DEN), "T_zero": float(T_ZERO), "G_zero": float(G_ZERO), "v_sup": float(V_SUP)},
        "pairings": {str(m): {"coarse": item[0], "fine": item[1]} for m, item in pairings.items()},
        "curves": curves,
        "checks": {
            "max_relative_width_change_between_resolutions": max(width_discrepancies),
            "max_relative_band_norm_discrepancy": max(item["norm_u_squared_relative_discrepancy"] for pair in pairings.values() for item in pair),
            "min_rank_one_denominator": min(min(curve["denominator"]) for curve in curves),
            "min_analytical_kernel_lower_bound_evaluated_in_floating_point": min(min(curve["analytical_kernel_lower_bound_evaluated_in_floating_point"]) for curve in curves),
            "max_budget_excess": max(max(np.array(curve["used_L2_radius"]) - curve["epsilon"]) for curve in curves),
        },
        "manuscript_comparison": compare_manuscript(args.manuscript, curves) if args.manuscript else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")
    print(json.dumps(report["checks"], indent=2))
    if report["manuscript_comparison"]:
        mismatch = report["manuscript_comparison"]["max_relative_coordinate_difference"]
        print(f"Maximum relative difference from manuscript coordinates: {mismatch:.6g}")
        if mismatch > 1.0e-9:
            raise SystemExit("Inline manuscript coordinates differ by more than 1e-9; inspect the JSON report.")
    if args.print_pgfplots:
        for curve in curves:
            print(f"% {curve['kernel_class']}, m={curve['moment_order']}")
            print(r"\addplot coordinates {")
            for x, y in zip(curve["xi"], curve["scaled_constructed_width"]):
                print(f"({x:.12e},{y:.12e})")
            print("};")


if __name__ == "__main__":
    main()
