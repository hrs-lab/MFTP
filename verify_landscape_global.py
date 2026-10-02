#!/usr/bin/env python3
"""Reproduce Appendix B checks for landscape_passage_uncertainty_revised.tex.

Requires Python 3 and NumPy only. Run:
    python verify_landscape_global.py
The report is written next to this script unless --output is supplied.
All calculations use ordinary double precision. Neither sampled inequalities
nor the reports below are interval-arithmetic or global-optimization certificates.
No optimizer is run and no global optimizing kernel is claimed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.legendre import leggauss


D, NU, A = 0.3, 0.7, 0.5
K = np.sqrt(NU / D)
DEN = 0.5 + np.tanh(K / 2) / K
M_ORDERS = (8, 32, 128, 512)
RADII = (1e-5, 1e-3, 1e-2, 1e-1, 1.0)


def baseline(x):
    """s, T and G_0, zero extended to the target interval."""
    x = np.asarray(x)
    inside = x < A
    s = np.where(inside, (1 - np.cosh(K * x) / np.cosh(K / 2)) / NU, 0)
    t = s / DEN
    s0 = (1 - 1 / np.cosh(K / 2)) / NU
    g = np.where(inside, np.sinh(K * (A - x)) / (D * K * np.cosh(K / 2))
                 + NU * s0 * t, 0)
    return s, t, g


def quadrature(n, left, right):
    z, w = leggauss(n)
    return left + (right - left) * (z + 1) / 2, w * (right - left) / 2


def band_values(x, m):
    """u_m via the three-term recurrence, without monomial coefficients."""
    z = 2 * np.asarray(x) - 1
    prev, cur = np.ones_like(z), z.copy()
    prev0, cur0 = 1.0, 0.0
    value = np.zeros_like(z)
    norm2 = 0.0
    for degree in range(2, 2 * m + 1):
        nxt = ((2 * degree - 1) * z * cur - (degree - 1) * prev) / degree
        nxt0 = -(degree - 1) * prev0 / degree
        if degree > m:
            value -= (2 * degree + 1) * nxt0 * nxt / m
            norm2 += (2 * degree + 1) * nxt0 * nxt0 / (m * m)
        prev, cur = cur, nxt
        prev0, cur0 = cur0, nxt0
    return value, norm2


def hermite_checks():
    """Construct U_n,L_n in T_j(2t-1), with scaled derivative rows."""
    out = []
    grid = np.linspace(0.0, 1.0, 100001)
    truth = np.sqrt(grid)
    for n in (1, 2, 4, 8, 16):
        degrees = np.arange(2 * n)
        theta = (2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n)
        nodes = (1 + np.cos(theta)) / 2
        values = np.cos(theta[:, None] * degrees)
        deriv = degrees * np.sin(theta[:, None] * degrees)
        # d/dt equations multiplied by sin(theta)/2.
        rhs = np.r_[np.sqrt(nodes), np.sin(theta) / (4 * np.sqrt(nodes))]
        upper = np.linalg.solve(np.vstack((values, deriv)), rhs)

        theta_l = np.arange(n + 1) * np.pi / n
        nodes_l = (1 + np.cos(theta_l)) / 2
        rows_l = np.cos(theta_l[:, None] * degrees)
        theta_i, nodes_i = theta_l[1:-1], nodes_l[1:-1]
        deriv_l = degrees * np.sin(theta_i[:, None] * degrees)
        rhs_l = np.r_[np.sqrt(nodes_l), np.sin(theta_i) / (4 * np.sqrt(nodes_i))]
        lower = np.linalg.solve(np.vstack((rows_l, deriv_l)), rhs_l)
        gap = float(np.pi * (upper[0] - lower[0]))
        exact = float(np.pi / (2 * n) * np.tan(np.pi / (8 * n)))
        out.append({
            "n": n, "weighted_gap_from_coefficients": gap,
            "weighted_gap_exact_formula": exact,
            "relative_gap_discrepancy": abs(gap / exact - 1),
            "minimum_upper_minus_sqrt_on_grid": float(np.min(chebval(2 * grid - 1, upper) - truth)),
            "minimum_sqrt_minus_lower_on_grid": float(np.min(truth - chebval(2 * grid - 1, lower))),
        })
    assert max(r["relative_gap_discrepancy"] for r in out) < 1e-9
    assert min(r["minimum_upper_minus_sqrt_on_grid"] for r in out) > -1e-12
    assert min(r["minimum_sqrt_minus_lower_on_grid"] for r in out) > -1e-12
    return out


def width_result(delta, alpha, left_pair, right_pair, left_norm, right_norm):
    """Rank-one pair width and bounds using nu*||R|| <= 2.

    alpha = nu <left,R right>, from quadrature. The interval bounds
    use its analytic absolute bound, evaluated in floating point.
    """
    numerator = float(2 * delta * NU * left_pair * right_pair)
    denominator = float(1 - (delta * alpha) ** 2)
    rho_bound = float(2 * delta * left_norm * right_norm)
    assert numerator > 0 and denominator > 0 and rho_bound < 1
    assert abs(delta * alpha) <= rho_bound * (1 + 1e-10)
    width = numerator / denominator
    return {"amplitude": float(delta), "quadratic_radius_used": float(delta * left_norm * right_norm),
            "quadrature_rank_one_width": width,
            "denominator_from_quadrature": denominator,
            "lower_width_from_resolvent_norm": numerator,
            "upper_width_from_resolvent_norm": numerator / (1 - rho_bound ** 2),
            "denominator_perturbation_absolute_bound": rho_bound}


def band_and_kernel_checks(gauss_points):
    xa, wa = quadrature(gauss_points, 0, A)
    xb, wb = quadrature(gauss_points, A, 1)
    x, w = np.r_[xa, xb], np.r_[wa, wb]
    s, t, g = baseline(x)
    s0, t0, g0 = (float(v) for v in baseline(0.0))
    jt = K * np.tanh(K / 2) / (NU * DEN)
    jg = 1 / (D * np.cosh(K / 2)) + (1 - 1 / np.cosh(K / 2)) * jt
    gmean = float(w @ g)
    assert abs(gmean - t0) < 1e-12
    # G_0 decreases on [0,a], is zero on B, and integral G_0 = T(0).
    v_sup = max(g0 - t0, t0)
    v = (g - t0) / v_sup
    vnorm = float(np.sqrt(w @ (v * v)))
    vg = float(w @ (g * v))

    # The Green kernel is continuous but has a diagonal derivative jump.
    # Its double-quadrature use below is a numerical check, not verified integration.
    small = np.minimum(xa[:, None], xa[None, :])
    large = np.maximum(xa[:, None], xa[None, :])
    h = np.cosh(K * small) * np.sinh(K * (A - large)) / (D * K * np.cosh(K / 2))
    rg = h + (NU / DEN) * np.outer(s[:gauss_points], s[:gauss_points])
    rv_a = rg @ (wa * v[:gauss_points])
    table, pairs, adaptive = [], [], []
    moment_residual_max = 0.0
    band_grid = np.linspace(0, 1, 10001)
    for m in M_ORDERS:
        u, unorm2 = band_values(x, m)
        ut, ug = float(w @ (u * t)), float(w @ (u * g))
        ugrid, _ = band_values(band_grid, m)
        assert np.max(np.abs(ugrid)) <= 10
        qnorm_error = abs(float(w @ (u * u)) / unorm2 - 1)
        # Verify orthogonality through m with an independent recurrence.
        z = 2 * x - 1
        pprev, pcur = np.ones_like(x), z
        residual = max(abs(float(w @ u)), abs(float(w @ (u * pcur))) * np.sqrt(3))
        for j in range(2, m + 1):
            pn = ((2 * j - 1) * z * pcur - (j - 1) * pprev) / j
            residual = max(residual, abs(float(w @ (u * pn))) * np.sqrt(2 * j + 1))
            pprev, pcur = pcur, pn
        moment_residual_max = max(moment_residual_max, residual)
        table.append({"m": m, "B_T": float(4 * np.pi * m * m * ut / jt),
                      "B_G": float(4 * np.pi * m * m * ug / jg),
                      "B_2": float(np.pi * m * unorm2 / 2),
                      "u_norm_squared_coefficient_formula": float(unorm2),
                      "quadrature_norm_relative_discrepancy": float(qnorm_error),
                      "largest_orthonormal_moment_residual": float(residual),
                      "sampled_sup_abs_u": float(np.max(np.abs(ugrid)))})
        u_scaled = u / 10
        unorm = float(np.sqrt(unorm2) / 10)
        up_t, up_g = ut / 10, ug / 10
        alpha_free = float(NU * (wa @ (u_scaled[:gauss_points] * rv_a)))
        ru_a = rg @ (wa * u_scaled[:gauss_points])
        alpha_rev = float(NU * (wa @ (u_scaled[:gauss_points] * ru_a)))
        free = width_result(0.5, alpha_free, up_t, vg, unorm, vnorm)
        rev = width_result(0.5, alpha_rev, up_t, up_g, unorm, unorm)
        # Actual leading terms of these particular scaled constructions.
        free_leading = 2 * 0.5 * NU * jt * vg / (40 * np.pi * m * m)
        rev_leading = NU * jt * jg / (1600 * np.pi ** 2 * m ** 4)
        for result, leading in ((free, free_leading), (rev, rev_leading)):
            result["asymptotic_leading_term"] = float(leading)
            result["width_to_leading_lower"] = result["lower_width_from_resolvent_norm"] / leading
            result["width_to_leading_upper"] = result["upper_width_from_resolvent_norm"] / leading
            result["analytical_kernel_lower_bound"] = 0.5
        pairs.append({"m": m, "unrestricted": free, "reversible": rev})
        for radius in RADII:
            delta_free = min(0.5, radius / (unorm * vnorm))
            delta_rev = min(0.5, radius / (unorm * unorm))
            ff = width_result(delta_free, alpha_free, up_t, vg, unorm, vnorm)
            rr = width_result(delta_rev, alpha_rev, up_t, up_g, unorm, unorm)
            for result in (ff, rr):
                result["analytical_kernel_lower_bound"] = 1 - result["amplitude"]
                result["budget_active"] = bool(abs(result["quadratic_radius_used"] - radius) <= 1e-12 * radius)
                assert result["quadratic_radius_used"] <= radius * (1 + 1e-12)
            adaptive.append({"m": m, "radius": radius, "unrestricted": ff, "reversible": rr})
    assert moment_residual_max < 1e-8
    assert max(row["quadrature_norm_relative_discrepancy"] for row in table) < 1e-8
    return {"kernel_definitions": {
                "band": "u_m(x)=-(1/m) sum_{n=m+1}^{2m} ell_n(1/2) ell_n(x)",
                "U": "U_m=u_m/10, using the manuscript's analytic uniform bound ||u_m||_infinity <= 10",
                "V": "V=(G_0-T(0))/||G_0-T(0)||_infinity",
                "fixed_pairs": "1 +/- (1/2) U_m(y)V(z), and 1 +/- (1/2) U_m(y)U_m(z)",
                "adaptive_pairs": "replace 1/2 by min(1/2,epsilon/(||left||_2 ||right||_2))",
                "width_bound_status": "Analytical resolvent-norm inequalities evaluated with non-validated floating-point pairings; not rigorous numerical enclosures."},
            "baseline": {"D": D, "nu": NU, "a": A, "J_T": float(jt), "J_G": float(jg),
                         "G_integral_minus_T_zero": gmean - t0,
                         "V_sup_normalizer": float(v_sup), "V_norm": vnorm},
            "band_table": table, "positive_pairs": pairs, "budget_adaptive_cases": adaptive,
            "largest_band_moment_residual": float(moment_residual_max)}


def matrix_checks():
    # Exact local diffusion coefficients for the two cells [0,1/2],[1/2,1].
    s0 = float(baseline(0.0)[0])
    sint = (A - np.tanh(K * A) / K) / NU
    svec = np.array([sint / A, 0.0])
    hmat = np.array([[sint / A, 0.0], [0.0, 0.0]])
    bvec = np.array([s0, 0.0])
    rows = []
    for amplitude in (-1, -0.75, -0.25, 0, 0.25, 0.75, 1):
        z = np.array([[1 + amplitude, 1 - amplitude], [1 - amplitude, 1 + amplitude]])
        mmat = 0.5 * z.T
        value = float(s0 + NU * bvec @ mmat @ np.linalg.solve(np.eye(2) - NU * hmat @ mmat, svec))
        scalar = float(s0 / (1 - NU * (1 + amplitude) * sint))
        rows.append({"t": amplitude, "matrix_objective": value, "scalar_objective": scalar,
                     "absolute_discrepancy": abs(value - scalar)})
    assert max(row["absolute_discrepancy"] for row in rows) < 1e-14
    return rows


def repair_checks(seed=20260926, examples_per_class=8):
    """Use exact degree-two cell quadrature for all polynomial quantities.

    The repaired kernels are piecewise polynomials, not block-constant and
    not claimed continuous. A proven lower bound also controls points
    between the quadrature nodes.
    """
    quad = [quadrature(24, j / 4, (j + 1) / 4) for j in range(4)]
    x = np.concatenate([item[0] for item in quad])
    w = np.concatenate([item[1] for item in quad])
    cells = np.repeat(np.arange(4), 24)
    phi = np.column_stack((np.sqrt(3) * (2 * x - 1), np.sqrt(5) * (6 * x * x - 6 * x + 1)))
    phi_sup = np.sqrt([3.0, 5.0])
    phi_l1 = np.array([np.sqrt(3) / 2, 2 * np.sqrt(5) / (3 * np.sqrt(3))])
    lam = float(phi_sup @ phi_l1)
    rng = np.random.default_rng(seed)
    output = []
    for reversible in (False, True):
        for index in range(examples_per_class):
            raw = rng.normal(size=(4, 4))
            if reversible:
                raw = (raw + raw.T) / 2
            q = raw - raw.mean(axis=0)[None, :] - raw.mean(axis=1)[:, None] + raw.mean()
            q *= 0.8 / np.max(np.abs(q))
            zmat = 1 + q
            qgrid = q[cells[:, None], cells[None, :]]
            coefficients = phi.T @ (w[:, None] * qgrid)
            eta = np.max(np.abs(coefficients), axis=1)
            beta_all = float(phi_sup @ eta)
            beta = (2 + lam) * beta_all if reversible else beta_all
            qc = qgrid - phi @ coefficients
            if reversible:
                qc = qc - (qc * w[None, :]) @ phi @ phi.T
            repaired_q = qc / (1 + beta)
            repaired_k = 1 + repaired_q
            weights = w[:, None] * w[None, :]
            original_norm = float(np.sqrt(np.mean(q * q)))
            repaired_norm = float(np.sqrt(np.sum(weights * repaired_q * repaired_q)))
            moment_error = float(np.max(np.abs(phi.T @ (w[:, None] * repaired_q))))
            destination_error = float(np.max(np.abs(w @ repaired_q)))
            source_error = float(np.max(np.abs(repaired_q @ w)))
            symmetry_error = float(np.max(np.abs(repaired_q - repaired_q.T))) if reversible else None
            displacement = float(np.sqrt(np.sum(weights * (repaired_q - qgrid) ** 2)))
            displacement_bound = beta * (1 + original_norm) / (1 + beta)
            analytical_lower_bound = float(np.min(zmat) / (1 + beta))
            row = {"reversible": reversible, "example_index": index,
                   "original_four_cell_kernel": zmat.tolist(),
                   "original_quadratic_radius": original_norm, "repaired_quadratic_radius": repaired_norm,
                   "beta": beta, "eta": eta.tolist(),
                   "largest_moment_residual": moment_error,
                   "destination_marginal_residual": destination_error,
                   "source_marginal_residual": source_error,
                   "symmetry_residual": symmetry_error,
                   "sampled_kernel_minimum": float(np.min(repaired_k)),
                   "analytical_kernel_lower_bound": analytical_lower_bound,
                   "repair_L2_displacement": displacement,
                   "repair_L2_displacement_bound": float(displacement_bound)}
            assert moment_error < 1e-12 and destination_error < 1e-12 and source_error < 1e-12
            assert repaired_norm <= original_norm * (1 + 1e-12)
            assert displacement <= displacement_bound * (1 + 1e-12)
            assert analytical_lower_bound > 0 and np.min(repaired_k) >= analytical_lower_bound - 1e-12
            assert not reversible or symmetry_error < 1e-12
            output.append(row)
    return {"seed": seed, "examples_per_class": examples_per_class,
            "quadrature_points_per_cell": 24, "Lambda_2": lam, "examples": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("landscape_global_verification.json"))
    parser.add_argument("--gauss-points", type=int, default=1600)
    args = parser.parse_args()
    if args.gauss_points < 2 * max(M_ORDERS) + 1:
        parser.error("--gauss-points must be at least 1025 for these degree-1024 bands")
    report = {"description": "Floating-point checks of formulas and feasible constructions; no global optimization or interval certification.",
              "software": {"python": platform.python_version(), "numpy": np.__version__},
              "numpy_version": np.__version__, "gauss_points_per_half": args.gauss_points,
              "hermite_grid_points": 100001,
              "hermite": hermite_checks(),
              "bands_and_kernels": band_and_kernel_checks(args.gauss_points),
              "two_cell_matrix": matrix_checks(), "four_cell_repair": repair_checks()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("All floating-point consistency checks passed.")
    print(f"Report: {args.output.resolve()}")
    for row in report["bands_and_kernels"]["band_table"]:
        print(f"m={row['m']:3d}: B_T={row['B_T']:.6f}, B_G={row['B_G']:.6f}, B_2={row['B_2']:.6f}")
    print("Hermite relative gap discrepancy:", max(r["relative_gap_discrepancy"] for r in report["hermite"]))
    print("Largest repair moment residual:", max(r["largest_moment_residual"] for r in report["four_cell_repair"]["examples"]))


if __name__ == "__main__":
    main()
