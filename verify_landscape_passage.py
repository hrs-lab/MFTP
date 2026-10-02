#!/usr/bin/env python3
"""Reproduce Appendix A checks for landscape_passage_uncertainty.

Requires only Python 3 and NumPy. Run:
    python verify_landscape_passage.py
Optional --output PATH overrides the JSON destination (default: beside script).
All calculations use ordinary floating-point arithmetic. They are numerical
consistency checks, not interval-arithmetic certificates or global optimization.
"""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
from numpy.polynomial.legendre import leggauss, legval, legvander

D = 0.3
NU = 0.7
TARGET = 0.5
ORDERS = (8, 32, 128, 512)
QUADRATURE_ORDER = 1600
KAPPA = np.sqrt(NU / D)
COSH_A = np.cosh(KAPPA * TARGET)
C = 1.0 - 1.0 / COSH_A
DENOM = 0.5 + np.tanh(KAPPA / 2.0) / KAPPA
JT = KAPPA * np.tanh(KAPPA / 2.0) / (NU * DENOM)
JG = 1.0 / (D * COSH_A) + C * JT


def refined_gauss_legendre(n):
    """Refine NumPy's nodes and recompute weights by the derivative formula.

    High-order residuals magnify small endpoint-weight errors. Four Newton
    steps followed by recurrence-based weights reduce this effect without
    extra dependencies. The result remains ordinary double precision.
    """
    nodes, _ = leggauss(n)
    for _ in range(4):
        previous, current = np.ones(n), nodes.copy()
        for j in range(2, n+1):
            previous, current = current, ((2*j-1)*nodes*current-(j-1)*previous)/j
        derivative = n*(nodes*current-previous)/(nodes*nodes-1)
        nodes -= current/derivative
    previous, current = np.ones(n), nodes.copy()
    for j in range(2, n+1):
        previous, current = current, ((2*j-1)*nodes*current-(j-1)*previous)/j
    derivative = n*(nodes*current-previous)/(nodes*nodes-1)
    weights = 2/((1-nodes*nodes)*derivative*derivative)
    weights *= 2/np.sum(weights)
    return nodes, weights


def baseline(z):
    z = np.asarray(z, dtype=float)
    inside = z < TARGET
    s = np.where(inside, (1.0 - np.cosh(KAPPA*z)/COSH_A)/NU, 0.0)
    t = s / DENOM
    g = np.where(inside, np.sinh(KAPPA*(TARGET-z))/(D*KAPPA*COSH_A), 0.0) + C*t
    return t, g, s


def tridiagonal_solve(lower, diagonal, upper, rhs):
    """Thomas elimination for the mixed-boundary finite-difference operator."""
    b = diagonal.copy()
    r = np.array(rhs, dtype=float, copy=True)
    for j in range(1, len(b)):
        factor = lower[j-1] / b[j-1]
        b[j] -= factor * upper[j-1]
        r[j] -= factor * r[j-1]
    r[-1] /= b[-1]
    for j in range(len(b)-2, -1, -1):
        r[j] = (r[j] - upper[j]*r[j+1]) / b[j]
    return r


def finite_difference(n, amplitude, sign, ufun, vfun):
    """Solve the discretized backward equation; no continuum resolvent used.

    Neumann reflection is imposed by a centered ghost point. The absorbing
    endpoint is eliminated. Both nonlocal integrals use trapezoidal weights.
    Two dense rank-one terms are handled by a two-by-two Woodbury solve.
    """
    h = TARGET/n
    z = np.arange(n)*h
    weights = np.full(n, h)
    weights[0] = h/2.0
    diagonal = np.full(n, NU + 2*D/h**2)
    lower = np.full(n-1, -D/h**2)
    upper = lower.copy()
    upper[0] *= 2
    u, v = ufun(z), vfun(z)
    forcing = np.column_stack((np.ones(n), v))
    k_forcing = tridiagonal_solve(lower, diagonal, upper, forcing)
    k_one = k_forcing[:, 0]
    nonlocal_rows = np.vstack((NU*weights, sign*amplitude*NU*weights*u))
    small = np.eye(2) - nonlocal_rows @ k_forcing
    solution = k_one + k_forcing @ np.linalg.solve(small, nonlocal_rows @ k_one)
    baseline_fd = k_one/(1.0-NU*weights@k_one)
    return z, solution, baseline_fd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('landscape_passage_verification.json'))
    args = parser.parse_args()
    nodes, weights0 = refined_gauss_legendre(QUADRATURE_ORDER)
    z = np.r_[(nodes+1)/4, (nodes+3)/4]
    weights = np.r_[weights0/4, weights0/4]
    t, g, s = baseline(z)
    t0, g0, _ = (float(item) for item in baseline(0.0))
    gmean = float(weights@g)
    b = g-gmean
    gamma = float(np.sqrt(weights@(b*b)))
    basis = legvander(2*z-1, max(ORDERS))*np.sqrt(2*np.arange(max(ORDERS)+1)+1)
    coeff_t, coeff_g = basis.T@(weights*t), basis.T@(weights*g)
    c_free = NU*JT*gamma/np.sqrt(24*np.pi)
    c_rev = NU*JT*JG/(24*np.pi)
    rows, checks = [], []
    residuals = {}
    for m in ORDERS:
        a = t-basis[:, :m+1]@coeff_t[:m+1]
        c = g-basis[:, :m+1]@coeff_g[:m+1]
        anorm = float(np.sqrt(weights@(a*a)))
        cnorm = float(np.sqrt(weights@(c*c)))
        A, B, E = a/anorm, b/gamma, c/cnorm
        aa, bb, ee = (float(weights@(f*f)) for f in (A, B, E))
        ae = float(weights@(A*E))
        normalizer = np.sqrt(2*(1+ae*ae))
        sf = NU*anorm*gamma
        sr = NU*anorm*cnorm*np.sqrt((1+ae*ae)/2)
        ma = basis[:, :m+1].T@(weights*A)
        me = basis[:, :m+1].T@(weights*E)
        mb = float(weights@B)
        # L2 norms in the remaining variable of every polynomial moment.
        rev_moments = np.sqrt(np.maximum(0, ma*ma*ee+me*me*aa+2*ma*me*ae))/normalizer
        free_pairing = NU*float(weights@(t*A))*float(weights@(g*B))
        rev_pairing = NU*(float(weights@(t*A))*float(weights@(g*E))
                          + float(weights@(t*E))*float(weights@(g*A)))/normalizer
        rows.append(dict(m=m, F_m=float(m**1.5*sf/c_free),
                         V_m=float(m**3*sr/c_rev),
                         alpha_m_radians=float(np.arccos(np.clip(ae, -1, 1))),
                         S_m=sf, S_m_reversible=sr))
        checks.append(dict(
            m=m,
            normalized_residual_polynomial_orthogonality_max=float(max(np.max(abs(ma)), np.max(abs(me)))),
            free_daughter_polynomial_moment_L2_max=float(np.max(abs(ma))*np.sqrt(bb)),
            free_normalization_L2=float(abs(ma[0])*np.sqrt(bb)),
            free_stationarity_L2=float(abs(mb)*np.sqrt(aa)),
            reversible_polynomial_moment_L2_max=float(np.max(rev_moments)),
            reversible_normalization_and_stationarity_L2=float(rev_moments[0]),
            free_tensor_budget=float(np.sqrt(aa*bb)),
            reversible_tensor_budget=float(np.sqrt(2*aa*ee+2*ae*ae)/normalizer),
            free_unprojected_gradient_relative_error=float(abs(free_pairing/sf-1)),
            reversible_unprojected_gradient_relative_error=float(abs(rev_pairing/sr-1))))
        if m in (8, 32):
            residuals[m] = (a, anorm, sf)

    # The normalized b is a combination of 1, cosh(k z), sinh(k z) on A.
    # This yields an explicit solution of (-D d^2/dz^2 + nu)Kv=v;
    # the resonant terms are z sinh(k z) and z cosh(k z).
    a0 = (C/(NU*DENOM)-gmean)/gamma
    ac = (np.tanh(KAPPA*TARGET)/(D*KAPPA)-C/(NU*DENOM*COSH_A))/gamma
    ash = -1/(D*KAPPA*gamma)
    homogeneous_sinh = ash/(2*NU)

    def particular(x):
        return a0/NU-ac*x*np.sinh(KAPPA*x)/(2*D*KAPPA)-ash*x*np.cosh(KAPPA*x)/(2*D*KAPPA)

    homogeneous_cosh = -(particular(TARGET)+homogeneous_sinh*np.sinh(KAPPA*TARGET))/COSH_A
    sv = float(weights@(s*(g-gmean)/gamma))

    def vfun(x):
        return (baseline(x)[1]-gmean)/gamma

    def rvfun(x):
        x = np.asarray(x)
        kv = particular(x)+homogeneous_cosh*np.cosh(KAPPA*x)+homogeneous_sinh*np.sinh(KAPPA*x)
        return np.where(x < TARGET, kv+NU/DENOM*baseline(x)[2]*sv, 0.0)

    path_results = []
    positivity_grid_intervals = 65536
    fine_z = np.linspace(0.0, 1.0, positivity_grid_intervals+1)
    for m in (8, 32):
        a_values, anorm, sharp = residuals[m]
        legendre_coeff = coeff_t[:m+1]*np.sqrt(2*np.arange(m+1)+1)

        def ufun(x):
            return (baseline(x)[0]-legval(2*np.asarray(x)-1, legendre_coeff))/anorm

        indices = np.arange(m+1)
        # |ell_n'| <= sqrt(2n+1)n(n+1) on [0,1], and |T'| <= JT.
        derivative_bound_u = (JT+np.sum(abs(coeff_t[:m+1])*np.sqrt(2*indices+1)*indices*(indices+1)))/anorm
        sampled_u_sup = float(np.max(abs(ufun(fine_z))))
        upper_u_sup = sampled_u_sup+derivative_bound_u/(2*positivity_grid_intervals)
        # G_0 decreases on A and vanishes on B: its endpoint extrema suffice.
        upper_v_sup = max(abs(g0-gmean), abs(gmean))/gamma
        upper_q_sup = upper_u_sup*upper_v_sup
        amplitude = min(0.05, 0.25/upper_q_sup)
        theta = float(NU*weights@((a_values/anorm)*rvfun(z)))
        grid_results = []
        for sign in (-1, 1):
            previous_displacement_error = None
            for n in (512, 1024, 2048, 4096):
                mesh, fd, fd_base = finite_difference(n, amplitude, sign, ufun, vfun)
                displacement_exact = sign*amplitude*NU*anorm*rvfun(mesh)/(1-sign*amplitude*theta)
                exact = baseline(mesh)[0]+displacement_exact
                passage_error = float(np.max(abs(fd-exact)))
                displacement_error = float(np.max(abs((fd-fd_base)-displacement_exact)))
                observed_order = (None if previous_displacement_error is None else
                                  float(np.log2(previous_displacement_error/displacement_error)))
                previous_displacement_error = displacement_error
                grid_results.append(dict(sign=sign, intervals=n,
                                         maximum_passage_time_error=passage_error,
                                         maximum_displacement_error=displacement_error,
                                         displacement_observed_order=observed_order))
        scalar_plus = t0+amplitude*sharp/(1-amplitude*theta)
        scalar_minus = t0-amplitude*sharp/(1+amplitude*theta)
        full_plus = t0+amplitude*NU*anorm*float(rvfun(0.0))/(1-amplitude*theta)
        full_minus = t0-amplitude*NU*anorm*float(rvfun(0.0))/(1+amplitude*theta)
        diameter = 2*amplitude*sharp/(1-amplitude**2*theta**2)
        path_results.append(dict(m=m, amplitude=float(amplitude), theta=theta,
                                 u_derivative_bound=float(derivative_bound_u),
                                 sampled_u_sup=sampled_u_sup, u_sup_upper_bound=float(upper_u_sup),
                                 v_sup_upper_bound=float(upper_v_sup),
                                 positivity_lower_bound_both_signs=float(1-amplitude*upper_q_sup),
                                 resolvent_smallness_upper_bound=float(2*amplitude),
                                 full_vs_scalar_formula_max_absolute_error=float(max(abs(full_plus-scalar_plus), abs(full_minus-scalar_minus))),
                                 diameter_formula_cancellation_check_absolute_error=float(abs((scalar_plus-scalar_minus)-diameter)),
                                 finite_difference=grid_results))

    finest = [item for path in path_results for item in path['finite_difference'] if item['intervals'] == 4096]
    summary = dict(
        largest_normalized_polynomial_orthogonality_residual=max(row['normalized_residual_polynomial_orthogonality_max'] for row in checks),
        largest_unprojected_gradient_relative_error=max(max(row['free_unprojected_gradient_relative_error'], row['reversible_unprojected_gradient_relative_error']) for row in checks),
        largest_tensor_budget_error=max(max(abs(row['free_tensor_budget']-1), abs(row['reversible_tensor_budget']-1)) for row in checks),
        largest_finest_grid_passage_time_error=max(row['maximum_passage_time_error'] for row in finest),
        minimum_finest_grid_displacement_order=min(row['displacement_observed_order'] for row in finest),
        minimum_positivity_lower_bound=min(row['positivity_lower_bound_both_signs'] for row in path_results))
    result = dict(
        purpose='Floating-point consistency checks for Appendix A; no interval certification or global optimization.',
        software=dict(python=platform.python_version(), numpy=np.__version__),
        parameters=dict(D=D, nu=NU, target=TARGET, initial_state=0.0,
                        moment_orders=list(ORDERS), gauss_nodes_per_half_interval=QUADRATURE_ORDER,
                        gauss_node_newton_refinements=4,
                        positivity_grid_intervals=positivity_grid_intervals,
                        finite_difference_intervals=[512, 1024, 2048, 4096]),
        baseline=dict(T_at_zero=t0, integral_G=gmean, integral_G_vs_T0_absolute_error=abs(gmean-t0),
                      gamma=gamma, J_T=float(JT), J_G=float(JG), C_free=float(c_free), C_reversible=float(c_rev),
                      Rv_at_zero_vs_gamma_absolute_error=abs(float(rvfun(0.0))-gamma)),
        table_A=rows, projection_and_tensor_checks=checks, rank_one_paths=path_results, summary=summary)
    # Tolerances are regression checks, not a proof of the theoretical claims.
    if summary['largest_normalized_polynomial_orthogonality_residual'] > 1e-7:
        raise AssertionError('Polynomial projection check exceeded tolerance.')
    if summary['largest_unprojected_gradient_relative_error'] > 1e-4:
        raise AssertionError('Gradient pairing check exceeded tolerance.')
    if summary['largest_tensor_budget_error'] > 1e-12:
        raise AssertionError('Tensor budget check exceeded tolerance.')
    if summary['largest_finest_grid_passage_time_error'] > 1e-7:
        raise AssertionError('Finite-difference path check exceeded tolerance.')
    if summary['minimum_finest_grid_displacement_order'] < 1.9:
        raise AssertionError('Finite-difference displacement convergence check failed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(table_A=rows, summary=summary), indent=2))
    print(f'Wrote {args.output.resolve()}')


if __name__ == '__main__':
    main()
