"""Independent stationary Langevin moment hierarchy through cubic order g**2.

This is a classical perturbative reference about a stable Gaussian process,
not a claim of a global stationary distribution for an unbounded cubic potential.
No Dyson/Born implementation is used to generate its stationary moments.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from negfpy.anharmonic import (
    CubicSCBAProblem, DenseCubicVertex, HarmonicReservoir, SignedFrequencyGrid,
    build_cubic_cut_current_operator, connected_cubic_moments,
)
from negfpy.core.types import Device1D, LeadBlocks


@lru_cache(maxsize=1)
def _hierarchy():
    """Solve polynomial stationarity equations, independent of time contractions."""
    n = 3
    d = np.array([[2.3, -.45, 0.], [-.45, 2., -.35], [0., -.35, 1.8]])
    gamma = np.array([.7, 0., .55])
    temperature = np.array([1.2, 0., .5])
    a = np.block([[np.zeros((n, n)), np.eye(n)], [-d, -np.diag(gamma)]])
    diffusion = np.zeros_like(a)
    diffusion[n:, n:] = np.diag(2 * gamma * temperature)
    s0 = solve_continuous_lyapunov(a, -diffusion)
    psi = np.zeros((n, n, n))
    for direction, strength in (([-1., 1., 0.], .6), ([0., -1., 1.], .4)):
        psi += strength * np.einsum("i,j,k->ijk", direction, direction, direction)

    # Cubic force = -g/2 Psi_i,j,k (u_j*u_k - S0_j,k). The subtracted
    # covariance is an explicitly fixed, local linear-potential counterterm.
    # Its mean is zero through O(g); a fixed counterterm leaves O(g**3) means.
    counterforce = .5 * np.einsum("ijk,jk->i", psi, s0[:n, :n])
    dim = 2 * n
    one_leg = np.zeros((dim, dim, dim))
    one_leg[n:] = -np.einsum("aij,ib,jc->abc", psi, s0[:n], s0[:n])
    third_source = one_leg + one_leg.transpose(1, 0, 2) + one_leg.transpose(1, 2, 0)
    identity = np.eye(dim)
    k3 = (np.kron(a, np.kron(identity, identity))
          + np.kron(identity, np.kron(a, identity))
          + np.kron(identity, np.kron(identity, a)))
    t1 = np.linalg.solve(k3, -third_source.reshape(-1)).reshape((dim,) * 3)
    one_covariance_leg = np.zeros((dim, dim))
    one_covariance_leg[n:] = -.5 * np.einsum("aij,ijb->ab", psi, t1[:n, :n])
    second_source = one_covariance_leg + one_covariance_leg.T
    s2 = solve_continuous_lyapunov(a, -second_source)
    return dict(n=n, d=d, gamma=gamma, temperature=temperature, a=a, diffusion=diffusion,
                s0=s0, psi=psi, t1=t1, s2=s2, counterforce=counterforce,
                third_source=third_source, second_source=second_source, k3=k3)


def _problem(h, strength):
    n = h["n"]
    device = Device1D([h["d"][i:i + 1, i:i + 1] for i in range(n)],
                      [h["d"][i:i + 1, i + 1:i + 2] for i in range(n - 1)])
    # The cut operator only uses device and vertex. These leads are placeholders;
    # this reference's physical baths are the Langevin drift and diffusion above.
    lead = LeadBlocks(np.array([[2.5]]), np.array([[-.3]]))
    return CubicSCBAProblem(
        device=device, left=HarmonicReservoir(lead, temperature=1.2),
        right=HarmonicReservoir(lead, temperature=.5),
        vertex=DenseCubicVertex(strength * h["psi"]),
        grid=SignedFrequencyGrid(step=.1, n_positive=3),
    )


def _time_lines(h, lag, epsilon):
    eigenvalues, eigenvectors = np.linalg.eig(h["a"])
    propagator = np.einsum("ia,ta,aj->tij", eigenvectors,
                          np.exp(lag[:, None] * eigenvalues), np.linalg.inv(eigenvectors))
    np.testing.assert_allclose(propagator.imag, 0., atol=2e-15)
    propagator = propagator.real
    n = h["n"]
    covariance = propagator @ h["s0"]
    # Positive-lag retarded displacement Green function is -E_up. epsilon
    # is a formal hbar, not a damping parameter or a quantum bath model.
    c = covariance[:, :n, :n] - .5j * epsilon * propagator[:, :n, n:]
    p = covariance[:, n:, :n] - .5j * epsilon * propagator[:, n:, n:]
    return c, p


@lru_cache(maxsize=1)
def build_reference():
    """Return JSON-compatible evidence, including explicit approximation scope."""
    h = _hierarchy()
    n, strength = h["n"], .12
    covariance = h["s0"] + strength**2 * h["s2"]
    contacts = h["gamma"] * (h["temperature"] - np.diag(covariance)[n:])
    problem = _problem(h, strength)
    cuts = []
    for cut in (1, 2):
        operator = build_cubic_cut_current_operator(problem, cut=cut)
        i, j = operator.harmonic_indices.T
        a, b, c = operator.cubic_indices.T
        result = operator.evaluate_moments(covariance[n + i, j],
                                          strength * h["t1"][n + a, b, c])
        cuts.append({"cut": cut, "harmonic": result["harmonic_solver"],
                     "cubic": result["cubic_solver"], "total": result["total_solver"],
                     "contact_prefix": float(np.sum(contacts[:cut])),
                     "continuity_residual": float(result["total_solver"]
                                                  - np.sum(contacts[:cut]))})

    indices = np.array([(i, j, k) for i in range(n) for j in range(n)
                        for k in range(j, n)])
    expected = h["t1"][n + indices[:, 0], indices[:, 1], indices[:, 2]]
    lag = np.linspace(0., 120., 24001)
    epsilon = 1e-4
    c, p = _time_lines(h, lag, epsilon)
    vertex = DenseCubicVertex(h["psi"])
    histories = []
    for end, stride in ((20., 2), (40., 2), (80., 2), (120., 2), (120., 1)):
        selector = slice(0, int(round(end / (lag[1] - lag[0]))) + 1, stride)
        result = connected_cubic_moments(vertex, indices, lag[selector],
                                         c[selector], p[selector])
        moment = result["connected_moments"] / epsilon
        histories.append({"memory": end, "step": float(lag[stride]),
                          "max_hierarchy_difference": float(np.max(abs(moment - expected))),
                          "max_endpoint_integrand": float(np.max(abs(
                              result["endpoint_integrand"] / epsilon))),
                          "moments": moment.tolist()})
    half_c, half_p = _time_lines(h, lag, epsilon / 2)
    half_hbar = connected_cubic_moments(vertex, indices, lag, half_c, half_p)
    half_moment = half_hbar["connected_moments"] / (epsilon / 2)
    # The contraction is cubic in epsilon; after dividing by epsilon its
    # quantum contamination is exactly proportional to epsilon**2 at fixed
    # lines and quadrature. The difference divided by 3 estimates the remaining
    # contamination of the half-epsilon result, independently of dt errors.
    hbar_difference = float(np.max(abs(half_moment - histories[-1]["moments"])))
    return {
        "model": "three-site-classical-Langevin-formal-cubic-perturbation",
        "strength": strength, "dynamical_matrix": h["d"].tolist(),
        "gamma": h["gamma"].tolist(), "temperature": h["temperature"].tolist(),
        "mean_policy": "fixed-local-linear-counterforce-cancels-leading-tadpole",
        "counterforce_coefficient": h["counterforce"].tolist(),
        "mean_order_remaining": "O(g**3)", "stationary_order": "through-g**2",
        "global_cubic_stationary_distribution_claimed": False,
        "quantum_scba_conservation_proof": False,
        "finite_hbar": epsilon, "classical_limit_error_order": "O(epsilon**2)",
        "half_hbar_moment_change": hbar_difference,
        "full_hbar_quantum_contamination_estimate": 4 * hbar_difference / 3,
        "contact_currents": contacts.tolist(), "cuts": cuts,
        "moment_indices": indices.tolist(), "hierarchy_moments": expected.tolist(),
        "histories": histories,
        "harmonic_stationarity_residual": float(np.max(abs(
            h["a"] @ h["s0"] + h["s0"] @ h["a"].T + h["diffusion"]))),
        "third_stationarity_residual": float(np.max(abs(
            h["k3"] @ h["t1"].reshape(-1) + h["third_source"].reshape(-1)))),
        "covariance_correction_residual": float(np.max(abs(
            h["a"] @ h["s2"] + h["s2"] @ h["a"].T + h["second_source"]))),
    }


def test_stationary_hierarchy_closes_contact_and_full_cut_currents():
    reference = build_reference()
    assert max(reference["contact_currents"]) > 0.01
    assert abs(sum(reference["contact_currents"])) < 1e-14
    for cut in reference["cuts"]:
        assert abs(cut["cubic"]) > 1e-6
        assert abs(cut["continuity_residual"]) < 1e-14
        # Omitting the cubic operator creates a resolved continuity defect.
        assert abs(cut["harmonic"] - cut["contact_prefix"]) > 1e-6
    for key in ("harmonic_stationarity_residual", "third_stationarity_residual",
                "covariance_correction_residual"):
        assert reference[key] < 1e-14


def test_classical_limit_three_point_memory_and_step_convergence():
    reference = build_reference()
    histories = reference["histories"]
    assert histories[-1]["max_hierarchy_difference"] < 2e-6
    assert histories[-1]["max_hierarchy_difference"] < (
        histories[-2]["max_hierarchy_difference"] / 3.9)
    assert histories[0]["max_hierarchy_difference"] > (
        histories[2]["max_hierarchy_difference"] * 2)
    assert histories[-1]["max_endpoint_integrand"] < 1e-12
    assert reference["full_hbar_quantum_contamination_estimate"] < 1e-9


def test_stationary_cyclic_third_moments_and_declared_mean_policy():
    h = _hierarchy()
    n = h["n"]
    third = h["t1"]
    np.testing.assert_allclose(third, third.transpose(1, 0, 2), atol=2e-15)
    np.testing.assert_allclose(third, third.transpose(2, 1, 0), atol=2e-15)
    cyclic = third[n:, :n, :n] + third[:n, n:, :n] + third[:n, :n, n:]
    np.testing.assert_allclose(cyclic, 0., atol=2e-15)
    gaussian_force = -.5 * np.einsum("ijk,jk->i", h["psi"], h["s0"][:n, :n])
    np.testing.assert_allclose(gaussian_force + h["counterforce"], 0., atol=1e-15)
