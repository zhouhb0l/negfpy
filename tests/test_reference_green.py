"""Synthetic full-matrix tests; no materials, checkpoints, or solver changes."""
import unittest

import numpy as np

from negfpp.reference.increment_green import evaluate_increment, prepare_reference, tangent_green


class IncrementGreenTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(9741)

    def matrix(self, n, scale=1.0):
        return scale * (self.rng.normal(size=(n, n))
                        + 1j*self.rng.normal(size=(n, n)))

    def test_full_nonsymmetric_complex_matches_direct_dyson_and_sources(self):
        n = 5
        k0 = self.matrix(n, .2) + (2.0+.6j)*np.eye(n)
        sc0, sd0 = self.matrix(n), self.matrix(n)
        reference = prepare_reference(k0, sc0, sd0)
        dr, dc, dd = (self.matrix(n, .03) for _ in range(3))
        current = evaluate_increment(reference, dr, dc, dd)
        direct_r = np.linalg.inv(k0-dr)
        direct_c = direct_r @ (sc0+dc) @ direct_r.conj().T
        direct_d = direct_r @ (sd0+dd) @ direct_r.conj().T
        for calculated, direct in ((current.retarded, direct_r),
                                   (current.statistical, direct_c),
                                   (current.spectral, direct_d)):
            np.testing.assert_allclose(calculated, direct, rtol=3e-15, atol=3e-15)
        self.assertGreater(np.linalg.norm(current.spectral
                           - (current.retarded-current.retarded.conj().T)), .1)
        self.assertLess(current.diagnostics['dyson_left_backward_residual'], 1e-15)
        self.assertLess(current.diagnostics['dyson_right_backward_residual'], 1e-15)

    def test_active_block_embedding_keeps_full_device_couplings(self):
        n, active = 6, [4, 1, 3]
        k0 = self.matrix(n, .15)+(2+.5j)*np.eye(n)
        sc0, sd0 = self.matrix(n), self.matrix(n)
        reference = prepare_reference(k0, sc0, sd0)
        blocks = tuple(self.matrix(3, .06) for _ in range(3))
        full = []
        for block in blocks:
            matrix = np.zeros((n, n), complex)
            matrix[np.ix_(active, active)] = block
            full.append(matrix)
        embedded = evaluate_increment(reference, *blocks, active_dofs=active)
        direct = evaluate_increment(reference, *full)
        for field in ('retarded', 'statistical', 'spectral'):
            np.testing.assert_array_equal(getattr(embedded, field), getattr(direct, field))
        # Solving the isolated active operator would omit actual device paths.
        self.assertGreater(np.linalg.norm(embedded.retarded[np.ix_(active, active)]
                           - np.linalg.inv(k0[np.ix_(active, active)]-blocks[0])), .001)

    def test_exact_scalar_zero_with_independent_nonzero_c(self):
        reference = prepare_reference([[2+.5j]], [[-3j]], [[0j]])
        current = evaluate_increment(reference, [[0j]], [[0j]], [[0j]])
        np.testing.assert_array_equal(current.spectral, [[0j]])
        np.testing.assert_array_equal(current.delta_spectral, [[0j]])
        np.testing.assert_array_equal(current.delta_retarded, [[0j]])
        np.testing.assert_array_equal(current.statistical, reference.statistical)
        self.assertNotEqual(current.statistical[0, 0], 0)
        derivative = tangent_green(current, [[.2]], [[.1j]], [[0j]])
        np.testing.assert_array_equal(derivative.spectral, [[0j]])

    def test_tiny_increment_retained_without_subtracting_large_green_fields(self):
        reference = prepare_reference([[1.]], [[2j]], [[3j]])
        change = 1e-18
        current = evaluate_increment(reference, [[change]], [[0j]], [[0j]])
        # The total rounds to its reference; the independently retained
        # increment still contains the analytically nonzero response.
        np.testing.assert_array_equal(current.retarded, reference.retarded)
        self.assertAlmostEqual(current.delta_retarded[0, 0].real/change, 1.)
        self.assertAlmostEqual(current.delta_statistical[0, 0].imag/change, 4.)

    def test_nearly_singular_scalar_reference_and_current(self):
        reference = prepare_reference([[1e-10+2e-12j]], [[3e-12j]], [[4e-12j]])
        changes = ([[-.2e-10+.1e-12j]], [[.2e-12j]], [[-.1e-12j]])
        current = evaluate_increment(reference, *changes)
        direct_r = 1/(reference.operator[0, 0]-changes[0][0][0])
        np.testing.assert_allclose(current.retarded, [[direct_r]], rtol=1e-15)
        np.testing.assert_allclose(current.statistical,
            [[abs(direct_r)**2*(reference.source_c[0, 0]+changes[1][0][0])]], rtol=2e-15)
        self.assertLess(current.diagnostics['dyson_left_backward_residual'], 1e-15)

    def test_analytic_tangent_has_second_order_central_difference_error(self):
        n = 4
        k0 = self.matrix(n, .1)+(1.5+.5j)*np.eye(n)
        reference = prepare_reference(k0, self.matrix(n), self.matrix(n))
        changes = tuple(self.matrix(n, .1) for _ in range(3))
        current = evaluate_increment(reference, *changes)
        directions = tuple(self.matrix(n, .2) for _ in range(3))
        tangent = tangent_green(current, *directions)
        errors = []
        for h in (1e-2, 5e-3):
            plus = evaluate_increment(reference, *(x+h*v for x, v in zip(changes, directions)))
            minus = evaluate_increment(reference, *(x-h*v for x, v in zip(changes, directions)))
            error = 0.
            for field in ('retarded', 'statistical', 'spectral'):
                numerical = (getattr(plus, field)-getattr(minus, field))/(2*h)
                error += np.linalg.norm(numerical-getattr(tangent, field))**2
            errors.append(error**.5)
        self.assertGreater(errors[0]/errors[1], 3.95)
        self.assertLess(errors[0]/errors[1], 4.05)

    def test_active_tangent_is_real_linear_and_not_complex_linear(self):
        reference = prepare_reference([[2+.2j, .1j], [.15, 1+.4j]],
                                      [[-1j, .2], [.3, -2j]], [[.4j, .1], [.2, .3j]])
        directions = tuple(self.matrix(1) for _ in range(3))
        tangent = tangent_green(reference, *directions, active_dofs=[1])
        real_scaled = tangent_green(reference, *(2.3*v for v in directions), active_dofs=[1])
        complex_scaled = tangent_green(reference, *(1j*v for v in directions), active_dofs=[1])
        for field in ('retarded', 'statistical', 'spectral'):
            np.testing.assert_allclose(getattr(real_scaled, field),
                                       2.3*getattr(tangent, field), rtol=1e-15, atol=1e-15)
        self.assertGreater(np.linalg.norm(complex_scaled.statistical
                           -1j*tangent.statistical), .1)

    def test_invalid_input_and_singular_operator_fail_without_repair(self):
        for bad in ([[np.nan]], [[np.inf]]):
            with self.assertRaises(ValueError):
                prepare_reference(bad, [[0]], [[0]])
        with self.assertRaises(ValueError):
            prepare_reference([[0]], [[0]], [[0]])
        reference = prepare_reference(np.eye(2), np.eye(2), np.eye(2))
        with self.assertRaises(ValueError):
            evaluate_increment(reference, np.eye(2), np.eye(2), np.eye(2))
        with self.assertRaises(ValueError):
            tangent_green(reference, [[0]], [[0]], [[0]], active_dofs=[1, 1])
        with self.assertRaises(ValueError):
            tangent_green(reference, [[0]], [[0]], [[0]], active_dofs=[True])


if __name__ == '__main__':
    unittest.main()
