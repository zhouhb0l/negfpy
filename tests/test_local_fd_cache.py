"""Generated dense oracles for the exact g-only cache; no material inputs."""
import weakref

import numpy as np
import pytest

from negfpp.local_fd.g_only_cache import Bands, GOnlyFactor, adjoint


from negfpp.local_fd import batched_blocks as legacy

def dense(bands, sizes):
    batch = bands.diagonal[0].shape[0]
    edges = np.cumsum([0]+list(sizes))
    result = np.zeros((batch, edges[-1], edges[-1]), complex)
    for i, block in enumerate(bands.diagonal):
        result[:, edges[i]:edges[i+1], edges[i]:edges[i+1]] = block
    for i, (up, low) in enumerate(zip(bands.upper, bands.lower)):
        result[:, edges[i]:edges[i+1], edges[i+1]:edges[i+2]] = up
        result[:, edges[i+1]:edges[i+2], edges[i]:edges[i+1]] = low
    return result


def bands_from_dense(value, sizes):
    edges = np.cumsum([0]+list(sizes))
    return Bands(
        [value[:, edges[i]:edges[i+1], edges[i]:edges[i+1]].copy() for i in range(len(sizes))],
        [value[:, edges[i]:edges[i+1], edges[i+1]:edges[i+2]].copy() for i in range(len(sizes)-1)],
        [value[:, edges[i+1]:edges[i+2], edges[i]:edges[i+1]].copy() for i in range(len(sizes)-1)],
    )


def assert_selected(actual, expected, sizes, atol=3e-12):
    target = bands_from_dense(expected, sizes)
    for observed, reference in zip(actual.diagonal+actual.upper+actual.lower,
                                   target.diagonal+target.upper+target.lower):
        np.testing.assert_allclose(observed, reference, rtol=2e-12, atol=atol)


def generated(sizes=(3, 4, 2), batch=2, seed=591):
    rng = np.random.default_rng(seed)
    def random(shape):
        return rng.normal(size=shape)+1j*rng.normal(size=shape)
    diagonal, upper, lower = [], [], []
    sd, su = [], []
    for size in sizes:
        value = .03*random((batch, size, size))
        diagonal.append((value+adjoint(value))/2+(2+1j)*np.eye(size))
        value = random((batch, size, size))
        sd.append((value+adjoint(value))/2)
    for a, b in zip(sizes, sizes[1:]):
        upper.append(.04*random((batch, a, b)))
        # General non-adjoint operator couplings are admitted by the algebra.
        lower.append(.04*random((batch, b, a)))
        su.append(.08*random((batch, a, b)))
    return Bands(diagonal, upper, lower), Bands(sd, su, [adjoint(a) for a in su])


@pytest.mark.parametrize('sizes', [(1,), (3, 4, 2), (48, 24), (48, 48, 24)])
def test_inverse_and_indefinite_source_match_independent_dense(sizes):
    operator, source = generated(sizes)
    factor = GOnlyFactor(operator.diagonal, operator.upper, operator.lower)
    inverse = np.linalg.inv(dense(operator, sizes))
    assert_selected(factor.selected_inverse(), inverse, sizes)
    result = factor.correlate(source)
    expected = inverse@dense(source, sizes)@adjoint(inverse)
    assert_selected(result, expected, sizes)
    for row in range(len(sizes)):
        for column in range(max(0, row-1), min(len(sizes), row+2)):
            target = bands_from_dense(inverse, sizes)
            reference = target.diagonal[row] if row == column else (
                target.upper[row] if column == row+1 else target.lower[column])
            np.testing.assert_allclose(factor.selected_block(row, column), reference, atol=3e-12)
    assert factor.stats()['source_action_block_solves'] == 0
    assert factor.stats()['retained_transfer_blocks'] == 0
    assert factor.stats()['retained_diagonal_green_blocks'] == 0


@pytest.mark.parametrize('sizes', [(2,), (3, 4, 2), (48, 24)])
def test_same_signed_response_as_original_compact_backend(sizes):
    operator, source = generated(sizes, batch=3)
    old = legacy.BatchedFactor(operator.diagonal, operator.upper, operator.lower).compact()
    new = GOnlyFactor(operator.diagonal, operator.upper, operator.lower)
    old_source = legacy.Bands(source.diagonal, source.upper, source.lower)
    for coefficient in (1., -3., 0.):
        scaled = Bands(*[[coefficient*a for a in family] for family in
                         (source.diagonal, source.upper, source.lower)])
        original_scaled = legacy.Bands(scaled.diagonal, scaled.upper, scaled.lower)
        expected, actual = old.correlate(original_scaled), new.correlate(scaled)
        for a, b in zip(actual.diagonal+actual.upper+actual.lower,
                        expected.diagonal+expected.upper+expected.lower):
            np.testing.assert_allclose(a, b, atol=3e-12)
    assert old_source.nbytes == source.nbytes


def test_optical_identity_signed_bath_currents_and_linear_current_action():
    sizes, batch = (48, 24), 2
    rng = np.random.default_rng(718)
    total = sum(sizes)
    supports = [np.arange(8), np.arange(total-8, total),
                np.r_[np.arange(20, 40), np.arange(48, 64)]]
    baths = []
    for support in supports:
        gamma = np.zeros((batch, total, total), complex)
        values = rng.normal(size=(batch, len(support), len(support)))
        values = values@values.swapaxes(-1, -2)/len(support)+np.eye(len(support))
        for f in range(batch):
            gamma[f][np.ix_(support, support)] = values[f]
        baths.append(gamma)
    gamma_sum = sum(baths)
    raw = rng.normal(size=(batch, total, total))+1j*rng.normal(size=(batch, total, total))
    harmonic = .02*(raw+adjoint(raw))/2
    operator_dense = 3*np.eye(total)-harmonic+.5j*gamma_sum
    operator = bands_from_dense(operator_dense, sizes)
    factor = GOnlyFactor(operator.diagonal, operator.upper, operator.lower)
    inverse = np.linalg.inv(operator_dense)
    spectral = 1j*(inverse-adjoint(inverse))
    optical = factor.correlate(bands_from_dense(gamma_sum, sizes), qualification=True)
    assert_selected(optical, spectral, sizes, atol=2e-11)

    def currents(weights):
        source = sum(w*g for w, g in zip(weights, baths))
        correlation = factor.correlate(bands_from_dense(source, sizes))
        full_correlation = dense(correlation, sizes)  # all blocks for two-block case
        current = np.stack([weights[i]*np.einsum('fij,fji->f', g, spectral)
                            -np.einsum('fij,fji->f', g, full_correlation)
                            for i, g in enumerate(baths)], axis=1)
        dense_current = np.stack([
            weights[i]*np.einsum('fij,fji->f', g, spectral)
            -np.einsum('fij,fji->f', g, inverse@source@adjoint(inverse))
            for i, g in enumerate(baths)], axis=1)
        np.testing.assert_allclose(current, dense_current, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(current.imag, 0., atol=2e-11)
        np.testing.assert_allclose(current.sum(axis=1), 0., atol=2e-11)
        return current

    direction = np.array([.5, -.5, -.14])
    second = np.array([-.2, .4, .3])
    base = currents(direction)
    np.testing.assert_allclose(currents(direction+17), base, atol=5e-11)
    np.testing.assert_allclose(currents(direction+second), base+currents(second), atol=3e-11)
    np.testing.assert_allclose(currents(np.ones(3)), 0., atol=2e-11)


def test_exact_family_aliases_share_but_equal_distinct_arrays_do_not():
    batch, size, count = 2, 4, 7
    diagonal = [(3+1j)*np.broadcast_to(np.eye(size), (batch, size, size)) for _ in range(count)]
    up = np.full((batch, size, size), .002+0j)
    low = up.copy()
    factor = GOnlyFactor(diagonal, [up]*(count-1), [low]*(count-1))
    assert len({id(a) for a in factor.raw_upper}) == 1
    assert len({id(a) for a in factor.raw_lower}) == 1
    assert factor.raw_upper[0] is not factor.raw_lower[0]
    assert factor.nbytes == (count+2)*batch*size**2*16
    separate = GOnlyFactor(diagonal, [up.copy() for _ in range(count-1)],
                          [low.copy() for _ in range(count-1)])
    assert separate.stats()['retained_coupling_families'] == 2*(count-1)
    assert separate.nbytes > factor.nbytes
    up[:] = 50
    assert np.max(np.abs(factor.raw_upper[0])) < 1
    for block in factor.left+factor.raw_upper+factor.raw_lower:
        assert not block.flags.writeable


def test_constructor_does_not_retain_diagonal_operator_or_full_green():
    operator, source = generated()
    references = [weakref.ref(a) for a in operator.diagonal]
    factor = GOnlyFactor(operator.diagonal, operator.upper, operator.lower)
    del operator
    assert all(ref() is None for ref in references)
    assert not hasattr(factor, 'green')
    assert not hasattr(factor, 'a')
    assert not hasattr(factor, 'u')
    assert factor.compact() is factor
    factor.correlate(source)
    expected = sum(a.nbytes for a in
                   {id(a): a for a in factor.left+factor.raw_upper+factor.raw_lower}.values())
    assert factor.nbytes == expected


def test_no_solves_or_inverse_during_actions_or_conversion(monkeypatch):
    operator, source = generated()
    old = legacy.BatchedFactor(operator.diagonal, operator.upper, operator.lower)
    standalone = GOnlyFactor(operator.diagonal, operator.upper, operator.lower)
    def forbidden(*args, **kwargs):
        raise AssertionError('A source/selection/conversion must not invert')
    monkeypatch.setattr(np.linalg, 'solve', forbidden)
    monkeypatch.setattr(np.linalg, 'inv', forbidden)
    converted = GOnlyFactor.from_existing_factor(old, source_point_qualified=True)
    before = standalone.correlate(source)
    after = converted.correlate(source)
    for a, b in zip(before.diagonal+before.upper+before.lower,
                    after.diagonal+after.upper+after.lower):
        np.testing.assert_array_equal(a, b)
    converted.selected_inverse()
    converted.selected_block(1, 2)
    assert converted.stats()['factor_block_solves'] == 0
    assert converted.stats()['inherited_factor_block_solves'] == 6
    assert converted.stats()['conversion_no_new_inversions'] is True


def test_conversion_exact_alias_checks_and_admission():
    operator, _ = generated((4, 4, 4), batch=1)
    shared = operator.upper[0]
    operator.upper[1] = shared
    old = legacy.BatchedFactor(operator.diagonal, operator.upper, operator.lower)
    with pytest.raises(ValueError, match='point-level'):
        GOnlyFactor.from_existing_factor(old)
    factor = GOnlyFactor.from_existing_factor(old, source_point_qualified=True,
                                              upper_aliases=[shared, shared])
    assert factor.raw_upper[0] is factor.raw_upper[1]
    wrong = shared.copy()
    wrong[0, 0, 0] += 1e-15
    with pytest.raises(ValueError, match='exact operator'):
        GOnlyFactor.from_existing_factor(old, source_point_qualified=True,
                                         upper_aliases=[shared, wrong])
    old.compact()
    with pytest.raises(ValueError, match='un-compacted'):
        GOnlyFactor.from_existing_factor(old, source_point_qualified=True)


def test_conversion_rejects_signed_zero_and_dtype_alias_changes():
    diagonal = [(3+1j)*np.eye(2, dtype=np.complex128)[None]]*2
    raw = np.zeros((1, 2, 2), dtype=np.complex128)
    old = legacy.BatchedFactor(diagonal, [raw], [raw])
    signed_zero = raw.copy()
    signed_zero.real[0, 0, 0] = -0.
    assert np.array_equal(old.a.upper[0], signed_zero)
    assert old.a.upper[0].tobytes() != signed_zero.tobytes()
    with pytest.raises(ValueError, match='exact operator'):
        GOnlyFactor.from_existing_factor(old, source_point_qualified=True,
                                         upper_aliases=[signed_zero])
    same_values_other_dtype = raw.astype(np.complex64)
    assert np.array_equal(old.a.upper[0], same_values_other_dtype)
    with pytest.raises(ValueError, match='exact operator'):
        GOnlyFactor.from_existing_factor(old, source_point_qualified=True,
                                         upper_aliases=[same_values_other_dtype])


@pytest.mark.parametrize('kind', ['count', 'shape', 'nonfinite', 'nonhermitian', 'adjoint'])
def test_bad_source_rejected_before_counters_change(kind):
    operator, source = generated()
    factor = GOnlyFactor(operator.diagonal, operator.upper, operator.lower)
    if kind == 'count':
        source.diagonal.pop()
    elif kind == 'shape':
        source.diagonal[0] = source.diagonal[0][:, :1, :1]
    elif kind == 'nonfinite':
        source.diagonal[0][0, 0, 0] = np.nan
    elif kind == 'nonhermitian':
        source.diagonal[0][0, 0, 0] += 1j
    else:
        source.upper[0][0, 0, 0] += 1
    before = factor.stats()
    with pytest.raises(ValueError):
        factor.correlate(source)
    assert factor.stats() == before


@pytest.mark.parametrize('row,column', [(-1, 0), (0, 3), (0, 2), (True, 0), (0, .5)])
def test_bad_selected_indices(row, column):
    operator, _ = generated()
    factor = GOnlyFactor(operator.diagonal, operator.upper, operator.lower)
    with pytest.raises(IndexError):
        factor.selected_block(row, column)


def test_product_counter_matches_explicit_actions_and_linear_length_count():
    operator, source = generated((3, 4, 2), batch=2)
    factor = GOnlyFactor(operator.diagonal, operator.upper, operator.lower)
    factor.correlate(source, qualification=True)
    factor.correlate(source)
    assert factor.stats()['source_calls'] == 2
    assert factor.stats()['qualification_source_calls'] == 1
    assert factor.stats()['source_block_products_per_call'] == 2*(21*3-19)
    assert factor.stats()['source_block_products_recorded_total'] == 4*(21*3-19)
