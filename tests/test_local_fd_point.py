"""Generated full-support point and q/frequency-worker oracles."""
from types import SimpleNamespace
import time
import weakref

import numpy as np
import pytest

from negfpp.local_fd.g_only_point import GOnlyFrozenPoint, exact_coupling_aliases
from negfpp.local_fd.g_only_engine import GOnlyParallelCurrents, _weights, partition_frequency_tiles
from negfpp.local_fd import batched_spectral, probe_core


def generated(batch=2, seed=74):
    rng = np.random.default_rng(seed)
    def complex_matrix(size):
        return (rng.normal(size=(size, size))+1j*rng.normal(size=(size, size)))/np.sqrt(size)
    z = complex_matrix(24)
    d00 = 2.3*np.eye(24)+.04*(z+z.conj().T)
    d01 = .08*complex_matrix(24)
    contacts = []
    for _ in range(2):
        z, noise = complex_matrix(24), complex_matrix(24)
        gamma = .25*np.eye(24)+.1*noise@noise.conj().T
        contacts.append(.03*(z+z.conj().T)-.5j*gamma)
    z, noise = complex_matrix(36), complex_matrix(36)
    gamma = .12*np.eye(36)+.05*noise@noise.conj().T
    sigma = .02*(z+z.conj().T)-.5j*gamma
    active = np.r_[np.arange(18, 24), np.arange(24, 48), np.arange(48, 54)]
    active = active[rng.permutation(36)]
    factors = np.linspace(.7, 1.2, batch)
    return dict(harmonic=(d00, d01, d01.conj().T),
                sigma_short=factors[:, None, None]*sigma,
                active_dofs=active,
                contacts=tuple((1+.2*np.arange(batch))[:, None, None]*c for c in contacts),
                frequencies=np.linspace(1.2, 2.2, batch))



def dense_oracle(arguments, centers, index):
    size = 24*(centers+2)
    d00, d01, d10 = arguments['harmonic']
    operator = np.zeros((size, size), complex)
    for cell in range(centers+2):
        sl = slice(24*cell, 24*(cell+1))
        operator[sl, sl] = arguments['frequencies'][index]**2*np.eye(24)-d00
        if cell < centers+1:
            nxt = slice(24*(cell+1), 24*(cell+2))
            operator[sl, nxt], operator[nxt, sl] = -d01, -d10
    supports = [np.arange(24), np.arange(size-24, size)]
    supports += [arguments['active_dofs']+24*j for j in range(centers)]
    retarded = [c[index] for c in arguments['contacts']]+[arguments['sigma_short'][index]]*centers
    for support, sigma in zip(supports, retarded):
        operator[np.ix_(support, support)] -= sigma
    green = np.linalg.inv(operator)
    gammas = [1j*(s-s.conj().T) for s in retarded]
    transmission = probe_core.transmission_matrix(green, gammas, supports)
    return green, supports, transmission



@pytest.mark.parametrize('centers', [1, 2, 3])
def test_point_current_selected_blocks_and_original_guards_match(centers):
    arguments = generated()
    old = batched_spectral.BatchedFrozenPoint(**arguments, center_layers=centers, compact=True)
    new = GOnlyFrozenPoint(**arguments, center_layers=centers)
    rng = np.random.default_rng(centers)
    weights = rng.normal(size=(2, centers+2))
    np.testing.assert_allclose(new.currents(weights), old.currents(weights), rtol=2e-12, atol=4e-13)
    if centers <= 5:
        for f in range(2):
            green, supports, transmission = dense_oracle(arguments, centers, f)
            expected = transmission.sum(axis=1)*weights[f]-transmission@weights[f]
            np.testing.assert_allclose(new.currents(weights)[f], expected, rtol=3e-12, atol=4e-13)
            for terminal, support in enumerate(supports):
                np.testing.assert_allclose(new.selected_on_bath(terminal)[f],
                                           green[np.ix_(support, support)], atol=4e-14)
    np.testing.assert_allclose(new.currents(weights+100), old.currents(weights), atol=1e-11)
    np.testing.assert_array_equal(new.currents(np.ones_like(weights)), np.zeros_like(weights))
    stats = new.stats()
    for name, value in old.diagnostics.items():
        if isinstance(value, (int, float)):
            assert new.diagnostics[name] == value
    assert stats['original_qualified_factor']['full_source_calls'] == 2
    assert stats['original_qualified_factor']['compact_qualification_source_calls'] == 1
    assert stats['factor']['qualification_source_calls'] == 1
    assert stats['factor']['factor_block_solves'] == 0
    assert stats['factor']['inherited_factor_block_solves'] == 2*len(new.block_sizes)
    assert new.diagnostics['g_only_optical_guard_ratio'] <= 1
    assert new.diagnostics['g_only_uniform_current_guard_ratio'] <= 1
    assert stats['exact_coupling_proof']['all_interfaces_bitwise_checked'] is True
    assert new.green is new.spectral is None
    if centers >= 9:
        assert stats['factor']['retained_coupling_families'] < 2*(len(new.block_sizes)-1)
        assert stats['stored_numeric_bytes'] < old.stats()['stored_numeric_bytes']



def test_exact_stencil_proof_rejects_wrong_interface_even_at_roundoff_scale():
    arguments = generated(batch=1)
    old = batched_spectral.BatchedFrozenPoint(**arguments, center_layers=9)
    upper, lower, proof = exact_coupling_aliases(old)
    assert proof['stencil_families'] < len(upper)
    assert len({id(a) for a in upper}) == proof['stencil_families']
    old.factor.a.upper[1][0, 0, 0] += 1e-30
    with pytest.raises(ValueError, match='construction proof failed'):
        exact_coupling_aliases(old)



def test_one_bad_frequency_passivity_and_new_qualification_failure_are_not_hidden(monkeypatch):
    arguments = generated()
    arguments['sigma_short'][1] = .1j*np.eye(36)
    with pytest.raises(ValueError, match='not passive'):
        GOnlyFrozenPoint(**arguments, center_layers=1)
    from negfpp.local_fd import g_only_point
    original = g_only_point.GOnlyFactor.correlate
    def corrupt(factor, source, **kwargs):
        result = original(factor, source, **kwargs)
        result.diagonal[0] = result.diagonal[0]+.01
        return result
    monkeypatch.setattr(g_only_point.GOnlyFactor, 'correlate', corrupt)
    with pytest.raises(ValueError, match='G-only full-source optical'):
        GOnlyFrozenPoint(**generated(), center_layers=1)



def test_conversion_releases_original_factor_and_redundant_arrays(monkeypatch):
    module = batched_spectral
    old_constructor = module.BatchedFrozenPoint
    references = []
    def capture(*args, **kwargs):
        point = old_constructor(*args, **kwargs)
        references.append(weakref.ref(point.factor))
        references.extend(weakref.ref(a) for a in point.factor.a.diagonal+point.green.upper+point.green.lower)
        return point
    monkeypatch.setattr(module, 'BatchedFrozenPoint', capture)
    point = GOnlyFrozenPoint(**generated(), center_layers=5)
    assert references and all(reference() is None for reference in references)
    assert point.stats()['g_only'] is True



def inputs(nq=3, batch=2):
    base = generated(batch=batch, seed=311)
    onsite, forward, reverse, spectra, contacts = [], [], [], [], [[], []]
    for q in range(nq):
        onsite.append(base['harmonic'][0]+q*.001*np.eye(24))
        forward.append(base['harmonic'][1]*(1+q*.002))
        reverse.append(forward[-1].conj().T)
        spectra.append(base['sigma_short']*(1+.01*q))
        for side in range(2):
            contacts[side].append(base['contacts'][side]*(1+.005*q))
    return SimpleNamespace(
        harmonic=SimpleNamespace(onsite=np.stack(onsite), forward=np.stack(forward), source_reverse=np.stack(reverse)),
        kernel=SimpleNamespace(retarded=np.stack(spectra), active_dofs=base['active_dofs']),
        frequencies=base['frequencies'], weights=np.linspace(.019, .021, batch),
        grid=SimpleNamespace(positive_indices=np.arange(batch)),
        contact_retarded=tuple(np.stack(a) for a in contacts),
        scale=SimpleNamespace(omega0=62831853071795.86, hbar=1.0545718176461565e-34))



def expected_integrated(data, centers, temperatures, direction=None, harmonic=False):
    core = probe_core
    transmissions = []
    for f in range(len(data.frequencies)):
        average = np.zeros((centers+2, centers+2))
        for q in range(len(data.kernel.retarded)):
            arguments = dict(harmonic=(data.harmonic.onsite[q], data.harmonic.forward[q],
                                      data.harmonic.forward[q].conj().T),
                             sigma_short=(np.zeros_like(data.kernel.retarded[q]) if harmonic else data.kernel.retarded[q]),
                             active_dofs=data.kernel.active_dofs, contacts=tuple(a[q] for a in data.contact_retarded),
                             frequencies=data.frequencies)
            _, _, transmission = dense_oracle(arguments, centers, f)
            average += transmission/len(data.kernel.retarded)
        transmissions.append(average)
    transmissions = np.stack(transmissions)
    if direction is None:
        return core.terminal_currents(data.frequencies, data.weights, transmissions, temperatures,
                                     omega0=data.scale.omega0, hbar=data.scale.hbar)
    return core.current_temperature_jacobian(data.frequencies, data.weights, transmissions, temperatures,
                                             omega0=data.scale.omega0, hbar=data.scale.hbar)@direction



@pytest.mark.parametrize('nq,workers', [(1, 1), (1, 2), (3, 2)])
def test_mesh_generic_engine_currents_jv_harmonic_and_cleanup(nq, workers):
    data, centers = inputs(nq=nq), 2
    temperature = np.array([300.5, 299.5, 300.1, 299.9])
    direction = np.array([.13, -.27, .38, -.21])
    expected = expected_integrated(data, centers, temperature)
    expected_jv = expected_integrated(data, centers, temperature, direction)
    expected_harmonic = expected_integrated(data, centers, np.array([300.5, 299.5, 300., 300.]), harmonic=True)
    with GOnlyParallelCurrents(data, centers, deadline=time.monotonic()+60, tile_size=1,
                              max_workers=workers, max_cache_bytes=64*1024**2) as engine:
        processes = list(engine.processes)
        np.testing.assert_allclose(engine.currents(temperature), expected, rtol=5e-10, atol=3e-22)
        tangent = engine.jacobian_action(temperature, direction)
        np.testing.assert_allclose(tangent, expected_jv, rtol=5e-10, atol=3e-22)
        np.testing.assert_allclose(engine.harmonic_current, expected_harmonic, rtol=5e-10, atol=3e-22)
        step = .002
        finite_difference = (engine.currents(temperature+step*direction)-engine.currents(temperature-step*direction))/(2*step)
        np.testing.assert_allclose(tangent, finite_difference, rtol=2e-8, atol=2e-21)
        result = engine.stats()
        assert result['workers_count'] == workers
        coverage = [(q, f) for group in result['frequency_task_groups']
                    for q, start, stop in group for f in range(start, stop)]
        assert sorted(coverage) == [(q, f) for q in range(nq) for f in range(len(data.frequencies))]
        assert result['callbacks'] == dict(currents=3, jacobian=1)
        for worker in result['workers']:
            assert worker['calls'] == 4
            for record in worker['q_records']:
                for tile in record['tiles']:
                    assert tile['factor']['source_calls'] == 5
                    assert tile['original_qualified_factor']['source_calls'] == 3
                    assert tile['total_source_correlations'] == 8
    assert not any(p.is_alive() for p in processes)



@pytest.mark.parametrize('failure', ['passivity', 'cache'])
def test_worker_startup_failure_closes_every_launched_process(monkeypatch, failure):
    data = inputs(nq=3)
    if failure == 'passivity':
        data.kernel.retarded[1, 0] = .2j*np.eye(36)
    launched = []
    original = GOnlyParallelCurrents.close
    def close(engine):
        launched.extend(engine.processes)
        return original(engine)
    monkeypatch.setattr(GOnlyParallelCurrents, 'close', close)
    with pytest.raises(RuntimeError, match='not passive|payload cap exceeded'):
        GOnlyParallelCurrents(data, 1, deadline=time.monotonic()+60, tile_size=1,
                             max_workers=2, max_cache_bytes=1 if failure == 'cache' else 64*1024**2)
    assert len(launched) == 2
    assert not any(p.is_alive() for p in launched)



def test_bose_weight_functions_are_exact_original_functions():
    core = probe_core
    omega, temperatures = np.array([.000001, .025, 3.])[:, None], np.array([200., 300., 700.])[None, :]
    for derivative, function in [(False, core.bose_energy_weight), (True, core.bose_energy_temperature_derivative)]:
        np.testing.assert_array_equal(_weights(omega, temperatures, 62831853071795.86, 1.0545718176461565e-34,
                                               derivative=derivative),
                                      function(omega, temperatures, omega0=62831853071795.86))


def test_gamma_frequency_workers_match_serial_and_dense_with_configured_temperatures():
    data, centers = inputs(nq=1, batch=4), 2
    temperature = np.array([320., 280., 307., 294.])
    direction = np.array([.11, -.23, .35, -.17])
    results = []
    for workers in (1, 2):
        with GOnlyParallelCurrents(
            data, centers, deadline=time.monotonic()+60, tile_size=1,
            max_workers=workers, max_cache_bytes=64*1024**2,
            left_temperature=320., right_temperature=280.,
            equilibrium_temperature=295.,
        ) as engine:
            results.append((engine.currents(temperature),
                            engine.jacobian_action(temperature, direction),
                            engine.harmonic_current.copy()))
            stats = engine.stats()
            assert stats['workers_count'] == workers
            assert stats['startup_temperatures_kelvin'] == [320., 280., 295.]
            coverage = [(q, f) for group in stats['frequency_task_groups']
                        for q, start, stop in group for f in range(start, stop)]
            assert sorted(coverage) == [(0, f) for f in range(4)]
            assert sum(w['work_counts']['interacting_tiles_completed']
                       for w in stats['workers']) == 4
    for serial, parallel in zip(*results):
        np.testing.assert_allclose(parallel, serial, rtol=2e-12, atol=2e-23)
    np.testing.assert_allclose(results[1][0], expected_integrated(data, centers, temperature),
                               rtol=5e-10, atol=3e-22)
    np.testing.assert_allclose(results[1][1], expected_integrated(data, centers, temperature, direction),
                               rtol=5e-10, atol=3e-22)
    np.testing.assert_allclose(results[1][2], expected_integrated(
        data, centers, np.array([320., 280., 295., 295.]), harmonic=True),
        rtol=5e-10, atol=3e-22)


def test_partition_covers_uneven_tiles_and_caps_worker_count():
    groups = partition_frequency_tiles(2, 5, 2, 20)
    assert len(groups) == 6
    assert all(len(group) == 1 for group in groups)
    coverage = [(q, f) for group in groups
                for q, start, stop in group for f in range(start, stop)]
    assert sorted(coverage) == [(q, f) for q in range(2) for f in range(5)]
    with pytest.raises(ValueError, match='positive integers'):
        partition_frequency_tiles(1, 4, 1, True)
