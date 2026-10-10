"""Buffer mapping and independent small-system embedding checks."""
import json
from dataclasses import replace

import numpy as np
import pytest
from test_anharmonic_cubic_cut_current import _problem as cubic_problem
from test_anharmonic_cubic_cut_current import _tensor
from test_anharmonic_harmonic_pointwise import _problem
from test_anharmonic_interacting_pointwise import _sigma

from negfpy.anharmonic import (
    FixedFieldGreenEvaluator,
    KeldyshSpectrum,
    SCBAConfig,
    SignedFrequencyGrid,
    SparseCubicVertex,
    add_harmonic_lead_buffers,
    build_cubic_cut_current_operator,
)
from negfpy.anharmonic._harmonic import assemble_device_dynamical_matrix
from negfpy.anharmonic.cubic import _contract_cubic_pair
from negfpy.core.types import LeadBlocks


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('counts', [(1, 1), (2, 0), (0, 2), (2, 3)])
def test_exact_interaction_and_field_embedding(sparse, counts):
    original = cubic_problem(_tensor(6, [((0, 2, 4), .07), ((1, 1, 3), -.03)]),
                             npl=2, sparse_vertex=sparse)
    embedded = add_harmonic_lead_buffers(original, left_layers=counts[0], right_layers=counts[1])
    new, ids = embedded.problem, embedded.original_indices
    shift = counts[0] * 2
    assert new.ndof == original.ndof + sum(counts)*2
    np.testing.assert_array_equal(embedded.original_cuts, np.arange(1,3)+counts[0])
    if sparse:
        assert isinstance(new.vertex, SparseCubicVertex)
        np.testing.assert_array_equal(new.vertex.indices, original.vertex.indices+shift)
        np.testing.assert_array_equal(new.vertex.coefficients, original.vertex.coefficients)
        before, after = original.vertex.to_dense(max_bytes=2**20).values, new.vertex.to_dense(max_bytes=2**20).values
    else:
        before, after = original.vertex.values, new.vertex.values
    np.testing.assert_array_equal(after[np.ix_(ids,ids,ids)], before)
    assert np.count_nonzero(after) == np.count_nonzero(before)
    rng = np.random.default_rng(5)
    fields = {n:rng.normal(size=(original.grid.n_points,original.ndof,original.ndof)).astype(complex)
              for n in ('retarded','lesser','greater')}
    sigma = KeldyshSpectrum(**fields)
    padded = embedded.embed_self_energy(sigma)
    for name, expected in fields.items():
        actual = getattr(padded, name)
        np.testing.assert_array_equal(actual[:,ids[:,None],ids], expected)
        assert np.count_nonzero(actual) == np.count_nonzero(expected)
        assert not actual.flags.writeable
        np.testing.assert_array_equal(getattr(sigma,name),expected)
    # Arbitrary new Green entries cannot invent interactions outside the crop.
    g1=rng.normal(size=(new.ndof,new.ndof))+1j*rng.normal(size=(new.ndof,new.ndof))
    g2=rng.normal(size=g1.shape)+1j*rng.normal(size=g1.shape)
    expected=_contract_cubic_pair(before,g1[np.ix_(ids,ids)],g2[np.ix_(ids,ids)])
    actual=_contract_cubic_pair(after,g1,g2)
    np.testing.assert_allclose(actual[np.ix_(ids,ids)],expected,rtol=1e-13,atol=1e-16)
    actual[np.ix_(ids,ids)]=0
    assert not np.any(actual)
    for old_cut,new_cut in zip(range(1,3),embedded.original_cuts):
        old=build_cubic_cut_current_operator(original,old_cut)
        shifted=build_cubic_cut_current_operator(new,int(new_cut))
        np.testing.assert_array_equal(shifted.cubic_indices,old.cubic_indices+shift)
        np.testing.assert_array_equal(shifted.cubic_coefficients,old.cubic_coefficients)
    assert not embedded.metadata['buffer_convergence_verified']
    json.dumps(embedded.metadata, allow_nan=False)


def test_complex_reordered_contacts_and_different_leads():
    original=_problem()
    right=replace(original.right, lead=LeadBlocks(original.right.lead.d00+.1*np.eye(2),
                                                 original.right.lead.d01*.9))
    original=replace(original,right=right)
    embedded=add_harmonic_lead_buffers(original,left_layers=2,right_layers=1)
    new=embedded.problem
    full=assemble_device_dynamical_matrix(new.device);ids=embedded.original_indices
    np.testing.assert_array_equal(full[np.ix_(ids,ids)],assemble_device_dynamical_matrix(original.device))
    c_left=np.zeros((original.ndof,2),complex);c_right=c_left.copy()
    c_left[[1,0]]=original.left.device_to_lead
    c_right[[3,2]]=original.right.device_to_lead
    np.testing.assert_array_equal(full[np.ix_(ids,[2,3])],c_left)
    np.testing.assert_array_equal(full[np.ix_(ids,[8,9])],c_right)
    np.testing.assert_array_equal(new.device.coupling_blocks[0],original.left.lead.d01)
    np.testing.assert_array_equal(new.right.lead.d00,original.right.lead.d00)
    assert new.left.temperature==original.left.temperature
    assert new.right.temperature==original.right.temperature
    assert not np.shares_memory(new.device.onsite_blocks[2],original.device.onsite_blocks[0])


def test_eliminated_buffer_matches_independent_contact_recursion():
    # Non-symmetric leads, complex interface, unequal reservoirs and nonzero
    # fixed pp fields. The reference eliminates each new layer analytically.
    from negfpy.core.surface_gf import surface_gf
    original=replace(_problem(temperatures=(.2,1.1)),grid=SignedFrequencyGrid(.1,30))
    sigma=_sigma(original.grid,original.ndof)
    embedded=add_harmonic_lead_buffers(original,left_layers=1,right_layers=1)
    config=SCBAConfig(eta_lead=.025)
    evaluator=FixedFieldGreenEvaluator(embedded.problem,config,embedded.embed_self_energy(sigma))
    omega=1.7
    pp=FixedFieldGreenEvaluator(original,config,sigma).interpolator.evaluate(omega)
    sr=np.zeros((original.ndof,original.ndof),complex);sl=sr.copy();sg=sr.copy()
    for side in ('left','right'):
        res=getattr(original,side);lead=res.lead
        forward=lead.d01.conj().T if side=='left' else lead.d01
        g=surface_gf(omega,lead.d00,forward,forward.conj().T,eta=config.eta_lead)
        outer=forward@g@forward.conj().T
        gr=np.linalg.inv(omega**2*np.eye(2)-lead.d00-outer)
        gamma=1j*(outer-outer.conj().T)
        n=1. / np.expm1(omega/res.temperature)
        coupling=res.device_to_lead
        ids=res.resolved_device_indices(dim=original.ndof,side=side)
        sr[np.ix_(ids,ids)] += coupling@gr@coupling.conj().T
        sl[np.ix_(ids,ids)] += coupling@gr@(-1j*n*gamma)@gr.conj().T@coupling.conj().T
        sg[np.ix_(ids,ids)] += coupling@gr@(-1j*(n+1)*gamma)@gr.conj().T@coupling.conj().T
    gr=np.linalg.inv(omega**2*np.eye(original.ndof)-assemble_device_dynamical_matrix(original.device)-sr-pp.retarded)
    actual=evaluator.evaluate(omega);ids=embedded.original_indices
    for name,expected in [('retarded',gr),('lesser',gr@(sl+pp.lesser)@gr.conj().T),('greater',gr@(sg+pp.greater)@gr.conj().T)]:
        np.testing.assert_allclose(getattr(actual,name)[np.ix_(ids,ids)],expected,rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('counts', [(True,1),(-1,1),(0,0),(1.5,1)])
def test_invalid_counts(counts):
    with pytest.raises((TypeError,ValueError)):
        add_harmonic_lead_buffers(_problem(),left_layers=counts[0],right_layers=counts[1])


def test_nonboundary_contact_and_nonhermitian_lead_rejected():
    p=_problem();p=replace(p,left=replace(p.left,device_indices=[0,2]))
    with pytest.raises(ValueError,match='boundary layer'):
        add_harmonic_lead_buffers(p,left_layers=1,right_layers=0)
    p=_problem();lead=replace(p.left.lead,d10=p.left.lead.d01)
    with pytest.raises(ValueError,match='forward adjoint'):
        add_harmonic_lead_buffers(replace(p,left=replace(p.left,lead=lead)),left_layers=1,right_layers=0)


def test_byte_limits_and_wrong_grid_fail_before_expansion():
    p=_problem()
    with pytest.raises(MemoryError,match='Dense padded vertex'):
        add_harmonic_lead_buffers(p,left_layers=1,right_layers=1,max_dense_vertex_bytes=1)
    e=add_harmonic_lead_buffers(p,left_layers=1,right_layers=1)
    with pytest.raises(MemoryError,match='self-energy arrays'):
        e.embed_self_energy(KeldyshSpectrum.zeros(n_frequency=p.grid.n_points,ndof=p.ndof),max_output_bytes=1)
    with pytest.raises(ValueError,match='shape differs'):
        e.embed_self_energy(KeldyshSpectrum.zeros(n_frequency=3,ndof=p.ndof))


def test_finite_eta_buffer_difference_decreases_with_eta():
    # This checks the regulator effect, not an SCBA convergence claim.
    original = _problem(acoustic=True, temperatures=(.2, 1.1))
    sigma = _sigma(original.grid, original.ndof, complex_matrix=False)
    embedded = add_harmonic_lead_buffers(original, left_layers=1, right_layers=1)
    padded = embedded.embed_self_energy(sigma)
    errors = []
    for eta in (.025, .0125, .00625):
        config = SCBAConfig(eta_lead=eta)
        before = FixedFieldGreenEvaluator(original, config, sigma).evaluate(.73)
        after = FixedFieldGreenEvaluator(embedded.problem, config, padded).evaluate(.73)
        ids = embedded.original_indices
        errors.append(np.linalg.norm(after.retarded[np.ix_(ids, ids)] - before.retarded))
    assert errors[0] > 1e-3
    assert 0 < errors[2] < .6 * errors[1] < .36 * errors[0]
    assert not embedded.metadata['buffer_convergence_verified']
