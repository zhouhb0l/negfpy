"""Exact g-only adapter preserving the historical silicon point guards.

The unchanged historical BatchedFrozenPoint performs all raw qualification.
Its original compact-cache optical/uniform tests also run, before replacement.
A further exact-source check qualifies the new representation at every tile.
No SCBA map, Born contraction, fitting or spectral repair is implemented here.
"""
from __future__ import annotations
import numpy as np

from .g_only_cache import GOnlyFactor
from . import batched_spectral, spectral_backend


def _bitwise_equal(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and left.tobytes(order='C') == right.tobytes(order='C'))


def exact_coupling_aliases(point):
    """Prove shared raw couplings from exact translated integer stencils.

    Each signature lists the ordered crossing probe scatters in the original
    construction.  Harmonic cross-group blocks have one fixed 24x24 insertion;
    contact supports do not cross.  Each distinct stencil is rebuilt using the
    exact original operation order, then EVERY actual block is checked by
    shape, dtype and C-order bytes.  This is not approximate periodicity or a
    search for numerically equal matrices.
    """
    error = spectral_backend.FrozenPointError
    raw = point.factor.a
    if raw is None or point.factor.compacted:
        raise error('Coupling proof requires the original full raw operator')
    crossings = [[] for _ in raw.upper]
    for terminal, bath in enumerate(point.baths):
        if len(bath.parts) == 1:
            continue
        if terminal < 2:
            raise error('A contact crosses grouped blocks unexpectedly')
        if len(bath.parts) != 2 or bath.parts[1].block != bath.parts[0].block+1:
            raise error('Probe support exceeds original grouped adjacency')
        left, right = bath.parts
        crossings[left.block].append(tuple(tuple(int(v) for v in array) for array in
                                          (left.local, left.source, right.local, right.source)))
    families = {}
    upper_aliases, lower_aliases = [], []
    for i, ordered in enumerate(crossings):
        signature = (point.block_sizes[i], point.block_sizes[i+1], tuple(ordered))
        if signature not in families:
            a, b = signature[:2]
            if a != 48 or b not in (24, 48):
                raise error('Unrecognized original-cell coupling geometry')
            upper = np.zeros((point.batch, a, b), complex)
            lower = np.zeros((point.batch, b, a), complex)
            upper[:, 24:48, :24] = -point.harmonic[1]
            lower[:, :24, 24:48] = -point.harmonic[2]
            for left_local, left_source, right_local, right_source in ordered:
                ll, ls, rl, rs = [np.asarray(values, dtype=np.int64) for values in
                                  (left_local, left_source, right_local, right_source)]
                # Same assignment/addition order as the original _scatter.
                upper[:, ll[:, None], rl] += -1.*point.sigma_short[:, ls[:, None], rs]
                lower[:, rl[:, None], ll] += -1.*point.sigma_short[:, rs[:, None], ls]
            upper.setflags(write=False)
            lower.setflags(write=False)
            families[signature] = (upper, lower)
        upper, lower = families[signature]
        if not _bitwise_equal(raw.upper[i], upper) or not _bitwise_equal(raw.lower[i], lower):
            raise error('Exact repeated-coupling construction proof failed; no merging permitted')
        upper_aliases.append(upper)
        lower_aliases.append(lower)
    proof = dict(passed=True, interfaces=len(crossings), stencil_families=len(families),
                 unique_raw_coupling_arrays=2*len(families), all_interfaces_bitwise_checked=True,
                 approximate_equality_used=False, source='ordered translated integer support stencils',
                 shared_family_numeric_bytes=sum(u.nbytes+l.nbytes for u, l in families.values()))
    if len(families) > 4:
        raise error('Exact coupling stencil family count exceeds admitted resource envelope')
    return upper_aliases, lower_aliases, proof


class GOnlyFrozenPoint:
    def __init__(self, *, harmonic, sigma_short, active_dofs, contacts,
                 center_layers, frequencies, rtol=1e-8, atol=1e-11):
        legacy = batched_spectral
        self._point = legacy.BatchedFrozenPoint(
            harmonic=harmonic, sigma_short=sigma_short, active_dofs=active_dofs,
            contacts=contacts, center_layers=center_layers, frequencies=frequencies,
            rtol=rtol, atol=atol, compact=False)
        point = self._point
        # Original constructor has already admitted raw Sigma, harmonic and
        # inverse/source/optical/uniform guards at the original tolerances.
        upper, lower, self.coupling_proof = exact_coupling_aliases(point)
        factor = GOnlyFactor.from_existing_factor(
            point.factor, source_point_qualified=True, upper_aliases=upper, lower_aliases=lower)
        original_spectral = point.spectral
        point.compact()  # ORIGINAL additional compact-source guards, unchanged.
        self.original_qualified_factor = point.factor.stats()
        if (self.original_qualified_factor['full_source_calls'] != 2
                or self.original_qualified_factor['compact_qualification_source_calls'] != 1):
            raise ValueError('Original startup work/qualification is incomplete')
        source = point._sources(np.ones((point.batch, point.nterminals)))
        propagated = factor.correlate(source, qualification=True)
        spectral_blocks = original_spectral.diagonal+original_spectral.upper+original_spectral.lower
        propagated_blocks = propagated.diagonal+propagated.upper+propagated.lower
        scale = np.maximum(legacy._per_frequency_max(spectral_blocks, point.batch),
                           legacy._per_frequency_max(propagated_blocks, point.batch))
        error = legacy._per_frequency_max([a-b for a, b in zip(spectral_blocks, propagated_blocks)], point.batch)
        allowance = point.atol+point.rtol*scale
        if np.any(error > allowance):
            raise ValueError('G-only full-source optical identity failed; no repair')
        uniform = np.stack([legacy._trace(propagated, bath) for bath in point.baths], axis=1)
        point._real_guard(uniform, 'g-only uniform full-source contractions')
        uerror = np.max(np.abs(point.spectral_traces-uniform.real), axis=1)
        uallow = point.atol+point.rtol*np.maximum(np.max(np.abs(point.spectral_traces), axis=1),
                                               np.max(np.abs(uniform.real), axis=1))
        if np.any(uerror > uallow):
            raise ValueError('G-only uniform-temperature raw current identity failed')
        added = dict(
            g_only_optical_max_abs=error,
            g_only_optical_guard_ratio=np.divide(error, allowance, out=np.zeros_like(error), where=allowance > 0),
            g_only_uniform_current_max_abs=uerror,
            g_only_uniform_current_guard_ratio=np.divide(uerror, uallow, out=np.zeros_like(uerror), where=uallow > 0))
        point.frequency_diagnostics.update(added)
        point.diagnostics.update({name: float(np.max(value)) for name, value in added.items()})
        point.factor = factor
        point.diagnostics['g_only_additional_startup_source_correlations'] = 1
        self.original_qualification_passed = True

    def __getattr__(self, name):
        return getattr(self._point, name)

    def currents(self, energy_weights):
        return self._point.currents(energy_weights)

    def stats(self):
        value = self._point.stats()
        value.update(g_only=True, exact_coupling_proof=dict(self.coupling_proof),
                     original_point_qualification_passed=self.original_qualification_passed,
                     original_qualified_factor=dict(self.original_qualified_factor),
                     startup_original_source_correlations=3,
                     startup_g_only_source_correlations=1,
                     total_source_correlations=self.original_qualified_factor['source_calls']+self.factor.source_calls)
        return value
