"""Exact reference/increment cubic Born arithmetic.

This module reuses the packed contraction primitive, q routing, finite
frequency grid, and normalization. No material input, field symmetry, physical guard, or
convergence threshold is altered here.

For independent statistical/spectral channels define the ordered bilinear map
``B(x,y).C = K(CC + DD/4)`` and ``B(x,y).D = K(CD + DC)``. Only the *second
vertex* is conjugated inside K; neither Green factor is conjugated. Expansion
and analytic directional derivatives are evaluated by cross contractions,
never by subtracting two large quadratic Born results.

The reference subtraction alone leaves the fixed-point eigenvalues unchanged.
Any feedback correction using these derivatives must still be evaluated and
accepted with the original SCBA equations and guards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.transverse_channels import StatisticalSpectral
from negfpy.anharmonic.transverse_packed import (
    PackedTransverseCubicVertex,
    _array,
    _contract_into,
    _finite,
    _integer,
    _packed,
    plan_packed_born_fft_dimensions,
)
from negfpy.anharmonic.transverse_q import difference_q_indices

_DEFAULT_BYTES = 64 * 1024**2
_DEFAULT_WORK = 100_000_000
_OPERATIONS = {
    "bilinear": ((0, 1),),
    "tangent": ((0, 1), (1, 0)),
    "polarized_increment": ((0, 1), (1, 0)),
    "increment": ((0, 1), (1, 0), (1, 1)),
    "anchored": ((0, 0), (0, 1), (1, 0), (1, 1)),
}


@dataclass(frozen=True, slots=True)
class BornExpansionPlan:
    """Numeric payload/work bound, excluding caller data and library scratch.

    Four coexisting input FFTs are included even for equal input objects; the
    executor does not rely on aliasing. All ordered pairs accumulate into one
    q-sized transform per output. Output C/D coexist. ``contraction_work`` uses
    the canonical complex multiply-accumulate count, not measured FLOPs/time.
    Caller-owned reference, increment, optional cached reference Born, vertex,
    FFT/BLAS internal scratch and Python overhead are separate; this is not an
    RSS guarantee. Cached reference Born must belong to the exact same grid,
    vertex and reference; its scientific binding is the caller's responsibility.
    """

    operation: str
    cached_reference: bool
    n_pairs: int
    n_q: int
    n_w: int
    n_fft: int
    n_active: int
    external_tile: int
    workers: int
    bin_tile: int
    effective_workers: int
    input_channel_bytes: int
    output_bytes: int
    transformed_channel_bytes: int
    q_transform_bytes: int
    vertex_bytes: int
    worker_workspace_bytes: int
    peak_owned_bytes: int
    contraction_work: int


def plan_born_expansion_dimensions(
    n_q: int, n_w: int, n_active: int, *, operation: str = "bilinear",
    cached_reference: bool = False, external_tile: int = 16,
    workers: int = 8, bin_tile: int = 1,
) -> BornExpansionPlan:
    """Plan without loading numerical fields; cached reference applies to anchored.

    One bilinear costs four canonical terms; a tangent costs eight, an exact
    expanded increment twelve, polarized increment eight, and an uncached
    complete anchored map sixteen. Caching B(reference,reference) reduces the
    expanded anchored work to twelve terms. Polarization forms its midpoint in
    FFT storage, so it needs no additional full frequency-field allocation.
    """
    if operation not in _OPERATIONS:
        raise ValueError(f"operation must be one of {tuple(_OPERATIONS)}.")
    if not isinstance(cached_reference, bool):
        raise TypeError("cached_reference must be a bool.")
    if cached_reference and operation != "anchored":
        raise ValueError("cached_reference is only valid for anchored arithmetic.")
    base = plan_packed_born_fft_dimensions(
        n_q, n_w, n_active, external_tile=external_tile,
        workers=workers, bin_tile=bin_tile,
    )
    n_pairs = len(_OPERATIONS[operation]) - int(cached_reference)
    return BornExpansionPlan(
        operation, cached_reference, n_pairs,
        base.n_q, base.n_w, base.n_fft, base.n_active,
        base.external_tile, base.workers, base.bin_tile, base.effective_workers,
        2 * base.input_green_bytes, base.output_bytes,
        4 * base.transformed_green_bytes, base.q_transform_bytes,
        base.vertex_bytes, base.worker_workspace_bytes,
        base.peak_owned_bytes + 3 * base.transformed_green_bytes,
        4 * n_pairs * base.per_component_contraction_work,
    )


def plan_born_expansion(
    grid: SignedFrequencyGrid, vertex: PackedTransverseCubicVertex, **kwargs,
) -> BornExpansionPlan:
    """Count a particular grid/vertex without reading fields or coefficients."""
    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    _packed(vertex)
    return plan_born_expansion_dimensions(
        vertex.mesh.n_qpoints, grid.n_points, vertex.ndof, **kwargs,
    )


def _channels(value: StatisticalSpectral, shape, name):
    if not isinstance(value, StatisticalSpectral):
        raise TypeError(f"{name} must be StatisticalSpectral.")
    return (
        _array(value.statistical, shape, f"{name}.statistical"),
        _array(value.spectral, shape, f"{name}.spectral"),
    )


def _evaluate(
    grid, vertex, first, second, *, operation, reference_born=None,
    external_tile=16, workers=8, bin_tile=1,
    max_bytes=_DEFAULT_BYTES, max_contraction_work=_DEFAULT_WORK,
):
    plan = plan_born_expansion(
        grid, vertex, operation=operation,
        cached_reference=reference_born is not None,
        external_tile=external_tile, workers=workers, bin_tile=bin_tile,
    )
    limit = _integer(max_bytes, "max_bytes")
    work_limit = _integer(max_contraction_work, "max_contraction_work")
    if plan.peak_owned_bytes > limit:
        raise ValueError("Born expansion peak owned payload exceeds max_bytes.")
    if plan.contraction_work > work_limit:
        raise ValueError("Born expansion exceeds max_contraction_work.")
    shape = (plan.n_q, plan.n_w, plan.n_active, plan.n_active)
    inputs = (_channels(first, shape, "first"), _channels(second, shape, "second"))
    constant = None if reference_born is None else _channels(reference_born, shape, "reference_born")
    for pair in inputs + (() if constant is None else (constant,)):
        for value in pair:
            _finite(value, "Born expansion input")
    outputs = tuple(
        np.zeros(shape, dtype=np.complex128) if constant is None
        else np.array(constant[index], dtype=np.complex128, copy=True)
        for index in range(2)
    )
    if not plan.n_active:
        return StatisticalSpectral(*outputs)
    routes = difference_q_indices(vertex.mesh, max_bytes=limit)
    pairs = _OPERATIONS[operation][int(reference_born is not None):]
    transformed = tuple(
        tuple(np.zeros((plan.n_q, plan.n_fft, plan.n_active, plan.n_active), dtype=np.complex128)
              for _ in range(2))
        for _ in range(2)
    )
    prefactor = 0.5j * grid.integration_weight / plan.n_q
    with threadpool_limits(limits=1, user_api="blas"):
        for source_pair, target_pair in zip(inputs, transformed, strict=True):
            for source, target in zip(source_pair, target_pair, strict=True):
                target[:, :plan.n_w] = source
                for q in range(plan.n_q):
                    target[q] = np.fft.fft(target[q], axis=0)
                _finite(target, "transformed Born expansion input")
        if operation == "polarized_increment":
            # B(h,r+h/2)+B(r+h/2,h) retains the quadratic h*h term
            # with two ordered bilinears. Form this midpoint in owned FFT
            # storage; no full reference+h field or large Born subtraction.
            midpoint_scratch = np.empty((plan.n_fft, plan.n_active, plan.n_active), dtype=np.complex128)
            for base_fft, step_fft in zip(transformed[0], transformed[1], strict=True):
                for q in range(plan.n_q):
                    np.multiply(step_fft[q], 0.5, out=midpoint_scratch)
                    np.add(base_fft[q], midpoint_scratch, out=base_fft[q])
                _finite(base_fft, "transformed Born midpoint")
            del midpoint_scratch
        for channel, output in enumerate(outputs):
            for q in range(plan.n_q):
                sigma = np.zeros((plan.n_fft, plan.n_active, plan.n_active), dtype=np.complex128)
                # DD first allows its common 1/4 weight without a second q buffer.
                channel_terms = ((1, 1), (0, 0)) if channel == 0 else ((0, 1), (1, 0))
                for term, (left_channel, right_channel) in enumerate(channel_terms):
                    for left_source, right_source in pairs:
                        left = transformed[left_source][left_channel]
                        right = transformed[right_source][right_channel]
                        for p in range(plan.n_q):
                            _contract_into(
                                vertex.coefficients[q, p], left[p], right[routes[q, p]],
                                sigma, plan.external_tile, plan.workers, plan.bin_tile,
                            )
                    if channel == 0 and term == 0:
                        sigma *= 0.25
                convolution = np.fft.ifft(sigma, axis=0)
                del sigma
                selected = convolution[grid.zero_index:grid.zero_index + plan.n_w]
                np.multiply(selected, prefactor, out=selected)
                np.add(output[q], selected, out=output[q])
                del selected, convolution
            _finite(output, "Born expansion output")
    return StatisticalSpectral(*outputs)


def born_bilinear(grid, vertex, first, second, **kwargs) -> StatisticalSpectral:
    """Ordered B(first,second), including exact q routes and four C/D terms."""
    return _evaluate(grid, vertex, first, second, operation="bilinear", **kwargs)


def tangent_born(grid, vertex, base, direction, **kwargs) -> StatisticalSpectral:
    """Analytic derivative d B(base,base)[direction] = B(base,h)+B(h,base).

    No finite differences or perturbed full Born maps are used in execution.
    This is a tangent through the Born functional only; the complete SCBA
    feedback derivative must also include the Dyson/Keldysh tangent upstream.
    """
    return _evaluate(grid, vertex, base, direction, operation="tangent", **kwargs)


def born_increment(grid, vertex, reference, increment, **kwargs) -> StatisticalSpectral:
    """Exact finite Born increment B(0,h)+B(h,0)+B(h,h), keeping quadratic h.

    Here 0 denotes the supplied reference, not a zero Green function. This
    avoids subtracting B(reference+h) and B(reference); it is not a linearization.
    """
    return _evaluate(grid, vertex, reference, increment, operation="increment", **kwargs)


def difference_born(grid, vertex, reference, increment, **kwargs) -> StatisticalSpectral:
    """Exact polarized finite increment, costing two rather than three bilinears.

    B(h,r+h/2)+B(r+h/2,h) = B(r,h)+B(h,r)+B(h,h). All finite
    increment terms are retained algebraically. The midpoint is formed in FFT
    storage, and no two large Born results are subtracted. As with all floating
    arithmetic, sub-ulp contributions can round during midpoint accumulation;
    this is not an arbitrary-precision guarantee.
    """
    return _evaluate(grid, vertex, reference, increment, operation="polarized_increment", **kwargs)


def anchored_born(
    grid, vertex, reference, increment, *, reference_born=None, **kwargs,
) -> StatisticalSpectral:
    """Exact B(reference+increment), with all four ordered expansion pieces.

    Optional reference_born is an immutable caller-cached B(reference,reference)
    for the same grid/vertex/reference. It is copied, never mutated. No cached
    mapped field from a different eta, coupling, grid, or Green input is valid.
    """
    return _evaluate(
        grid, vertex, reference, increment, operation="anchored",
        reference_born=reference_born, **kwargs,
    )
