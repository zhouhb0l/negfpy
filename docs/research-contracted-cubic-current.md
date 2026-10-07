# Direct connected cubic-current contraction

`build_contracted_cubic_cut` replaces the slow loop over individual
three-point moments when only one cut current is needed. It uses the same
Hamiltonian cut operator, the same one-vertex approximation and the same
finite-lag trapezoids as the existing implementation. It does not substitute
the layer EOM diagnostic or introduce a new physical closure.

## Exact rearrangement

Let the compiled cubic cut operator have terms `w_t W(p_i u_j u_k)`.
For each collected potential monomial `h_m*u_a*u_b*u_c`, add `h_m` to all
six permutations of the active response tensor `V`, including repeated
permutations when indices coincide. `V` is the polynomial's third derivative.
For exactly symmetric input it equals `Psi`. Small differences among ordered
input coefficients enter their actual polynomial sum; no stored vertex is
modified or replaced by a representative permutation.

The connected contribution is

\[
J_{\rm conn}=2\operatorname{Im}\int d\tau\sum_t w_t
\sum_{abc}P_{i_ta}(\tau)V_{abc}C_{j_tb}(\tau)C_{k_tc}(\tau).
\]

There is no additional `1/6` and no complex conjugation. The response includes
all active vertex coordinates, including interactions entirely on one side of
the cut. Only the external operator rows are restricted to those needed by
this cut. Dropping internal response vertices would change the result.

At each lag, two matrix products form

\[
T_{abk}=\sum_cV_{abc}C_{kc},\qquad
X_{ajk}=\sum_bC_{jb}T_{abk}.
\]

The implementation gathers the required `X[:,j_t,k_t]`, multiplies by
`P[i_t,:]`, and contracts the cut coefficients. Gathers are tiled in groups
of 256 operator terms. It never forms a full `M_ijk` tensor or performs a
separate vertex sum for every requested moment. With A active coordinates,
J displacement rows and N cut terms, the principal work per lag is
`O(A^3*J + A^2*J^2 + A*N)`.

## Streaming interfaces

```python
from negfpy.anharmonic import build_contracted_cubic_cut

plan = build_contracted_cubic_cut(problem, cut=3)
connected = plan.integrate_from_green(
    problem.grid, green, lag_times, block_size=16,
)
```

This method selects required greater Green entries once, then streams the
Fourier correlations in bounded time blocks. A material sampler can avoid
retaining full Green matrices by calling
`integrate_from_selected_spectra(grid, G_displacement, G_momentum, lag_times)`.

Both spectral arguments contain **centered displacement greater Green
functions**, without a zero-frequency mean delta. Their shapes are
`(Nw,len(plan.displacement_rows),len(plan.active_indices))` and
`(Nw,len(plan.momentum_rows),len(plan.active_indices))`.
The second argument selects rows where a time derivative will be taken;
it must not already contain a momentum derivative. The code computes
`C=i*integral(exp(-i*omega*t)*G>)` and
`P=integral(exp(-i*omega*t)*omega*G>)` with the existing rectangle weights.

For other correlation providers, `plan.integrate(lag_times, provider)` calls
`provider(times_block)`, expecting `(C,P)` with the same selected rows and
columns and with time replacing frequency as the first axis. These must be
centered correlations and their time derivatives in the same state.

Only scalar integrands are retained across blocks. Their final trapezoid
integration includes intervals crossing block boundaries. Nonuniform lag
spacing and a short final block are supported. A cut with no cubic terms
returns exactly zero connected cubic contribution without invoking the provider.

`connected_cubic_solver` remains in solver units. `connected_cubic` includes
the operator's physical current scale once. **Neither is a full current.**
Harmonic current and disconnected mean terms must still be added explicitly
in matching units and from the same state. The previous
`combine_cubic_moments` can supply disconnected moments by passing a zero
connected-moment vector and explicit means/covariances; contracting those
against the operator is cheap and does not require the expensive connected
moment calculation. Nonzero velocity means also affect harmonic raw moments.

## Storage and limits

The response tensor is dense only on active coordinates, stored as complex128
to avoid repeated real-to-complex conversions for BLAS. Its size is `16*A^3`
bytes. The default `max_dense_tensor_bytes=256*2**20` is checked before this
allocation. It limits only that tensor, not process memory or input arrays.
The sparse input is never expanded into a full-device cubic tensor.

The plan reports response storage, explicit transform/gather storage and
selected correlation block sizes separately. These omit compiled operator
indices, input spectra, BLAS-internal scratch, allocator overhead and any
buffers retained by the caller. They are not process-RSS bounds.

At 120 active coordinates and 120 selected rows, response storage is
27,648,000 bytes; the conservative sum of explicit transform/gather arrays
is 56,279,040 bytes. A one-point selected C/P block is 460,800 bytes, or
7,372,800 bytes for 16 points. Retaining the same selected C/P history at
4,801 points would instead take 2,212,300,800 bytes. The new path avoids
that history allocation. These dimensions are a scaling illustration;
actual material operator/row counts have not yet been compiled here.

The Fourier half-recurrence guard is unchanged. With the accepted material
frequency step .025, lag endpoints must be below `pi/.025=125.6637`.
Streaming does not improve frequency resolution, remove finite-domain tails,
or make a larger memory window valid on the same grid. Endpoint integrands
and small current differences are not rigorous tail bounds.

## Validation and benchmark

Independent tests compare the direct integrand with all six ordered Wick
pairings and the previous per-moment integrals. Complex nonsymmetric lines
expose accidental conjugation or transpose changes. Tests cover repeated
indices, inactive coordinates, internal response vertices, sparse storage,
physical scaling, unequal ordered coefficients, nonuniform time steps and
multiple block sizes. Invalid shapes, nonfinite lines, memory limits and
recurrence violations are rejected.

A separate dressed Drude steady-state check reproduces both previously
verified cuts with the new streamed Fourier/current path and retains the
harmonic-only negative control. Source derivation and mean/closure limitations
remain those in [the moment approximation](research-three-point-moments.md)
and [the steady reference](research-steady-cubic-current.md).

`scripts/benchmark_contracted_cubic_current.py` compares a resolved nonzero
current on 12 active coordinates against the original method. It times both
paths three times and records agreement and median speedup. A separate
120-coordinate synthetic dense-support case measures only nine lag evaluations
in total. It is denser than the material, uses no material fields, and is not
a prediction of full material runtime. Run with one BLAS thread and `src`
on `PYTHONPATH`; timing is hardware dependent.

The next step is a bounded selected-block diagnostic from the accepted fixed
interaction field, with a declared mean policy and frequency/memory checks.
No material archive is loaded, SCBA calculation repeated, or cluster job
submitted by this development step. Existing acceptance gates and legacy
cross-cut guards remain unchanged; material spatial validation is pending.
