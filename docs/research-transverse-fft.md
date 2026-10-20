# Frequency FFT for the coupled transverse-q Born contraction

The new `transverse_fft` module evaluates the same finite-grid lesser and
greater Born sums as the [direct transverse-q reference](research-transverse-q-prototype.md).
It replaces the frequency-pair loop with a zero-padded FFT convolution and
restricts the contraction to the exact support of the cubic vertex. Momentum
routing, the interaction model, the quadrature weight, and the returned device
coordinates stay the same. This is a contraction backend; it does not perform
Dyson solves, retarded reconstruction, SCBA iteration, or current integration.

The module is separate from the frozen material-input pilot and the historical
Nq=1 workflows. Its synthetic development does not reduce the user-selected
broadening or alter accepted fields. A material input rejected by a contact
classification check remains rejected independently of this optimization.

## API and conventions

```python
from negfpy.anharmonic.transverse_fft import cubic_born_self_energy_q_fft

sigma_lesser, sigma_greater = cubic_born_self_energy_q_fft(
    grid,
    vertex,
    g_lesser,
    g_greater,
    max_pair_evaluations=100_000,
    max_bytes=64 * 1024**2,
)
```

The arguments use the existing `SignedFrequencyGrid` and
`TransverseCubicVertex`. Each Green function and output has shape
`(Nq, Nomega, D, D)`. The vertex keeps its established shape
`(Nq, Nq, D, D, D)`, solver units, and cell-periodic Fourier convention:

\[
V_{acd}(q,p)=\sum_{s,t}\Psi_{0a,sc,td}
e^{i[p\cdot s+(q-p)\cdot t]}.
\]

The complete transverse mesh has third period one; transport is not Fourier
transformed. Integer `difference_q_indices` routes the second internal line to
\(q-p\). The caller includes any coupling-strength multiplier in the vertex;
the kernel does not apply \(g\) itself. Scaling the supplied vertex by a real
\(g\) scales the result by \(g^2\).

The second vertex is complex conjugated. Neither Green function is conjugated
or transposed in the contraction. The routines accept arbitrary finite complex
Green arrays for algebraic verification; validation of physical correlation
identities belongs to the solver layer. Inputs are not modified.

## Linear frequency convolution

Let \(N_\omega=2M+1\), with zero frequency at array index \(M\), and define
\(L\) as the smallest power of two satisfying \(L\ge 2N_\omega-1\). Pad each
matrix-valued frequency sequence with zeros at its end and apply NumPy's
forward FFT along that array axis:

\[
\widehat G^x_{ce}(p,k)=\sum_{j=0}^{N_\omega-1}
G^x_{ce}(p,j)e^{-2\pi i k j/L}.
\]

No `fftshift` is needed. At each Fourier bin the code forms

\[
\widehat C^x_{ab}(q,k)=\sum_{p,cdef}
V_{acd}(q,p)\widehat G^x_{ce}(p,k)
\widehat G^x_{df}(q-p,k)V_{bef}(q,p)^*.
\]

The inverse FFT includes its usual \(1/L\). Cropping indices
`M:M+Nomega` gives the required signed-frequency convolution:

\[
\Sigma^x_{ab}(q,i)=\frac{i\,\Delta\omega}{4\pi N_q}
C^x_{ab}(q,i+M).
\]

Thus the second Green index in the expanded sum is \(i-j+M\), exactly as in
the direct kernel. Values outside the supplied band are zero. Frequencies do
not wrap around; momenta do. The \(1/N_q\) mesh weight occurs once, and no
additional FFT normalization is inserted. At one q point this reduces to the
existing one-q discrete Born expression. Floating-point FFT and summation
differences permit tolerance agreement, not byte equality with the direct sum.

Each bin uses three staged contractions, with intermediate shapes
\((A,A,A)\), \((A,A,A)\), and \((A,A)\), where \(A\) is the active coordinate
count. No rank-four \(A^4\) or \(D^4\) coordinate tensor is constructed.

## Exact active-coordinate support

The active set is the union of indices appearing on any of the three legs of
an exactly nonzero coefficient across every \((q,p)\) block. Support discovery
uses matrix-size Boolean masks instead of a full vertex-size mask. There is
no numerical threshold, coefficient clipping, permutation completion, or
projection. An arbitrarily small nonzero coefficient remains active.

The contraction uses the corresponding principal submatrix of every Green
function. This is exact because a coordinate outside that union has zero
coefficient on every internal and external leg. Couplings in a supplied full
Green matrix involving inactive coordinates cannot enter this bubble. Returned
arrays retain full device dimensions, and inactive rows and columns are exact
zeros. An identically zero vertex returns zero arrays without any FFT work.

This compression does not remove harmonic buffers from a device Dyson solve,
change an interaction crop, or compress the stored input vertex. The dense
`TransverseCubicVertex` still occupies \(16N_q^2D^3\) bytes.

## Resource limits

All shapes, finite-value checks, and resource guards run before output
allocation or transforms. Both limits must be positive integers.

`max_pair_evaluations` bounds \(N_q^2L\) staged contractions per contour
component, or zero for an identically zero vertex. Lesser and greater are
processed sequentially and each uses that partition. The name matches the
direct sibling, but its counted work differs: the direct kernel counts
\(N_q^2(3M^2+3M+1)\) supported frequency pairs per component.

`max_bytes` separately bounds each full output's complex128 payload,
\(16N_qN_\omega D^2\), and the planned explicit FFT workspace. The workspace
account includes two transformed arrays, a projected input, four staged
cubic arrays, two matrix temporaries, routing, and small support/index arrays:

\[
B_{\mathrm{work}}=16\left[
2N_qLA^2+N_qN_\omega A^2+4A^3+2A^2\right]
+8N_q^2+8A+D^2+D.
\]

Both output arrays coexist. Input arrays, Python overhead, and library-internal
FFT/BLAS scratch are outside this workspace estimate; the limit is not a total
process-RSS guarantee. Validation and support scans use bounded masks, and
Green inputs are projected before conversion to complex128 to avoid copying a
full real input spectrum into complex storage. These API limits are independent
of the existing direct kernel's combined `max_output_bytes` limit.

The leading contraction cost changes from
\(O(N_q^2N_\omega^2D^4)\) to \(O(N_q^2LA^4)\), plus the FFT cost
\(O(N_qLA^2\log L)\). For 601 frequencies, the direct count is 270901
frequency pairs per momentum pair and the FFT count is 2048 bins. This is a
reduction in contraction count, not a measured speedup. Dense vertex storage,
large active-coordinate contractions, and full-device spectra remain important
material-scale costs.

## Verification and next integration step

Fourteen focused synthetic cases are covered by thirteen initial passes and
one passing guard recheck. The first run took 1.74 seconds and had one failure:
the test expected the word `exceed`, while the correctly raised byte-budget
exception said `must fit max_bytes`. Only that test's message matcher changed;
the single affected guard case passed in 1.00 second. The original failed log
and JUnit report are preserved. Production source stayed unchanged, and no
numerical case was repeated.

The passing cases compare against the direct kernel for one-q, four-q complex,
and two-by-two meshes using independent real-space polynomial vertices and
their Fourier transforms. They also cover frequency-edge routing without
wraparound, opposite-edge normalization, quadratic coupling scaling, an
inactive-coordinate embedding under a 32 KiB budget, unequal complex contour
inputs, unchanged read-only inputs, exact-zero fast return, and a nonzero
coefficient of \(10^{-40}\). Byte/work guards and malformed inputs are rejected
before transforms or output allocation. Ruff checks pass.

Evidence is stored under
`reports/phonon-phonon-interaction/multiq-fft-development-20261021/` in the
workspace root, including `fft-tests-execution.json` and
`fft-guard-recheck-execution.json` with stable source hashes for each run.

A bounded synthetic scaling experiment can measure runtime and explicit
payloads after those comparisons pass. Solver integration and material-scale
SCBA remain subsequent work. This backend alone establishes neither a material
transport solution nor frequency, momentum, broadening, or physical convergence.

Implementation: [`transverse_fft.py`](../src/negfpy/anharmonic/transverse_fft.py).
Reference: [`transverse_q.py`](../src/negfpy/anharmonic/transverse_q.py).
Focused tests: [`test_anharmonic_transverse_fft.py`](../tests/test_anharmonic_transverse_fft.py).
