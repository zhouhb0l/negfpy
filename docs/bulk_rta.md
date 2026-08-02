# Bulk phonons and three-phonon SMRTA transport

`negfpy.bulk` is the NEGFPy 1.0 public interface for periodic-crystal
phonons and intrinsic three-phonon thermal conductivity. It provides:

- audited phono3py FC2 and FC3 ingestion with immutable input hashes;
- phonon bands and total/atom/Cartesian-projected density of states (DOS);
- analytical group velocities;
- on-shell three-phonon linewidths, population lifetimes, and mean free paths;
- mode-, band-index-, and tensor-resolved single-mode relaxation-time
  approximation (SMRTA) conductivity; and
- cumulative conductivity versus frequency or mean free path.

This is a bulk phonon-Boltzmann calculation. It is deliberately separate from
open-device NEGF: there is no junction, contact self-energy, finite interaction
region, or anharmonic lead in an SMRTA result.

## Installation

The validated material path uses phono3py to reconstruct crystal mappings and
HDF5 to read force constants and archive RTA results:

```bash
python -m pip install -e '.[material-oracle]'
```

The 1.0 compatibility convention is pinned to phono3py 4.4.0. Do not mix a
different frequency conversion, linewidth convention, or crystal mapping into
a claimed same-input validation.

## Safe FC2-only starting point

Use `BulkHarmonicProblem` for band structures, uniform-mesh phonons, DOS, and
group velocities. This type accepts no FC3 path, so an ordinary exploratory
calculation cannot accidentally start a cubic contraction.

```python
import negfpy.bulk as bulk

harmonic = bulk.BulkHarmonicProblem.from_phono3py(
    "phono3py_disp.yaml",
    "fc2.hdf5",
)
```

Construction fails closed unless the FC2 passes both its schema audit and the
mapped physical audit. The latter checks the crystal mapping, reciprocity and
time reversal, uncorrected dynamical-matrix Hermiticity, the translational
acoustic sum rule, the rotational sum-rule residual, and stability on a fixed
deterministic q-point sample. For full FC2, it also checks every redundant
first-axis row through explicit pure-translation permutations; unused corrupt
rows are not ignored. It never repairs or symmetrizes the input.

For both FC2 and FC3, every consumed HDF5 path component must be an internal
hard link. Soft links and external links are rejected, including links in
optional mapping or metadata fields, and the container SHA256 must remain
unchanged across the read. This ensures recorded source provenance binds the
actual tensor and mapping bytes rather than a mutable external target.

These checks are necessary release preflights, not a proof that the IFC is a
complete physical model. A user must still establish supercell/range,
electronic-structure, displacement-amplitude, and numerical convergence for
the actual material. Sampled stability also does not replace a sufficiently
dense scan of the Brillouin zone.

### Band structure

Band vertices are fractional coordinates in the reciprocal basis of the
primitive cell stored in `harmonic.mapping`. Frequencies are cyclic THz, and
the present implementation is non-NAC.

```python
bands = bulk.compute_phonon_band_structure(
    harmonic.fc2,
    harmonic.mapping,
    vertices=[
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ],
    labels=[r"$\Gamma$", "X", r"$\Gamma$", "L"],
    points_per_segment=51,
    frequency_factor_to_thz=harmonic.frequency_factor_to_thz,
)

x = bands.path.distances_inv_angstrom
frequency_thz = bands.frequencies_thz
```

Those example vertices use the reciprocal basis of the pinned fcc-silicon
primitive cell. Choose a path appropriate to any other primitive basis.

### Full-mesh phonons and DOS

A band path is not a Brillouin-zone integration mesh. Build a complete
uniform mesh explicitly:

```python
spectrum = harmonic.compute_uniform_mesh_phonons((11, 11, 11))
dos = bulk.compute_phonon_dos(spectrum, sigma_thz=0.1)
```

`BulkPhononSpectrum` records `sampling_kind`. `compute_phonon_dos` rejects a
`band_path` spectrum. An arbitrary q-point list is accepted only with explicit
`qpoint_weights` and a nonempty `sampling_description`; a full uniform mesh
has complete equal weights by construction.

`dos.total_states_per_thz_cell` has units states/(THz primitive cell).
`dos.projected_states_per_thz_cell` has shape `(atom, Cartesian, frequency)`
and sums to the total. The default frequency window extends five Gaussian
widths beyond the sampled frequency range. Its integral must recover
`3 * N_atom` within `normalization_tolerance`, otherwise the calculation
fails. Setting `allow_truncated_window=True` is an explicit noncertifying
escape hatch; the returned `normalization_certified` will then be false.

The state-count gate checks frequency-window normalization only. A physically
resolved DOS additionally requires convergence with q mesh, Gaussian width,
and frequency-grid spacing. A dynamically unstable frequency below the stated
tolerance is rejected rather than hidden in the DOS.

### Group velocity and degeneracies

```python
velocities = bulk.compute_bulk_group_velocities(
    harmonic.fc2,
    harmonic.mapping,
    spectrum.qpoints,
    frequency_factor_to_thz=harmonic.frequency_factor_to_thz,
    cutoff_frequency_thz=1.0e-4,
    degenerate_tolerance_thz=1.0e-4,
    symmetrize=True,
)
```

The FC2 dynamical-matrix derivative is analytical. Degenerate subspaces are
resolved by deterministic first-order perturbation theory, followed by
little-group symmetrization when requested. Results are stored in both
`values_thz_angstrom` and `values_m_per_s`, using

\[
1\;\mathrm{THz\,\mathring{A}}=100\;\mathrm{m\,s^{-1}}.
\]

Eigenvectors inside an exactly or nearly degenerate subspace are not unique:
they may acquire phases or rotate within the subspace. Individual mode
projections, velocities, and array-index labels can therefore be gauge
dependent. Validation must compare projectors or sums over matched degenerate
subspaces, not gauge-dependent eigenvector columns. Total observables and
complete degenerate-subspace sums remain the appropriate invariants.

## Explicit heavy FC3/SMRTA path

Construct `BulkTransportProblem` only when a three-phonon calculation is
intended:

```python
problem = bulk.BulkTransportProblem.from_phono3py(
    "phono3py_disp.yaml",
    "fc2.hdf5",
    "fc3.hdf5",
    mesh=(11, 11, 11),
    sigma_thz=0.1,
    make_r0_average=True,
)
```

This repeats the mandatory FC2 audits and adds a read-only translated-FC3
audit: mapping/schedule completeness, finiteness, the three independent
Gamma-point cubic acoustic sum rules, exact nonzero support, pairwise support
range, and the recorded phase/anchor convention. It does not symmetrize,
threshold, truncate, or correct FC3.

`make_r0_average=True` is the 1.0 release default and the convention validated
against phono3py 4.4.0. It averages equivalent first-leg origins while
building the explicit translational representation. Record the setting with
the input hashes; `False` is a different convention study, not an invisible
performance switch.

`sigma_thz` is the Gaussian width used for the three-phonon energy delta
functions. It is a numerical control rather than a fitted material property.

### Linewidth, lifetime, and mean free path

```python
linewidths = bulk.compute_three_phonon_linewidths(
    problem.three_phonon,
    temperatures=[100.0, 300.0, 500.0],
)
relaxation = bulk.compute_three_phonon_lifetimes(linewidths)

velocities = bulk.compute_bulk_group_velocities(
    problem.three_phonon.fc2,
    problem.three_phonon.fc2_mapping,
    linewidths.interactions.phonons.qpoints,
    frequency_factor_to_thz=problem.three_phonon.frequency_factor_to_thz,
    cutoff_frequency_thz=problem.three_phonon.cutoff_frequency_thz,
    degenerate_tolerance_thz=problem.three_phonon.degenerate_tolerance_thz,
    symmetrize=True,
)
mean_free_path = bulk.compute_mean_free_paths(velocities, relaxation)
```

The leading linewidth axes are `(temperature, q point, band index)`. The
linewidth convention is exact:

- `gamma_thz_hwhm` is positive-frequency cyclic-THz HWHM;
- `gamma_normal_thz_hwhm + gamma_umklapp_thz_hwhm` is the total;
- the angular-frequency FWHM is `4 * pi * gamma` in ps\(^{-1}\); and
- the population lifetime is

\[
\boxed{\tau_{qj}[\mathrm{ps}]
       =\frac{1}{4\pi\,\gamma_{qj}[\mathrm{THz,HWHM}]}}.
\]

Because THz is ps\(^{-1}\), no additional power of ten belongs in this
conversion. The mean-free-path vector is

\[
\boldsymbol\Lambda_{qj}[\mathring{A}]
=\mathbf v_{qj}[\mathrm{THz\,\mathring{A}}]\tau_{qj}[\mathrm{ps}].
\]

Modes below `cutoff_frequency_thz` are excluded consistently. The strict
conductivity path rejects any included mode with an unresolved nonpositive
linewidth instead of inventing a lifetime. `gamma_floor_thz` is a resolution
threshold, not an added scattering mechanism.

### SMRTA conductivity and archive

For normalized q-point weights \(w_q\), NEGFPy evaluates

\[
\kappa_{\alpha\beta}(T)=\frac{1}{V}
\sum_{qj}w_q C_{qj}(T)
v_{qj,\alpha}v_{qj,\beta}\tau_{qj}(T),
\]

with the quantum harmonic heat capacity

\[
C_{qj}=k_B x^2\frac{e^x}{(e^x-1)^2},\qquad
x=\frac{h\nu_{qj}}{k_BT}.
\]

The complete operation computes a missing full-mesh linewidth result, group
velocities, heat capacities, lifetimes, mean free paths, and conductivity:

```python
rta = bulk.compute_rta_thermal_conductivity(
    problem,
    temperatures=[100.0, 300.0, 500.0],
    linewidths=linewidths,  # optional reuse; must match the exact problem
)

archive = bulk.write_rta_result_hdf5("silicon-rta.h5", rta)
reloaded = bulk.read_rta_result_hdf5(archive)
```

Supplying `linewidths` avoids repeating the expensive FC3 contraction. The
function verifies the complete q mesh, every band index, temperatures, and an
identity hash covering the FC2/FC3 mappings, mesh, smearing, cutoff, and
conventions. The HDF5 writer is atomic, refuses overwrite by default, records
the schema and metadata, and stores a payload hash that is verified on read.
An ordinary API result never self-certifies a release: only the external
fail-closed release driver may combine authenticated reference evidence,
environment identity, exact source revision, and all scientific gates.

Full mode-space FC3 interaction tensors can occupy several gigabytes even
though RTA post-processing needs only the harmonic mesh and on-shell rates.
Convert a fresh linewidth result to the portable lightweight representation
and archive it separately:

```python
rates = bulk.summarize_three_phonon_linewidths(linewidths)
rate_file = bulk.write_rate_spectrum_hdf5("silicon-rates.h5", rates)
rates = bulk.read_rate_spectrum_hdf5(rate_file)

rta = bulk.compute_rta_thermal_conductivity(
    problem,
    temperatures=[100.0, 300.0, 500.0],
    linewidths=rates,
)
```

`BulkThreePhononRateSpectrum` retains total, normal, and Umklapp gamma, phase
space, q points, frequencies, controls, and a path-independent physical-input
digest, but not matrix elements. Its checked archive is self-contained:
external and soft links, extra fields, wrong dtypes, and payload changes are
rejected. Source/evidence authentication is still required when converting a
legacy archive that predates the portable digest.

Important result fields are:

| Field | Shape | Meaning |
| --- | --- | --- |
| `kappa_w_mk` | `(temperature, 3, 3)` | total conductivity tensor |
| `kappa_voigt_w_mk` | `(temperature, 6)` | `xx, yy, zz, yz, xz, xy` |
| `mode_kappa_w_mk` | `(temperature, q, band index, 3, 3)` | mode contributions |
| `band_index_kappa_w_mk` | `(temperature, band index, 3, 3)` | q-sum at a fixed eigenvalue-array index |
| `gamma_thz_hwhm` | `(temperature, q, band index)` | total cyclic-HWHM linewidth |
| `population_lifetime_ps` | `(temperature, q, band index)` | population lifetime |
| `mean_free_path_angstrom` | `(temperature, q, band index)` | MFP magnitude |

`band_index_kappa_w_mk` is intentionally not called “branch conductivity.” A
fixed sorted eigenvalue index is not a globally tracked phonon branch and can
exchange character at crossings or degeneracies. Use total tensors,
mode-character analysis, or sums over matched subspaces for scientific
interpretation.

Cumulative tensors can be built in memory:

```python
by_frequency = bulk.compute_cumulative_thermal_conductivity(
    rta,
    coordinate="frequency_thz",
)
by_mfp = bulk.compute_cumulative_thermal_conductivity(
    rta,
    coordinate="mean_free_path_angstrom",
)
```

The last threshold must reproduce the total tensor up to floating-point
summation order.

## Safe silicon example

The default example performs only the audited FC2 path:

```bash
python examples/bulk_rta_silicon.py \
  --crystal /path/to/phono3py_disp.yaml \
  --fc2 /path/to/fc2.hdf5
```

It cannot read FC3. A user must provide both independent opt-ins before the
heavy calculation is reachable:

```bash
python examples/bulk_rta_silicon.py \
  --crystal /path/to/phono3py_disp.yaml \
  --fc2 /path/to/fc2.hdf5 \
  --fc3 /path/to/fc3.hdf5 \
  --run-fc3 --allow-heavy \
  --output /path/to/artifacts/silicon-rta.h5
```

The default 3x3x3 mesh is bounded but deliberately coarse. It exercises the
API; it does not establish a physical silicon conductivity.

## Implementation validation versus physical convergence

Same-input **implementation validation** compares NEGFPy with the pinned
phono3py reference using identical IFCs and numerical controls:

- FC2 dynamical matrices and analytical group velocities;
- quantum mode heat capacities;
- sampled gauge-invariant three-phonon interaction strengths;
- on-shell linewidths, including normal/Umklapp sums; and
- the final SMRTA tensor and unit conversion.

This establishes conventions, indexing, units, and implementation agreement.
It does not establish that one quoted conductivity is physically converged.
A publication result additionally requires, at minimum:

1. independent q-mesh and Gaussian-width convergence;
2. stable phonons and passing IFC mapping/physical audits;
3. convergence with IFC supercell, range, and underlying force quality;
4. an appropriate temperature range for fixed IFCs and quasiparticles; and
5. an explicit assessment of scattering mechanisms omitted from SMRTA.

Coarse-mesh values may vary non-monotonically by orders of magnitude.
Agreement on the same coarse mesh validates the code path, not experiment.
Archive input hashes, mesh, smearing, cutoff, temperatures, code revision,
environment, and the complete convergence ladder for every reported value.

## Approximation boundary

The 1.0 bulk result is intrinsic cubic SMRTA:

- harmonic quasiparticle frequencies and velocities;
- on-shell three-phonon scattering from FC3;
- independent relaxation of each mode; and
- a total linewidth containing both normal and Umklapp events.

Including normal events in the SMRTA linewidth does not assert that they are
fully resistive in the coupled BTE. SMRTA does not recover collective drift or
the redistribution among modes produced by an iterative LBTE solution.

The following are excluded unless a future separately named method says
otherwise:

- iterative LBTE;
- isotope, boundary, defect, electron-phonon, and four-phonon scattering;
- quartic renormalization, thermal expansion, and temperature-dependent IFCs;
- polar non-analytical corrections (NAC);
- Wigner/coherence transport;
- a two-dimensional effective-thickness convention; and
- open-device contacts and junction physics.

These omissions must not be hidden by fitting a linewidth floor or prefactor.

## Boundary to NEGFPy 2.0

NEGFPy 2.0 will target real-material anharmonic NEGF: an open junction with
harmonic reservoir self-energies and a conserving cubic SCBA self-energy in a
finite interaction region, followed by transverse-momentum-coupled material
geometries. It is not obtained by inserting a bulk SMRTA lifetime into a
harmonic transmission formula. The 1.0 bulk implementation remains an
independent FC2/FC3, linewidth, unit, and weak-scattering oracle for 2.0, while
the two solvers retain separate physical meanings and certification gates.
