# Crystal IFC images to a finite transverse-periodic device

The bounded adapter in
[`transverse_material.py`](../src/negfpy/anharmonic/transverse_material.py)
connects audited crystal FC2 and full FC3 image data to the explicit
[transverse input interface](research-transverse-q-mapping.md). It retains
integer transverse winding while mapping transport positions into finite
device DOFs. All 23 focused synthetic unit and independent reference checks
pass. No actual material dataset has been mapped or solved by this
milestone.

## Input path and API

The existing source builders remain the entry point:

```python
from negfpy.anharmonic.layered import build_layered_device_mapping
from negfpy.anharmonic.mapped_fc3 import mass_normalize_translational_fc3
from negfpy.anharmonic.translational_fc3 import convert_fc3_to_translational
from negfpy.anharmonic.transverse_material import (
    map_material_cubic,
    material_transverse_harmonic,
)
from negfpy.anharmonic.transverse_mapping import fourier_transverse_cubic

geometry = build_layered_device_mapping(
    fc2_crystal,
    raw_fc2,
    transport_supercell_matrix=transport_matrix,
)
images = convert_fc3_to_translational(
    raw_full_fc3,
    fc3_crystal,
    make_r0_average=use_declared_r0_average,
)
normalized = mass_normalize_translational_fc3(
    images,
    fc3_crystal,
    source_certification=full_fc3_validation,
    output_units="si",
)
mapped = map_material_cubic(
    normalized,
    geometry,
    n_device_layers=n_layers,
    interaction_layers=interaction_layers,
    scale=scale,
)
vertex = fourier_transverse_cubic(mesh, mapped.ndof, mapped.terms)
device, lead = material_transverse_harmonic(
    geometry,
    n_device_layers=n_layers,
    scale=scale,
)
```

`MappedTransverseCubic` contains the complete finite-device dimension,
an ordered tuple of `TransverseCubicTerm` contributions, and source,
geometry, crop, scaling, and expansion metadata. Its coefficients use solver
units; its term weights remain unapplied until `fourier_transverse_cubic`.
Do not pass these terms through `mass_normalize_cubic_terms` again.

The normalized FC3 builder currently requires a passing, complete
permutation audit of the full raw source. Compact FC3 remains unsupported
by that path until its paired certification is implemented. FC2 uses the
full or compact layout already accepted by `LayeredDeviceMapping`.
FC2 and FC3 may use their respective audited image tables, but must share
the same primitive crystal identity, including atom ordering and masses.

The adapter checks the embedded FC2 and crystal content against the
geometry's source identities and checks the normalized FC3 content against
its recorded digest. A caller-supplied tensor without those source
builders and identities is not an equivalent input.

## Integer transport geometry

Use row-vector lattice coordinates. If the primitive lattice is \(A\) and
the nonsingular integer transport supercell matrix is \(S\), then

\[
A_{\mathrm{tr}}=SA,\qquad y=xS^{-1}.
\]

The first component of \(y\) is transport; the remaining two components are
transverse. This permits an integer transport cell with nontrivial cosets,
rather than requiring a primitive Cartesian axis. The Cartesian tensor
components retain the original coordinate basis; the lattice coordinate
change does not rotate the displacement components.

The existing geometry enumerates one representative \(r\) of each direct
coset of the primitive translation lattice modulo the transport lattice.
For primitive atom \(a\) at \(\tau_a\), it constructs a transport basis
position and an associated integer primitive-cell representative. There
are \(|\det S|N_{\mathrm{primitive}}\) atoms in this transport basis.

Each effective FC3 image supplies three physical positions
\(x_0,x_1,x_2\) in primitive fractional coordinates, together with their
primitive atom identities. For every direct coset representative, the
adapter maps all three positions:

\[
(x_\ell+r)S^{-1}=C_\ell+\rho_{b_\ell},
\qquad \ell=0,1,2.
\]

Here \(C_\ell\) is an integer transport-cell vector and \(b_\ell\) is the
matched transport-basis atom. The implementation uses the existing
fractional snapping and unique basis-match tolerances. A periodic basis
match can carry an additional integer winding; that winding is added to
the decomposed cell. Dropping it would move a near-boundary atom to the
wrong cell and alter a Fourier phase or transport layer.

Basis matching is constrained by primitive atom identity, not just
geometrical proximity. Fractional shortest vectors cannot be rounded
directly into cell offsets: their basis-position contribution must first
be resolved by this mapping.

## Finite transport crop and transverse reanchoring

Let \(p\) be `geometry.principal_cells`, \(B\) the number of transport-basis
atoms, and \(L=3pB\) the principal-layer DOF count. For a longitudinal
translation \(h\), define

\[
X_\ell=C_{\ell,0}+h,\qquad
j_\ell=\left\lfloor X_\ell/p\right\rfloor,\qquad
u_\ell=X_\ell\bmod p.
\]

The finite-device Cartesian DOF is

\[
d_\ell=j_\ell L+3(u_\ell B+b_\ell)+\alpha_\ell.
\]

`interaction_layers` must be a nonempty, ordered contiguous subset of the
device layers. If its first and last layers are \(j_{\min},j_{\max}\), all
three legs must obey

\[
p j_{\min}\le C_{\ell,0}+h
\le p(j_{\max}+1)-1.
\]

The mapper enumerates the intersection of these three integer anchor
intervals. There is no periodic transport wrapping. A term with no allowed
anchor is outside the declared interaction model. Testing only the first
leg against the crop would retain boundary-crossing interactions with
missing partners.

The returned transverse offsets are relative to the first leg:

\[
s=C_{1,\perp}-C_{0,\perp},\qquad
t=C_{2,\perp}-C_{0,\perp}.
\]

Thus one contribution represents \(\Psi_{0d_0,s d_1,t d_2}\).
Only its common transverse origin is removed; longitudinal positions stay
in the finite-device DOFs. Changing a direct coset representative by a
transport-lattice vector changes all three cell labels together. The
relative transverse offsets are unchanged, and relabeling \(h\) preserves
the same allowed finite-device positions.

Every coset and allowed longitudinal anchor represents an actual copy in
the enlarged basis or finite device. There is no division by \(|\det S|\),
\(p\), or the device length. The later Fourier mapper uses

\[
V_{acd}(q,k)=
\sum_{\mathrm{terms}}w\,c\,
e^{i[k\cdot s+(q-k)\cdot t]},
\]

with no additional atomic-position phase. The Born kernel supplies its
single \(1/N_q\) momentum weight. A mesh `(ny, nz, 1)` samples the two
transverse directions of the chosen transport lattice.

## Source weights, permutations, and units

The explicit source image expansion retains the shortest-vector
multiplicity weight, anchor weight, and optional `r0` representation.
The adapter reuses the existing effective-image expansion from
[`mapped_fc3.py`](../src/negfpy/anharmonic/mapped_fc3.py).
With `make_r0_average=False`, an image has one effective leg. With it
enabled, the source's audited `all_shortest` flag selects either the
original leg with factor one or three consistently permuted leg choices
with factor \(1/3\) each.

The stored scalar term weight is

\[
w=w_{\mathrm{multiplicity}}\,
w_{\mathrm{anchor}}\,
w_{\mathrm{r0}}.
\]

This weight is applied once, by the later Fourier sum. The source tensor
component is not multiplied by it during normalization or device mapping.
The adapter neither generates a missing complete permutation set nor
repairs source coefficients. The resulting reduced vertex must pass the
existing reality and leg-exchange checks.

`MassNormalizedTranslationalFC3` already contains
\(\Phi^{(3)}_{\mathrm{SI}}/\sqrt{m_a m_b m_c}\), with kilogram masses.
The only additional coefficient conversion is

\[
c_{\mathrm{solver}}=
\frac{\sqrt{\hbar}}{\omega_0^{5/2}}\,
c_{\mathrm{normalized\ SI}}.
\]

Zero source components are omitted exactly; no magnitude threshold is
used to remove small interactions. An unrepresentable nonzero coefficient
is rejected. Cropping changes the interaction model and can alter its
sum-rule behavior. No acoustic-sum-rule, interaction-boundary, or
symmetrization correction is added.

## Harmonic images and contact boundary

`material_transverse_harmonic` rebuilds the FC2 images from the audited raw
source and shortest-vector table embedded in the same geometry. The old
Nq=1 longitudinal terms have already summed transverse winding and cannot
be used to recover it.

For each canonical row-basis atom, the mapper verifies that its origin
maps to transport cell zero. Each destination shortest image is mapped
with the same basis and winding convention used for FC3. Its coefficient
is divided by its shortest-image multiplicity once. For each principal-cell
subcell \(u\), a destination with transport shift \(C_0\) gives

\[
\Delta j=\left\lfloor (u+C_0)/p\right\rfloor,\qquad
u'=(u+C_0)\bmod p.
\]

The corresponding FC00, FC01, or FC10 matrix contribution retains the
transverse key \(C_\perp\). Support outside \(\Delta j=-1,0,1\) is rejected
because it is incompatible with the declared principal layer.
Distinct source images that map to the same entry accumulate.

The existing harmonic adapter then applies declared FC2/SI, mass, and
\(\omega_0^{-2}\) conversion exactly once. It also rejects inconsistent
opposite-image FC00 blocks and inconsistent explicit FC01/FC10 blocks.
All device layers have the source principal-layer composition and masses.
The returned `DeviceKSpace` and `LeadKSpace` accept the same two radian
transverse coordinates as the cubic input interface.

Reservoir temperatures, broadening, and contact coupling choices remain
explicit caller inputs to `build_transverse_problem`. This adapter does not
define an acoustic zero-frequency prescription. The existing gapped contact
helper is suitable only for leads that satisfy its gap guard. The audited
Nq=1 acoustic treatment already exists in
[`contacts.py`](../src/negfpy/anharmonic/contacts.py) and
[`acoustic_contacts.py`](../src/negfpy/anharmonic/acoustic_contacts.py).
Extending and verifying that path for complex paired-q contacts remains
separate work; the new geometry adapter does not imply that extension.

## Bounds and validation status

The default cubic limits are 100,000 returned scalar terms and 1,000,000
effective-image/coset combinations. Source `r0` expansion is bounded before
copying its optional leg variants. Harmonic construction defaults to
1,000,000 image/subcell evaluations and 64 MiB of real image-block storage.
The downstream Fourier and solver limits still apply separately. These
guards bound particular payloads and work counts, not total process memory
or production scaling.

Both focused files passed on their first runs without production fixes:
19 unit tests in 2.04 seconds and four independent reference cases in
1.29 seconds, for 23 distinct passing checks. These were separate runs,
not a repeated combined execution. Ruff passed after removing one blank
line from the reference import block; the reference file's Python AST
was verified unchanged. No numerical replay was needed.

The independent reference in
[`test_anharmonic_transverse_material_reference.py`](../tests/test_anharmonic_transverse_material_reference.py)
starts from a synthetic raw bond polynomial with unequal atomic masses and
declared physical units. It enumerates and crops real-space interactions
independently of the finite IFC mappers, then compares complete-supercell
Fourier transforms with the mapped cubic vertex and harmonic device/lead
blocks. Its four cases cover:

- Gamma-only reduction.
- A four-q mesh with nonzero imaginary cubic phases.
- A sheared transport matrix with determinant two, duplicated images,
  and a synthetic flag choice exercising all three `r0` leg branches.
- A two-by-two mesh sampling both transverse directions.

The checks retain open transport endpoints and require every cubic leg to
stay inside the declared crop. They verify image multiplicity and solver
unit factors without adding a coset or principal-cell normalization.

The unit tests in
[`test_anharmonic_transverse_material.py`](../tests/test_anharmonic_transverse_material.py)
cover agreement with the existing Nq=1 finite crop under two transport
matrices, harmonic physical-unit conversion, invalid crops, work/storage
bounds, changed source content, and mismatched primitive identities.
The synthetic `r0` case exercises representation weights; it is not a
validation of shortest-image selection for an actual material dataset.
These new reference cases validate the raw-input mapping and Fourier
transforms; they do not add a material SCBA calculation.

Realistic IFC dataset integration, complex paired-q acoustic-contact
validation, interaction-crop and transverse-mesh convergence,
strong-coupling robustness, and computational scaling remain later stages.
This code introduces no material transport or release claim and does not
change the accepted Nq=1 results.
