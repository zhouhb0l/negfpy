# One full-device Dyson and packed Born evaluation

`transverse_material_map.evaluate_packed_born_map` connects the exact-support
packed executor to full harmonic-device physics. It evaluates one Born image;
it does not iterate, mix fields, update accepted results or declare convergence.
The existing small dense solver and its 64 MiB limits remain unchanged.

`transverse_compact_contacts.build_compact_transverse_contact` resolves each
reservoir's original side-dependent coupling and device addresses, then uses
the existing paired-q contact builder in the boundary coordinate basis. The
wrapper retains the original ordered addresses, including noncontiguous ones.
The fixed-eta zero-frequency limit and all lead classification checks are
unchanged. Four q sectors, 601 frequencies and 24 boundary coordinates need
66,502,656 bytes for three fields and zero limits, fitting the existing cap.
All 216 harmonic coordinates still enter the subsequent device inverse.

The evaluator forms and inverts one full-device matrix at a time:

```
G_r = inverse(omega**2 I - D - Sigma_L,r - Sigma_R,r - Sigma_input,r)
G_<,> = G_r (Sigma_L,<,> + Sigma_R,<,> + Sigma_input,<,>) G_a
```

Contacts and optional input interaction self-energy are embedded at their own
explicit supports. No device eta is added. Only active principal Green blocks
are retained across frequencies. Each full matrix is checked for the local
Keldysh/anti-Hermitian/PSD identities; retained active fields also undergo the
joint opposite-q and signed-frequency checks. Full-device reflection is not
independently checked through a second set of inversions.

The packed FFT Born executor computes both contour components. A real coupling
`g` multiplies the resulting image once as `g**2`; the supplied vertex must be
in solver units before g. Optional `sigma_input` is already scaled. Retarded
reconstruction calls the existing finite-grid zero-padded FFT Hilbert routine,
one q at a time. Explicit input self-energy must pass the same causal relation.
Empty interaction support and zero coupling give zero images.

`input_currents` are positive-frequency rectangle currents per transverse cell
for the state actually used in Dyson. Boundary traces use boundary Green blocks;
collision power uses the input interaction self-energy and active Green blocks.
The freshly mapped self-energy is not substituted into these traces. A harmonic
input can have balanced currents while its Born image is nonzero. This does not
establish SCBA convergence.

`plan_material_born_map` computes a conservative additional numeric allocation
plan before Dyson. It includes three retained Green arrays, three mapped arrays,
full-device single-frequency workspaces, the packed FFT plan and qwise retarded
reconstruction. The last Dyson matrices remain counted during later phases.
Caller-owned vertex/contact/input arrays are reported separately. Python and
library-internal scratch are excluded; this is not an RSS limit. Contact and
raw material mapping construction have separate bounds. Work gates include
both complete contour contractions and the number of full-device inverses.

New focused references compare the full pipeline against an independently
embedded full-device direct Born contraction and direct PV reconstruction on
one and four q sectors. They include complex phases, unordered support,
inactive harmonic buffers, nonzero input interaction self-energy and physical
current conversion. Guard tests check preallocation rejection, causality,
singular failure/thread restoration and zero coupling. No historical benchmark
or scientific workflow needs to be replayed for packaging.

`scripts/run_transverse_material_map.py` is the new actual-silicon experiment.
It fixes nine harmonic layers, interaction layers 2..6, mesh 4x1, the 601-node
signed grid with step .025, 305/295 K, g=.03, eta_lead=.0015625 and eta_device=0.
The packed material mapper avoids the full-device cubic tensor. One exact-zero
input interaction state is evaluated. The driver saves the unscaled vertex,
addresses, harmonic matrix, compact contacts, active input Green fields and
mapped self-energy as hashed NumPy files for subsequent development. These are
explicitly labeled input versus mapped state, not an accepted SCBA restart.

The actual material experiment remains distinct from synthetic tests. Full
material SCBA, restart acceptance, q/grid/crop accuracy and converged transport
remain subsequent work. The completed synthetic slice timings do not establish
the runtime or speedup of this full map.
