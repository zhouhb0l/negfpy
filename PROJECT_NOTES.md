# Version 1.2.0 additions

The public one-JSON installed command accepts saved, qualified complete-energy
IFC2/IFC3 data, rolling evaluated-input checkpoints and explicit authenticated
historical initial guesses. Rebuild every target with unchanged complete support,
physical guards and all six convergence screens. A q/temperature migrated state
is a guess, never an accepted result. Rolling checkpoints contain only evaluated
INPUT channels; rejected latest trials and selected anchors remain distinct.
Original numerical source selection, qualifications and prior releases survive.

# Package contract

Version 1.1.0 adds an optional authenticated external evaluated self-energy
initial guess. Its original input/report/fields and fresh target identities
must match. Only R/C/D are loaded; all history is fresh, and new original-six
and physical checks remain mandatory. The public restart contract is documented
in docs/RESTART.md. Prior releases and campaign archives remain unchanged.

Stable entity: `sw-negfpp`. Public entry points are `negfpp INPUT.json` and `python -m negfpp INPUT.json`. The numerical namespace `negfpy` is retained internally to preserve the existing implementation and imports.

Keep the original cubic convention, paired thermal contacts, complete endpoint support, independent channel fields and all six convergence screens. In-progress outputs and numerical nonconvergence must remain distinguishable from an accepted state. Do not replace the existing solver with fitted lifetimes, diagonal scattering, wrapper-specific physics or tolerance shortcuts.

`SOURCE_PROVENANCE.json` binds selected source files to their origin. `examples/silicon24/input-provenance.json` binds the compact example to its source dataset and saved energy model. Large campaign outputs and environments remain outside the source archive. Write new runs to a new output directory.

Selected unit checks exercise the existing numerical building blocks and the new command/input/output contracts. The silicon example is a bounded installation/calculation check, not an accuracy-refinement campaign.

Version 1.0.1 adds the existing same-run reference/increment SCBA controller with exact Green/Born increments and bounded analytic feedback. It preserves the original 18 nonlinear-attempt and six tangent-attempt ceilings. No foreign checkpoint initialization is implied by the reference-seeded command. The five-full-layer silicon example contains 40 atoms and three complete interacting center layers; keep that geometry explicit.

Version 1.0.2 adds the existing Method 1 frozen-retarded extension with local fluctuation–dissipation/Bose probes. It uses a separately authenticated short-device evaluated input, retaining complete overlapping support. The original recursive current/temperature methods are selected into normal package modules; frequency tiles may be distributed across CPU workers with every q/frequency point counted once. The approximately 1µm silicon example is a coarse Gamma/101-node, g=0.10 execution check. Donor six-screen SCBA convergence and target nonlinear/linear probe convergence are distinct. No long-device Born maps, physical-kernel repairs or full long-device SCBA claim are permitted.
