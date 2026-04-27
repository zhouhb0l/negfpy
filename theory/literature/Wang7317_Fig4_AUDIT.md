# Wang 7317 Fig. 4 Audit

Target paper:
- J.-S. Wang, B. K. Agarwalla, H. Li, J. Thingna, "Nonequilibrium Green's function method for quantum thermal transport"
- local copy: `literature/1303.7317v1.pdf`
- target figure: `Fig. 4`

Goal:
- keep the quartic SCMF benchmark framework general enough for future material-based models
- keep the Wang Fig. 4 benchmark path paper-specific and auditable
- avoid mixing paper conventions into the general inelastic theory layer

## Exact items now matched in code

1. Approximation class
- We target quartic `SCMF`, not `SCBA`.
- Code path: `src/negfpy/inelastic/wang.py`

2. Nonlinear self-energy structure
- Paper Eq. (126):
  `Sigma_n(τ,τ')_{jj'} = 3 i ħ δ(τ,τ') Σ_{kl} T_{jj'kl} G_{kl}(τ,τ)`
- Code:
  `sigma_new = 3.0 * einsum("ijkl,kl->ij", quartic, covariance)`
- Here `covariance = i ħ ∫ G^< dω / (2π)`, so the Eq. (126) structure is preserved.

3. Equal-time covariance structure
- Paper Eq. (129):
  `<u u^T> = i ħ ∫ G^< [ω] dω / (2π)`
- Code:
  `_covariance_from_open_system(...)`

4. Current formula strategy
- Paper text states the current can still be calculated using the Caroli/Landauer formula with `Sigma_n^r` incorporated into `G^r`.
- Code:
  `quartic_scmf_current_vs_temperature(...)`

5. Lorentz-Drude bath functional form
- Paper caption:
  `J_alpha(ω) = ε² ω / (1 + ω² / ω_D²)`
- Code now uses a dedicated helper:
  `wang_lorentz_drude_bath_from_epsilon(...)`
- This avoids the earlier mismatch where the Wang presets were feeding `ε` directly into the bath prefactor instead of `ε²`.

6. Paper parameters
- One-particle:
  - `Omega² = 60.321 meV/(Å² u)`
  - `T1111 = 0.241, 1.2, 2.4 eV/(Å⁴ u²)`
- Two-particle:
  - `K11 = K22 = 60.321 meV/(Å² u)`
  - `K12 = K21 = -30.165 meV/(Å² u)`
  - black: `T1111 = T2222 = 0.483`, `T{1,1,1,2} = -0.241`, `T{1,1,2,2} = 0.241`
  - red: `T1111 = T2222 = 2.4`, `T{1,1,1,2} = -1.2`, `T{1,1,2,2} = 1.2`
  - blue: `T1111 = T2222 = 4.8`, `T{1,1,1,2} = -2.4`, `T{1,1,2,2} = 2.4`
- The curly-bracket permutation convention is matched by `_quartic_tensor_with_permutations(...)`.

7. Temperature protocol
- Paper:
  `T_L = 1.25 T`, `T_R = 0.75 T`
- Code:
  default factors in `quartic_scmf_current_vs_temperature(...)`

## Items still under audit

1. Exact SI normalization
- The paper reports current on the `10^-9 W` axis.
- Our current code has the right structure and ordering, but the absolute scale still does not match that figure.
- Conclusion: exact paper-native-to-SI normalization is not yet proven.

2. Absolute bath normalization in SI
- The functional form is matched.
- The exact prefactor mapping from paper units into the dynamical units used by the solver is still under audit.

3. Exact quantitative reproduction
- We should not yet claim that the benchmark "matches the published result".
- The current status is:
  - formal structure: mostly matched
  - paper coefficients: matched
  - bath functional form: matched
  - final output scale: not yet matched

## Recommended next steps

1. Work only on the one-particle case first.
2. Re-derive the paper-native unit convention into the solver units without using heuristic SI conversions.
3. Only after the one-particle current scale is correct, move to the two-particle case.
4. Keep all Wang-specific logic isolated in `src/negfpy/inelastic/wang.py`.
