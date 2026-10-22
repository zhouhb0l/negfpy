# Bounded diagnosis of the silicon Green identity failure

The closed SCBA continuation 25908656 completed trajectory states 13 and 14,
then failed the full-device Green Keldysh identity screen at candidate 15.
Its normalized identity allowance ratio was 11.966648348881517. The old logs
lack the frequency, raw defect and matrix-solve information needed for diagnosis.

This development adds an opt-in diagnostic path to the existing evaluator. The
ordinary SCBA/Born path retains its equations and thresholds. Diagnostic mode
uses the same validation, full-device Dyson inverse and lesser/greater products,
then stops before Born evaluation even when all Green checks pass. It records
scalar diagnostics only and does not advance the SCBA trajectory.

Reconstruct the exact failed candidate as 0.9 times evaluated input Sigma14 plus
0.1 times its raw Born image. The candidate must match all three raw-block hashes
recorded for failed attempt 3. Authenticate the original 19 material arrays and
six saved checkpoint arrays before using them. The previous run failed, but its
last complete checkpoint was published and verified by the wrapper; that
checkpoint is a diagnostic input, not an accepted converged restart.

Preserve g=.03, eta_lead=.0015625, eta_device=0, 305/295 K, nine layers with crop
2..6, 4 q sectors and 601 signed-frequency nodes. Identity rtol=1e-8 and atol=1e-11
remain unchanged. No clipping, inverse regularization, retry or mixing change.

The reported global Keldysh ratio uses independent maxima of absolute defect
and reference scale. Keep that original acceptance definition and also report
the maximum locally normalized ratio separately. Locate extrema by q index,
frequency index/value and matrix entry. Retain at most two points of matrix
context; never retain the full-device frequency stack for diagnostics. At those
points report left/right inverse backward residuals, norm(A)*norm(Gretarded) as
an inverse-based conditioning estimate, the source spectral identity defect,
its propagated Green contribution, the remaining defect, and lesser/greater
subtraction magnitudes. These quantities support diagnosis; they do not by
themselves prove intrinsic physical instability or certify a solver repair.

The result distinguishes completed diagnostic execution from passing the Green
identity. A captured failed identity can be a successful diagnostic execution.
Neither outcome is SCBA convergence or accepted transport. No Born map, new
checkpoint, accepted current, material-array output or historical-field update
is produced. Tiny synthetic checks and mock driver/wrapper checks validate this
new path; material evaluation runs only in the allocated Aspire2A job.
