"""Original associated binary64 contact-derivative diagnostic."""
import numpy as np
from . import acoustic_contacts as _acoustic

def _equation_metrics(derivative, g0, a, b, condition):
    # EXACT canonical multiplication association, denominator and thresholds.
    lhs_correction = g0 @ a @ derivative @ b @ g0
    equation_error = derivative - lhs_correction + g0 @ g0
    denominator = max(
        float(np.linalg.norm(derivative, ord="fro"))
        + float(np.linalg.norm(lhs_correction, ord="fro"))
        + float(np.linalg.norm(g0 @ g0, ord="fro")),
        _acoustic._TINY,
    )
    residual = float(np.linalg.norm(equation_error, ord="fro")) / denominator
    product = condition * residual
    passed = bool(np.isfinite(residual) and np.isfinite(product)
                  and residual <= 1e-10 and product <= 1e-8)
    return equation_error, {"derivative_residual":residual,
        "condition_times_residual":product,"original_guard_passed":passed}
