"""Outward decimal enclosure of one stored contact-derivative residual.

This isolated feasibility module does not solve a contact, evaluate a potential,
or modify production acceptance.  Stored binary64 components are interpreted as
exact numbers.  At 60 decimal digits it encloses precisely the associated matrix
expression ``X - (((g0 @ a) @ X) @ b) @ g0 + g0 @ g0`` and its original Frobenius
normalization.  The condition estimate is an exact binary64 *screen multiplier*,
not a certified condition-number bound.  No true forward-error certificate is
claimed.  A PASS here uses a different arithmetic contract from the original
binary64 residual screen and cannot retroactively change its recorded outcome.

Only the Python standard library is imported.  Matrix inputs may be ordinary
square nested sequences or array-like objects supporting len/indexing; every
real and imaginary component must already be exactly representable in binary64.
No input is changed.  Work and monotonic deadlines are mandatory and fail closed.
"""

from dataclasses import dataclass
from decimal import (
    Context, Decimal, DivisionByZero, InvalidOperation, Overflow,
    ROUND_CEILING, ROUND_FLOOR,
)
from math import isfinite
from time import monotonic


ALGORITHM_VERSION = "exact-binary64-decimal60-directed-residual-v1"
DECIMAL_PRECISION = 60
MAX_DIMENSION = 96
RESIDUAL_THRESHOLD = Decimal("1e-10")
CONDITION_SCREEN_THRESHOLD = Decimal("1e-8")
ARITHMETIC_CONTRACT = "exact stored binary64 inputs; decimal60 outward interval residual"
_ZERO = Decimal(0)
_ONE = Decimal(1)


class ResidualCertificateLimitError(RuntimeError):
    """A resource bound prevents completing an enclosure; there is no result."""


@dataclass(frozen=True)
class ContactResidualCertificate:
    """Decimal endpoints remain Decimal (or strings in as_dict), never floats."""

    dimension: int
    numerator_upper: Decimal
    denominator_lower: Decimal
    rho_upper: Decimal
    condition_estimate_exact: Decimal
    condition_screen_upper: Decimal
    residual_screen_passed: bool
    condition_screen_passed: bool
    work_planned: int
    work_completed: int
    elapsed_seconds: float
    algorithm_version: str = ALGORITHM_VERSION
    decimal_precision: int = DECIMAL_PRECISION
    arithmetic_contract: str = ARITHMETIC_CONTRACT
    condition_is_certified_bound: bool = False
    true_forward_error_certified: bool = False
    original_binary64_arithmetic_contract: bool = False

    @property
    def passed(self):
        return self.residual_screen_passed and self.condition_screen_passed

    def as_dict(self):
        """JSON-ready proof scalars; exact decimal bounds are serialized as text."""
        answer = dict(self.__dict__)
        for name in (
            "numerator_upper", "denominator_lower", "rho_upper",
            "condition_estimate_exact", "condition_screen_upper",
        ):
            answer[name] = str(answer[name])
        answer.update(
            passed=self.passed,
            residual_threshold=str(RESIDUAL_THRESHOLD),
            condition_screen_threshold=str(CONDITION_SCREEN_THRESHOLD),
            inputs_interpreted_as_exact_binary64=True,
            residual_association="X - (((g0 @ a) @ X) @ b) @ g0 + g0 @ g0",
            work_unit="one real interval primitive (not wall time or Decimal instruction)",
            scope="residual enclosure and estimated-condition screens only",
        )
        return answer


@dataclass(frozen=True)
class _Interval:
    lo: Decimal
    hi: Decimal


_IZERO = _Interval(_ZERO, _ZERO)


class _Arithmetic:
    """Known scalar operations with independently directed endpoint rounding."""

    def __init__(self, *, max_work, deadline_monotonic):
        self.max_work = max_work
        self.deadline = deadline_monotonic
        self.work = 0
        # Even five products of extreme finite binary64 values and subsequent
        # squares fit very comfortably inside these explicit exponent bounds.
        self.down = Context(prec=DECIMAL_PRECISION, rounding=ROUND_FLOOR,
                            Emin=-999999, Emax=999999)
        self.up = Context(prec=DECIMAL_PRECISION, rounding=ROUND_CEILING,
                          Emin=-999999, Emax=999999)
        for ctx in (self.down, self.up):
            for signal in (InvalidOperation, DivisionByZero, Overflow):
                ctx.traps[signal] = True

    def check_deadline(self):
        if monotonic() >= self.deadline:
            raise TimeoutError("decimal contact residual reached its monotonic deadline")

    def _tick(self):
        self.work += 1
        if self.work > self.max_work:
            raise ResidualCertificateLimitError("decimal contact residual exceeded max_work")
        if self.work % 64 == 0:
            self.check_deadline()

    def add(self, x, y):
        self._tick()
        return _Interval(self.down.add(x.lo, y.lo), self.up.add(x.hi, y.hi))

    def sub(self, x, y):
        self._tick()
        return _Interval(self.down.subtract(x.lo, y.hi), self.up.subtract(x.hi, y.lo))

    def mul(self, x, y):
        self._tick()
        pairs = ((x.lo, y.lo), (x.lo, y.hi), (x.hi, y.lo), (x.hi, y.hi))
        return _Interval(min(self.down.multiply(a, b) for a, b in pairs),
                         max(self.up.multiply(a, b) for a, b in pairs))

    def square(self, x):
        self._tick()
        if x.lo <= 0 <= x.hi:
            lo = _ZERO
        else:
            lo = min(self.down.multiply(x.lo, x.lo), self.down.multiply(x.hi, x.hi))
        hi = max(self.up.multiply(x.lo, x.lo), self.up.multiply(x.hi, x.hi))
        return _Interval(lo, hi)

    def sqrt(self, x):
        self._tick()
        if x.lo < 0:
            raise ValueError("cannot enclose a square root of a negative interval")
        # Decimal.sqrt is correctly rounded to nearest (half even), regardless
        # of a Context's rounding mode.  Expanding to adjacent representable
        # decimals therefore encloses its exact value in either direction.
        lo = (_ZERO if x.lo == 0 else self.down.next_minus(self.down.sqrt(x.lo)))
        hi = (_ZERO if x.hi == 0 else self.up.next_plus(self.up.sqrt(x.hi)))
        return _Interval(lo, hi)

    def positive_upper_ratio(self, numerator, denominator):
        self._tick()
        if numerator < 0 or denominator <= 0:
            raise ValueError("positive residual denominator could not be certified")
        return self.up.divide(numerator, denominator)

    def positive_upper_product(self, x, y):
        self._tick()
        if x < 0 or y < 0:
            raise ValueError("screen factors must be nonnegative")
        return self.up.multiply(x, y)


def _complex_add(arith, x, y):
    return arith.add(x[0], y[0]), arith.add(x[1], y[1])


def _complex_sub(arith, x, y):
    return arith.sub(x[0], y[0]), arith.sub(x[1], y[1])


def _complex_mul(arith, x, y):
    # Keeping four real products explicitly makes the enclosure independent of
    # a BLAS implementation's complex reduction order or hidden precision.
    rr = arith.mul(x[0], y[0])
    ii = arith.mul(x[1], y[1])
    ri = arith.mul(x[0], y[1])
    ir = arith.mul(x[1], y[0])
    return arith.sub(rr, ii), arith.add(ri, ir)


def _matmul(arith, a, b):
    n = len(a)
    out = []
    for i in range(n):
        arith.check_deadline()
        row = []
        for j in range(n):
            value = (_IZERO, _IZERO)
            for k in range(n):
                value = _complex_add(arith, value, _complex_mul(arith, a[i][k], b[k][j]))
            row.append(value)
        out.append(row)
    return out


def _frobenius(arith, matrix):
    norm_squared = _IZERO
    for row in matrix:
        for real, imag in row:
            entry = arith.add(arith.square(real), arith.square(imag))
            norm_squared = arith.add(norm_squared, entry)
    return arith.sqrt(norm_squared)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive Python integer")
    return value


def planned_work(dimension):
    """Exact interval-primitive count; shape validation precedes numeric allocation."""
    n = _positive_integer(dimension, "dimension")
    if n > MAX_DIMENSION:
        raise ValueError(f"dimension exceeds the hard limit {MAX_DIMENSION}")
    return 40 * n**3 + 20 * n**2 + 8


def _exact_float_component(value):
    if isinstance(value, bool):
        raise ValueError("boolean components are not binary64 matrix data")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("matrix components must be finite binary64 values") from exc
    if not isfinite(number) or value != number:
        raise ValueError("matrix components must already be exactly representable in binary64")
    return Decimal.from_float(number)


def _read_matrix(matrix, n, arith):
    out = []
    for i in range(n):
        arith.check_deadline()
        row = []
        for j in range(n):
            value = matrix[i][j]
            if isinstance(value, bool):
                raise ValueError("boolean entries are not binary64 matrix data")
            try:
                real, imag = value.real, value.imag
            except AttributeError as exc:
                raise ValueError("matrix entries must be real or complex scalar numbers") from exc
            real = _exact_float_component(real)
            imag = _exact_float_component(imag)
            row.append((_Interval(real, real), _Interval(imag, imag)))
        out.append(row)
    return out


def certify_contact_residual(g0, a, b, derivative, *, condition_estimate,
                             max_work, deadline_monotonic, max_dimension=MAX_DIMENSION):
    """Enclose one candidate's residual; do not solve, repair, or mutate anything.

    ``max_work`` bounds real interval primitive calls, exactly ``planned_work(n)``
    for a completed evaluation. ``max_dimension`` may only tighten the hard 96
    bound. ``deadline_monotonic`` is an absolute, finite monotonic deadline.
    Missing/invalid bounds, nonfinite inputs, uncertain zero denominator, expiry,
    or Decimal arithmetic exceptions raise instead of returning a PASS result.
    """
    started = monotonic()
    max_work = _positive_integer(max_work, "max_work")
    max_dimension = _positive_integer(max_dimension, "max_dimension")
    if max_dimension > MAX_DIMENSION:
        raise ValueError(f"max_dimension cannot exceed the hard limit {MAX_DIMENSION}")
    if isinstance(deadline_monotonic, bool):
        raise ValueError("deadline_monotonic must be finite")
    try:
        deadline = float(deadline_monotonic)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("deadline_monotonic must be finite") from exc
    if not isfinite(deadline):
        raise ValueError("deadline_monotonic must be finite")
    arith = _Arithmetic(max_work=max_work, deadline_monotonic=deadline)
    arith.check_deadline()
    try:
        n = len(g0)
        matrices = (g0, a, b, derivative)
        if not 1 <= n <= max_dimension:
            raise ValueError("matrix dimension is empty or exceeds max_dimension")
        for matrix in matrices:
            arith.check_deadline()
            if len(matrix) != n or any(len(matrix[i]) != n for i in range(n)):
                raise ValueError("g0, a, b, derivative must be equally sized square matrices")
    except TypeError as exc:
        raise ValueError("matrix inputs must be bounded square nested sequences") from exc
    work = planned_work(n)
    if work > max_work:
        raise ResidualCertificateLimitError(
            f"planned work {work} exceeds max_work {max_work} before numeric allocation")
    condition = _exact_float_component(condition_estimate)
    # This is the pre-existing nonsingularity screen, not a verified condition
    # number.  Its threshold uses binary64 eps = 2**-52.
    if not _ONE <= condition < Decimal(2**52):
        raise ValueError("condition estimate must satisfy the original finite [1, 1/eps) screen")
    g, aa, bb, x = tuple(_read_matrix(matrix, n, arith) for matrix in matrices)
    product = _matmul(arith, g, aa)
    product = _matmul(arith, product, x)
    product = _matmul(arith, product, bb)
    product = _matmul(arith, product, g)
    square = _matmul(arith, g, g)
    error = [[_complex_add(arith, _complex_sub(arith, x[i][j], product[i][j]), square[i][j])
              for j in range(n)] for i in range(n)]
    error_norm = _frobenius(arith, error)
    x_norm = _frobenius(arith, x)
    product_norm = _frobenius(arith, product)
    square_norm = _frobenius(arith, square)
    denominator = arith.add(arith.add(x_norm, product_norm), square_norm)
    rho = arith.positive_upper_ratio(error_norm.hi, denominator.lo)
    condition_product = arith.positive_upper_product(condition, rho)
    arith.check_deadline()
    if arith.work != work:
        raise RuntimeError("interval work accounting does not match the declared algorithm")
    bounds = (error_norm.hi, denominator.lo, rho, condition_product)
    if not all(value.is_finite() for value in bounds):
        raise ValueError("a finite residual enclosure could not be established")
    return ContactResidualCertificate(
        dimension=n, numerator_upper=error_norm.hi,
        denominator_lower=denominator.lo, rho_upper=rho,
        condition_estimate_exact=condition, condition_screen_upper=condition_product,
        residual_screen_passed=bool(rho <= RESIDUAL_THRESHOLD),
        condition_screen_passed=bool(condition_product <= CONDITION_SCREEN_THRESHOLD),
        work_planned=work, work_completed=arith.work, elapsed_seconds=monotonic() - started,
    )
