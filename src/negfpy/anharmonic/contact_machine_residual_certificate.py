"""Decimal60 residual enclosure of exact machine dyadic components.

This NEW feasibility arithmetic contract preserves contact_residual_certificate
unchanged.  Real/imaginary input components are read via as_integer_ratio(),
including numpy.longdouble without casting them to float.  Directed rational
division encloses those exact machine values before the original associated
expression is evaluated with the preserved interval primitives.  The original
binary64 condition estimate remains only an estimated screen multiplier.
"""

from dataclasses import dataclass
from decimal import Decimal
import hashlib
from math import isfinite
from pathlib import Path
from time import monotonic

from . import contact_residual_certificate as original


ALGORITHM_VERSION = "exact-machine-dyadic-decimal60-directed-residual-v1"
ORIGINAL_SOURCE_SHA256 = "e663d48faec0ea740248ab3eb623a5c39af81b66085f2c9d6c741f84c770a762"
MAX_RATIO_INTEGER_BITS = 32768
ARITHMETIC_CONTRACT = "exact machine dyadic components via as_integer_ratio; decimal60 outward residual"


@dataclass(frozen=True)
class MachineContactResidualCertificate(original.ContactResidualCertificate):
    maximum_observed_ratio_integer_bits: int = 0

    def as_dict(self):
        result = super().as_dict()
        result.update(inputs_interpreted_as_exact_binary64=False,
            inputs_interpreted_as_exact_machine_dyadic_rationals=True,
            component_conversion="as_integer_ratio; separate FLOOR/CEILING Decimal rational division",
            input_components_cast_to_binary64=False,
            maximum_ratio_integer_bits_allowed=MAX_RATIO_INTEGER_BITS,
            original_binary64_arithmetic_contract=False)
        return result


class _MachineArithmetic(original._Arithmetic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.maximum_ratio_bits = 0

    def machine_component(self, value):
        if isinstance(value, bool):
            raise ValueError("boolean components are not machine floating-point data")
        try:
            numerator, denominator = value.as_integer_ratio()
        except (AttributeError, ValueError, OverflowError, TypeError) as error:
            raise ValueError("finite machine components with as_integer_ratio are required") from error
        if (type(numerator) is not int or type(denominator) is not int or denominator <= 0
                or denominator & (denominator-1)):
            raise ValueError("machine components must have finite dyadic integer ratios")
        bits = max(abs(numerator).bit_length(), denominator.bit_length())
        if bits > MAX_RATIO_INTEGER_BITS:
            raise ValueError("machine component integer-ratio resource bound exceeded")
        self.maximum_ratio_bits = max(self.maximum_ratio_bits, bits)
        self._tick()
        n, d = Decimal(numerator), Decimal(denominator)
        return original._Interval(self.down.divide(n, d), self.up.divide(n, d))


def planned_work(dimension):
    n = original._positive_integer(dimension, "dimension")
    return original.planned_work(n)+8*n*n


def _read_matrix(matrix, n, arithmetic):
    out = []
    for i in range(n):
        arithmetic.check_deadline()
        row = []
        for j in range(n):
            value = matrix[i][j]
            if isinstance(value, bool):
                raise ValueError("boolean matrix entries are invalid")
            try:
                real, imag = value.real, value.imag
            except AttributeError as error:
                raise ValueError("machine real/complex matrix entries are required") from error
            row.append((arithmetic.machine_component(real), arithmetic.machine_component(imag)))
        out.append(row)
    return out


def certify_machine_contact_residual(g0, a, b, derivative, *, condition_estimate,
        max_work, deadline_monotonic, max_dimension=original.MAX_DIMENSION):
    """Enclose the SAME E using exact machine inputs, retaining longdouble bits.

    This is a new residual-arithmetic screen, not original-binary64 acceptance,
    a contact descriptor qualification, or a rigorous condition/forward bound.
    Work includes all 8*n*n directed component conversions.  The fixed 60-digit
    precision and hard shape/ratio-bit/work/deadline bounds cannot be loosened.
    """
    started = monotonic()
    max_work = original._positive_integer(max_work,"max_work")
    max_dimension = original._positive_integer(max_dimension,"max_dimension")
    if max_dimension>original.MAX_DIMENSION:
        raise ValueError("max_dimension exceeds the hard limit 96")
    if isinstance(deadline_monotonic,bool):
        raise ValueError("finite explicit monotonic deadline required")
    try:
        deadline = float(deadline_monotonic)
    except (TypeError,ValueError,OverflowError) as error:
        raise ValueError("finite explicit monotonic deadline required") from error
    if not isfinite(deadline):
        raise ValueError("finite explicit monotonic deadline required")
    arith = _MachineArithmetic(max_work=max_work,deadline_monotonic=deadline)
    arith.check_deadline()
    if hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()!=ORIGINAL_SOURCE_SHA256:
        raise ValueError("Preserved reviewed binary64 interval source differs")
    try:
        n = len(g0)
        matrices = (g0,a,b,derivative)
        if not 1<=n<=max_dimension:
            raise ValueError("matrix dimension is empty or exceeds cap")
        for matrix in matrices:
            arith.check_deadline()
            if len(matrix)!=n or any(len(matrix[i])!=n for i in range(n)):
                raise ValueError("equal square machine matrices are required")
    except TypeError as error:
        raise ValueError("bounded square nested machine matrices required") from error
    work = planned_work(n)
    if work>max_work:
        raise original.ResidualCertificateLimitError("machine residual work exceeds cap before numeric allocation")
    condition = original._exact_float_component(condition_estimate)
    if not Decimal(1)<=condition<Decimal(2**52):
        raise ValueError("original finite condition screen failed")
    g,aa,bb,x = tuple(_read_matrix(matrix,n,arith) for matrix in matrices)
    product = original._matmul(arith,g,aa)
    product = original._matmul(arith,product,x)
    product = original._matmul(arith,product,bb)
    product = original._matmul(arith,product,g)
    square = original._matmul(arith,g,g)
    error = [[original._complex_add(arith,original._complex_sub(arith,x[i][j],product[i][j]),square[i][j])
              for j in range(n)] for i in range(n)]
    numerator = original._frobenius(arith,error)
    denominator = arith.add(arith.add(original._frobenius(arith,x),
                                     original._frobenius(arith,product)),
                            original._frobenius(arith,square))
    rho = arith.positive_upper_ratio(numerator.hi,denominator.lo)
    condition_product = arith.positive_upper_product(condition,rho)
    arith.check_deadline()
    if arith.work!=work:
        raise RuntimeError("machine residual work accounting differs")
    if not all(value.is_finite() for value in (numerator.hi,denominator.lo,rho,condition_product)):
        raise ValueError("finite machine residual enclosure could not be established")
    return MachineContactResidualCertificate(
        dimension=n,numerator_upper=numerator.hi,denominator_lower=denominator.lo,
        rho_upper=rho,condition_estimate_exact=condition,condition_screen_upper=condition_product,
        residual_screen_passed=bool(rho<=original.RESIDUAL_THRESHOLD),
        condition_screen_passed=bool(condition_product<=original.CONDITION_SCREEN_THRESHOLD),
        work_planned=work,work_completed=arith.work,elapsed_seconds=monotonic()-started,
        algorithm_version=ALGORITHM_VERSION,arithmetic_contract=ARITHMETIC_CONTRACT,
        maximum_observed_ratio_integer_bits=arith.maximum_ratio_bits)
