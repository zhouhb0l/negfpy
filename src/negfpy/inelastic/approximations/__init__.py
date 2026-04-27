"""Approximation-first inelastic theory modules.

This subpackage exposes the six development tracks explicitly:
- third-order lowest order
- third-order mean field
- third-order SCBA
- fourth-order lowest order
- fourth-order mean field
- fourth-order SCBA

Toy models such as FPU-alpha and phi^4 live outside this folder in
``negfpy.inelastic`` so that the theory layer stays general and can later be
applied to material-derived interaction tensors.
"""

from .fourth_order_lowest_order import (
    FourthOrderInteraction,
    FourthOrderLowestOrderModel,
    fourth_order_lowest_order_model_from_covariance,
    fourth_order_lowest_order_model_from_device,
    fourth_order_lowest_order_self_energy,
)
from .fourth_order_mean_field import (
    FourthOrderMeanFieldModel,
    fourth_order_mean_field_model_from_covariance,
    fourth_order_mean_field_model_from_device,
    fourth_order_mean_field_self_energy,
)
from .fourth_order_scba import FourthOrderSCBAModel, fourth_order_scba_model_from_device, fourth_order_scba_self_energy
from .third_order_lowest_order import (
    ThirdOrderInteraction,
    ThirdOrderLowestOrderModel,
    third_order_lowest_order_model_from_device,
    third_order_lowest_order_self_energy,
)
from .third_order_mean_field import (
    ThirdOrderMeanFieldModel,
    third_order_mean_field_model_from_displacement,
    third_order_mean_field_self_energy,
)
from .third_order_scba import ThirdOrderSCBAModel, third_order_scba_model_from_device, third_order_scba_self_energy

__all__ = [
    "ThirdOrderInteraction",
    "ThirdOrderLowestOrderModel",
    "third_order_lowest_order_self_energy",
    "third_order_lowest_order_model_from_device",
    "ThirdOrderMeanFieldModel",
    "third_order_mean_field_self_energy",
    "third_order_mean_field_model_from_displacement",
    "ThirdOrderSCBAModel",
    "third_order_scba_model_from_device",
    "third_order_scba_self_energy",
    "FourthOrderInteraction",
    "FourthOrderLowestOrderModel",
    "fourth_order_lowest_order_model_from_covariance",
    "fourth_order_lowest_order_model_from_device",
    "fourth_order_lowest_order_self_energy",
    "FourthOrderMeanFieldModel",
    "fourth_order_mean_field_self_energy",
    "fourth_order_mean_field_model_from_covariance",
    "fourth_order_mean_field_model_from_device",
    "FourthOrderSCBAModel",
    "fourth_order_scba_model_from_device",
    "fourth_order_scba_self_energy",
]
