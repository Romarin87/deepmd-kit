# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.fitting.dipole_fitting import DipoleFitting as DipoleFittingNetDP
from deepmd.dpmodel.fitting.dos_fitting import DOSFittingNet as DOSFittingNetDP
from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet as EnergyFittingNetDP
from deepmd.dpmodel.fitting.polarizability_fitting import (
    PolarFitting as PolarFittingNetDP,
)
from deepmd.dpmodel.fitting.property_fitting import (
    PropertyFittingNet as PropertyFittingNetDP,
)
from deepmd.dpmodel.fitting.sezm_ener_fitting import (
    SeZMEnergyFittingNet as SeZMEnergyFittingNetDP,
)

from ..common import (
    array_api_strict_module,
    to_array_api_strict_array,
)
from ..utils import exclude_mask as _strict_exclude_mask  # noqa: F401
from ..utils import network as _strict_network  # noqa: F401
from ..utils.exclude_mask import (
    AtomExcludeMask,
)
from ..utils.network import (
    NetworkCollection,
)


def setattr_for_general_fitting(name: str, value: Any) -> Any:
    if name in {
        "bias_atom_e",
        "fparam_avg",
        "fparam_inv_std",
        "aparam_avg",
        "aparam_inv_std",
        "case_embd",
        "default_fparam_tensor",
    }:
        value = to_array_api_strict_array(value)
    elif name == "emask":
        value = AtomExcludeMask(value.ntypes, value.exclude_types)
    elif name == "nets":
        value = NetworkCollection.deserialize(value.serialize())
    return value


@array_api_strict_module
class EnergyFittingNet(EnergyFittingNetDP):
    def __setattr__(self, name: str, value: Any) -> None:
        value = setattr_for_general_fitting(name, value)
        return super().__setattr__(name, value)


@array_api_strict_module
class SeZMEnergyFittingNet(SeZMEnergyFittingNetDP):
    def __setattr__(self, name: str, value: Any) -> None:
        value = setattr_for_general_fitting(name, value)
        return super().__setattr__(name, value)


@array_api_strict_module
class PropertyFittingNet(PropertyFittingNetDP):
    def __setattr__(self, name: str, value: Any) -> None:
        value = setattr_for_general_fitting(name, value)
        return super().__setattr__(name, value)


@array_api_strict_module
class DOSFittingNet(DOSFittingNetDP):
    def __setattr__(self, name: str, value: Any) -> None:
        value = setattr_for_general_fitting(name, value)
        return super().__setattr__(name, value)


@array_api_strict_module
class DipoleFittingNet(DipoleFittingNetDP):
    def __setattr__(self, name: str, value: Any) -> None:
        value = setattr_for_general_fitting(name, value)
        return super().__setattr__(name, value)


@array_api_strict_module
class PolarFittingNet(PolarFittingNetDP):
    def __setattr__(self, name: str, value: Any) -> None:
        value = setattr_for_general_fitting(name, value)
        if name in {
            "scale",
            "constant_matrix",
        }:
            value = to_array_api_strict_array(value)
        return super().__setattr__(name, value)
