# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    ClassVar,
)

import deepmd.jax.utils.exclude_mask as _jax_exclude_mask  # noqa: F401
import deepmd.jax.utils.network as _jax_network  # noqa: F401
from deepmd.dpmodel.fitting.dipole_fitting import DipoleFitting as DipoleFittingNetDP
from deepmd.dpmodel.fitting.dos_fitting import DOSFittingNet as DOSFittingNetDP
from deepmd.dpmodel.fitting.dpa4_ener import (
    GLUFittingNet as GLUFittingNetDP,
)
from deepmd.dpmodel.fitting.dpa4_ener import (
    SeZMNetworkCollection as SeZMNetworkCollectionDP,
)
from deepmd.dpmodel.fitting.dpa4_ener import (
    SeZMEnergyFittingNet as SeZMEnergyFittingNetDP,
)
from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet as EnergyFittingNetDP
from deepmd.dpmodel.fitting.polarizability_fitting import (
    PolarFitting as PolarFittingNetDP,
)
from deepmd.dpmodel.fitting.property_fitting import (
    PropertyFittingNet as PropertyFittingNetDP,
)
from deepmd.jax.common import (
    flax_module,
    register_dpmodel_mapping,
)
from deepmd.jax.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.jax.utils.network import (
    NativeLayer,
)


@flax_module
class GLUFittingNet(GLUFittingNetDP):
    _jax_data_list_attrs: ClassVar[set[str]] = {"hidden_layers"}

    def __setattr__(self, name: str, value) -> None:  # noqa: ANN001
        if name == "hidden_layers":
            value = [
                item
                if isinstance(item, NativeLayer)
                else NativeLayer.deserialize(item.serialize())
                for item in value
            ]
        elif name == "output_layer" and value is not None and not isinstance(
            value, NativeLayer
        ):
            value = NativeLayer.deserialize(value.serialize())
        return super().__setattr__(name, value)


@flax_module
class SeZMNetworkCollection(SeZMNetworkCollectionDP):
    NETWORK_TYPE_MAP = {
        "sezm_fitting_network": GLUFittingNet,
    }


@BaseFitting.register("ener")
@flax_module
class EnergyFittingNet(EnergyFittingNetDP):
    pass


@BaseFitting.register("dpa4_ener")
@BaseFitting.register("sezm_ener")
@flax_module
class SeZMEnergyFittingNet(SeZMEnergyFittingNetDP):
    pass


register_dpmodel_mapping(
    GLUFittingNetDP,
    lambda v: GLUFittingNet.deserialize(v.serialize()),
)

register_dpmodel_mapping(
    SeZMNetworkCollectionDP,
    lambda v: SeZMNetworkCollection.deserialize(v.serialize()),
)


@BaseFitting.register("property")
@flax_module
class PropertyFittingNet(PropertyFittingNetDP):
    pass


@BaseFitting.register("dos")
@flax_module
class DOSFittingNet(DOSFittingNetDP):
    pass


@BaseFitting.register("dipole")
@flax_module
class DipoleFittingNet(DipoleFittingNetDP):
    pass


@BaseFitting.register("polar")
@flax_module
class PolarFittingNet(PolarFittingNetDP):
    pass
