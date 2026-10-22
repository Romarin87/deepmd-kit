# SPDX-License-Identifier: LGPL-3.0-or-later
"""Trainable JAX wrappers for the portable DPA4 GLU energy head."""

from typing import ClassVar

from deepmd.dpmodel.fitting.dpa4_ener import GLUFittingNet as GLUFittingNetDP
from deepmd.dpmodel.fitting.dpa4_ener import (
    SeZMEnergyFittingNet as SeZMEnergyFittingNetDP,
)
from deepmd.dpmodel.fitting.dpa4_ener import (
    SeZMNetworkCollection as SeZMNetworkCollectionDP,
)
from deepmd.jax.common import flax_module
from deepmd.jax.fitting.base_fitting import BaseFitting
import deepmd.jax.utils.network as _jax_network  # noqa: F401


@flax_module
class GLUFittingNet(GLUFittingNetDP):
    pass


@flax_module
class SeZMNetworkCollection(SeZMNetworkCollectionDP):
    _jax_data_list_attrs: ClassVar[set[str]] = {"_networks"}
    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "sezm_fitting_network": GLUFittingNet,
    }


@BaseFitting.register("dpa4_ener")
@BaseFitting.register("sezm_ener")
@flax_module
class SeZMEnergyFittingNet(SeZMEnergyFittingNetDP):
    pass
