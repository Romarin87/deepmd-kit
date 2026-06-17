# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    ClassVar,
)

from deepmd.dpmodel.descriptor.dpa4_nn.radial import RadialMLP as RadialMLPDP
from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Linear as SO2LinearDP
from deepmd.jax.common import (
    flax_module,
    register_dpmodel_mapping,
)


@flax_module
class RadialMLP(RadialMLPDP):
    _jax_data_list_attrs: ClassVar[set[str]] = {
        "layers",
        "norms",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layers = list(self.layers)
        self.norms = list(self.norms)


@flax_module
class SO2Linear(SO2LinearDP):
    _jax_data_list_attrs: ClassVar[set[str]] = {
        "weight_m",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.weight_m = list(self.weight_m)


register_dpmodel_mapping(
    RadialMLPDP,
    lambda v: RadialMLP.deserialize(v.serialize()),
)

register_dpmodel_mapping(
    SO2LinearDP,
    lambda v: SO2Linear.deserialize(v.serialize()),
)
