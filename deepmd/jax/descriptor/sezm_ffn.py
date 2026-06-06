# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.sezm_ffn import (
    EquivariantFFN as EquivariantFFNDP,
)
from deepmd.jax.common import (
    flax_module,
)
from deepmd.jax.descriptor.sezm_so2 import (
    GatedActivation,
)
from deepmd.jax.descriptor.sezm_so3 import (
    SO3Linear,
)


@flax_module
class EquivariantFFN(EquivariantFFNDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"so3_linear_1", "so3_linear_2"}:
            value = (
                value
                if isinstance(value, SO3Linear)
                else SO3Linear.deserialize(value.serialize())
            )
        elif name in {"act"}:
            value = (
                value
                if isinstance(value, GatedActivation)
                else GatedActivation.deserialize(value.serialize())
            )
        return super().__setattr__(name, value)
