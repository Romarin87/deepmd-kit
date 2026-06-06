# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.sezm_ffn import (
    EquivariantFFN as EquivariantFFNDP,
    S2GridProjector as S2GridProjectorDP,
    SwiGLUS2Activation as SwiGLUS2ActivationDP,
)
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    to_jax_array,
)
from deepmd.jax.descriptor.sezm_so2 import (
    GatedActivation,
)
from deepmd.jax.descriptor.sezm_so3 import (
    FocusLinear,
    SO3Linear,
)


@flax_module
class S2GridProjector(S2GridProjectorDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"to_grid_mat", "from_grid_mat"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
        return super().__setattr__(name, value)


@flax_module
class SwiGLUS2Activation(SwiGLUS2ActivationDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"scalar_gate"} and value is not None:
            value = (
                value
                if isinstance(value, FocusLinear)
                else FocusLinear.deserialize(value.serialize())
            )
        elif name in {"projector"} and value is not None:
            value = (
                value
                if isinstance(value, S2GridProjector)
                else S2GridProjector.deserialize(value.serialize())
            )
        return super().__setattr__(name, value)


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
            if isinstance(value, (GatedActivation, SwiGLUS2Activation)):
                pass
            elif value.serialize().get("@class") == "SwiGLUS2Activation":
                value = SwiGLUS2Activation.deserialize(value.serialize())
            else:
                value = GatedActivation.deserialize(value.serialize())
        return super().__setattr__(name, value)
