# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

from typing import (
    Any,
)

from packaging.version import (
    Version,
)

from deepmd.dpmodel.descriptor.sezm_block import (
    SeZMInteractionBlock as SeZMInteractionBlockDP,
)
from deepmd.jax.common import (
    flax_module,
)
from deepmd.jax.descriptor.sezm_ffn import (
    EquivariantFFN,
)
from deepmd.jax.descriptor.sezm_norm import (
    EquivariantRMSNorm,
)
from deepmd.jax.descriptor.sezm_so2 import (
    SO2Convolution,
)
from deepmd.jax.env import (
    flax_version,
    nnx,
)


def _maybe_nnx_list(value: list[Any]) -> Any:
    if Version(flax_version) >= Version("0.12.0"):
        return nnx.List(value)
    return value


@flax_module
class SeZMInteractionBlock(SeZMInteractionBlockDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"pre_so2_norm", "post_so2_norm"} and value is not None:
            value = (
                value
                if isinstance(value, EquivariantRMSNorm)
                else EquivariantRMSNorm.deserialize(value.serialize())
            )
        elif name in {"so2_conv"}:
            value = (
                value
                if isinstance(value, SO2Convolution)
                else SO2Convolution.deserialize(value.serialize())
            )
        elif name in {"pre_ffn_norms", "post_ffn_norms"}:
            value = [
                None
                if item is None
                else item
                if isinstance(item, EquivariantRMSNorm)
                else EquivariantRMSNorm.deserialize(item.serialize())
                for item in value
            ]
            value = _maybe_nnx_list(value)
        elif name in {"ffns"}:
            value = [
                item
                if isinstance(item, EquivariantFFN)
                else EquivariantFFN.deserialize(item.serialize())
                for item in value
            ]
            value = _maybe_nnx_list(value)
        return super().__setattr__(name, value)
