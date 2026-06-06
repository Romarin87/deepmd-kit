# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.sezm_so3 import (
    ChannelLinear as ChannelLinearDP,
    FocusLinear as FocusLinearDP,
    SO3Linear as SO3LinearDP,
)
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    to_jax_array,
)
from deepmd.jax.utils.network import (
    ArrayAPIParam,
)


def _to_jax_parameter(owner: Any, value: Any) -> Any:
    value = to_jax_array(value)
    if value is None:
        return None
    if getattr(owner, "trainable", True):
        return ArrayAPIParam(value)
    return ArrayAPIVariable(value)


@flax_module
class ChannelLinear(ChannelLinearDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"weight", "bias"}:
            value = _to_jax_parameter(self, value)
        return super().__setattr__(name, value)


@flax_module
class FocusLinear(FocusLinearDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"weight", "bias"}:
            value = _to_jax_parameter(self, value)
        return super().__setattr__(name, value)


@flax_module
class SO3Linear(SO3LinearDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"weight", "bias"}:
            value = _to_jax_parameter(self, value)
        elif name in {"expand_index"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
        return super().__setattr__(name, value)
