# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

from typing import (
    Any,
)

from packaging.version import (
    Version,
)

from deepmd.dpmodel.descriptor.sezm_so2 import (
    DynamicRadialDegreeMixer as DynamicRadialDegreeMixerDP,
    SO2Linear as SO2LinearDP,
)
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    to_jax_array,
)
from deepmd.jax.env import (
    flax_version,
    nnx,
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


def _maybe_nnx_list(value: list[Any]) -> Any:
    if Version(flax_version) >= Version("0.12.0"):
        return nnx.List(value)
    return value


@flax_module
class SO2Linear(SO2LinearDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"weight_m0", "bias0"}:
            value = _to_jax_parameter(self, value)
        elif name in {"weight_m"}:
            value = [_to_jax_parameter(self, item) for item in value]
            value = _maybe_nnx_list(value)
        elif name in {"m0_idx", "pos_indices", "neg_indices"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
        return super().__setattr__(name, value)


@flax_module
class DynamicRadialDegreeMixer(DynamicRadialDegreeMixerDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"weight", "channel_basis"}:
            value = _to_jax_parameter(self, value)
        elif name in {"kernel_compact_index", "kernel_dense_index"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
        return super().__setattr__(name, value)
