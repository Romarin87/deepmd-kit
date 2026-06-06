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
    GatedActivation as GatedActivationDP,
    SO2Convolution as SO2ConvolutionDP,
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
from deepmd.jax.descriptor.sezm_so3 import (
    ChannelLinear,
    FocusLinear,
    SO3Linear,
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


@flax_module
class GatedActivation(GatedActivationDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"gate_linear"} and value is not None:
            value = (
                value
                if isinstance(value, FocusLinear)
                else FocusLinear.deserialize(value.serialize())
            )
        elif name in {"expand_index"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
        return super().__setattr__(name, value)


@flax_module
class SO2Convolution(SO2ConvolutionDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {
            "coeff_index_m",
            "degree_index_m",
            "degree_index_full",
            "rotate_inv_rescale_full",
        }:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
        elif name in {"so2_linears"}:
            value = [
                item
                if isinstance(item, SO2Linear)
                else SO2Linear.deserialize(item.serialize())
                for item in value
            ]
            value = _maybe_nnx_list(value)
        elif name in {"non_linearities"}:
            value = [
                None
                if item is None
                else item
                if isinstance(item, GatedActivation)
                else GatedActivation.deserialize(item.serialize())
                for item in value
            ]
            value = _maybe_nnx_list(value)
        elif name in {"radial_hidden_proj"} and value is not None:
            value = (
                value
                if isinstance(value, ChannelLinear)
                else ChannelLinear.deserialize(value.serialize())
            )
        elif name in {"radial_degree_mixer"} and value is not None:
            value = (
                value
                if isinstance(value, DynamicRadialDegreeMixer)
                else DynamicRadialDegreeMixer.deserialize(value.serialize())
            )
        elif name in {"pre_focus_mix", "post_focus_mix"}:
            value = (
                value
                if isinstance(value, SO3Linear)
                else SO3Linear.deserialize(value.serialize())
            )
        return super().__setattr__(name, value)
