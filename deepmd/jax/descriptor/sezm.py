# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import numpy as np
from packaging.version import (
    Version,
)

from deepmd.dpmodel.descriptor.sezm import (
    C3CutoffEnvelope as C3CutoffEnvelopeDP,
    DescrptSeZM as DescrptSeZMDP,
    EnvironmentInitialEmbedding as EnvironmentInitialEmbeddingDP,
    GeometricInitialEmbedding as GeometricInitialEmbeddingDP,
    RadialBasis as RadialBasisDP,
    RadialMLP as RadialMLPDP,
    RMSNorm as RMSNormDP,
    SeZMTypeEmbedding as SeZMTypeEmbeddingDP,
)
from deepmd.dpmodel.descriptor.sezm_wignerd import (
    WignerDCalculator as WignerDCalculatorDP,
    build_edge_quaternion,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_rotation_matrix,
    quaternion_z_rotation,
)
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    to_jax_array,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.descriptor.sezm_block import (
    SeZMInteractionBlock,
)
from deepmd.jax.descriptor.sezm_ffn import (
    EquivariantFFN,
)
from deepmd.jax.env import (
    flax_version,
    nnx,
)
from deepmd.jax.utils.network import (
    ArrayAPIParam,
    NativeLayer,
)


def _maybe_nnx_list(value: list[Any]) -> Any:
    if Version(flax_version) >= Version("0.12.0"):
        return nnx.List(value)
    return value


def _freeze_static_arrays(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _freeze_static_arrays(value.tolist())
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_freeze_static_arrays(item) for item in value))
    if isinstance(value, list | tuple):
        return tuple(_freeze_static_arrays(item) for item in value)
    return value


@flax_module
class SeZMTypeEmbedding(SeZMTypeEmbeddingDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"embedding"}:
            value = to_jax_array(value)
            if value is not None:
                if getattr(self, "trainable", True):
                    value = ArrayAPIParam(value)
                else:
                    value = ArrayAPIVariable(value)
        return super().__setattr__(name, value)


@flax_module
class C3CutoffEnvelope(C3CutoffEnvelopeDP):
    pass


@flax_module
class RadialBasis(RadialBasisDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"freqs"}:
            value = to_jax_array(value)
            if value is not None:
                if getattr(self, "trainable", True):
                    value = ArrayAPIParam(value)
                else:
                    value = ArrayAPIVariable(value)
        elif name in {"envelope"}:
            value = C3CutoffEnvelope.deserialize(value.serialize())
        return super().__setattr__(name, value)


@flax_module
class RMSNorm(RMSNormDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"scale"}:
            value = to_jax_array(value)
            if value is not None:
                if getattr(self, "trainable", True):
                    value = ArrayAPIParam(value)
                else:
                    value = ArrayAPIVariable(value)
        return super().__setattr__(name, value)


@flax_module
class RadialMLP(RadialMLPDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"layers"}:
            value = [
                layer
                if isinstance(layer, NativeLayer)
                else NativeLayer.deserialize(layer.serialize())
                for layer in value
            ]
            value = _maybe_nnx_list(value)
        elif name in {"norms"}:
            value = [
                norm
                if isinstance(norm, RMSNorm)
                else RMSNorm.deserialize(norm.serialize())
                for norm in value
            ]
            value = _maybe_nnx_list(value)
        return super().__setattr__(name, value)


@flax_module
class GeometricInitialEmbedding(GeometricInitialEmbeddingDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {
            "non_scalar_row_index",
            "zonal_m0_col_index_for_row",
            "radial_slot_index_for_row",
        }:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
        return super().__setattr__(name, value)


@flax_module
class EnvironmentInitialEmbedding(EnvironmentInitialEmbeddingDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {
            "rbf_proj_layer1",
            "rbf_proj_layer2",
            "g_layer1",
            "g_layer2",
            "output_proj",
        }:
            if not isinstance(value, NativeLayer):
                value = NativeLayer.deserialize(value.serialize())
        elif name in {"env_type_embed"}:
            if not isinstance(value, SeZMTypeEmbedding):
                value = SeZMTypeEmbedding.deserialize(value.serialize())
        return super().__setattr__(name, value)


@flax_module
class WignerDCalculator(WignerDCalculatorDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"l1_perm", "l1_sign_outer"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
        elif name in {"poly_coeffs", "poly_basis"}:
            value = _freeze_static_arrays(value)
        return super().__setattr__(name, value)


@BaseDescriptor.register("SeZM")
@BaseDescriptor.register("sezm")
@BaseDescriptor.register("DPA4")
@BaseDescriptor.register("dpa4")
@flax_module
class DescrptSeZM(DescrptSeZMDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"mean", "stddev"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
            elif Version(flax_version) >= Version("0.12.0"):
                value = nnx.data(value)
        elif name in {"type_embedding"}:
            if not isinstance(value, SeZMTypeEmbedding):
                value = SeZMTypeEmbedding.deserialize(value.serialize())
        elif name in {"radial_basis"}:
            if not isinstance(value, RadialBasis):
                value = RadialBasis.deserialize(value.serialize())
        elif name in {"edge_envelope"}:
            if not isinstance(value, C3CutoffEnvelope):
                value = C3CutoffEnvelope.deserialize(value.serialize())
        elif name in {"radial_embedding"}:
            if not isinstance(value, RadialMLP):
                value = RadialMLP.deserialize(value.serialize())
        elif name in {"wigner_calc"}:
            if not isinstance(value, WignerDCalculator):
                value = WignerDCalculator.deserialize(value.serialize())
        elif name in {"env_seed_embedding"} and value is not None:
            if not isinstance(value, EnvironmentInitialEmbedding):
                value = EnvironmentInitialEmbedding.deserialize(value.serialize())
        elif name in {"film_scale_norm", "film_shift_norm"} and value is not None:
            if not isinstance(value, RMSNorm):
                value = RMSNorm.deserialize(value.serialize())
        elif name in {"film_scale_strength_log", "film_shift_strength_log"}:
            value = to_jax_array(value)
            if value is not None:
                if getattr(self, "trainable", True):
                    value = ArrayAPIParam(value)
                else:
                    value = ArrayAPIVariable(value)
        elif name in {"gie"} and value is not None:
            if not isinstance(value, GeometricInitialEmbedding):
                value = GeometricInitialEmbedding.deserialize(value.serialize())
        elif name in {"blocks"}:
            value = [
                block
                if isinstance(block, SeZMInteractionBlock)
                else SeZMInteractionBlock.deserialize(block.serialize())
                for block in value
            ]
            value = _maybe_nnx_list(value)
        elif name in {"output_ffn"} and value is not None:
            if not isinstance(value, EquivariantFFN):
                value = EquivariantFFN.deserialize(value.serialize())
        return super().__setattr__(name, value)
