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


def _pt_value(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.ndim > 1 and arr.shape[0] == 1:
        return np.squeeze(arr, axis=0)
    return value


def _set_native_layer_from_pt(layer: Any, variables: dict[str, Any], prefix: str) -> None:
    if f"{prefix}.matrix" in variables:
        layer.w = variables[f"{prefix}.matrix"]
    if f"{prefix}.bias" in variables:
        layer.b = variables[f"{prefix}.bias"]
    if f"{prefix}.idt" in variables:
        layer.idt = variables[f"{prefix}.idt"]


def _set_channel_linear_from_pt(
    layer: Any, variables: dict[str, Any], prefix: str
) -> None:
    if f"{prefix}.weight" in variables:
        layer.weight = variables[f"{prefix}.weight"]
    if f"{prefix}.bias" in variables:
        layer.bias = variables[f"{prefix}.bias"]


def _apply_pt_radial_embedding_state(
    radial_embedding: Any, variables: dict[str, Any]
) -> None:
    layer_idx = 0
    norm_idx = 0
    for key in sorted(variables):
        if not key.startswith("radial_embedding.net."):
            continue
        parts = key.split(".")
        if len(parts) < 4:
            continue
        module_idx = parts[2]
        if parts[3] == "matrix":
            _set_native_layer_from_pt(
                radial_embedding.layers[layer_idx],
                variables,
                f"radial_embedding.net.{module_idx}",
            )
            layer_idx += 1
        elif parts[3] == "adam_scale":
            radial_embedding.norms[norm_idx].scale = _pt_value(variables[key])
            norm_idx += 1


def _apply_pt_env_seed_state(
    env_seed_embedding: Any, variables: dict[str, Any]
) -> None:
    prefix = "env_seed_embedding."
    if f"{prefix}env_type_embed.adam_type_embedding" in variables:
        env_seed_embedding.env_type_embed.embedding = variables[
            f"{prefix}env_type_embed.adam_type_embedding"
        ]
    for name in (
        "rbf_proj_layer1",
        "rbf_proj_layer2",
        "g_layer1",
        "g_layer2",
        "output_proj",
    ):
        _set_native_layer_from_pt(
            getattr(env_seed_embedding, name),
            variables,
            f"{prefix}{name}",
        )


def _apply_pt_so3_linear_state(
    layer: Any, variables: dict[str, Any], prefix: str
) -> None:
    if f"{prefix}.weight" in variables:
        layer.weight = variables[f"{prefix}.weight"]
    if f"{prefix}.bias" in variables:
        layer.bias = variables[f"{prefix}.bias"]
    if f"{prefix}.expand_index" in variables:
        layer.expand_index = variables[f"{prefix}.expand_index"]


def _apply_pt_gated_activation_state(
    act: Any, variables: dict[str, Any], prefix: str
) -> None:
    if f"{prefix}.expand_index" in variables:
        act.expand_index = variables[f"{prefix}.expand_index"]
    if hasattr(act, "gate_linear") and act.gate_linear is not None:
        _set_channel_linear_from_pt(act.gate_linear, variables, f"{prefix}.gate_linear")


def _apply_pt_ffn_state(ffn: Any, variables: dict[str, Any], prefix: str) -> None:
    _apply_pt_so3_linear_state(ffn.so3_linear_1, variables, f"{prefix}.so3_linear_1")
    _apply_pt_so3_linear_state(ffn.so3_linear_2, variables, f"{prefix}.so3_linear_2")
    if f"{prefix}.act.scalar_gate.weight" in variables and hasattr(
        ffn.act, "scalar_gate"
    ):
        _set_channel_linear_from_pt(
            ffn.act.scalar_gate,
            variables,
            f"{prefix}.act.scalar_gate",
        )
    if hasattr(ffn.act, "projector") and ffn.act.projector is not None:
        if f"{prefix}.act.projector.to_grid_mat" in variables:
            ffn.act.projector.to_grid_mat = variables[
                f"{prefix}.act.projector.to_grid_mat"
            ]
        if f"{prefix}.act.projector.from_grid_mat" in variables:
            ffn.act.projector.from_grid_mat = variables[
                f"{prefix}.act.projector.from_grid_mat"
            ]
    if hasattr(ffn.act, "gate_linear"):
        _apply_pt_gated_activation_state(ffn.act, variables, f"{prefix}.act")


def _apply_pt_equivariant_norm_state(
    norm: Any | None, variables: dict[str, Any], prefix: str
) -> None:
    if norm is None:
        return
    for name in ("adam_scale", "bias", "expand_index", "balance_weight"):
        key = f"{prefix}.{name}"
        if key in variables:
            setattr(norm, name, variables[key])


def _apply_pt_scalar_norm_state(
    norm: Any | None, variables: dict[str, Any], prefix: str
) -> None:
    if norm is None:
        return
    if f"{prefix}.adam_scale" in variables:
        norm.adam_scale = variables[f"{prefix}.adam_scale"]


def _apply_pt_so2_linear_state(
    layer: Any, variables: dict[str, Any], prefix: str
) -> None:
    if f"{prefix}.weight_m0" in variables:
        layer.weight_m0 = variables[f"{prefix}.weight_m0"]
    if f"{prefix}.bias0" in variables:
        layer.bias0 = variables[f"{prefix}.bias0"]
    for name in ("m0_idx", "pos_indices", "neg_indices"):
        key = f"{prefix}.{name}"
        if key in variables:
            setattr(layer, name, variables[key])
    weights = []
    for idx in range(layer.mmax):
        key = f"{prefix}.weight_m.{idx}"
        if key in variables:
            weights.append(variables[key])
    if weights:
        layer.weight_m = weights


def _apply_pt_radial_degree_mixer_state(
    mixer: Any | None, variables: dict[str, Any], prefix: str
) -> None:
    if mixer is None:
        return
    for name in (
        "weight",
        "channel_basis",
        "kernel_compact_index",
        "kernel_dense_index",
    ):
        key = f"{prefix}.{name}"
        if key in variables:
            setattr(mixer, name, variables[key])


def _apply_pt_so2_conv_state(
    conv: Any, variables: dict[str, Any], prefix: str
) -> None:
    for name in ("coeff_index_m", "degree_index_m", "rotate_inv_rescale_full"):
        key = f"{prefix}.{name}"
        if key in variables:
            setattr(conv, name, variables[key])
    for idx, layer in enumerate(conv.so2_linears):
        _apply_pt_so2_linear_state(layer, variables, f"{prefix}.so2_linears.{idx}")
    for idx, act in enumerate(conv.non_linearities):
        if act is not None:
            _apply_pt_gated_activation_state(
                act,
                variables,
                f"{prefix}.non_linearities.{idx}",
            )
    if conv.radial_hidden_proj is not None:
        _set_channel_linear_from_pt(
            conv.radial_hidden_proj,
            variables,
            f"{prefix}.radial_hidden_proj",
        )
    _apply_pt_radial_degree_mixer_state(
        conv.radial_degree_mixer,
        variables,
        f"{prefix}.radial_degree_mixer",
    )
    _apply_pt_scalar_norm_state(conv.attn_qk_norm, variables, f"{prefix}.attn_qk_norm")
    _apply_pt_scalar_norm_state(
        conv.attn_output_gate_norm,
        variables,
        f"{prefix}.attn_output_gate_norm",
    )
    if conv.attn_q_proj is not None:
        _set_channel_linear_from_pt(conv.attn_q_proj, variables, f"{prefix}.attn_q_proj")
    if conv.attn_k_proj is not None:
        _set_channel_linear_from_pt(conv.attn_k_proj, variables, f"{prefix}.attn_k_proj")
    if f"{prefix}.adamw_attn_logit_w" in variables:
        conv.attn_logit_w = variables[f"{prefix}.adamw_attn_logit_w"]
    if f"{prefix}.adamw_attn_z_bias_raw" in variables:
        conv.attn_z_bias_raw = variables[f"{prefix}.adamw_attn_z_bias_raw"]
    if f"{prefix}.adamw_attn_gate_w" in variables:
        conv.attn_gate_w = variables[f"{prefix}.adamw_attn_gate_w"]
    _apply_pt_so3_linear_state(conv.pre_focus_mix, variables, f"{prefix}.pre_focus_mix")
    _apply_pt_so3_linear_state(
        conv.post_focus_mix,
        variables,
        f"{prefix}.post_focus_mix",
    )


def _apply_pt_block_state(block: Any, variables: dict[str, Any], prefix: str) -> None:
    _apply_pt_equivariant_norm_state(
        block.pre_so2_norm,
        variables,
        f"{prefix}.pre_so2_norm",
    )
    _apply_pt_equivariant_norm_state(
        block.post_so2_norm,
        variables,
        f"{prefix}.post_so2_norm",
    )
    _apply_pt_so2_conv_state(block.so2_conv, variables, f"{prefix}.so2_conv")
    for idx, norm in enumerate(block.pre_ffn_norms):
        _apply_pt_equivariant_norm_state(
            norm,
            variables,
            f"{prefix}.pre_ffn_norms.{idx}",
        )
    for idx, norm in enumerate(block.post_ffn_norms):
        _apply_pt_equivariant_norm_state(
            norm,
            variables,
            f"{prefix}.post_ffn_norms.{idx}",
        )
    for idx, ffn in enumerate(block.ffns):
        _apply_pt_ffn_state(ffn, variables, f"{prefix}.ffns.{idx}")


def _apply_pt_flat_descriptor_state(obj: Any, variables: dict[str, Any]) -> None:
    if "type_embedding.adam_type_embedding" in variables:
        obj.type_embedding.embedding = variables["type_embedding.adam_type_embedding"]
    if "radial_basis.adam_freqs" in variables:
        obj.radial_basis.freqs = variables["radial_basis.adam_freqs"]
    _apply_pt_radial_embedding_state(obj.radial_embedding, variables)
    if obj.env_seed_embedding is not None:
        _apply_pt_env_seed_state(obj.env_seed_embedding, variables)
    if obj.film_scale_norm is not None and "film_scale_norm.adam_scale" in variables:
        obj.film_scale_norm.scale = _pt_value(variables["film_scale_norm.adam_scale"])
    if obj.film_shift_norm is not None and "film_shift_norm.adam_scale" in variables:
        obj.film_shift_norm.scale = _pt_value(variables["film_shift_norm.adam_scale"])
    if "film_scale_strength_log" in variables:
        obj.film_scale_strength_log = variables["film_scale_strength_log"]
    if "film_shift_strength_log" in variables:
        obj.film_shift_strength_log = variables["film_shift_strength_log"]
    if obj.gie is not None:
        for name in (
            "non_scalar_row_index",
            "zonal_m0_col_index_for_row",
            "radial_slot_index_for_row",
        ):
            key = f"gie.{name}"
            if key in variables:
                setattr(obj.gie, name, variables[key])
    if obj.output_ffn is not None:
        _apply_pt_ffn_state(obj.output_ffn, variables, "output_ffn")
    for idx, block in enumerate(obj.blocks):
        _apply_pt_block_state(block, variables, f"blocks.{idx}")


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

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "DescrptSeZM":
        obj = super().deserialize(data)
        variables = data.get("@variables", {})
        if "type_embedding.adam_type_embedding" in variables:
            _apply_pt_flat_descriptor_state(obj, variables)
        return obj
