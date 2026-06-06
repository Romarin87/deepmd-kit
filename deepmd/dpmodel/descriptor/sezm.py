# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM/DPA4 descriptor scaffold for dpmodel and JAX backends."""

from __future__ import annotations

from typing import (
    Any,
    NamedTuple,
)

import math

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    Array,
    xp_add_at,
    xp_bincount,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.network import (
    NativeLayer,
    get_activation_fn,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.dpmodel.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.finetune import (
    map_pair_exclude_types,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .sezm_block import (
    SeZMInteractionBlock,
)
from .sezm_ffn import (
    EquivariantFFN,
)
from .sezm_indexing import (
    get_so3_dim_of_lmax,
    map_degree_idx,
)
from .sezm_wignerd import (
    WignerDCalculator,
    build_edge_quaternion,
    quaternion_multiply,
    quaternion_z_rotation,
)


SUPPORTED_RADIAL_SO2_MODE = "degree_channel"
SUPPORTED_LMAX = 3
SUPPORTED_MMAX = 1


class EdgeFeatureCache(NamedTuple):
    src: Array
    dst: Array
    edge_type_feat: Array
    edge_vec: Array
    edge_len: Array
    edge_rbf: Array
    edge_env: Array
    deg: Array
    inv_sqrt_deg: Array
    D_full: Array | None
    Dt_full: Array | None


class SeZMTypeEmbedding(NativeOP):
    def __init__(
        self,
        *,
        ntypes: int,
        embed_dim: int,
        precision: str = DEFAULT_PRECISION,
        seed: int | list[int] | None = None,
        trainable: bool = True,
        padding: bool = True,
    ) -> None:
        self.ntypes = int(ntypes)
        self.embed_dim = int(embed_dim)
        self.precision = precision
        self.seed = seed
        self.trainable = bool(trainable)
        self.padding = bool(padding)
        if self.ntypes <= 0:
            raise ValueError("`ntypes` must be positive")
        if self.embed_dim <= 0:
            raise ValueError("`embed_dim` must be positive")
        n_rows = self.ntypes + int(self.padding)
        rng = np.random.default_rng(child_seed(seed, 0))
        init_std = 1.0 / np.sqrt(float(self.ntypes + self.embed_dim))
        embedding = rng.normal(0.0, init_std, (n_rows, self.embed_dim)).astype(
            PRECISION_DICT[self.precision.lower()]
        )
        if self.padding:
            embedding[self.ntypes, :] = 0.0
        self.embedding = embedding

    def call(self, atype: Array) -> Array:
        embedding = self.embedding[...]
        xp = array_api_compat.array_namespace(atype, embedding)
        index = xp.astype(atype, xp.int64)
        if self.padding:
            index = xp.where(index < 0, xp.asarray(self.ntypes, dtype=xp.int64), index)
        return xp.take(embedding, index, axis=0)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SeZMTypeEmbedding",
            "@version": 1,
            "ntypes": self.ntypes,
            "embed_dim": self.embed_dim,
            "precision": self.precision,
            "seed": self.seed,
            "trainable": self.trainable,
            "padding": self.padding,
            "@variables": {
                "embedding": to_numpy_array(self.embedding[...]),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SeZMTypeEmbedding":
        data = data.copy()
        data.pop("@class", None)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        variables = data.pop("@variables")
        obj = cls(**data)
        obj.embedding = variables["embedding"]
        return obj


class C3CutoffEnvelope(NativeOP):
    def __init__(
        self,
        rcut: float,
        exponent: int = 5,
        precision: str = DEFAULT_PRECISION,
    ) -> None:
        if rcut <= 0.0:
            raise ValueError("`rcut` must be positive")
        if exponent <= 0:
            raise ValueError("`exponent` must be positive")
        self.rcut = float(rcut)
        self.exponent = int(exponent)
        self.precision = precision
        p = self.exponent
        self.coeff_a = -((p + 1) * (p + 2) * (p + 3)) / 6.0
        self.coeff_b = (p * (p + 2) * (p + 3)) / 2.0
        self.coeff_c = -(p * (p + 1) * (p + 3)) / 2.0
        self.coeff_d = (p * (p + 1) * (p + 2)) / 6.0

    def call(self, distance: Array) -> Array:
        xp = array_api_compat.array_namespace(distance)
        scaled = xp.clip(distance / self.rcut, 0.0, 1.0)
        poly = self.coeff_a + scaled * (
            self.coeff_b + scaled * (self.coeff_c + scaled * self.coeff_d)
        )
        value = 1.0 + scaled**self.exponent * poly
        return xp.where(scaled < 1.0, value, xp.zeros_like(value))

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "C3CutoffEnvelope",
            "@version": 1,
            "rcut": self.rcut,
            "exponent": self.exponent,
            "precision": self.precision,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "C3CutoffEnvelope":
        data = data.copy()
        data.pop("@class", None)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        return cls(**data)


class RadialBasis(NativeOP):
    def __init__(
        self,
        rcut: float,
        basis_type: str = "bessel",
        n_radial: int = 16,
        precision: str = DEFAULT_PRECISION,
        exponent: int = 7,
        trainable: bool = True,
    ) -> None:
        self.rcut = float(rcut)
        self.basis_type = str(basis_type).lower()
        self.n_radial = int(n_radial)
        self.precision = precision
        self.exponent = int(exponent)
        self.trainable = bool(trainable)
        if self.rcut <= 0.0:
            raise ValueError("`rcut` must be positive")
        if self.n_radial <= 0:
            raise ValueError("`n_radial` must be positive")
        if self.basis_type not in {"bessel", "gaussian"}:
            raise ValueError("`basis_type` must be either 'bessel' or 'gaussian'")
        dtype = PRECISION_DICT[self.precision.lower()]
        if self.basis_type == "bessel":
            freqs = np.arange(1, self.n_radial + 1, dtype=dtype) * (
                np.pi / self.rcut
            )
        else:
            freqs = np.linspace(0.0, self.rcut, self.n_radial, dtype=dtype)
        self.freqs = freqs.reshape(1, self.n_radial)
        gaussian_width = self.rcut / max(self.n_radial - 1, 1)
        self.gaussian_coeff = -0.5 / (gaussian_width * gaussian_width)
        self.envelope = C3CutoffEnvelope(
            rcut=self.rcut,
            exponent=self.exponent,
            precision=self.precision,
        )

    def call(self, distance: Array) -> Array:
        freqs = self.freqs[...]
        xp = array_api_compat.array_namespace(distance, freqs)
        if self.basis_type == "bessel":
            x = distance * freqs
            small_x = xp.abs(x) < 1e-7
            safe_x = xp.where(small_x, xp.ones_like(x), x)
            raw_sinc = xp.where(
                small_x,
                1.0 - (x * x) / 6.0,
                xp.sin(x) / safe_x,
            )
            raw = freqs * raw_sinc
        else:
            dr = distance - freqs
            raw = xp.exp(dr * dr * self.gaussian_coeff)
        return raw * self.envelope(distance)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "RadialBasis",
            "@version": 1,
            "config": {
                "rcut": self.rcut,
                "basis_type": self.basis_type,
                "n_radial": self.n_radial,
                "precision": self.precision,
                "exponent": self.exponent,
                "trainable": self.trainable,
            },
            "@variables": {
                "freqs": to_numpy_array(self.freqs[...]),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "RadialBasis":
        data = data.copy()
        data.pop("@class", None)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables", {})
        obj = cls(**config)
        if "freqs" in variables:
            obj.freqs = variables["freqs"]
        return obj


class RMSNorm(NativeOP):
    def __init__(
        self,
        *,
        channels: int,
        eps: float = 1e-7,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
    ) -> None:
        self.channels = int(channels)
        self.eps = float(eps)
        self.precision = precision
        self.trainable = bool(trainable)
        self.scale = np.ones(self.channels, dtype=PRECISION_DICT[precision.lower()])

    def call(self, x: Array) -> Array:
        scale = self.scale[...]
        xp = array_api_compat.array_namespace(x, scale)
        inv_rms = 1.0 / xp.sqrt(xp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * inv_rms * scale

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "RMSNorm",
            "@version": 1,
            "channels": self.channels,
            "eps": self.eps,
            "precision": self.precision,
            "trainable": self.trainable,
            "@variables": {
                "scale": to_numpy_array(self.scale[...]),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "RMSNorm":
        data = data.copy()
        data.pop("@class", None)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        variables = data.pop("@variables")
        obj = cls(**data)
        obj.scale = variables["scale"]
        return obj


class RadialMLP(NativeOP):
    def __init__(
        self,
        mlp_layers: list[int],
        *,
        activation_function: str = "silu",
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        if len(mlp_layers) < 2:
            raise ValueError("`mlp_layers` must have at least two entries")
        self.mlp_layers = [int(x) for x in mlp_layers]
        self.activation_function = str(activation_function)
        self.precision = precision
        self.trainable = bool(trainable)
        self.layers = [
            NativeLayer(
                self.mlp_layers[ii],
                self.mlp_layers[ii + 1],
                bias=False,
                use_timestep=False,
                activation_function=None,
                resnet=False,
                precision=self.precision,
                seed=child_seed(seed, ii),
                trainable=self.trainable,
            )
            for ii in range(len(self.mlp_layers) - 1)
        ]
        self.norms = [
            RMSNorm(
                channels=self.mlp_layers[ii + 1],
                precision=self.precision,
                trainable=self.trainable,
            )
            for ii in range(len(self.mlp_layers) - 2)
        ]

    def call(self, x: Array) -> Array:
        activation = get_activation_fn(self.activation_function)
        for ii, layer in enumerate(self.layers):
            x = layer(x)
            if ii < len(self.layers) - 1:
                x = activation(self.norms[ii](x))
        return x

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "RadialMLP",
            "@version": 1,
            "mlp_layers": self.mlp_layers,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "trainable": self.trainable,
            "layers": [layer.serialize() for layer in self.layers],
            "norms": [norm.serialize() for norm in self.norms],
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "RadialMLP":
        data = data.copy()
        data.pop("@class", None)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        layers = data.pop("layers")
        norms = data.pop("norms")
        obj = cls(**data)
        obj.layers = [NativeLayer.deserialize(layer) for layer in layers]
        obj.norms = [RMSNorm.deserialize(norm) for norm in norms]
        return obj


class GeometricInitialEmbedding(NativeOP):
    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        precision: str = DEFAULT_PRECISION,
    ) -> None:
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.precision = precision
        self.ebed_dim = get_so3_dim_of_lmax(self.lmax)
        if self.lmax > 0:
            packed_degree = map_degree_idx(self.lmax)
            self.non_scalar_row_index = np.arange(
                1,
                self.ebed_dim,
                dtype=np.int64,
            )
            non_scalar_degree = packed_degree[1:]
            self.zonal_m0_col_index_for_row = (
                non_scalar_degree * (non_scalar_degree + 1)
            ).astype(np.int64)
            self.radial_slot_index_for_row = (non_scalar_degree - 1).astype(np.int64)
        else:
            self.non_scalar_row_index = np.empty(0, dtype=np.int64)
            self.zonal_m0_col_index_for_row = np.empty(0, dtype=np.int64)
            self.radial_slot_index_for_row = np.empty(0, dtype=np.int64)

    def call(
        self,
        *,
        n_nodes: int,
        edge_cache: EdgeFeatureCache,
        radial_feat: Array,
    ) -> Array:
        xp = array_api_compat.array_namespace(edge_cache.edge_vec, radial_feat)
        out = xp.zeros(
            (n_nodes, self.ebed_dim, self.channels),
            dtype=edge_cache.edge_vec.dtype,
        )
        if self.lmax == 0 or edge_cache.dst.shape[0] == 0:
            return out
        if edge_cache.Dt_full is None:
            raise ValueError("GeometricInitialEmbedding requires Dt_full in edge cache")
        row_idx = xp.asarray(self.non_scalar_row_index[...], dtype=xp.int64)
        col_idx = xp.asarray(self.zonal_m0_col_index_for_row[...], dtype=xp.int64)
        slot_idx = xp.asarray(self.radial_slot_index_for_row[...], dtype=xp.int64)
        zonal = edge_cache.Dt_full[:, row_idx, col_idx]
        radial_per_row = xp.take(radial_feat, slot_idx, axis=1)
        message = zonal[..., None] * radial_per_row
        non_scalar_out = xp.zeros(
            (n_nodes, row_idx.shape[0], self.channels),
            dtype=message.dtype,
        )
        non_scalar_out = xp_add_at(non_scalar_out, edge_cache.dst, message)
        if hasattr(out, "at"):
            out = out.at[:, row_idx, :].set(non_scalar_out)
        else:
            out[:, row_idx, :] = non_scalar_out
        return out * edge_cache.inv_sqrt_deg

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "GeometricInitialEmbedding",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "channels": self.channels,
                "precision": self.precision,
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "GeometricInitialEmbedding":
        data = data.copy()
        data_cls = data.pop("@class", None)
        if data_cls != "GeometricInitialEmbedding":
            raise ValueError(f"Invalid class for GeometricInitialEmbedding: {data_cls}")
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        return cls(**data.pop("config"))


class EnvironmentInitialEmbedding(NativeOP):
    def __init__(
        self,
        *,
        ntypes: int,
        n_radial: int,
        channels: int,
        embed_dim: int = 64,
        axis_dim: int = 8,
        type_dim: int = 16,
        hidden_dim: int = 64,
        mlp_bias: bool = False,
        activation_function: str = "silu",
        eps: float = 1e-7,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.ntypes = int(ntypes)
        self.n_radial = int(n_radial)
        self.channels = int(channels)
        self.embed_dim = int(embed_dim)
        self.axis_dim = int(axis_dim)
        self.type_dim = int(type_dim)
        self.hidden_dim = int(hidden_dim)
        self.mlp_bias = bool(mlp_bias)
        self.activation_function = str(activation_function)
        self.eps = float(eps)
        self.precision = precision
        self.trainable = bool(trainable)
        if self.axis_dim >= self.embed_dim:
            raise ValueError("`axis_dim` must be < `embed_dim`")
        self.rbf_out_dim = max(32, self.embed_dim - 2 * self.type_dim)

        seed_rbf_proj = child_seed(seed, 0)
        self.rbf_proj_layer1 = NativeLayer(
            self.n_radial,
            self.rbf_out_dim,
            bias=self.mlp_bias,
            activation_function=self.activation_function,
            precision=self.precision,
            trainable=self.trainable,
            seed=child_seed(seed_rbf_proj, 0),
        )
        self.rbf_proj_layer2 = NativeLayer(
            self.rbf_out_dim,
            self.rbf_out_dim,
            bias=self.mlp_bias,
            activation_function=None,
            precision=self.precision,
            trainable=self.trainable,
            seed=child_seed(seed_rbf_proj, 1),
        )
        self.env_type_embed = SeZMTypeEmbedding(
            ntypes=self.ntypes,
            embed_dim=self.type_dim,
            precision=self.precision,
            seed=child_seed(seed, 1),
            trainable=self.trainable,
        )
        g_in_dim = self.rbf_out_dim + 2 * self.type_dim
        seed_g_net = child_seed(seed, 2)
        self.g_layer1 = NativeLayer(
            g_in_dim,
            self.hidden_dim,
            bias=self.mlp_bias,
            activation_function=self.activation_function,
            precision=self.precision,
            trainable=self.trainable,
            seed=child_seed(seed_g_net, 0),
        )
        self.g_layer2 = NativeLayer(
            self.hidden_dim,
            self.embed_dim,
            bias=self.mlp_bias,
            activation_function=None,
            precision=self.precision,
            trainable=self.trainable,
            seed=child_seed(seed_g_net, 1),
        )
        self.output_proj = NativeLayer(
            self.embed_dim * self.axis_dim,
            2 * self.channels,
            bias=False,
            activation_function=None,
            precision=self.precision,
            trainable=self.trainable,
            seed=child_seed(seed, 3),
        )
        self.output_proj.w = np.zeros_like(self.output_proj.w)

    def call(
        self,
        *,
        edge_cache: EdgeFeatureCache,
        atype_flat: Array,
        n_nodes: int,
    ) -> Array:
        xp = array_api_compat.array_namespace(
            edge_cache.edge_vec,
            edge_cache.edge_rbf,
            atype_flat,
        )
        edge_vec = edge_cache.edge_vec
        edge_rbf = edge_cache.edge_rbf
        edge_env = edge_cache.edge_env

        r_sq = xp.sum(edge_vec * edge_vec, axis=-1, keepdims=True)
        inv_r = 1.0 / xp.sqrt(r_sq + self.eps * self.eps)
        s = edge_env * inv_r
        r_hat = edge_vec * inv_r
        r_tilde = xp.concat([s, s * r_hat], axis=-1)

        atype_src = xp.take(atype_flat, edge_cache.src, axis=0)
        atype_dst = xp.take(atype_flat, edge_cache.dst, axis=0)
        type_src = self.env_type_embed(atype_src)
        type_dst = self.env_type_embed(atype_dst)
        rbf_proj = self.rbf_proj_layer2(self.rbf_proj_layer1(edge_rbf))
        g_input = xp.concat([rbf_proj, type_src, type_dst], axis=-1)
        g = self.g_layer2(self.g_layer1(g_input))

        outer = xp.einsum("ei,ej->eij", r_tilde, g)
        outer_flat = xp.reshape(outer, (outer.shape[0], 4 * self.embed_dim))
        env_agg = xp.zeros(
            (n_nodes, 4 * self.embed_dim),
            dtype=outer_flat.dtype,
        )
        env_agg = xp_add_at(env_agg, edge_cache.dst, outer_flat)
        env_agg = xp.reshape(env_agg, (n_nodes, 4, self.embed_dim))
        env_agg = env_agg * edge_cache.inv_sqrt_deg
        env_agg_t = xp.permute_dims(env_agg, (0, 2, 1))
        env_agg_axis = env_agg[:, :, : self.axis_dim]
        d_matrix = xp.matmul(env_agg_t, env_agg_axis)
        d_flat = xp.reshape(d_matrix, (n_nodes, self.embed_dim * self.axis_dim))
        return self.output_proj(d_flat)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "EnvironmentInitialEmbedding",
            "@version": 1,
            "config": {
                "ntypes": self.ntypes,
                "n_radial": self.n_radial,
                "channels": self.channels,
                "embed_dim": self.embed_dim,
                "axis_dim": self.axis_dim,
                "type_dim": self.type_dim,
                "hidden_dim": self.hidden_dim,
                "mlp_bias": self.mlp_bias,
                "activation_function": self.activation_function,
                "eps": self.eps,
                "precision": self.precision,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": {
                "rbf_proj_layer1": self.rbf_proj_layer1.serialize(),
                "rbf_proj_layer2": self.rbf_proj_layer2.serialize(),
                "env_type_embed": self.env_type_embed.serialize(),
                "g_layer1": self.g_layer1.serialize(),
                "g_layer2": self.g_layer2.serialize(),
                "output_proj": self.output_proj.serialize(),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "EnvironmentInitialEmbedding":
        data = data.copy()
        data_cls = data.pop("@class", None)
        if data_cls != "EnvironmentInitialEmbedding":
            raise ValueError(f"Invalid class for EnvironmentInitialEmbedding: {data_cls}")
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj.rbf_proj_layer1 = NativeLayer.deserialize(variables["rbf_proj_layer1"])
        obj.rbf_proj_layer2 = NativeLayer.deserialize(variables["rbf_proj_layer2"])
        obj.env_type_embed = SeZMTypeEmbedding.deserialize(variables["env_type_embed"])
        obj.g_layer1 = NativeLayer.deserialize(variables["g_layer1"])
        obj.g_layer2 = NativeLayer.deserialize(variables["g_layer2"])
        obj.output_proj = NativeLayer.deserialize(variables["output_proj"])
        return obj


def _normalize_bool_pair(value: bool | list[bool] | None, default: bool) -> list[bool]:
    if value is None:
        value = default
    if isinstance(value, bool):
        return [value, value]
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("expected a bool or a list[bool] of length 2")
    if any(not isinstance(flag, bool) for flag in value):
        raise ValueError("expected a bool or a list[bool] of length 2")
    return list(value)


@BaseDescriptor.register("SeZM")
@BaseDescriptor.register("sezm")
@BaseDescriptor.register("DPA4")
@BaseDescriptor.register("dpa4")
class DescrptSeZM(NativeOP, BaseDescriptor):
    """JAX/dpmodel SeZM descriptor interface.

    The eager forward port is intentionally added in stages. This class first
    establishes the exact constructor, aliases, serialization, and model wiring
    for the conservative energy path used by DPA4/SeZM.
    """

    LATEST_VERSION = 1.1

    def __init__(
        self,
        ntypes: int,
        sel: list[int] | int,
        rcut: float = 6.0,
        env_exp: list[int] | None = None,
        channels: int = 64,
        basis_type: str = "bessel",
        n_radial: int = 16,
        radial_mlp: list[int] | None = None,
        use_env_seed: bool = True,
        random_gamma: bool = True,
        lmax: int = 3,
        l_schedule: list[int] | None = None,
        mmax: int | None = 1,
        m_schedule: list[int] | None = None,
        n_blocks: int = 3,
        so2_norm: bool = False,
        so2_layers: int = 4,
        so2_attn_res: str = "none",
        radial_so2_mode: str = "degree_channel",
        radial_so2_rank: int = 1,
        n_focus: int = 1,
        focus_dim: int = 0,
        n_atten_head: int = 1,
        atten_f_mix: bool = False,
        atten_v_proj: bool = False,
        atten_o_proj: bool = False,
        ffn_neurons: int = 0,
        grid_mlp: bool = False,
        ffn_blocks: int = 1,
        sandwich_norm: list[bool] | None = None,
        mlp_bias: bool = False,
        layer_scale: bool = False,
        full_attn_res: str = "none",
        block_attn_res: str = "none",
        s2_activation: list[bool] | None = None,
        lebedev_quadrature: bool | list[bool] | None = True,
        activation_function: str = "silu",
        glu_activation: bool = True,
        use_amp: bool = True,
        exclude_types: list[tuple[int, int]] | None = None,
        precision: str = DEFAULT_PRECISION,
        eps: float = 1e-7,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        type_map: list[str] | None = None,
        inner_clamp_r_inner: float | None = None,
        inner_clamp_r_outer: float | None = None,
        add_chg_spin_ebd: bool = False,
        default_chg_spin: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("_comment", None)
        if kwargs:
            raise TypeError(f"Unsupported SeZM descriptor options: {kwargs}")
        self.version = self.LATEST_VERSION
        self.ntypes = int(ntypes)
        self.sel = [int(sel)] if isinstance(sel, int) else [int(x) for x in sel]
        self.rcut = float(rcut)
        self.env_exp = [7, 5] if env_exp is None else [int(x) for x in env_exp]
        if len(self.env_exp) != 2:
            raise ValueError("`env_exp` must contain [rbf_env_exp, edge_env_exp]")
        self.channels = int(channels)
        self.basis_type = str(basis_type).lower()
        self.n_radial = int(n_radial)
        radial_mlp = [0] if radial_mlp is None else [int(x) for x in radial_mlp]
        self.radial_mlp = [self.channels if x == 0 else x for x in radial_mlp]
        self.use_env_seed = bool(use_env_seed)
        self.random_gamma = bool(random_gamma)
        self.env_seed_embed_dim = min(self.channels, 128)
        self.env_seed_type_dim = min(32, max(8, self.channels // 4))
        axis_dim = 4 if self.env_seed_embed_dim < 64 else 8
        self.env_seed_axis_dim = min(
            axis_dim,
            max(1, self.env_seed_embed_dim - 1),
        )
        rbf_out_dim = max(
            32,
            self.env_seed_embed_dim - 2 * self.env_seed_type_dim,
        )
        g_in_dim = rbf_out_dim + 2 * self.env_seed_type_dim
        self.env_seed_hidden_dim = min(
            256,
            max(2 * self.env_seed_embed_dim, g_in_dim),
        )
        self.lmax = int(lmax)
        self.n_blocks = int(n_blocks)
        self.l_schedule = (
            [self.lmax for _ in range(self.n_blocks)]
            if l_schedule is None
            else [int(x) for x in l_schedule]
        )
        self.mmax = self.lmax if mmax is None else int(mmax)
        self.m_schedule = (
            [self.mmax for _ in self.l_schedule]
            if m_schedule is None
            else [int(x) for x in m_schedule]
        )
        self.so2_norm = bool(so2_norm)
        self.so2_layers = int(so2_layers)
        self.so2_attn_res = str(so2_attn_res).lower()
        self.radial_so2_mode = str(radial_so2_mode).lower()
        self.radial_so2_rank = int(radial_so2_rank)
        self.n_focus = int(n_focus)
        self.focus_dim = int(focus_dim)
        self.n_atten_head = int(n_atten_head)
        self.atten_f_mix = bool(atten_f_mix)
        self.atten_v_proj = bool(atten_v_proj)
        self.atten_o_proj = bool(atten_o_proj)
        self.ffn_neurons = int(ffn_neurons)
        self.grid_mlp = bool(grid_mlp)
        self.ffn_blocks = int(ffn_blocks)
        self.sandwich_norm = (
            [False, True, True, False]
            if sandwich_norm is None
            else [bool(x) for x in sandwich_norm]
        )
        self.mlp_bias = bool(mlp_bias)
        self.layer_scale = bool(layer_scale)
        self.full_attn_res = str(full_attn_res).lower()
        self.block_attn_res = str(block_attn_res).lower()
        self.s2_activation = (
            [False, True] if s2_activation is None else [bool(x) for x in s2_activation]
        )
        self.lebedev_quadrature = _normalize_bool_pair(lebedev_quadrature, True)
        self.activation_function = str(activation_function)
        self.glu_activation = bool(glu_activation)
        self.so2_s2_activation = self.s2_activation[0]
        self.ffn_s2_activation = self.s2_activation[1]
        self.so2_lebedev_quadrature = self.lebedev_quadrature[0]
        self.ffn_lebedev_quadrature = self.lebedev_quadrature[1]
        self.so2_activation_function = (
            "silu" if self.so2_s2_activation else self.activation_function
        )
        self.ffn_activation_function = (
            "silu" if self.ffn_s2_activation else self.activation_function
        )
        self.ffn_glu_activation = (
            True if self.ffn_s2_activation else self.glu_activation
        )
        self.out_activation_function = self.activation_function
        self.out_glu_activation = self.glu_activation
        self.use_amp = bool(use_amp)
        self.exclude_types = [] if exclude_types is None else list(exclude_types)
        self.precision = str(precision)
        if self.precision.lower() not in PRECISION_DICT:
            raise ValueError(f"Unsupported precision {precision!r}")
        self.eps = float(eps)
        self.trainable = bool(trainable)
        self.seed = seed
        self.type_map = type_map
        self.inner_clamp_r_inner = inner_clamp_r_inner
        self.inner_clamp_r_outer = inner_clamp_r_outer
        self.add_chg_spin_ebd = bool(add_chg_spin_ebd)
        self.default_chg_spin = (
            None if default_chg_spin is None else [float(x) for x in default_chg_spin]
        )
        self.mean = np.zeros(0, dtype=PRECISION_DICT[self.precision.lower()])
        self.stddev = np.ones(0, dtype=PRECISION_DICT[self.precision.lower()])
        self._validate_v1_path()
        self.type_embedding = SeZMTypeEmbedding(
            ntypes=self.ntypes,
            embed_dim=self.channels,
            precision=self.precision,
            seed=child_seed(self.seed, 0),
            trainable=self.trainable,
        )
        self.radial_basis = RadialBasis(
            rcut=self.rcut,
            basis_type=self.basis_type,
            n_radial=self.n_radial,
            precision=self.precision,
            exponent=self.env_exp[0],
            trainable=self.trainable,
        )
        self.edge_envelope = C3CutoffEnvelope(
            rcut=self.rcut,
            exponent=self.env_exp[1],
            precision=self.precision,
        )
        radial_out_dim = (self.lmax + 1) * self.channels
        self.radial_embedding = RadialMLP(
            [self.n_radial, *self.radial_mlp, radial_out_dim],
            activation_function=self.activation_function,
            precision=self.precision,
            trainable=self.trainable,
            seed=child_seed(self.seed, 3),
        )
        self.wigner_calc = WignerDCalculator(
            lmax=self.l_schedule[0],
            eps=self.eps,
            precision=self.precision,
        )
        if self.use_env_seed:
            self.env_seed_embedding = EnvironmentInitialEmbedding(
                ntypes=self.ntypes,
                n_radial=self.n_radial,
                channels=self.channels,
                embed_dim=self.env_seed_embed_dim,
                axis_dim=self.env_seed_axis_dim,
                type_dim=self.env_seed_type_dim,
                hidden_dim=self.env_seed_hidden_dim,
                mlp_bias=self.mlp_bias,
                activation_function=self.activation_function,
                eps=self.eps,
                precision=self.precision,
                trainable=self.trainable,
                seed=child_seed(self.seed, 4),
            )
            self.film_scale_norm = RMSNorm(
                channels=self.channels,
                eps=self.eps,
                precision=self.precision,
                trainable=self.trainable,
            )
            self.film_shift_norm = RMSNorm(
                channels=self.channels,
                eps=self.eps,
                precision=self.precision,
                trainable=self.trainable,
            )
            dtype = PRECISION_DICT[self.precision.lower()]
            self.film_scale_strength_log = np.asarray(
                [math.log(0.01)],
                dtype=dtype,
            )
            self.film_shift_strength_log = np.asarray(
                [math.log(0.01)],
                dtype=dtype,
            )
        else:
            self.env_seed_embedding = None
            self.film_scale_norm = None
            self.film_shift_norm = None
            self.film_scale_strength_log = None
            self.film_shift_strength_log = None
        self.use_gie = self.use_env_seed and self.l_schedule[0] > 0
        self.gie = (
            GeometricInitialEmbedding(
                lmax=self.l_schedule[0],
                channels=self.channels,
                precision=self.precision,
            )
            if self.use_gie
            else None
        )
        self.block_ffn_neurons = self._resolve_ffn_neurons(
            self.ffn_neurons,
            glu_activation=self.ffn_glu_activation,
        )
        self.out_ffn_neurons = self._resolve_ffn_neurons(
            self.ffn_neurons,
            glu_activation=self.out_glu_activation,
        )
        self._baseline_forward_supported = self._is_baseline_forward_supported()
        if self._baseline_forward_supported:
            self.blocks = [
                SeZMInteractionBlock(
                    lmax=ll,
                    mmax=mm,
                    channels=self.channels,
                    n_focus=self.n_focus,
                    focus_dim=self.focus_dim,
                    focus_compete=False,
                    so2_norm=self.so2_norm,
                    so2_layers=self.so2_layers,
                    so2_attn_res=self.so2_attn_res,
                    radial_so2_mode=self.radial_so2_mode,
                    radial_so2_rank=self.radial_so2_rank,
                    n_atten_head=self.n_atten_head,
                    atten_f_mix=self.atten_f_mix,
                    atten_v_proj=self.atten_v_proj,
                    atten_o_proj=self.atten_o_proj,
                    so2_pre_norm=self.sandwich_norm[0],
                    so2_post_norm=self.sandwich_norm[1],
                    ffn_pre_norm=self.sandwich_norm[2],
                    ffn_post_norm=self.sandwich_norm[3],
                    ffn_neurons=self.block_ffn_neurons,
                    grid_mlp=self.grid_mlp,
                    ffn_blocks=self.ffn_blocks,
                    layer_scale=self.layer_scale,
                    full_attn_res=self.full_attn_res,
                    block_attn_res=self.block_attn_res,
                    so2_s2_activation=self.so2_s2_activation,
                    ffn_s2_activation=self.ffn_s2_activation,
                    so2_lebedev_quadrature=self.so2_lebedev_quadrature,
                    ffn_lebedev_quadrature=self.ffn_lebedev_quadrature,
                    so2_activation_function=self.so2_activation_function,
                    ffn_activation_function=self.ffn_activation_function,
                    ffn_glu_activation=self.ffn_glu_activation,
                    mlp_bias=self.mlp_bias,
                    eps=self.eps,
                    precision=self.precision,
                    trainable=self.trainable,
                    seed=child_seed(self.seed, 10 + ii),
                )
                for ii, (ll, mm) in enumerate(zip(self.l_schedule, self.m_schedule))
            ]
            self.output_ffn = EquivariantFFN(
                lmax=0,
                channels=self.channels,
                hidden_channels=self.out_ffn_neurons,
                grid_mlp=False,
                precision=self.precision,
                s2_activation=False,
                lebedev_quadrature=False,
                activation_function=self.out_activation_function,
                glu_activation=self.out_glu_activation,
                mlp_bias=self.mlp_bias,
                trainable=self.trainable,
                seed=child_seed(self.seed, 2),
            )
        else:
            self.blocks = []
            self.output_ffn = None

    def _validate_v1_path(self) -> None:
        if self.lmax != SUPPORTED_LMAX or any(x != SUPPORTED_LMAX for x in self.l_schedule):
            raise NotImplementedError("JAX SeZM v1 supports only lmax=3.")
        if self.mmax != SUPPORTED_MMAX or any(x != SUPPORTED_MMAX for x in self.m_schedule):
            raise NotImplementedError("JAX SeZM v1 supports only mmax=1.")
        if self.radial_so2_mode != SUPPORTED_RADIAL_SO2_MODE:
            raise NotImplementedError(
                "JAX SeZM v1 supports only radial_so2_mode='degree_channel'."
            )
        unsupported = {
            "so2_attn_res": self.so2_attn_res != "none",
            "full_attn_res": self.full_attn_res != "none",
            "block_attn_res": self.block_attn_res != "none",
            "grid_mlp": self.grid_mlp,
            "layer_scale": self.layer_scale,
            "atten_f_mix": self.atten_f_mix,
            "atten_v_proj": self.atten_v_proj,
            "atten_o_proj": self.atten_o_proj,
            "inner_clamp": self.inner_clamp_r_inner is not None
            or self.inner_clamp_r_outer is not None,
            "charge_spin": self.add_chg_spin_ebd,
            "exclude_types": bool(self.exclude_types),
        }
        enabled = [name for name, active in unsupported.items() if active]
        if enabled:
            raise NotImplementedError(
                "JAX SeZM v1 does not support: " + ", ".join(enabled)
            )

    def _resolve_ffn_neurons(self, ffn_neurons: int, *, glu_activation: bool) -> int:
        resolved = int(ffn_neurons)
        if resolved < 0:
            raise ValueError("`ffn_neurons` must be >= 0")
        if resolved > 0:
            return resolved
        base_width = (
            (8.0 * float(self.channels) / 3.0)
            if glu_activation
            else (4.0 * float(self.channels))
        )
        return int(32 * math.ceil(base_width / 32.0))

    def _unsupported_forward_features(self) -> list[str]:
        unsupported = {
            "atten_f_mix": self.atten_f_mix,
            "atten_v_proj": self.atten_v_proj,
            "atten_o_proj": self.atten_o_proj,
            "so2_norm": self.so2_norm,
            "so2_attn_res": self.so2_attn_res != "none",
            "full_attn_res": self.full_attn_res != "none",
            "block_attn_res": self.block_attn_res != "none",
            "layer_scale": self.layer_scale,
            "grid_mlp": self.grid_mlp,
            "so2_s2_activation": self.so2_s2_activation,
            "ffn_s2_activation_without_lebedev": (
                self.ffn_s2_activation and not self.ffn_lebedev_quadrature
            ),
            "mlp_bias": self.mlp_bias,
            "charge_spin": self.add_chg_spin_ebd,
            "exclude_types": bool(self.exclude_types),
        }
        return [name for name, active in unsupported.items() if active]

    def _is_baseline_forward_supported(self) -> bool:
        return not self._unsupported_forward_features()

    def get_rcut(self) -> float:
        return self.rcut

    def get_rcut_smth(self) -> float:
        return self.rcut

    def get_sel(self) -> list[int]:
        return self.sel

    def get_ntypes(self) -> int:
        return self.ntypes

    def get_type_map(self) -> list[str]:
        return [] if self.type_map is None else self.type_map

    def get_dim_chg_spin(self) -> int:
        return 2 if self.add_chg_spin_ebd else 0

    def has_default_chg_spin(self) -> bool:
        return self.default_chg_spin is not None

    def get_default_chg_spin(self) -> list[float] | None:
        return self.default_chg_spin

    def get_dim_out(self) -> int:
        return self.channels

    def get_dim_emb(self) -> int:
        return self.channels

    def mixed_types(self) -> bool:
        return True

    def has_message_passing(self) -> bool:
        return bool(len(self.l_schedule) > 0 and self.lmax > 0)

    def need_sorted_nlist_for_lower(self) -> bool:
        return False

    def get_env_protection(self) -> float:
        return self.eps

    def share_params(
        self, base_class: Any, shared_level: Any, resume: bool = False
    ) -> None:
        raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any | None = None
    ) -> None:
        if self.type_map is None:
            self.type_map = type_map
        old_map = self.type_map
        remap = [old_map.index(tt) if tt in old_map else -1 for tt in type_map]
        self.exclude_types = map_pair_exclude_types(self.exclude_types, remap)
        self.type_map = type_map
        self.ntypes = len(type_map)

    def set_stat_mean_and_stddev(self, mean: Any, stddev: Any) -> None:
        self.mean = mean
        self.stddev = stddev

    def get_stat_mean_and_stddev(self) -> tuple[Array, Array]:
        return self.mean, self.stddev

    def compute_input_stats(
        self,
        merged: Any,
        path: Any | None = None,
    ) -> None:
        # SeZM uses internal equivariant normalization; descriptor mean/stddev
        # are kept only for interface compatibility and checkpoint schema.
        return None

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        local_jdata_cpy = local_jdata.copy()
        min_nbor_dist, sel = UpdateSel().update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["rcut"],
            local_jdata_cpy["sel"],
            True,
        )
        local_jdata_cpy["sel"] = sel[0]
        return local_jdata_cpy, min_nbor_dist

    def _reshape_coord(self, coord_ext: Array) -> Array:
        xp = array_api_compat.array_namespace(coord_ext)
        if len(coord_ext.shape) == 2:
            return xp.reshape(
                coord_ext,
                (coord_ext.shape[0], coord_ext.shape[1] // 3, 3),
            )
        if len(coord_ext.shape) != 3 or coord_ext.shape[-1] != 3:
            raise ValueError(
                "`coord_ext` must have shape [nf, nall * 3] or [nf, nall, 3]"
            )
        return coord_ext

    def _build_edge_cache(
        self,
        coord_ext: Array,
        atype_ext: Array,
        nlist: Array,
        mapping: Array | None = None,
        include_wigner: bool = False,
    ) -> EdgeFeatureCache:
        coord = self._reshape_coord(coord_ext)
        xp = array_api_compat.array_namespace(coord, atype_ext, nlist)
        nf, nloc, nnei = nlist.shape
        nall = coord.shape[1]
        if mapping is None and nall != nloc:
            raise NotImplementedError(
                "JAX SeZM baseline forward requires `mapping` when extended "
                "atoms are present."
            )
        slot_count = nf * nloc * nnei
        slot_idx = xp.arange(slot_count, dtype=xp.int64)
        frame_idx = slot_idx // (nloc * nnei)
        rem_idx = slot_idx - frame_idx * nloc * nnei
        loc_idx = rem_idx // nnei
        neighbor_ext_raw = xp.reshape(nlist, (-1,))
        valid = neighbor_ext_raw >= 0
        valid_f = xp.astype(xp.reshape(valid, (-1, 1)), coord.dtype)
        neighbor_ext = xp.where(valid, neighbor_ext_raw, xp.zeros_like(neighbor_ext_raw))
        dst = frame_idx * nloc + loc_idx
        if mapping is not None:
            mapped = xp.take(
                xp.reshape(mapping, (-1,)),
                frame_idx * mapping.shape[1] + neighbor_ext,
                axis=0,
            )
            src = frame_idx * nloc + mapped
        else:
            src = frame_idx * nloc + neighbor_ext

        coord_flat = xp.reshape(coord, (nf * nall, 3))
        center_idx = frame_idx * nall + loc_idx
        neighbor_idx = frame_idx * nall + neighbor_ext
        edge_vec = xp.take(coord_flat, neighbor_idx, axis=0) - xp.take(
            coord_flat, center_idx, axis=0
        )
        edge_len = xp.sqrt(
            xp.sum(edge_vec * edge_vec, axis=-1, keepdims=True) + self.eps * self.eps
        )
        edge_rbf = self.radial_basis(edge_len) * valid_f
        edge_env = self.edge_envelope(edge_len) * valid_f

        atype_local = atype_ext[:, :nloc]
        type_embedding = xp.reshape(self.type_embedding(atype_local), (nf * nloc, -1))
        edge_type_feat = (
            xp.take(type_embedding, src, axis=0) + xp.take(type_embedding, dst, axis=0)
        ) * valid_f

        edge_weight = xp.reshape(edge_env * edge_env, (-1,))
        deg = xp_bincount(dst, weights=edge_weight, minlength=nf * nloc)
        inv_sqrt_deg = 1.0 / xp.sqrt(
            xp.reshape(deg + xp.asarray(0.25, dtype=deg.dtype), (nf * nloc, 1, 1))
        )
        D_full: Array | None = None
        Dt_full: Array | None = None
        if include_wigner:
            D_full, Dt_full = self._build_edge_wigner(edge_vec, edge_len)
        return EdgeFeatureCache(
            src=src,
            dst=dst,
            edge_type_feat=edge_type_feat,
            edge_vec=edge_vec,
            edge_len=edge_len,
            edge_rbf=edge_rbf,
            edge_env=edge_env,
            deg=deg,
            inv_sqrt_deg=inv_sqrt_deg,
            D_full=D_full,
            Dt_full=Dt_full,
        )

    def _build_edge_wigner(self, edge_vec: Array, edge_len: Array) -> tuple[Array, Array]:
        edge_quat = build_edge_quaternion(
            edge_vec,
            edge_len=edge_len,
            eps=self.eps,
        )
        if self.random_gamma:
            xp = array_api_compat.array_namespace(edge_quat)
            edge_index = xp.arange(edge_quat.shape[0], dtype=edge_quat.dtype)
            c0 = xp.asarray(12.9898, dtype=edge_quat.dtype)
            c1 = xp.asarray(78.233, dtype=edge_quat.dtype)
            c2 = xp.asarray(43758.5453, dtype=edge_quat.dtype)
            two_pi = xp.asarray(2.0 * math.pi, dtype=edge_quat.dtype)
            raw = xp.sin(edge_index * c0 + c1) * c2
            gamma = (raw - xp.floor(raw)) * two_pi
            edge_quat = quaternion_multiply(quaternion_z_rotation(gamma), edge_quat)
        D_full, Dt_full = self.wigner_calc(edge_quat)
        xp = array_api_compat.array_namespace(D_full, Dt_full, edge_vec)
        return xp.astype(D_full, edge_vec.dtype), xp.astype(Dt_full, edge_vec.dtype)

    def call(
        self,
        coord_ext: Array,
        atype_ext: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        comm_dict: dict | None = None,
        charge_spin: Array | None = None,
    ) -> tuple[Array, Array, Array, Array, Array]:
        if not self._baseline_forward_supported:
            raise NotImplementedError(
                "JAX SeZM descriptor baseline forward does not support: "
                + ", ".join(self._unsupported_forward_features())
            )
        if charge_spin is not None:
            raise NotImplementedError("JAX SeZM baseline forward ignores charge_spin.")
        coord = self._reshape_coord(coord_ext)
        xp = array_api_compat.array_namespace(coord, atype_ext, nlist)
        nf, nloc, _ = nlist.shape
        n_nodes = nf * nloc
        atype_loc = atype_ext[:, :nloc]
        type_feat = xp.reshape(self.type_embedding(atype_loc), (n_nodes, self.channels))

        edge_cache = self._build_edge_cache(
            coord,
            atype_ext,
            nlist,
            mapping,
            include_wigner=True,
        )
        ebed_dim_0 = get_so3_dim_of_lmax(self.l_schedule[0])
        x0_out = type_feat
        radial_feat_for_gie = None
        if self.use_env_seed and edge_cache.src.shape[0] > 0:
            atype_flat = xp.reshape(atype_loc, (n_nodes,))
            film = self.env_seed_embedding(
                edge_cache=edge_cache,
                atype_flat=atype_flat,
                n_nodes=n_nodes,
            )
            scale_logits = film[:, : self.channels]
            shift_logits = film[:, self.channels :]
            scale_hat = self.film_scale_norm(scale_logits)
            shift_hat = self.film_shift_norm(shift_logits)
            scale_strength = xp.astype(
                xp.exp(self.film_scale_strength_log[...]),
                type_feat.dtype,
            )
            shift_strength = xp.astype(
                xp.exp(self.film_shift_strength_log[...]),
                type_feat.dtype,
            )
            one = xp.asarray(1.0, dtype=type_feat.dtype)
            scale = one + scale_strength * xp.tanh(scale_hat)
            shift = shift_strength * xp.tanh(shift_hat)
            x0_out = type_feat * scale + shift

        x = xp.zeros(
            (n_nodes, ebed_dim_0, 1, self.channels),
            dtype=type_feat.dtype,
        )
        x = _set_l0_features(x, x0_out)

        radial_feat_flat = self.radial_embedding(edge_cache.edge_rbf)
        radial_feat = xp.reshape(
            radial_feat_flat,
            (edge_cache.edge_rbf.shape[0], self.lmax + 1, self.channels),
        )
        radial_feat = radial_feat * xp.expand_dims(edge_cache.edge_env, axis=-1)
        radial_feat_for_gie = radial_feat
        radial_feat = radial_feat + xp.expand_dims(edge_cache.edge_type_feat, axis=1)
        radial_feat_per_block = [
            radial_feat[:, : ll + 1, :] for ll in self.l_schedule
        ]

        if self.use_gie and radial_feat_for_gie is not None:
            x = x + xp.expand_dims(
                self.gie(
                    n_nodes=n_nodes,
                    edge_cache=edge_cache,
                    radial_feat=radial_feat_for_gie[:, 1:, :],
                ),
                axis=2,
            )

        if edge_cache.src.shape[0] > 0:
            for block, block_radial in zip(
                self.blocks,
                radial_feat_per_block,
                strict=True,
            ):
                x, _, _, _ = block(x, edge_cache, block_radial)

        x_scalar = xp.reshape(x[:, 0:1, :, :], (n_nodes, 1, 1, self.channels))
        x_scalar = x_scalar + self.output_ffn(x_scalar)
        descriptor = xp.reshape(x_scalar, (nf, nloc, self.channels))
        empty = xp.zeros((0,), dtype=descriptor.dtype)
        return descriptor, empty, empty, empty, empty

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "Descriptor",
            "type": "SeZM",
            "@version": self.version,
            "config": {
                "ntypes": self.ntypes,
                "sel": self.sel,
                "rcut": self.rcut,
                "env_exp": self.env_exp,
                "channels": self.channels,
                "basis_type": self.basis_type,
                "n_radial": self.n_radial,
                "radial_mlp": self.radial_mlp,
                "use_env_seed": self.use_env_seed,
                "random_gamma": self.random_gamma,
                "lmax": self.lmax,
                "l_schedule": self.l_schedule,
                "mmax": self.mmax,
                "m_schedule": self.m_schedule,
                "n_blocks": self.n_blocks,
                "so2_norm": self.so2_norm,
                "so2_layers": self.so2_layers,
                "so2_attn_res": self.so2_attn_res,
                "radial_so2_mode": self.radial_so2_mode,
                "radial_so2_rank": self.radial_so2_rank,
                "n_focus": self.n_focus,
                "focus_dim": self.focus_dim,
                "n_atten_head": self.n_atten_head,
                "atten_f_mix": self.atten_f_mix,
                "atten_v_proj": self.atten_v_proj,
                "atten_o_proj": self.atten_o_proj,
                "ffn_neurons": self.ffn_neurons,
                "grid_mlp": self.grid_mlp,
                "ffn_blocks": self.ffn_blocks,
                "sandwich_norm": self.sandwich_norm,
                "mlp_bias": self.mlp_bias,
                "layer_scale": self.layer_scale,
                "full_attn_res": self.full_attn_res,
                "block_attn_res": self.block_attn_res,
                "s2_activation": self.s2_activation,
                "lebedev_quadrature": self.lebedev_quadrature,
                "activation_function": self.activation_function,
                "glu_activation": self.glu_activation,
                "use_amp": self.use_amp,
                "exclude_types": self.exclude_types,
                "precision": self.precision,
                "eps": self.eps,
                "trainable": self.trainable,
                "seed": self.seed,
                "type_map": self.type_map,
                "inner_clamp_r_inner": self.inner_clamp_r_inner,
                "inner_clamp_r_outer": self.inner_clamp_r_outer,
                "add_chg_spin_ebd": self.add_chg_spin_ebd,
                "default_chg_spin": self.default_chg_spin,
            },
            "@variables": {
                "mean": to_numpy_array(self.mean),
                "stddev": to_numpy_array(self.stddev),
                "type_embedding": self.type_embedding.serialize(),
                "radial_basis": self.radial_basis.serialize(),
                "edge_envelope": self.edge_envelope.serialize(),
                "radial_embedding": self.radial_embedding.serialize(),
                "wigner_calc": self.wigner_calc.serialize(),
                "env_seed_embedding": (
                    None
                    if self.env_seed_embedding is None
                    else self.env_seed_embedding.serialize()
                ),
                "film_scale_norm": (
                    None
                    if self.film_scale_norm is None
                    else self.film_scale_norm.serialize()
                ),
                "film_shift_norm": (
                    None
                    if self.film_shift_norm is None
                    else self.film_shift_norm.serialize()
                ),
                "film_scale_strength_log": to_numpy_array(
                    None
                    if self.film_scale_strength_log is None
                    else self.film_scale_strength_log[...]
                ),
                "film_shift_strength_log": to_numpy_array(
                    None
                    if self.film_shift_strength_log is None
                    else self.film_shift_strength_log[...]
                ),
                "gie": None if self.gie is None else self.gie.serialize(),
                "blocks": [block.serialize() for block in self.blocks],
                "output_ffn": (
                    None if self.output_ffn is None else self.output_ffn.serialize()
                ),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "DescrptSeZM":
        data = data.copy()
        data.pop("@class", None)
        type_val = data.pop("type")
        if type_val not in {"SeZM", "sezm", "DPA4", "dpa4"}:
            raise ValueError(f"Invalid SeZM descriptor type {type_val!r}")
        version = data.pop("@version", 1)
        check_version_compatibility(version, cls.LATEST_VERSION, 1)
        config = data.pop("config")
        variables = data.pop("@variables", {})
        obj = cls(**config)
        obj.version = version
        if "mean" in variables:
            obj.mean = variables["mean"]
        if "stddev" in variables:
            obj.stddev = variables["stddev"]
        if "type_embedding" in variables:
            obj.type_embedding = SeZMTypeEmbedding.deserialize(variables["type_embedding"])
        if "radial_basis" in variables:
            obj.radial_basis = RadialBasis.deserialize(variables["radial_basis"])
        if "edge_envelope" in variables:
            obj.edge_envelope = C3CutoffEnvelope.deserialize(variables["edge_envelope"])
        if "radial_embedding" in variables:
            obj.radial_embedding = RadialMLP.deserialize(variables["radial_embedding"])
        if "wigner_calc" in variables:
            obj.wigner_calc = WignerDCalculator.deserialize(variables["wigner_calc"])
        if "env_seed_embedding" in variables and variables["env_seed_embedding"] is not None:
            obj.env_seed_embedding = EnvironmentInitialEmbedding.deserialize(
                variables["env_seed_embedding"]
            )
        if "film_scale_norm" in variables and variables["film_scale_norm"] is not None:
            obj.film_scale_norm = RMSNorm.deserialize(variables["film_scale_norm"])
        if "film_shift_norm" in variables and variables["film_shift_norm"] is not None:
            obj.film_shift_norm = RMSNorm.deserialize(variables["film_shift_norm"])
        if "film_scale_strength_log" in variables:
            obj.film_scale_strength_log = variables["film_scale_strength_log"]
        if "film_shift_strength_log" in variables:
            obj.film_shift_strength_log = variables["film_shift_strength_log"]
        if "gie" in variables and variables["gie"] is not None:
            obj.gie = GeometricInitialEmbedding.deserialize(variables["gie"])
        if "blocks" in variables:
            obj.blocks = [
                SeZMInteractionBlock.deserialize(block) for block in variables["blocks"]
            ]
        if "output_ffn" in variables and variables["output_ffn"] is not None:
            obj.output_ffn = EquivariantFFN.deserialize(variables["output_ffn"])
        return obj


def _set_l0_features(x: Array, values: Array) -> Array:
    xp = array_api_compat.array_namespace(x, values)
    values = xp.astype(values, x.dtype)
    if hasattr(x, "at"):
        return x.at[:, 0, 0, :].set(values)
    x[:, 0, 0, :] = values
    return x
