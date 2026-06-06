# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM/DPA4 descriptor scaffold for dpmodel and JAX backends."""

from __future__ import annotations

from typing import (
    Any,
    NamedTuple,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    Array,
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
    ) -> EdgeFeatureCache:
        coord = self._reshape_coord(coord_ext)
        xp = array_api_compat.array_namespace(coord, atype_ext, nlist)
        nf, nloc, nnei = nlist.shape
        nall = coord.shape[1]
        valid = nlist >= 0
        frame_idx, loc_idx, nei_idx = xp.nonzero(valid)
        neighbor_ext = xp.take(
            xp.reshape(nlist, (-1,)),
            frame_idx * nloc * nnei + loc_idx * nnei + nei_idx,
            axis=0,
        )
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
        edge_rbf = self.radial_basis(edge_len)
        edge_env = self.edge_envelope(edge_len)

        atype_local = atype_ext[:, :nloc]
        type_embedding = xp.reshape(self.type_embedding(atype_local), (nf * nloc, -1))
        edge_type_feat = xp.take(type_embedding, src, axis=0) + xp.take(
            type_embedding, dst, axis=0
        )

        edge_weight = xp.reshape(edge_env * edge_env, (-1,))
        deg = xp_bincount(dst, weights=edge_weight, minlength=nf * nloc)
        inv_sqrt_deg = 1.0 / xp.sqrt(
            xp.reshape(deg + xp.asarray(0.25, dtype=deg.dtype), (nf * nloc, 1, 1))
        )
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
        )

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
        raise NotImplementedError(
            "JAX SeZM descriptor forward is being ported; construction and "
            "serialization are available, but eager forward is not complete yet."
        )

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
        return obj
