# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM/DPA4 descriptor scaffold for dpmodel and JAX backends."""

from __future__ import annotations

from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
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
        self.radial_mlp = [0] if radial_mlp is None else [int(x) for x in radial_mlp]
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
        return obj
