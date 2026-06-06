# SPDX-License-Identifier: LGPL-3.0-or-later
"""Interaction blocks for the staged JAX/dpmodel SeZM port."""

from __future__ import annotations

from typing import (
    Any,
)

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    NativeOP,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .sezm_ffn import (
    EquivariantFFN,
)
from .sezm_norm import (
    EquivariantRMSNorm,
)
from .sezm_so2 import (
    SO2Convolution,
)


class SeZMInteractionBlock(NativeOP):
    """Baseline SeZM block with SO2 and FFN residual shortcuts."""

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        channels: int,
        n_focus: int = 1,
        focus_dim: int = 0,
        focus_compete: bool = False,
        so2_norm: bool = False,
        so2_layers: int = 1,
        so2_attn_res: str = "none",
        radial_so2_mode: str = "none",
        radial_so2_rank: int = 0,
        n_atten_head: int = 0,
        atten_f_mix: bool = False,
        atten_v_proj: bool = False,
        atten_o_proj: bool = False,
        so2_pre_norm: bool = False,
        so2_post_norm: bool = False,
        ffn_pre_norm: bool = False,
        ffn_post_norm: bool = False,
        ffn_neurons: int = 96,
        grid_mlp: bool = False,
        ffn_blocks: int = 1,
        layer_scale: bool = False,
        full_attn_res: str = "none",
        block_attn_res: str = "none",
        so2_s2_activation: bool = False,
        ffn_s2_activation: bool = False,
        so2_lebedev_quadrature: bool = False,
        ffn_lebedev_quadrature: bool = False,
        so2_activation_function: str = "silu",
        ffn_activation_function: str = "silu",
        ffn_glu_activation: bool = True,
        mlp_bias: bool = False,
        eps: float = 1e-7,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.focus_dim = int(focus_dim)
        self.focus_compete = bool(focus_compete)
        self.so2_norm = bool(so2_norm)
        self.so2_layers = int(so2_layers)
        self.so2_attn_res = str(so2_attn_res).lower()
        self.radial_so2_mode = str(radial_so2_mode).lower()
        self.radial_so2_rank = int(radial_so2_rank)
        self.n_atten_head = int(n_atten_head)
        self.atten_f_mix = bool(atten_f_mix)
        self.atten_v_proj = bool(atten_v_proj)
        self.atten_o_proj = bool(atten_o_proj)
        self.so2_pre_norm = bool(so2_pre_norm)
        self.so2_post_norm = bool(so2_post_norm)
        self.ffn_pre_norm = bool(ffn_pre_norm)
        self.ffn_post_norm = bool(ffn_post_norm)
        self.ffn_neurons = int(ffn_neurons)
        self.grid_mlp = bool(grid_mlp)
        self.ffn_blocks = int(ffn_blocks)
        if self.ffn_blocks < 1:
            raise ValueError("`ffn_blocks` must be >= 1")
        self.layer_scale = bool(layer_scale)
        self.full_attn_res = str(full_attn_res).lower()
        self.block_attn_res = str(block_attn_res).lower()
        self.so2_s2_activation = bool(so2_s2_activation)
        self.ffn_s2_activation = bool(ffn_s2_activation)
        self.so2_lebedev_quadrature = bool(so2_lebedev_quadrature)
        self.ffn_lebedev_quadrature = bool(ffn_lebedev_quadrature)
        self.so2_activation_function = str(so2_activation_function)
        self.ffn_activation_function = str(ffn_activation_function)
        self.ffn_glu_activation = bool(ffn_glu_activation)
        self.mlp_bias = bool(mlp_bias)
        self.eps = float(eps)
        self.precision = precision
        self.trainable = bool(trainable)
        self._validate_baseline_path()

        self.pre_so2_norm = (
            EquivariantRMSNorm(
                self.lmax,
                self.channels,
                n_focus=1,
                eps=self.eps,
                precision=self.precision,
                trainable=self.trainable,
            )
            if self.so2_pre_norm
            else None
        )
        self.post_so2_norm = (
            EquivariantRMSNorm(
                self.lmax,
                self.channels,
                n_focus=1,
                eps=self.eps,
                precision=self.precision,
                trainable=self.trainable,
            )
            if self.so2_post_norm
            else None
        )
        self.so2_conv = SO2Convolution(
            lmax=self.lmax,
            mmax=self.mmax,
            channels=self.channels,
            n_focus=self.n_focus,
            focus_dim=self.focus_dim,
            focus_compete=self.focus_compete,
            so2_norm=self.so2_norm,
            so2_layers=self.so2_layers,
            so2_attn_res=self.so2_attn_res,
            radial_so2_mode=self.radial_so2_mode,
            radial_so2_rank=self.radial_so2_rank,
            n_atten_head=self.n_atten_head,
            atten_f_mix=self.atten_f_mix,
            atten_v_proj=self.atten_v_proj,
            atten_o_proj=self.atten_o_proj,
            s2_activation=self.so2_s2_activation,
            lebedev_quadrature=self.so2_lebedev_quadrature,
            activation_function=self.so2_activation_function,
            mlp_bias=self.mlp_bias,
            eps=self.eps,
            precision=self.precision,
            trainable=self.trainable,
            seed=child_seed(seed, 0),
        )
        self.pre_ffn_norms = [
            EquivariantRMSNorm(
                self.lmax,
                self.channels,
                n_focus=1,
                eps=self.eps,
                precision=self.precision,
                trainable=self.trainable,
            )
            if self.ffn_pre_norm
            else None
            for _ in range(self.ffn_blocks)
        ]
        self.post_ffn_norms = [
            EquivariantRMSNorm(
                self.lmax,
                self.channels,
                n_focus=1,
                eps=self.eps,
                precision=self.precision,
                trainable=self.trainable,
            )
            if self.ffn_post_norm
            else None
            for _ in range(self.ffn_blocks)
        ]
        self.ffns = [
            EquivariantFFN(
                lmax=self.lmax,
                channels=self.channels,
                hidden_channels=self.ffn_neurons,
                grid_mlp=self.grid_mlp,
                precision=self.precision,
                s2_activation=self.ffn_s2_activation,
                lebedev_quadrature=self.ffn_lebedev_quadrature,
                activation_function=self.ffn_activation_function,
                glu_activation=self.ffn_glu_activation,
                mlp_bias=self.mlp_bias,
                trainable=self.trainable,
                seed=child_seed(seed, 100 + ii),
            )
            for ii in range(self.ffn_blocks)
        ]

    def _validate_baseline_path(self) -> None:
        unsupported = {
            "full_attn_res": self.full_attn_res != "none",
            "block_attn_res": self.block_attn_res != "none",
            "layer_scale": self.layer_scale,
            "grid_mlp": self.grid_mlp,
            "ffn_s2_activation": self.ffn_s2_activation,
            "so2_s2_activation": self.so2_s2_activation,
        }
        enabled = [name for name, active in unsupported.items() if active]
        if enabled:
            raise NotImplementedError(
                "JAX SeZMInteractionBlock baseline path does not support: "
                + ", ".join(enabled)
            )

    def _run_so2_unit(self, x: Any, edge_cache: Any, radial_feat: Any) -> Any:
        x_pre = self.pre_so2_norm(x) if self.pre_so2_norm is not None else x
        n_node, ebed_dim, _, channels = x_pre.shape
        y = self.so2_conv(
            x_pre.reshape(n_node, ebed_dim, channels),
            edge_cache,
            radial_feat,
        )
        y = y.reshape(n_node, ebed_dim, 1, channels)
        return self.post_so2_norm(y) if self.post_so2_norm is not None else y

    def _run_ffn_unit(self, x: Any, unit_idx: int) -> Any:
        pre_norm = self.pre_ffn_norms[unit_idx]
        post_norm = self.post_ffn_norms[unit_idx]
        y = pre_norm(x) if pre_norm is not None else x
        y = self.ffns[unit_idx](y)
        return post_norm(y) if post_norm is not None else y

    def call(
        self,
        x: Any,
        edge_cache: Any,
        radial_feat: Any,
        unit_history: list[Any] | None = None,
    ) -> tuple[Any, None, None, None]:
        so2_state = x + self._run_so2_unit(x, edge_cache, radial_feat)
        ffn_state = so2_state
        for unit_idx in range(self.ffn_blocks):
            ffn_state = ffn_state + self._run_ffn_unit(ffn_state, unit_idx)
        return ffn_state, None, None, None

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SeZMInteractionBlock",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "channels": self.channels,
                "n_focus": self.n_focus,
                "focus_dim": self.focus_dim,
                "focus_compete": self.focus_compete,
                "so2_norm": self.so2_norm,
                "so2_layers": self.so2_layers,
                "so2_attn_res": self.so2_attn_res,
                "radial_so2_mode": self.radial_so2_mode,
                "radial_so2_rank": self.radial_so2_rank,
                "n_atten_head": self.n_atten_head,
                "atten_f_mix": self.atten_f_mix,
                "atten_v_proj": self.atten_v_proj,
                "atten_o_proj": self.atten_o_proj,
                "so2_pre_norm": self.so2_pre_norm,
                "so2_post_norm": self.so2_post_norm,
                "ffn_pre_norm": self.ffn_pre_norm,
                "ffn_post_norm": self.ffn_post_norm,
                "ffn_neurons": self.ffn_neurons,
                "grid_mlp": self.grid_mlp,
                "ffn_blocks": self.ffn_blocks,
                "layer_scale": self.layer_scale,
                "full_attn_res": self.full_attn_res,
                "block_attn_res": self.block_attn_res,
                "so2_s2_activation": self.so2_s2_activation,
                "ffn_s2_activation": self.ffn_s2_activation,
                "so2_lebedev_quadrature": self.so2_lebedev_quadrature,
                "ffn_lebedev_quadrature": self.ffn_lebedev_quadrature,
                "so2_activation_function": self.so2_activation_function,
                "ffn_activation_function": self.ffn_activation_function,
                "ffn_glu_activation": self.ffn_glu_activation,
                "mlp_bias": self.mlp_bias,
                "eps": self.eps,
                "precision": self.precision,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": {
                "pre_so2_norm": (
                    None if self.pre_so2_norm is None else self.pre_so2_norm.serialize()
                ),
                "post_so2_norm": (
                    None
                    if self.post_so2_norm is None
                    else self.post_so2_norm.serialize()
                ),
                "so2_conv": self.so2_conv.serialize(),
                "pre_ffn_norms": [
                    None if norm is None else norm.serialize()
                    for norm in self.pre_ffn_norms
                ],
                "post_ffn_norms": [
                    None if norm is None else norm.serialize()
                    for norm in self.post_ffn_norms
                ],
                "ffns": [ffn.serialize() for ffn in self.ffns],
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SeZMInteractionBlock":
        data = data.copy()
        data_cls = data.pop("@class", None)
        if data_cls != "SeZMInteractionBlock":
            raise ValueError(f"Invalid class for SeZMInteractionBlock: {data_cls}")
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj.pre_so2_norm = (
            None
            if variables["pre_so2_norm"] is None
            else EquivariantRMSNorm.deserialize(variables["pre_so2_norm"])
        )
        obj.post_so2_norm = (
            None
            if variables["post_so2_norm"] is None
            else EquivariantRMSNorm.deserialize(variables["post_so2_norm"])
        )
        obj.so2_conv = SO2Convolution.deserialize(variables["so2_conv"])
        obj.pre_ffn_norms = [
            None if item is None else EquivariantRMSNorm.deserialize(item)
            for item in variables["pre_ffn_norms"]
        ]
        obj.post_ffn_norms = [
            None if item is None else EquivariantRMSNorm.deserialize(item)
            for item in variables["post_ffn_norms"]
        ]
        obj.ffns = [EquivariantFFN.deserialize(item) for item in variables["ffns"]]
        return obj
