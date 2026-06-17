# SPDX-License-Identifier: LGPL-3.0-or-later
"""SO(2) channel mixing layers for the JAX/dpmodel SeZM port."""

from __future__ import annotations

from typing import (
    Any,
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
    xp_sigmoid,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.network import (
    get_activation_fn,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .sezm_so3 import (
    ChannelLinear,
    FocusLinear,
    SO3Linear,
    _trunc_normal,
)
from .sezm_norm import (
    ScalarRMSNorm,
)
from .sezm_indexing import (
    build_m_major_index,
    build_m_major_l_index,
    build_rotate_inv_rescale,
    get_so3_dim_of_lmax,
    map_degree_idx,
    project_D_to_m,
    project_Dt_from_m,
)


class SO2Linear(NativeOP):
    """SO(2)-equivariant linear mixing in SeZM's reduced m-major layout."""

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        in_channels: int,
        out_channels: int,
        n_focus: int = 1,
        precision: str = DEFAULT_PRECISION,
        mlp_bias: bool = False,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_focus = int(n_focus)
        self.precision = precision
        self.mlp_bias = bool(mlp_bias)
        self.trainable = bool(trainable)
        dtype = PRECISION_DICT[self.precision.lower()]

        m0_size = self.lmax + 1
        self.m0_idx = np.arange(m0_size, dtype=np.int64)
        pos_indices: list[np.ndarray] = []
        neg_indices: list[np.ndarray] = []
        m_ranges: list[tuple[int, int, int]] = []
        offset = m0_size
        for m_order in range(1, self.mmax + 1):
            num_l = self.lmax - m_order + 1
            neg_start = offset
            pos_start = offset + num_l
            neg_indices.append(
                np.arange(neg_start, neg_start + num_l, dtype=np.int64)
            )
            pos_indices.append(
                np.arange(pos_start, pos_start + num_l, dtype=np.int64)
            )
            m_ranges.append((neg_start, pos_start, num_l))
            offset += 2 * num_l

        self.reduced_dim = int(offset)
        self.neg_indices = (
            np.concatenate(neg_indices)
            if neg_indices
            else np.empty(0, dtype=np.int64)
        )
        self.pos_indices = (
            np.concatenate(pos_indices)
            if pos_indices
            else np.empty(0, dtype=np.int64)
        )
        self._m_ranges = m_ranges

        self._m0_in = (self.lmax + 1) * self.in_channels
        self._m0_out = (self.lmax + 1) * self.out_channels
        self._block_slices: list[tuple[int, int, int, int, int, int, int, int]] = []
        for neg_start, pos_start, num_l in self._m_ranges:
            ib = num_l * self.in_channels
            ob = num_l * self.out_channels
            self._block_slices.append(
                (
                    neg_start * self.in_channels,
                    neg_start * self.in_channels + ib,
                    pos_start * self.in_channels,
                    pos_start * self.in_channels + ib,
                    neg_start * self.out_channels,
                    neg_start * self.out_channels + ob,
                    pos_start * self.out_channels,
                    pos_start * self.out_channels + ob,
                )
            )

        num_m0 = self.lmax + 1
        num_in_m0 = num_m0 * self.in_channels
        num_out_m0 = num_m0 * self.out_channels
        weight_m0 = np.empty(
            (num_in_m0, self.n_focus * num_out_m0),
            dtype=dtype,
        )
        weight_m0_view = weight_m0.reshape(
            num_in_m0,
            self.n_focus,
            num_out_m0,
        )
        for focus_idx in range(self.n_focus):
            std = 1.0 / np.sqrt(float(num_in_m0 + num_out_m0))
            weight_m0_view[:, focus_idx, :] = _trunc_normal(
                np.random.default_rng(child_seed(seed, 1000 + focus_idx)),
                weight_m0_view[:, focus_idx, :].shape,
                std=std,
                dtype=dtype,
            )
        self.weight_m0 = weight_m0

        self.bias0 = (
            np.zeros(self.n_focus * self.out_channels, dtype=dtype)
            if self.mlp_bias
            else None
        )

        weight_m: list[Array] = []
        for m_order in range(1, self.mmax + 1):
            num_l = self.lmax - m_order + 1
            num_in = num_l * self.in_channels
            num_out = 2 * num_l * self.out_channels
            weight = np.empty((num_in, self.n_focus * num_out), dtype=dtype)
            weight_view = weight.reshape(num_in, self.n_focus, num_out)
            std = 1.0 / np.sqrt(float(num_in + num_out))
            for focus_idx in range(self.n_focus):
                weight_view[:, focus_idx, :] = _trunc_normal(
                    np.random.default_rng(
                        child_seed(seed, 2000 + m_order * 100 + focus_idx)
                    ),
                    weight_view[:, focus_idx, :].shape,
                    std=std,
                    dtype=dtype,
                )
            weight *= 1.0 / np.sqrt(2.0)
            weight_m.append(weight)
        self.weight_m = weight_m

    def _build_so2_weight(self) -> Array:
        weight_m0_arr = self.weight_m0[...]
        xp = array_api_compat.array_namespace(weight_m0_arr)
        in_total = self.reduced_dim * self.in_channels
        out_total = self.reduced_dim * self.out_channels
        weight = xp.zeros(
            (in_total, self.n_focus, out_total),
            dtype=weight_m0_arr.dtype,
        )
        weight_m0 = xp.reshape(
            weight_m0_arr,
            (self._m0_in, self.n_focus, self._m0_out),
        )
        weight = _set_so2_block(weight, 0, self._m0_in, 0, self._m0_out, weight_m0)

        for m_idx, w_param in enumerate(self.weight_m):
            ni0, ni1, pi0, pi1, no0, no1, po0, po1 = self._block_slices[m_idx]
            ib = ni1 - ni0
            ob = no1 - no0
            w = xp.reshape(w_param[...], (ib, self.n_focus, 2 * ob))
            w_u = w[:, :, :ob]
            w_v = w[:, :, ob:]
            weight = _set_so2_block(weight, ni0, ni1, no0, no1, w_u)
            weight = _set_so2_block(weight, ni0, ni1, po0, po1, w_v)
            weight = _set_so2_block(weight, pi0, pi1, no0, no1, -w_v)
            weight = _set_so2_block(weight, pi0, pi1, po0, po1, w_u)
        return weight

    def call(self, x: Array) -> Array:
        xp = array_api_compat.array_namespace(x, self.weight_m0)
        n_edge = x.shape[0]
        x_flat = xp.reshape(
            x,
            (n_edge, self.n_focus, self.reduced_dim * self.in_channels),
        )
        weight = self._build_so2_weight()
        out_flat = xp.einsum("efi,ifo->efo", x_flat, weight)
        out = xp.reshape(
            out_flat,
            (n_edge, self.n_focus, self.reduced_dim, self.out_channels),
        )
        if self.mlp_bias:
            bias0 = xp.reshape(
                self.bias0[...],
                (self.n_focus, self.out_channels),
            )
            if array_api_compat.is_jax_namespace(xp):
                degree_idx = xp.arange(
                    out.shape[2],
                    dtype=xp.int64,
                    device=array_api_compat.device(out),
                )
                degree_mask = xp.astype(
                    degree_idx[None, None, :, None] == 0,
                    out.dtype,
                )
                out = out + degree_mask * bias0[None, :, None, :]
            else:
                out = (
                    out.at[:, :, 0, :].add(bias0[None, :, :])
                    if hasattr(out, "at")
                    else _add_l0_bias_numpy(out, bias0)
                )
        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SO2Linear",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "n_focus": self.n_focus,
                "precision": self.precision,
                "mlp_bias": self.mlp_bias,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": {
                "weight_m0": to_numpy_array(self.weight_m0[...]),
                "bias0": to_numpy_array(
                    None if self.bias0 is None else self.bias0[...]
                ),
                "weight_m": [to_numpy_array(weight[...]) for weight in self.weight_m],
                "m0_idx": to_numpy_array(self.m0_idx[...]),
                "pos_indices": to_numpy_array(self.pos_indices[...]),
                "neg_indices": to_numpy_array(self.neg_indices[...]),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SO2Linear":
        data = data.copy()
        data_cls = data.pop("@class", None)
        if data_cls != "SO2Linear":
            raise ValueError(f"Invalid class for SO2Linear: {data_cls}")
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj.weight_m0 = variables["weight_m0"]
        obj.bias0 = variables.get("bias0")
        if "weight_m" in variables:
            obj.weight_m = variables["weight_m"]
        else:
            obj.weight_m = [
                variables[f"weight_m.{m_idx}"] for m_idx in range(obj.mmax)
            ]
        obj.m0_idx = variables.get("m0_idx", obj.m0_idx)
        obj.pos_indices = variables.get("pos_indices", obj.pos_indices)
        obj.neg_indices = variables.get("neg_indices", obj.neg_indices)
        return obj


def _set_so2_block(
    weight: Array,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    value: Array,
) -> Array:
    xp = array_api_compat.array_namespace(weight, value)
    if array_api_compat.is_jax_namespace(xp):
        row_idx = xp.arange(
            weight.shape[0],
            dtype=xp.int64,
            device=array_api_compat.device(weight),
        )
        col_idx = xp.arange(
            weight.shape[2],
            dtype=xp.int64,
            device=array_api_compat.device(weight),
        )
        block_rows = xp.arange(row0, row1, dtype=xp.int64, device=array_api_compat.device(weight))
        block_cols = xp.arange(col0, col1, dtype=xp.int64, device=array_api_compat.device(weight))
        row_mask = xp.astype(row_idx[:, None] == block_rows[None, :], weight.dtype)
        col_mask = xp.astype(block_cols[:, None] == col_idx[None, :], weight.dtype)
        update = xp.einsum("ri,ifj,jc->rfc", row_mask, value, col_mask)
        full_mask = xp.sum(row_mask, axis=1)[:, None, None] * xp.sum(
            col_mask,
            axis=0,
        )[None, None, :]
        return weight * (1.0 - full_mask) + update
    if hasattr(weight, "at"):
        return weight.at[row0:row1, :, col0:col1].set(value)
    weight[row0:row1, :, col0:col1] = value
    return weight


def _add_l0_bias_numpy(x: Array, bias: Array) -> Array:
    x[:, :, 0, :] = x[:, :, 0, :] + bias[None, :, :]
    return x


class DynamicRadialDegreeMixer(NativeOP):
    """Edge-conditioned degree mixer in SeZM's reduced SO(2) local layout."""

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        channels: int,
        mode: str,
        rank: int = 0,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.channels = int(channels)
        if self.channels < 1:
            raise ValueError("`channels` must be positive")
        self.mode = str(mode).lower()
        if self.mode not in {"degree", "degree_channel"}:
            raise ValueError("`mode` must be one of 'degree' or 'degree_channel'")
        self.rank = int(rank)
        if self.rank < 0:
            raise ValueError("`rank` must be non-negative")
        self.precision = precision
        self.trainable = bool(trainable)
        dtype = PRECISION_DICT[self.precision.lower()]

        self.reduced_dim = (self.lmax + 1) + sum(
            2 * (self.lmax - m_order + 1)
            for m_order in range(1, self.mmax + 1)
        )
        self.degree_kernel_size = sum(
            (self.lmax - m_order + 1) ** 2
            for m_order in range(self.mmax + 1)
        )
        self.input_dim = (self.lmax + 1) * self.channels
        if self.mode == "degree":
            self.proj_out_dim = self.degree_kernel_size
        elif self.rank > 0:
            self.proj_out_dim = self.degree_kernel_size * self.rank
        else:
            self.proj_out_dim = self.degree_kernel_size * self.channels

        std = 1.0 / np.sqrt(float(self.input_dim + self.proj_out_dim))
        self.weight = _trunc_normal(
            np.random.default_rng(child_seed(seed, 0)),
            (self.input_dim, self.proj_out_dim),
            std=std,
            dtype=dtype,
        )

        if self.mode == "degree_channel" and self.rank > 0:
            std_basis = 1.0 / np.sqrt(float(self.rank + self.channels))
            self.channel_basis = _trunc_normal(
                np.random.default_rng(child_seed(seed, 1)),
                (self.rank, self.channels),
                std=std_basis,
                dtype=dtype,
            )
        else:
            self.channel_basis = None

        compact_idx, dense_idx = self._build_dense_scatter_indices()
        self.kernel_compact_index = compact_idx
        self.kernel_dense_index = dense_idx

    def _build_dense_scatter_indices(self) -> tuple[np.ndarray, np.ndarray]:
        compact_indices: list[int] = []
        dense_indices: list[int] = []
        compact_offset = 0
        reduced_dim = self.reduced_dim

        def append_block(start_in: int, start_out: int, num_l: int) -> None:
            for l_in in range(num_l):
                for l_out in range(num_l):
                    compact_indices.append(compact_offset + l_in * num_l + l_out)
                    dense_indices.append(
                        (start_out + l_out) * reduced_dim + start_in + l_in
                    )

        num_l0 = self.lmax + 1
        append_block(0, 0, num_l0)
        compact_offset += num_l0 * num_l0

        offset = num_l0
        for m_order in range(1, self.mmax + 1):
            num_l = self.lmax - m_order + 1
            neg_start = offset
            pos_start = offset + num_l
            append_block(neg_start, neg_start, num_l)
            append_block(pos_start, pos_start, num_l)
            compact_offset += num_l * num_l
            offset += 2 * num_l

        return (
            np.asarray(compact_indices, dtype=np.int64),
            np.asarray(dense_indices, dtype=np.int64),
        )

    def _project_radial(self, radial_feat: Array) -> Array:
        xp = array_api_compat.array_namespace(radial_feat, self.weight)
        radial_m0 = xp.reshape(
            radial_feat[:, : self.lmax + 1, :],
            (radial_feat.shape[0], self.input_dim),
        )
        return xp.matmul(radial_m0, self.weight[...])

    def _scatter_degree_kernel(self, compact: Array) -> Array:
        xp = array_api_compat.array_namespace(
            compact,
            self.kernel_compact_index,
            self.kernel_dense_index,
        )
        n_edge = compact.shape[0]
        dense = xp.zeros(
            (n_edge, self.reduced_dim * self.reduced_dim),
            dtype=compact.dtype,
        )
        compact_index = xp.asarray(self.kernel_compact_index[...], dtype=xp.int64)
        dense_index = xp.asarray(self.kernel_dense_index[...], dtype=xp.int64)
        source = xp.take(compact, compact_index, axis=1)
        dense = _set_flat_columns(dense, dense_index, source)
        return xp.reshape(dense, (n_edge, self.reduced_dim, self.reduced_dim))

    def _scatter_rank_kernel(self, compact: Array) -> Array:
        xp = array_api_compat.array_namespace(
            compact,
            self.kernel_compact_index,
            self.kernel_dense_index,
        )
        n_edge = compact.shape[0]
        dense = xp.zeros(
            (n_edge, self.reduced_dim * self.reduced_dim, self.rank),
            dtype=compact.dtype,
        )
        compact_index = xp.asarray(self.kernel_compact_index[...], dtype=xp.int64)
        dense_index = xp.asarray(self.kernel_dense_index[...], dtype=xp.int64)
        source = xp.take(compact, compact_index, axis=1)
        dense = _set_flat_columns(dense, dense_index, source)
        return xp.reshape(dense, (n_edge, self.reduced_dim, self.reduced_dim, self.rank))

    def _scatter_channel_kernel(self, compact: Array) -> Array:
        xp = array_api_compat.array_namespace(
            compact,
            self.kernel_compact_index,
            self.kernel_dense_index,
        )
        n_edge = compact.shape[0]
        dense = xp.zeros(
            (n_edge, self.reduced_dim * self.reduced_dim, self.channels),
            dtype=compact.dtype,
        )
        compact_index = xp.asarray(self.kernel_compact_index[...], dtype=xp.int64)
        dense_index = xp.asarray(self.kernel_dense_index[...], dtype=xp.int64)
        source = xp.take(compact, compact_index, axis=1)
        dense = _set_flat_columns(dense, dense_index, source)
        return xp.reshape(
            dense,
            (n_edge, self.reduced_dim, self.reduced_dim, self.channels),
        )

    def call(self, x_local: Array, radial_feat: Array) -> Array:
        if x_local.shape != radial_feat.shape:
            raise ValueError("`x_local` and `radial_feat` must have the same shape")
        if x_local.shape[1] != self.reduced_dim or x_local.shape[2] != self.channels:
            raise ValueError("Input shape is incompatible with this mixer")

        xp = array_api_compat.array_namespace(x_local, radial_feat, self.weight)
        kernel_flat = self._project_radial(radial_feat)
        if self.mode == "degree":
            kernel = self._scatter_degree_kernel(kernel_flat)
            return xp.einsum("eoi,eic->eoc", kernel, x_local)

        if self.rank > 0:
            compact = xp.reshape(
                kernel_flat,
                (x_local.shape[0], self.degree_kernel_size, self.rank),
            )
            kernel = self._scatter_rank_kernel(compact)
            mixed = xp.einsum("eoir,eic->eorc", kernel, x_local)
            channel_basis = xp.reshape(
                self.channel_basis[...],
                (1, 1, self.rank, self.channels),
            )
            return xp.sum(mixed * channel_basis, axis=2)

        compact = xp.reshape(
            kernel_flat,
            (x_local.shape[0], self.degree_kernel_size, self.channels),
        )
        kernel = self._scatter_channel_kernel(compact)
        return xp.einsum("eoic,eic->eoc", kernel, x_local)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "DynamicRadialDegreeMixer",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "channels": self.channels,
                "mode": self.mode,
                "rank": self.rank,
                "precision": self.precision,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": {
                "weight": to_numpy_array(self.weight[...]),
                "channel_basis": to_numpy_array(
                    None if self.channel_basis is None else self.channel_basis[...]
                ),
                "kernel_compact_index": to_numpy_array(
                    self.kernel_compact_index[...]
                ),
                "kernel_dense_index": to_numpy_array(self.kernel_dense_index[...]),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "DynamicRadialDegreeMixer":
        data = data.copy()
        data_cls = data.pop("@class", None)
        if data_cls != "DynamicRadialDegreeMixer":
            raise ValueError(
                f"Invalid class for DynamicRadialDegreeMixer: {data_cls}"
            )
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj.weight = variables["weight"]
        obj.channel_basis = variables.get("channel_basis")
        obj.kernel_compact_index = variables.get(
            "kernel_compact_index",
            obj.kernel_compact_index,
        )
        obj.kernel_dense_index = variables.get(
            "kernel_dense_index",
            obj.kernel_dense_index,
        )
        return obj


def _set_flat_columns(target: Array, index: Array, source: Array) -> Array:
    xp = array_api_compat.array_namespace(target, index, source)
    if array_api_compat.is_jax_namespace(xp):
        col_idx = xp.arange(
            target.shape[1],
            dtype=index.dtype,
            device=array_api_compat.device(index),
        )
        mask = xp.astype(index[:, None] == col_idx[None, :], target.dtype)
        col_mask = xp.sum(mask, axis=0)
        if target.ndim == 2:
            update = xp.matmul(source, mask)
            return target * (1.0 - col_mask[None, :]) + update
        update = xp.einsum("rkc,kj->rjc", source, mask)
        return target * (1.0 - col_mask[None, :, None]) + update
    if hasattr(target, "at"):
        if target.ndim == 2:
            return target.at[:, index].set(source)
        return target.at[:, index, :].set(source)
    if target.ndim == 2:
        target[:, index] = source
    else:
        target[:, index, :] = source
    return target


class GatedActivation(NativeOP):
    """Degree-wise gated activation for full or reduced SeZM layouts."""

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        channels: int,
        n_focus: int = 1,
        precision: str = DEFAULT_PRECISION,
        activation_function: str = "silu",
        mlp_bias: bool = False,
        layout: str = "nfdc",
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.lmax = int(lmax)
        self.mmax = None if mmax is None else int(mmax)
        if self.mmax is not None:
            if self.mmax < 0:
                raise ValueError("`mmax` must be non-negative")
            if self.mmax > self.lmax:
                raise ValueError("`mmax` must be <= `lmax`")
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.precision = precision
        self.activation_function = str(activation_function)
        self.mlp_bias = bool(mlp_bias)
        self.layout = str(layout).lower()
        if self.layout not in {"nfdc", "ndfc"}:
            raise ValueError("`layout` must be either 'nfdc' or 'ndfc'")
        self.trainable = bool(trainable)
        dtype = PRECISION_DICT[self.precision.lower()]

        if self.lmax > 0:
            if self.mmax is None:
                expand_index = map_degree_idx(self.lmax)[1:] - 1
            else:
                expand_index = build_m_major_l_index(self.lmax, self.mmax)[1:] - 1
            gate_linear = FocusLinear(
                in_channels=self.channels,
                out_channels=self.lmax * self.channels,
                n_focus=self.n_focus,
                precision=self.precision,
                bias=self.mlp_bias,
                trainable=self.trainable,
                seed=seed,
            )
            rng = np.random.default_rng(child_seed(seed, 1))
            gate_linear.weight = rng.normal(
                0.0,
                0.01,
                gate_linear.weight.shape,
            ).astype(dtype)
            if gate_linear.bias is not None:
                gate_linear.bias = np.zeros(gate_linear.bias.shape, dtype=dtype)
            self.gate_linear = gate_linear
        else:
            expand_index = np.zeros(0, dtype=np.int64)
            self.gate_linear = None
        self.expand_index = np.asarray(expand_index, dtype=np.int64)

    def call(self, x: Array, gate: Array | None = None) -> Array:
        xp = array_api_compat.array_namespace(x)
        degree_axis = 1 if self.layout == "ndfc" else 2
        scalar_act = get_activation_fn(self.activation_function)

        if self.layout == "ndfc":
            gate_scalar_source = gate[:, 0, :, :] if gate is not None else x[:, 0, :, :]
            if gate is not None:
                x0 = x[:, :1, :, :] * scalar_act(gate[:, :1, :, :])
            else:
                x0 = scalar_act(x[:, :1, :, :])
            x_rest = x[:, 1:, :, :]
        else:
            gate_scalar_source = gate[:, :, 0, :] if gate is not None else x[:, :, 0, :]
            if gate is not None:
                x0 = x[:, :, :1, :] * scalar_act(gate[:, :, :1, :])
            else:
                x0 = scalar_act(x[:, :, :1, :])
            x_rest = x[:, :, 1:, :]

        if self.lmax == 0:
            return x0

        gating_scalars = xp_sigmoid(self.gate_linear(gate_scalar_source))
        gating_scalars = xp.reshape(
            gating_scalars,
            (x.shape[0], gate_scalar_source.shape[1], self.lmax, self.channels),
        )
        expand_index = xp.asarray(self.expand_index[...], dtype=xp.int64)
        gates = xp.take(gating_scalars, expand_index, axis=2)
        if self.layout == "ndfc":
            gates = xp.permute_dims(gates, (0, 2, 1, 3))
        return xp.concat([x0, x_rest * gates], axis=degree_axis)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "GatedActivation",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "channels": self.channels,
                "n_focus": self.n_focus,
                "precision": self.precision,
                "activation_function": self.activation_function,
                "mlp_bias": self.mlp_bias,
                "layout": self.layout,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": {
                "gate_linear": (
                    None if self.gate_linear is None else self.gate_linear.serialize()
                ),
                "expand_index": to_numpy_array(self.expand_index[...]),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "GatedActivation":
        data = data.copy()
        data_cls = data.pop("@class", None)
        if data_cls != "GatedActivation":
            raise ValueError(f"Invalid class for GatedActivation: {data_cls}")
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj.gate_linear = (
            None
            if variables["gate_linear"] is None
            else FocusLinear.deserialize(variables["gate_linear"])
        )
        obj.expand_index = variables.get("expand_index", obj.expand_index)
        return obj


class SO2Convolution(NativeOP):
    """Minimal eager SO(2) message convolution for the staged JAX SeZM port."""

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
        layer_scale: bool = False,
        n_atten_head: int = 0,
        atten_f_mix: bool = False,
        atten_v_proj: bool = False,
        atten_o_proj: bool = False,
        s2_activation: bool = False,
        lebedev_quadrature: bool = False,
        activation_function: str = "silu",
        mlp_bias: bool = False,
        radial_so2_mode: str = "none",
        radial_so2_rank: int = 0,
        eps: float = 1e-7,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        if self.n_focus < 1:
            raise ValueError("`n_focus` must be >= 1")
        self.focus_dim = int(focus_dim)
        if self.focus_dim < 0:
            raise ValueError("`focus_dim` must be >= 0")
        self.so2_focus_dim = self.channels if self.focus_dim == 0 else self.focus_dim
        self.hidden_channels = int(self.n_focus * self.so2_focus_dim)
        self.use_hidden_projection = self.hidden_channels != self.channels
        self.focus_compete = bool(focus_compete)
        self.so2_norm = bool(so2_norm)
        self.so2_layers = int(so2_layers)
        if self.so2_layers < 1:
            raise ValueError("`so2_layers` must be >= 1")
        self.so2_attn_res = str(so2_attn_res).lower()
        self.layer_scale = bool(layer_scale)
        self.n_atten_head = int(n_atten_head)
        if self.n_atten_head < 0:
            raise ValueError("`n_atten_head` must be non-negative")
        self.atten_f_mix = bool(atten_f_mix)
        self.atten_v_proj = bool(atten_v_proj)
        self.atten_o_proj = bool(atten_o_proj)
        self.attn_n_focus = (
            1 if self.atten_f_mix and self.n_atten_head > 0 else self.n_focus
        )
        self.attn_focus_dim = (
            self.hidden_channels
            if self.atten_f_mix and self.n_atten_head > 0
            else self.so2_focus_dim
        )
        if self.n_atten_head > 0 and self.attn_focus_dim % self.n_atten_head != 0:
            raise ValueError("`n_atten_head` must divide the attention width")
        self.head_dim = (
            None
            if self.n_atten_head == 0
            else int(self.attn_focus_dim // self.n_atten_head)
        )
        self.s2_activation = bool(s2_activation)
        self.lebedev_quadrature = bool(lebedev_quadrature)
        self.activation_function = str(activation_function)
        self.mlp_bias = bool(mlp_bias)
        self.radial_so2_mode = str(radial_so2_mode).lower()
        if self.radial_so2_mode not in {"none", "degree", "degree_channel"}:
            raise ValueError(
                "`radial_so2_mode` must be one of 'none', 'degree', or 'degree_channel'"
            )
        self.radial_so2_rank = int(radial_so2_rank)
        if self.radial_so2_rank < 0:
            raise ValueError("`radial_so2_rank` must be non-negative")
        self.eps = float(eps)
        self.precision = precision
        self.trainable = bool(trainable)
        self._validate_minimal_path()

        dtype = PRECISION_DICT[self.precision.lower()]
        self.ebed_dim_full = get_so3_dim_of_lmax(self.lmax)
        self.coeff_index_m = build_m_major_index(self.lmax, self.mmax)
        self.degree_index_m = build_m_major_l_index(self.lmax, self.mmax)
        self.degree_index_full = map_degree_idx(self.lmax)
        self.rotate_inv_rescale_full = build_rotate_inv_rescale(
            self.lmax,
            self.mmax,
            self.degree_index_full,
            dtype=dtype,
        )
        self.reduced_dim = int(self.coeff_index_m.shape[0])

        seed_so2_stack = child_seed(seed, 0)
        seed_so3_pre = child_seed(seed, 2)
        seed_so3_post = child_seed(seed, 3)
        seed_gate = child_seed(seed, 4)
        seed_radial_hidden = child_seed(seed, 6)
        seed_radial_degree = child_seed(seed, 7)

        self.so2_linears = [
            SO2Linear(
                lmax=self.lmax,
                mmax=self.mmax,
                in_channels=self.so2_focus_dim,
                out_channels=self.so2_focus_dim,
                n_focus=self.n_focus,
                precision=self.precision,
                mlp_bias=self.mlp_bias,
                trainable=self.trainable,
                seed=child_seed(seed_so2_stack, ii),
            )
            for ii in range(self.so2_layers)
        ]
        self.non_linearities = [
            GatedActivation(
                lmax=self.lmax,
                mmax=self.mmax,
                channels=self.so2_focus_dim,
                n_focus=self.n_focus,
                precision=self.precision,
                activation_function=self.activation_function,
                mlp_bias=self.mlp_bias,
                layout="nfdc",
                trainable=self.trainable,
                seed=child_seed(seed_so2_stack, 1000 + ii),
            )
            for ii in range(max(0, self.so2_layers - 1))
        ] + [None]
        self.radial_hidden_proj = (
            ChannelLinear(
                in_channels=self.channels,
                out_channels=self.hidden_channels,
                precision=self.precision,
                bias=False,
                trainable=self.trainable,
                seed=seed_radial_hidden,
            )
            if self.use_hidden_projection
            else None
        )
        self.radial_degree_mixer = (
            DynamicRadialDegreeMixer(
                lmax=self.lmax,
                mmax=self.mmax,
                channels=self.hidden_channels,
                mode=self.radial_so2_mode,
                rank=self.radial_so2_rank,
                precision=self.precision,
                trainable=self.trainable,
                seed=seed_radial_degree,
            )
            if self.radial_so2_mode != "none"
            else None
        )
        if self.n_atten_head > 0:
            self.attn_qk_norm = ScalarRMSNorm(
                channels=self.attn_focus_dim,
                n_focus=self.attn_n_focus,
                eps=self.eps,
                precision=self.precision,
                trainable=self.trainable,
            )
            self.attn_q_proj = FocusLinear(
                in_channels=self.attn_focus_dim,
                out_channels=self.attn_focus_dim,
                n_focus=self.attn_n_focus,
                precision=self.precision,
                bias=False,
                trainable=self.trainable,
                seed=child_seed(seed_gate, 0),
            )
            self.attn_k_proj = FocusLinear(
                in_channels=self.attn_focus_dim,
                out_channels=self.attn_focus_dim,
                n_focus=self.attn_n_focus,
                precision=self.precision,
                bias=False,
                trainable=self.trainable,
                seed=child_seed(seed_gate, 1),
            )
            self.attn_logit_w = np.random.default_rng(child_seed(seed_gate, 2)).normal(
                0.0,
                0.01,
                (self.attn_focus_dim, self.attn_n_focus, self.n_atten_head),
            ).astype(dtype)
            self.attn_z_bias_raw = np.full(
                (self.attn_n_focus, self.n_atten_head),
                0.5413,
                dtype=dtype,
            )
            self.attn_output_gate_norm = ScalarRMSNorm(
                channels=self.attn_focus_dim,
                n_focus=self.attn_n_focus,
                eps=self.eps,
                precision=self.precision,
                trainable=self.trainable,
            )
            self.attn_gate_w = np.random.default_rng(child_seed(seed_gate, 3)).normal(
                0.0,
                0.01,
                (self.attn_focus_dim, self.attn_n_focus, self.n_atten_head),
            ).astype(dtype)
        else:
            self.attn_qk_norm = None
            self.attn_q_proj = None
            self.attn_k_proj = None
            self.attn_output_gate_norm = None
            self.attn_logit_w = None
            self.attn_z_bias_raw = None
            self.attn_gate_w = None
        self.pre_focus_mix = SO3Linear(
            lmax=self.lmax,
            in_channels=self.channels,
            out_channels=self.hidden_channels,
            n_focus=1,
            precision=self.precision,
            mlp_bias=self.mlp_bias,
            trainable=self.trainable,
            seed=seed_so3_pre,
        )
        self.post_focus_mix = SO3Linear(
            lmax=self.lmax,
            in_channels=self.hidden_channels,
            out_channels=self.channels,
            n_focus=1,
            precision=self.precision,
            mlp_bias=self.mlp_bias,
            trainable=self.trainable,
            seed=seed_so3_post,
            init_std=0.0,
        )

    def _validate_minimal_path(self) -> None:
        unsupported = {
            "focus_compete": self.focus_compete and self.n_focus > 1,
            "so2_norm": self.so2_norm,
            "so2_attn_res": self.so2_attn_res != "none",
            "layer_scale": self.layer_scale,
            "atten_f_mix": self.atten_f_mix,
            "atten_v_proj": self.atten_v_proj,
            "atten_o_proj": self.atten_o_proj,
            "s2_activation": self.s2_activation,
            "mlp_bias": self.mlp_bias,
        }
        enabled = [name for name, active in unsupported.items() if active]
        if enabled:
            raise NotImplementedError(
                "JAX SO2Convolution minimal path does not support: "
                + ", ".join(enabled)
            )

    def call(
        self,
        x: Array,
        edge_cache: Any,
        radial_feat: Array,
    ) -> Array:
        if edge_cache.D_full is None or edge_cache.Dt_full is None:
            raise ValueError("SO2Convolution requires Wigner D matrices in edge_cache")
        xp = array_api_compat.array_namespace(
            x,
            radial_feat,
            edge_cache.src,
            edge_cache.dst,
            edge_cache.D_full,
            self.coeff_index_m,
        )
        src = xp.astype(edge_cache.src, xp.int64)
        dst = xp.astype(edge_cache.dst, xp.int64)
        n_node = x.shape[0]
        n_edge = src.shape[0]

        x_wide = self.pre_focus_mix(xp.expand_dims(x, axis=2))
        x_wide = xp.squeeze(x_wide, axis=2)

        D_m_prime = project_D_to_m(
            edge_cache.D_full,
            self.coeff_index_m[...],
            self.ebed_dim_full,
        )
        x_src = xp.take(x_wide, src, axis=0)
        x_local = xp.matmul(D_m_prime, x_src)

        degree_index = xp.asarray(self.degree_index_m[...], dtype=xp.int64)
        rad_feat = xp.take(radial_feat, degree_index, axis=1)
        if self.radial_hidden_proj is not None:
            rad_feat = self.radial_hidden_proj(rad_feat)
        if self.radial_degree_mixer is None:
            x_local = x_local * rad_feat
        else:
            x_local = self.radial_degree_mixer(x_local, rad_feat)

        x_local = xp.reshape(
            x_local,
            (n_edge, self.reduced_dim, self.n_focus, self.so2_focus_dim),
        )
        x_local = xp.permute_dims(x_local, (0, 2, 1, 3))
        for so2_linear, non_linear in zip(
            self.so2_linears,
            self.non_linearities,
            strict=True,
        ):
            residual = x_local
            update = so2_linear(x_local)
            if non_linear is not None:
                update = non_linear(update)
            x_local = residual + update

        x_local = xp.permute_dims(x_local, (0, 2, 1, 3))
        x_local = xp.reshape(
            x_local,
            (n_edge, self.reduced_dim, self.hidden_channels),
        )

        Dt_from_m = project_Dt_from_m(
            edge_cache.Dt_full,
            self.coeff_index_m[...],
            self.ebed_dim_full,
        )
        x_message = xp.matmul(Dt_from_m, x_local)
        x_message = x_message * xp.reshape(
            self.rotate_inv_rescale_full[...],
            (1, self.ebed_dim_full, 1),
        )
        if self.n_atten_head == 0:
            x_message = x_message * xp.expand_dims(edge_cache.edge_env, axis=-1)
            out = xp.zeros(
                (n_node, self.ebed_dim_full, self.hidden_channels),
                dtype=x_message.dtype,
            )
            out = _scatter_add_first_axis(out, dst, x_message)
            out = out * edge_cache.inv_sqrt_deg
        else:
            head_dim = int(self.head_dim)
            x_l0_node = xp.reshape(
                x_wide[:, 0, :],
                (n_node, self.attn_n_focus, self.attn_focus_dim),
            )
            qk_input = self.attn_qk_norm(x_l0_node)
            q_node = self.attn_q_proj(qk_input)
            k_node = self.attn_k_proj(qk_input)
            q_edge = xp.reshape(
                xp.take(q_node, dst, axis=0),
                (
                    n_edge,
                    self.attn_n_focus,
                    self.n_atten_head,
                    head_dim,
                ),
            )
            k_edge = xp.reshape(
                xp.take(k_node, src, axis=0),
                (
                    n_edge,
                    self.attn_n_focus,
                    self.n_atten_head,
                    head_dim,
                ),
            )
            radial_l0 = xp.reshape(
                rad_feat[:, 0, :],
                (n_edge, self.attn_n_focus, self.attn_focus_dim),
            )
            radial_bias = xp.einsum(
                "efi,ifo->efo",
                radial_l0,
                self.attn_logit_w[...],
            )
            attn_logits = xp.sum(q_edge * k_edge, axis=-1) * (
                float(head_dim) ** -0.5
            )
            attn_logits = attn_logits + radial_bias
            attn_alpha = _segment_envelope_gated_softmax(
                attn_logits,
                edge_cache.edge_env,
                dst,
                n_node,
                self.attn_z_bias_raw[...],
                self.eps,
            )
            value_heads = xp.reshape(
                x_message,
                (
                    n_edge,
                    self.ebed_dim_full,
                    self.attn_n_focus,
                    self.n_atten_head,
                    head_dim,
                ),
            )
            weighted_value = value_heads * xp.reshape(
                attn_alpha,
                (n_edge, 1, self.attn_n_focus, self.n_atten_head, 1),
            )
            out_heads = xp.zeros(
                (
                    n_node,
                    self.ebed_dim_full,
                    self.attn_n_focus,
                    self.n_atten_head,
                    head_dim,
                ),
                dtype=weighted_value.dtype,
            )
            out_heads = _scatter_add_first_axis(out_heads, dst, weighted_value)
            attn_output_gate = xp_sigmoid(
                xp.einsum(
                    "nfi,ifo->nfo",
                    self.attn_output_gate_norm(x_l0_node),
                    self.attn_gate_w[...],
                )
            )
            out_heads = out_heads * xp.reshape(
                attn_output_gate,
                (n_node, 1, self.attn_n_focus, self.n_atten_head, 1),
            )
            out = xp.reshape(
                out_heads,
                (n_node, self.ebed_dim_full, self.hidden_channels),
            )
        out = self.post_focus_mix(xp.expand_dims(out, axis=2))
        return xp.squeeze(out, axis=2)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SO2Convolution",
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
                "layer_scale": self.layer_scale,
                "n_atten_head": self.n_atten_head,
                "atten_f_mix": self.atten_f_mix,
                "atten_v_proj": self.atten_v_proj,
                "atten_o_proj": self.atten_o_proj,
                "s2_activation": self.s2_activation,
                "lebedev_quadrature": self.lebedev_quadrature,
                "activation_function": self.activation_function,
                "mlp_bias": self.mlp_bias,
                "radial_so2_mode": self.radial_so2_mode,
                "radial_so2_rank": self.radial_so2_rank,
                "eps": self.eps,
                "precision": self.precision,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": {
                "coeff_index_m": to_numpy_array(self.coeff_index_m[...]),
                "degree_index_m": to_numpy_array(self.degree_index_m[...]),
                "degree_index_full": to_numpy_array(self.degree_index_full[...]),
                "rotate_inv_rescale_full": to_numpy_array(
                    self.rotate_inv_rescale_full[...]
                ),
                "so2_linears": [layer.serialize() for layer in self.so2_linears],
                "non_linearities": [
                    None if layer is None else layer.serialize()
                    for layer in self.non_linearities
                ],
                "radial_hidden_proj": (
                    None
                    if self.radial_hidden_proj is None
                    else self.radial_hidden_proj.serialize()
                ),
                "radial_degree_mixer": (
                    None
                    if self.radial_degree_mixer is None
                    else self.radial_degree_mixer.serialize()
                ),
                "attn_qk_norm": (
                    None
                    if self.attn_qk_norm is None
                    else self.attn_qk_norm.serialize()
                ),
                "attn_q_proj": (
                    None
                    if self.attn_q_proj is None
                    else self.attn_q_proj.serialize()
                ),
                "attn_k_proj": (
                    None
                    if self.attn_k_proj is None
                    else self.attn_k_proj.serialize()
                ),
                "attn_output_gate_norm": (
                    None
                    if self.attn_output_gate_norm is None
                    else self.attn_output_gate_norm.serialize()
                ),
                "attn_logit_w": to_numpy_array(
                    None if self.attn_logit_w is None else self.attn_logit_w[...]
                ),
                "attn_z_bias_raw": to_numpy_array(
                    None
                    if self.attn_z_bias_raw is None
                    else self.attn_z_bias_raw[...]
                ),
                "attn_gate_w": to_numpy_array(
                    None if self.attn_gate_w is None else self.attn_gate_w[...]
                ),
                "pre_focus_mix": self.pre_focus_mix.serialize(),
                "post_focus_mix": self.post_focus_mix.serialize(),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SO2Convolution":
        data = data.copy()
        data_cls = data.pop("@class", None)
        if data_cls != "SO2Convolution":
            raise ValueError(f"Invalid class for SO2Convolution: {data_cls}")
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj.coeff_index_m = variables.get("coeff_index_m", obj.coeff_index_m)
        obj.degree_index_m = variables.get("degree_index_m", obj.degree_index_m)
        obj.degree_index_full = variables.get(
            "degree_index_full",
            obj.degree_index_full,
        )
        obj.rotate_inv_rescale_full = variables.get(
            "rotate_inv_rescale_full",
            obj.rotate_inv_rescale_full,
        )
        obj.so2_linears = [
            SO2Linear.deserialize(item) for item in variables["so2_linears"]
        ]
        obj.non_linearities = [
            None if item is None else GatedActivation.deserialize(item)
            for item in variables.get("non_linearities", obj.non_linearities)
        ]
        obj.radial_hidden_proj = (
            None
            if variables["radial_hidden_proj"] is None
            else ChannelLinear.deserialize(variables["radial_hidden_proj"])
        )
        obj.radial_degree_mixer = (
            None
            if variables["radial_degree_mixer"] is None
            else DynamicRadialDegreeMixer.deserialize(
                variables["radial_degree_mixer"]
            )
        )
        obj.attn_qk_norm = (
            None
            if variables.get("attn_qk_norm") is None
            else ScalarRMSNorm.deserialize(variables["attn_qk_norm"])
        )
        obj.attn_q_proj = (
            None
            if variables.get("attn_q_proj") is None
            else FocusLinear.deserialize(variables["attn_q_proj"])
        )
        obj.attn_k_proj = (
            None
            if variables.get("attn_k_proj") is None
            else FocusLinear.deserialize(variables["attn_k_proj"])
        )
        obj.attn_output_gate_norm = (
            None
            if variables.get("attn_output_gate_norm") is None
            else ScalarRMSNorm.deserialize(variables["attn_output_gate_norm"])
        )
        obj.attn_logit_w = variables.get("attn_logit_w", obj.attn_logit_w)
        obj.attn_z_bias_raw = variables.get(
            "attn_z_bias_raw",
            obj.attn_z_bias_raw,
        )
        obj.attn_gate_w = variables.get("attn_gate_w", obj.attn_gate_w)
        obj.pre_focus_mix = SO3Linear.deserialize(variables["pre_focus_mix"])
        obj.post_focus_mix = SO3Linear.deserialize(variables["post_focus_mix"])
        return obj


def _segment_envelope_gated_softmax(
    logits: Array,
    edge_env: Array,
    dst: Array,
    n_nodes: int,
    z_bias_raw: Array,
    eps: float,
) -> Array:
    xp = array_api_compat.array_namespace(logits, edge_env, dst, z_bias_raw)
    n_edge, n_focus, n_head = logits.shape
    n_channel = n_focus * n_head
    logits_2d = xp.reshape(logits, (n_edge, n_channel))
    edge_weight_sq = xp.reshape(xp.maximum(edge_env, 0.0), (n_edge,)) ** 2
    zeta = _softplus(xp.reshape(z_bias_raw, (1, n_channel)))
    dst = xp.astype(dst, xp.int64)
    neg_large = xp.asarray(-1.0e30, dtype=logits_2d.dtype)
    logits_for_max = xp.where(
        xp.reshape(edge_weight_sq > 0.0, (n_edge, 1)),
        logits_2d,
        xp.full(logits_2d.shape, neg_large, dtype=logits_2d.dtype),
    )
    group_max = xp.full(
        (n_nodes, n_channel),
        neg_large,
        dtype=logits_2d.dtype,
    )
    group_max = _scatter_max_first_axis(group_max, dst, logits_for_max)
    edge_max = xp.take(group_max, dst, axis=0)
    edge_has_value = edge_max > neg_large * 0.5
    edge_max = xp.where(edge_has_value, edge_max, xp.zeros_like(edge_max))
    group_has_value = group_max > neg_large * 0.5
    group_max_safe = xp.where(
        group_has_value,
        group_max,
        xp.zeros_like(group_max),
    )
    edge_weighted_exp = xp.reshape(edge_weight_sq, (n_edge, 1)) * xp.exp(
        logits_2d - edge_max
    )
    denom_sum = xp.zeros((n_nodes, n_channel), dtype=logits_2d.dtype)
    denom_sum = _scatter_add_first_axis(denom_sum, dst, edge_weighted_exp)
    denom = denom_sum + zeta * xp.exp(-group_max_safe)
    alpha = edge_weighted_exp / (xp.take(denom, dst, axis=0) + float(eps))
    return xp.reshape(alpha, (n_edge, n_focus, n_head))


def _softplus(x: Array) -> Array:
    xp = array_api_compat.array_namespace(x)
    return xp.log1p(xp.exp(-xp.abs(x))) + xp.maximum(x, 0.0)


def _scatter_add_first_axis(target: Array, index: Array, source: Array) -> Array:
    xp = array_api_compat.array_namespace(target, index, source)
    if array_api_compat.is_jax_namespace(xp):
        node_idx = xp.arange(
            target.shape[0],
            dtype=index.dtype,
            device=array_api_compat.device(index),
        )
        weights = xp.astype(index[:, None] == node_idx[None, :], source.dtype)
        source_flat = xp.reshape(source, (source.shape[0], -1))
        update_flat = xp.matmul(xp.permute_dims(weights, (1, 0)), source_flat)
        update = xp.reshape(update_flat, target.shape)
        return target + update
    if hasattr(target, "at"):
        return target.at[index].add(source)
    np.add.at(target, index, source)
    return target


def _scatter_max_first_axis(target: Array, index: Array, source: Array) -> Array:
    xp = array_api_compat.array_namespace(target, index, source)
    if array_api_compat.is_jax_namespace(xp):
        node_idx = xp.arange(
            target.shape[0],
            dtype=index.dtype,
            device=array_api_compat.device(index),
        )
        mask = index[:, None] == node_idx[None, :]
        source_flat = xp.reshape(source, (source.shape[0], -1))
        target_flat = xp.reshape(target, (target.shape[0], -1))
        masked = xp.where(mask[:, :, None], source_flat[:, None, :], target_flat)
        return xp.reshape(xp.max(masked, axis=0), target.shape)
    if hasattr(target, "at"):
        return target.at[index].max(source)
    np.maximum.at(target, index, source)
    return target
