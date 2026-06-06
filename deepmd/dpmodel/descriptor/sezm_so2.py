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
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .sezm_so3 import (
    _trunc_normal,
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
    if hasattr(target, "at"):
        if target.ndim == 2:
            return target.at[:, index].set(source)
        return target.at[:, index, :].set(source)
    if target.ndim == 2:
        target[:, index] = source
    else:
        target[:, index, :] = source
    return target
