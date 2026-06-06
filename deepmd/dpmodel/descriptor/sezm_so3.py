# SPDX-License-Identifier: LGPL-3.0-or-later
"""SO(3) channel mixing layers for the JAX/dpmodel SeZM port."""

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

from .sezm_indexing import (
    get_so3_dim_of_lmax,
    map_degree_idx,
)


def _trunc_normal(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    *,
    std: float,
    dtype: Any,
) -> np.ndarray:
    values = rng.normal(0.0, std, shape)
    values = np.clip(values, -3.0 * std, 3.0 * std)
    return values.astype(dtype)


class ChannelLinear(NativeOP):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        precision: str = DEFAULT_PRECISION,
        bias: bool = True,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        init_std: float | None = None,
    ) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.precision = precision
        self.use_bias = bool(bias)
        self.trainable = bool(trainable)
        dtype = PRECISION_DICT[self.precision.lower()]
        rng = np.random.default_rng(seed)
        if init_std is None:
            bound = 1.0 / np.sqrt(float(self.in_channels))
            self.weight = rng.uniform(
                -bound,
                bound,
                (self.in_channels, self.out_channels),
            ).astype(dtype)
        else:
            self.weight = rng.normal(
                0.0,
                init_std,
                (self.in_channels, self.out_channels),
            ).astype(dtype)
        self.bias = (
            np.zeros(self.out_channels, dtype=dtype) if self.use_bias else None
        )

    def call(self, x: Array) -> Array:
        xp = array_api_compat.array_namespace(x, self.weight)
        out = xp.matmul(x, self.weight[...])
        if self.use_bias:
            out = out + self.bias[...]
        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "ChannelLinear",
            "@version": 1,
            "config": {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "precision": self.precision,
                "bias": self.use_bias,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": {
                "weight": to_numpy_array(self.weight[...]),
                "bias": to_numpy_array(None if self.bias is None else self.bias[...]),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "ChannelLinear":
        data = data.copy()
        data.pop("@class", None)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj.weight = variables["weight"]
        obj.bias = variables["bias"]
        return obj


class FocusLinear(NativeOP):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        n_focus: int,
        precision: str = DEFAULT_PRECISION,
        bias: bool = True,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        init_std: float | None = None,
    ) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_focus = int(n_focus)
        self.precision = precision
        self.use_bias = bool(bias)
        self.trainable = bool(trainable)
        dtype = PRECISION_DICT[self.precision.lower()]
        rng = np.random.default_rng(seed)
        if init_std is None:
            bound = 1.0 / np.sqrt(float(self.in_channels))
            self.weight = rng.uniform(
                -bound,
                bound,
                (self.in_channels, self.n_focus * self.out_channels),
            ).astype(dtype)
        else:
            self.weight = rng.normal(
                0.0,
                init_std,
                (self.in_channels, self.n_focus * self.out_channels),
            ).astype(dtype)
        self.bias = (
            np.zeros(self.n_focus * self.out_channels, dtype=dtype)
            if self.use_bias
            else None
        )

    def call(self, x: Array) -> Array:
        xp = array_api_compat.array_namespace(x, self.weight)
        weight = xp.reshape(
            self.weight[...],
            (self.in_channels, self.n_focus, self.out_channels),
        )
        out = xp.einsum("bfi,ifo->bfo", x, weight)
        if self.use_bias:
            bias = xp.reshape(self.bias[...], (self.n_focus, self.out_channels))
            out = out + bias[None, :, :]
        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "FocusLinear",
            "@version": 1,
            "config": {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "n_focus": self.n_focus,
                "precision": self.precision,
                "bias": self.use_bias,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": {
                "weight": to_numpy_array(self.weight[...]),
                "bias": to_numpy_array(None if self.bias is None else self.bias[...]),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "FocusLinear":
        data = data.copy()
        data.pop("@class", None)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj.weight = variables["weight"]
        obj.bias = variables["bias"]
        return obj


class SO3Linear(NativeOP):
    def __init__(
        self,
        *,
        lmax: int,
        in_channels: int,
        out_channels: int,
        n_focus: int = 1,
        precision: str = DEFAULT_PRECISION,
        mlp_bias: bool = False,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        init_std: float | None = None,
    ) -> None:
        self.lmax = int(lmax)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_focus = int(n_focus)
        self.precision = precision
        self.mlp_bias = bool(mlp_bias)
        self.trainable = bool(trainable)
        self.ebed_dim = get_so3_dim_of_lmax(self.lmax)
        dtype = PRECISION_DICT[self.precision.lower()]
        num_l = self.lmax + 1
        self.weight = np.empty(
            (num_l, self.in_channels, self.n_focus * self.out_channels),
            dtype=dtype,
        )
        if init_std is not None:
            rng = np.random.default_rng(seed)
            if init_std == 0.0:
                self.weight.fill(0.0)
            else:
                self.weight = rng.normal(
                    0.0,
                    init_std,
                    self.weight.shape,
                ).astype(dtype)
        else:
            for l_idx in range(num_l):
                rng = np.random.default_rng(child_seed(seed, 1000 + l_idx))
                std = 1.0 / np.sqrt(
                    float(self.in_channels + self.n_focus * self.out_channels)
                )
                self.weight[l_idx] = _trunc_normal(
                    rng,
                    self.weight[l_idx].shape,
                    std=std,
                    dtype=dtype,
                )
        self.bias = (
            np.zeros(self.n_focus * self.out_channels, dtype=dtype)
            if self.mlp_bias
            else None
        )
        self.expand_index = map_degree_idx(self.lmax)

    def call(self, x: Array) -> Array:
        xp = array_api_compat.array_namespace(x, self.weight, self.expand_index)
        weight = xp.reshape(
            self.weight[...],
            (
                self.lmax + 1,
                self.in_channels,
                self.n_focus,
                self.out_channels,
            ),
        )
        expand_index = xp.asarray(self.expand_index[...], dtype=xp.int64)
        weight_expanded = xp.take(weight, expand_index, axis=0)
        out = xp.einsum("ndfi,difo->ndfo", x, weight_expanded)
        if self.mlp_bias:
            bias = xp.reshape(self.bias[...], (self.n_focus, self.out_channels))
            zeros = xp.zeros_like(out)
            zeros = zeros.at[:, 0, :, :].add(bias[None, :, :]) if hasattr(zeros, "at") else _add_l0_bias_numpy(zeros, bias)
            out = out + zeros
        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SO3Linear",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "n_focus": self.n_focus,
                "precision": self.precision,
                "mlp_bias": self.mlp_bias,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": {
                "weight": to_numpy_array(self.weight[...]),
                "bias": to_numpy_array(None if self.bias is None else self.bias[...]),
                "expand_index": to_numpy_array(self.expand_index[...]),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SO3Linear":
        data = data.copy()
        data.pop("@class", None)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj.weight = variables["weight"]
        obj.bias = variables["bias"]
        obj.expand_index = variables["expand_index"]
        return obj


def _add_l0_bias_numpy(x: Array, bias: Array) -> Array:
    x[:, 0, :, :] = x[:, 0, :, :] + bias[None, :, :]
    return x
