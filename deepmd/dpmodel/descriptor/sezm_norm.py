# SPDX-License-Identifier: LGPL-3.0-or-later
"""Normalization layers for the staged JAX/dpmodel SeZM port."""

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
from deepmd.utils.version import (
    check_version_compatibility,
)

from .sezm_indexing import (
    map_degree_idx,
)


class EquivariantRMSNorm(NativeOP):
    """Degree-balanced RMSNorm on packed SO(3) coefficient layout."""

    def __init__(
        self,
        lmax: int,
        channels: int,
        n_focus: int = 1,
        *,
        eps: float = 1e-5,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
    ) -> None:
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.eps = float(eps)
        self.precision = precision
        self.trainable = bool(trainable)
        dtype = PRECISION_DICT[self.precision.lower()]
        self.adam_scale = np.ones(
            (self.lmax + 1, self.n_focus, self.channels),
            dtype=dtype,
        )
        self.bias = np.zeros((self.n_focus, self.channels), dtype=dtype)
        self.expand_index = map_degree_idx(self.lmax)
        weights: list[float] = []
        scale = 1.0 / ((self.lmax + 1) * self.channels)
        for degree in range(self.lmax + 1):
            weights.extend([scale / (2 * degree + 1)] * (2 * degree + 1))
        self.balance_weight = np.asarray(weights, dtype=dtype)

    def call(self, x: Array) -> Array:
        xp = array_api_compat.array_namespace(
            x,
            self.adam_scale,
            self.bias,
            self.expand_index,
            self.balance_weight,
        )
        x0 = x[:, :1, :, :]
        xt = x[:, 1:, :, :]
        x0 = x0 - xp.mean(x0, axis=-1, keepdims=True)
        balance_weight = self.balance_weight[...]
        mean_variance = xp.sum(x0 * x0, axis=(1, 3)) * balance_weight[0]
        if xt.shape[1] > 0:
            mean_variance = mean_variance + xp.einsum(
                "ndfc,d->nf",
                xt * xt,
                balance_weight[1:],
            )
        inv_rms = 1.0 / xp.sqrt(mean_variance + self.eps)
        inv_rms = xp.expand_dims(xp.expand_dims(inv_rms, axis=1), axis=-1)
        x0 = x0 * inv_rms
        if xt.shape[1] > 0:
            xt = xt * inv_rms

        expand_index = xp.asarray(self.expand_index[...], dtype=xp.int64)
        scale = xp.take(self.adam_scale[...], expand_index, axis=0)
        scale = xp.expand_dims(scale, axis=0)
        x0 = x0 * scale[:, :1, :, :]
        if xt.shape[1] > 0:
            xt = xt * scale[:, 1:, :, :]
        x0 = x0 + xp.reshape(self.bias[...], (1, 1, self.n_focus, self.channels))
        if xt.shape[1] == 0:
            return x0
        return xp.concat([x0, xt], axis=1)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "EquivariantRMSNorm",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "channels": self.channels,
                "n_focus": self.n_focus,
                "eps": self.eps,
                "precision": self.precision,
                "trainable": self.trainable,
            },
            "@variables": {
                "adam_scale": to_numpy_array(self.adam_scale[...]),
                "bias": to_numpy_array(self.bias[...]),
                "expand_index": to_numpy_array(self.expand_index[...]),
                "balance_weight": to_numpy_array(self.balance_weight[...]),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "EquivariantRMSNorm":
        data = data.copy()
        data_cls = data.pop("@class", None)
        if data_cls != "EquivariantRMSNorm":
            raise ValueError(f"Invalid class for EquivariantRMSNorm: {data_cls}")
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj.adam_scale = variables["adam_scale"]
        obj.bias = variables["bias"]
        obj.expand_index = variables["expand_index"]
        obj.balance_weight = variables["balance_weight"]
        return obj


class ScalarRMSNorm(NativeOP):
    """Per-focus RMSNorm for scalar attention/gate branches."""

    def __init__(
        self,
        *,
        channels: int,
        n_focus: int = 1,
        eps: float = 1e-7,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
    ) -> None:
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.eps = float(eps)
        self.precision = precision
        self.trainable = bool(trainable)
        dtype = PRECISION_DICT[self.precision.lower()]
        self.adam_scale = np.ones((self.n_focus, self.channels), dtype=dtype)

    def call(self, x: Array) -> Array:
        xp = array_api_compat.array_namespace(x, self.adam_scale)
        inv_rms = 1.0 / xp.sqrt(xp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        x = x * inv_rms
        if x.ndim == 2:
            return x * self.adam_scale[0]
        return x * xp.expand_dims(self.adam_scale[...], axis=0)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "ScalarRMSNorm",
            "@version": 1,
            "config": {
                "channels": self.channels,
                "n_focus": self.n_focus,
                "eps": self.eps,
                "precision": self.precision,
                "trainable": self.trainable,
            },
            "@variables": {
                "adam_scale": to_numpy_array(self.adam_scale[...]),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "ScalarRMSNorm":
        data = data.copy()
        data_cls = data.pop("@class", None)
        if data_cls != "ScalarRMSNorm":
            raise ValueError(f"Invalid class for ScalarRMSNorm: {data_cls}")
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj.adam_scale = variables["adam_scale"]
        return obj
