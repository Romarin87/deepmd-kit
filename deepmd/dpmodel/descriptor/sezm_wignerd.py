# SPDX-License-Identifier: LGPL-3.0-or-later
"""Quaternion edge-frame utilities for the JAX/dpmodel SeZM port."""

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


def _safe_norm_nd(x: Array, eps: float = 1e-7) -> Array:
    xp = array_api_compat.array_namespace(x)
    return xp.sqrt(xp.sum(x * x, axis=-1, keepdims=True) + eps * eps)


def quaternion_normalize(q: Array, eps: float = 1e-7) -> Array:
    return q / _safe_norm_nd(q, eps)


def quaternion_multiply(q1: Array, q2: Array) -> Array:
    xp = array_api_compat.array_namespace(q1, q2)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return xp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quaternion_to_rotation_matrix(q: Array) -> Array:
    xp = array_api_compat.array_namespace(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return xp.stack(
        [
            xp.stack(
                [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                axis=-1,
            ),
            xp.stack(
                [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
                axis=-1,
            ),
            xp.stack(
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)],
                axis=-1,
            ),
        ],
        axis=-2,
    )


def quaternion_z_rotation(gamma: Array) -> Array:
    xp = array_api_compat.array_namespace(gamma)
    half_gamma = 0.5 * gamma
    return xp.stack(
        [
            xp.cos(half_gamma),
            xp.zeros_like(gamma),
            xp.zeros_like(gamma),
            xp.sin(half_gamma),
        ],
        axis=-1,
    )


def _smooth_step_cinf(x: Array) -> Array:
    xp = array_api_compat.array_namespace(x)
    x_clamped = xp.clip(x, 0.0, 1.0)
    eps = xp.asarray(1e-12, dtype=x_clamped.dtype)
    left = xp.exp(-1.0 / xp.clip(x_clamped, eps, 1.0))
    right = xp.exp(-1.0 / xp.clip(1.0 - x_clamped, eps, 1.0))
    interior = left / (left + right)
    return xp.where(
        x_clamped <= 0.0,
        xp.zeros_like(x_clamped),
        xp.where(x_clamped >= 1.0, xp.ones_like(x_clamped), interior),
    )


def quaternion_nlerp(
    q0: Array,
    q1: Array,
    weight: Array,
    *,
    eps: float = 1e-7,
) -> Array:
    xp = array_api_compat.array_namespace(q0, q1, weight)
    dot = xp.sum(q0 * q1, axis=-1, keepdims=True)
    q1_aligned = xp.where(dot < 0.0, -q1, q1)
    weight = weight[..., None]
    blended = (1.0 - weight) * q0 + weight * q1_aligned
    return quaternion_normalize(blended, eps)


def _build_edge_quaternion_chart_pos_z(edge_unit: Array, eps: float) -> Array:
    xp = array_api_compat.array_namespace(edge_unit)
    x = edge_unit[..., 0]
    y = edge_unit[..., 1]
    z = edge_unit[..., 2]
    q = xp.stack([1.0 + z, y, -x, xp.zeros_like(x)], axis=-1)
    return quaternion_normalize(q, eps)


def _build_edge_quaternion_chart_neg_z(edge_unit: Array, eps: float) -> Array:
    xp = array_api_compat.array_namespace(edge_unit)
    x = edge_unit[..., 0]
    y = edge_unit[..., 1]
    z = edge_unit[..., 2]
    q = xp.stack([-x, xp.zeros_like(x), 1.0 - z, y], axis=-1)
    return quaternion_normalize(q, eps)


def build_edge_quaternion(
    edge_vec: Array,
    *,
    edge_len: Array | None = None,
    eps: float = 1e-7,
) -> Array:
    xp = array_api_compat.array_namespace(edge_vec)
    if edge_len is None:
        edge_len = _safe_norm_nd(edge_vec, eps)
    else:
        edge_len = xp.sqrt(edge_len * edge_len + eps * eps)
    edge_unit = edge_vec / edge_len
    q_pos = _build_edge_quaternion_chart_pos_z(edge_unit, eps)
    q_neg = _build_edge_quaternion_chart_neg_z(edge_unit, eps)
    blend = _smooth_step_cinf(0.5 * (edge_unit[..., 2] + 1.0))
    return quaternion_nlerp(q_neg, q_pos, blend, eps=eps)


class WignerDCalculator(NativeOP):
    """Packed Wigner-D calculator for the staged JAX SeZM port.

    The current implementation covers the scalar block and the ``l=1`` vector
    block. Higher-order blocks are intentionally left explicit until the
    low-order polynomial kernels are ported.
    """

    def __init__(
        self,
        lmax: int,
        *,
        eps: float = 1e-7,
        precision: str = DEFAULT_PRECISION,
    ) -> None:
        self.lmax = int(lmax)
        if self.lmax < 0:
            raise ValueError("`lmax` must be non-negative")
        self.eps = float(eps)
        self.precision = precision
        self.dim_full = (self.lmax + 1) ** 2
        dtype = PRECISION_DICT[self.precision.lower()]
        self.l1_perm = np.asarray([1, 2, 0], dtype=np.int64)
        l1_sign = np.asarray([-1.0, -1.0, 1.0], dtype=dtype)
        self.l1_sign_outer = np.asarray(np.outer(l1_sign, l1_sign), dtype=dtype)

    def call(self, edge_quaternion: Array) -> tuple[Array, Array]:
        xp = array_api_compat.array_namespace(edge_quaternion, self.l1_sign_outer)
        edge_quaternion = quaternion_normalize(edge_quaternion, eps=self.eps)
        n_edge = edge_quaternion.shape[0]
        D0 = xp.ones((n_edge, 1, 1), dtype=edge_quaternion.dtype)
        if self.lmax == 0:
            return D0, D0
        if self.lmax > 1:
            raise NotImplementedError(
                "JAX SeZM WignerD currently supports only lmax<=1; "
                "l=2/3 polynomial kernels are the next porting step."
            )
        D1 = self._compute_l1_block(edge_quaternion)
        zeros_01 = xp.zeros((n_edge, 1, 3), dtype=edge_quaternion.dtype)
        zeros_10 = xp.zeros((n_edge, 3, 1), dtype=edge_quaternion.dtype)
        top = xp.concat([D0, zeros_01], axis=-1)
        bottom = xp.concat([zeros_10, D1], axis=-1)
        D_full = xp.concat([top, bottom], axis=-2)
        return D_full, xp.matrix_transpose(D_full)

    def _compute_l1_block(self, edge_quaternion: Array) -> Array:
        rot_mat = quaternion_to_rotation_matrix(edge_quaternion)
        rot_perm = self.l1_perm[...]
        sign_outer = self.l1_sign_outer[...]
        xp = array_api_compat.array_namespace(rot_mat, rot_perm, sign_outer)
        rot_mat = xp.take(rot_mat, rot_perm, axis=-2)
        rot_mat = xp.take(rot_mat, rot_perm, axis=-1)
        return rot_mat * sign_outer

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "WignerDCalculator",
            "@version": 1,
            "lmax": self.lmax,
            "eps": self.eps,
            "precision": self.precision,
            "@variables": {
                "l1_perm": to_numpy_array(self.l1_perm[...]),
                "l1_sign_outer": to_numpy_array(self.l1_sign_outer[...]),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "WignerDCalculator":
        data = data.copy()
        data.pop("@class", None)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        variables = data.pop("@variables", {})
        obj = cls(**data)
        if "l1_perm" in variables:
            obj.l1_perm = variables["l1_perm"]
        if "l1_sign_outer" in variables:
            obj.l1_sign_outer = variables["l1_sign_outer"]
        return obj
