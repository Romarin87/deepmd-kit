# SPDX-License-Identifier: LGPL-3.0-or-later
"""Packed SO(3) indexing helpers for the JAX/dpmodel SeZM port."""

from __future__ import annotations

from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    Array,
)


def get_so3_dim_of_lmax(lmax: int) -> int:
    return int((int(lmax) + 1) ** 2)


def so3_packed_index(degree: int, m: int) -> int:
    degree = int(degree)
    m = int(m)
    if degree < 0:
        raise ValueError("`degree` must be non-negative")
    if abs(m) > degree:
        raise ValueError("`m` must satisfy -degree <= m <= degree")
    return degree * degree + degree + m


def map_degree_idx(lmax: int) -> np.ndarray:
    lmax_i = int(lmax)
    if lmax_i < 0:
        raise ValueError("`lmax` must be non-negative")
    values: list[int] = []
    for degree in range(lmax_i + 1):
        values.extend([degree] * (2 * degree + 1))
    return np.asarray(values, dtype=np.int64)


def build_l_major_index(lmax: int, mmax: int) -> np.ndarray:
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    _validate_l_m(lmax_i, mmax_i)
    indices: list[int] = []
    for degree in range(lmax_i + 1):
        m_keep = min(mmax_i, degree)
        for m in range(-m_keep, m_keep + 1):
            indices.append(so3_packed_index(degree, m))
    return np.asarray(indices, dtype=np.int64)


def build_m_major_index(lmax: int, mmax: int) -> np.ndarray:
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    _validate_l_m(lmax_i, mmax_i)
    indices: list[int] = []
    for degree in range(lmax_i + 1):
        indices.append(so3_packed_index(degree, 0))
    for m in range(1, mmax_i + 1):
        for degree in range(m, lmax_i + 1):
            indices.append(so3_packed_index(degree, -m))
        for degree in range(m, lmax_i + 1):
            indices.append(so3_packed_index(degree, m))
    return np.asarray(indices, dtype=np.int64)


def build_m_major_l_index(lmax: int, mmax: int) -> np.ndarray:
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    _validate_l_m(lmax_i, mmax_i)
    degrees: list[int] = []
    for degree in range(lmax_i + 1):
        degrees.append(degree)
    for m in range(1, mmax_i + 1):
        for degree in range(m, lmax_i + 1):
            degrees.append(degree)
        for degree in range(m, lmax_i + 1):
            degrees.append(degree)
    return np.asarray(degrees, dtype=np.int64)


def build_rotate_inv_rescale(
    lmax: int,
    mmax: int,
    degree_index: Array | np.ndarray,
    *,
    dtype: Any = np.float64,
) -> Array:
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    _validate_l_m(lmax_i, mmax_i)
    xp = array_api_compat.array_namespace(degree_index)
    degrees = xp.astype(degree_index, xp.int64)
    rescale = xp.ones(degrees.shape, dtype=dtype)
    if mmax_i == lmax_i:
        return rescale
    mask = degrees > mmax_i
    denom = float(2 * mmax_i + 1)
    degree_values = xp.astype(degrees, dtype)
    updated = xp.sqrt((2.0 * degree_values + 1.0) / denom)
    return xp.where(mask, updated, rescale)


def project_D_to_m(
    D_full: Array,
    coeff_index_m: Array,
    ebed_dim_full: int,
) -> Array:
    xp = array_api_compat.array_namespace(D_full, coeff_index_m)
    D_block = D_full[:, :ebed_dim_full, :ebed_dim_full]
    return xp.take(D_block, xp.astype(coeff_index_m, xp.int64), axis=1)


def project_Dt_from_m(
    Dt_full: Array,
    coeff_index_m: Array,
    ebed_dim_full: int,
) -> Array:
    xp = array_api_compat.array_namespace(Dt_full, coeff_index_m)
    Dt_block = Dt_full[:, :ebed_dim_full, :ebed_dim_full]
    return xp.take(Dt_block, xp.astype(coeff_index_m, xp.int64), axis=2)


def _validate_l_m(lmax: int, mmax: int) -> None:
    if lmax < 0:
        raise ValueError("`lmax` must be non-negative")
    if mmax < 0:
        raise ValueError("`mmax` must be non-negative")
    if mmax > lmax:
        raise ValueError("`mmax` must be <= `lmax`")
