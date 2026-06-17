# SPDX-License-Identifier: LGPL-3.0-or-later
"""Quaternion edge-frame utilities for the JAX/dpmodel SeZM port."""

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
    xp_add_at,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.safe_gradient import (
    safe_for_sqrt,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class CaseCoefficients(NamedTuple):
    coeff: np.ndarray
    horner: np.ndarray
    poly_len: np.ndarray
    ra_exp: np.ndarray
    rb_exp: np.ndarray
    sign: np.ndarray
    valid_mask: np.ndarray
    horner_step_mask: np.ndarray
    signed_coeff: np.ndarray


class WignerPolynomialCoefficients(NamedTuple):
    lmin: int
    lmax: int
    size: int
    max_poly_len: int
    n_primary: int
    n_derived: int
    primary_row: np.ndarray
    primary_col: np.ndarray
    primary_flat: np.ndarray
    case1: CaseCoefficients
    case2: CaseCoefficients
    mp_plus_m: np.ndarray
    m_minus_mp: np.ndarray
    diagonal_mask: np.ndarray
    anti_diagonal_mask: np.ndarray
    special_2m: np.ndarray
    anti_diag_sign: np.ndarray
    derived_row: np.ndarray
    derived_col: np.ndarray
    derived_flat: np.ndarray
    derived_primary_idx: np.ndarray
    derived_sign: np.ndarray


def _xp_asarray(xp: Any, value: Any, dtype: Any | None = None) -> Array:
    try:
        value = value[...]
    except TypeError:
        pass
    if dtype is None:
        return xp.asarray(value)
    return xp.asarray(value, dtype=dtype)


def _safe_norm_nd(x: Array, eps: float = 1e-7) -> Array:
    xp = array_api_compat.array_namespace(x)
    return safe_for_sqrt(xp.sum(x * x, axis=-1, keepdims=True) + eps * eps)


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
        edge_len = safe_for_sqrt(edge_len * edge_len + eps * eps)
    edge_unit = edge_vec / edge_len
    q_pos = _build_edge_quaternion_chart_pos_z(edge_unit, eps)
    q_neg = _build_edge_quaternion_chart_neg_z(edge_unit, eps)
    blend = _smooth_step_cinf(0.5 * (edge_unit[..., 2] + 1.0))
    return quaternion_nlerp(q_neg, q_pos, blend, eps=eps)


class WignerDCalculator(NativeOP):
    """Packed quaternion Wigner-D calculator for the staged JAX SeZM port."""

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
        self.poly_lmin = 2
        self.poly_offset = self.poly_lmin * self.poly_lmin
        if self.lmax >= self.poly_lmin:
            self.poly_coeffs = self._precompute_wigner_coefficients(
                self.lmax,
                lmin=self.poly_lmin,
            )
            self.poly_basis = self._assemble_block_diagonal_real_basis(
                self._precompute_real_basis_blocks(
                    lmin=self.poly_lmin,
                    lmax=self.lmax,
                    dtype=dtype,
                )
            )

    def call(self, edge_quaternion: Array) -> tuple[Array, Array]:
        xp = array_api_compat.array_namespace(edge_quaternion, self.l1_sign_outer)
        edge_quaternion = quaternion_normalize(edge_quaternion, eps=self.eps)
        n_edge = edge_quaternion.shape[0]
        D0 = xp.ones((n_edge, 1, 1), dtype=edge_quaternion.dtype)
        if self.lmax == 0:
            return D0, D0
        D1 = self._compute_l1_block(edge_quaternion)
        if self.lmax == 1:
            zeros_01 = xp.zeros((n_edge, 1, 3), dtype=edge_quaternion.dtype)
            zeros_10 = xp.zeros((n_edge, 3, 1), dtype=edge_quaternion.dtype)
            top = xp.concat([D0, zeros_01], axis=-1)
            bottom = xp.concat([zeros_10, D1], axis=-1)
            D_full = xp.concat([top, bottom], axis=-2)
            return D_full, xp.matrix_transpose(D_full)

        D_poly = self._compute_polynomial_blocks(edge_quaternion)
        poly_size = D_poly.shape[-1]
        zeros_01 = xp.zeros((n_edge, 1, 3), dtype=edge_quaternion.dtype)
        zeros_0p = xp.zeros((n_edge, 1, poly_size), dtype=edge_quaternion.dtype)
        zeros_10 = xp.zeros((n_edge, 3, 1), dtype=edge_quaternion.dtype)
        zeros_1p = xp.zeros((n_edge, 3, poly_size), dtype=edge_quaternion.dtype)
        zeros_p0 = xp.zeros((n_edge, poly_size, 1), dtype=edge_quaternion.dtype)
        zeros_p1 = xp.zeros((n_edge, poly_size, 3), dtype=edge_quaternion.dtype)
        top = xp.concat([D0, zeros_01, zeros_0p], axis=-1)
        middle = xp.concat([zeros_10, D1, zeros_1p], axis=-1)
        bottom = xp.concat([zeros_p0, zeros_p1, D_poly], axis=-1)
        D_full = xp.concat([top, middle, bottom], axis=-2)
        return D_full, xp.matrix_transpose(D_full)

    def _compute_l1_block(self, edge_quaternion: Array) -> Array:
        rot_mat = quaternion_to_rotation_matrix(edge_quaternion)
        rot_perm = self.l1_perm[...]
        sign_outer = self.l1_sign_outer[...]
        xp = array_api_compat.array_namespace(rot_mat, rot_perm, sign_outer)
        rot_mat = xp.take(rot_mat, rot_perm, axis=-2)
        rot_mat = xp.take(rot_mat, rot_perm, axis=-1)
        return rot_mat * sign_outer

    def _compute_polynomial_blocks(self, edge_quaternion: Array) -> Array:
        ra_re, ra_im, rb_re, rb_im = self._quaternion_to_ra_rb_real(edge_quaternion)
        D_re, D_im = self._wigner_d_matrix_realpair(
            ra_re,
            ra_im,
            rb_re,
            rb_im,
            self.poly_coeffs,
            output_dtype=edge_quaternion.dtype,
        )
        return self._wigner_d_pair_to_real(
            D_re,
            D_im,
            self.poly_basis,
        )

    @staticmethod
    def _factorial_table(nmax: int) -> np.ndarray:
        table = np.zeros(nmax + 1, dtype=np.float64)
        table[0] = 1.0
        for ii in range(1, nmax + 1):
            table[ii] = table[ii - 1] * ii
        return table

    @staticmethod
    def _binomial(nval: int, kval: int, factorial: np.ndarray) -> float:
        if kval < 0 or kval > nval:
            return 0.0
        return float(factorial[nval] / (factorial[kval] * factorial[nval - kval]))

    @staticmethod
    def _allocate_case_coeffs(
        n_primary: int,
        max_poly_len: int,
    ) -> dict[str, np.ndarray]:
        return {
            "coeff": np.zeros(n_primary, dtype=np.float64),
            "horner": np.zeros((n_primary, max_poly_len), dtype=np.float64),
            "poly_len": np.zeros(n_primary, dtype=np.int64),
            "ra_exp": np.zeros(n_primary, dtype=np.float64),
            "rb_exp": np.zeros(n_primary, dtype=np.float64),
            "sign": np.zeros(n_primary, dtype=np.float64),
        }

    @staticmethod
    def _compute_case_coefficients(
        case: dict[str, np.ndarray],
        idx: int,
        ell: int,
        mp: int,
        mval: int,
        sqrt_factor: float,
        factorial: np.ndarray,
        *,
        is_case1: bool,
    ) -> None:
        if is_case1:
            rho_min = max(0, mp - mval)
            rho_max = min(ell + mp, ell - mval)
        else:
            rho_min = max(0, -(mp + mval))
            rho_max = min(ell - mval, ell - mp)
        if rho_min > rho_max:
            return

        if is_case1:
            binom1 = WignerDCalculator._binomial(ell + mp, rho_min, factorial)
            binom2 = WignerDCalculator._binomial(
                ell - mp,
                ell - mval - rho_min,
                factorial,
            )
        else:
            binom1 = WignerDCalculator._binomial(
                ell + mp,
                ell - mval - rho_min,
                factorial,
            )
            binom2 = WignerDCalculator._binomial(ell - mp, rho_min, factorial)
        case["coeff"][idx] = sqrt_factor * binom1 * binom2

        poly_len = rho_max - rho_min + 1
        case["poly_len"][idx] = poly_len
        for ii, rho in enumerate(range(rho_max, rho_min, -1)):
            if is_case1:
                n1 = ell + mp - rho + 1
                n2 = ell - mval - rho + 1
                d1 = rho
                d2 = mval - mp + rho
            else:
                n1 = ell - mval - rho + 1
                n2 = ell - mp - rho + 1
                d1 = rho
                d2 = mp + mval + rho
            if d1 != 0 and d2 != 0:
                case["horner"][idx, ii] = (n1 * n2) / (d1 * d2)

        if is_case1:
            case["ra_exp"][idx] = 2 * ell + mp - mval - 2 * rho_min
            case["rb_exp"][idx] = mval - mp + 2 * rho_min
            case["sign"][idx] = (-1) ** rho_min
        else:
            case["ra_exp"][idx] = mp + mval + 2 * rho_min
            case["rb_exp"][idx] = 2 * ell - mp - mval - 2 * rho_min
            case["sign"][idx] = ((-1) ** (ell - mval)) * ((-1) ** rho_min)

    @staticmethod
    def _finalize_case_coefficients(
        case: dict[str, np.ndarray],
        max_poly_len: int,
    ) -> CaseCoefficients:
        step_count = np.clip(case["poly_len"] - 1, 0, None)
        if max_poly_len > 1:
            horner_step_mask = (
                np.arange(max_poly_len - 1, dtype=np.int64)[None, :]
                < step_count[:, None]
            )
        else:
            horner_step_mask = np.zeros((case["poly_len"].shape[0], 0), dtype=bool)
        return CaseCoefficients(
            coeff=case["coeff"],
            horner=case["horner"],
            poly_len=case["poly_len"],
            ra_exp=case["ra_exp"],
            rb_exp=case["rb_exp"],
            sign=case["sign"],
            valid_mask=case["poly_len"] > 0,
            horner_step_mask=horner_step_mask,
            signed_coeff=case["sign"] * case["coeff"],
        )

    @staticmethod
    def _precompute_wigner_coefficients(
        lmax: int,
        *,
        lmin: int = 0,
    ) -> WignerPolynomialCoefficients:
        if lmin < 0:
            raise ValueError("`lmin` must be non-negative")
        if lmax < lmin:
            raise ValueError("`lmax` must be >= `lmin`")

        factorial = WignerDCalculator._factorial_table(2 * lmax + 1)
        n_total = sum((2 * ell + 1) ** 2 for ell in range(lmin, lmax + 1))
        n_primary = sum(
            1
            for ell in range(lmin, lmax + 1)
            for mp in range(-ell, ell + 1)
            for mval in range(-ell, ell + 1)
            if mp + mval > 0 or (mp + mval == 0 and mp >= 0)
        )
        n_derived = n_total - n_primary
        max_poly_len = lmax + 1
        size = (lmax + 1) ** 2 - lmin * lmin

        primary_row = np.zeros(n_primary, dtype=np.int64)
        primary_col = np.zeros(n_primary, dtype=np.int64)
        mp_plus_m = np.zeros(n_primary, dtype=np.float64)
        m_minus_mp = np.zeros(n_primary, dtype=np.float64)
        diagonal_mask = np.zeros(n_primary, dtype=bool)
        anti_diagonal_mask = np.zeros(n_primary, dtype=bool)
        special_2m = np.zeros(n_primary, dtype=np.float64)
        anti_diag_sign = np.zeros(n_primary, dtype=np.float64)
        case1 = WignerDCalculator._allocate_case_coeffs(n_primary, max_poly_len)
        case2 = WignerDCalculator._allocate_case_coeffs(n_primary, max_poly_len)
        derived_row = np.zeros(n_derived, dtype=np.int64)
        derived_col = np.zeros(n_derived, dtype=np.int64)
        derived_primary_idx = np.zeros(n_derived, dtype=np.int64)
        derived_sign = np.zeros(n_derived, dtype=np.float64)

        primary_map: dict[tuple[int, int], int] = {}
        primary_idx = 0
        block_start = 0
        for ell in range(lmin, lmax + 1):
            block_size = 2 * ell + 1
            for mp_local in range(block_size):
                mp = mp_local - ell
                for m_local in range(block_size):
                    mval = m_local - ell
                    row = block_start + mp_local
                    col = block_start + m_local
                    is_primary = (mp + mval > 0) or (mp + mval == 0 and mp >= 0)
                    if not is_primary:
                        continue
                    primary_map[(row, col)] = primary_idx
                    primary_row[primary_idx] = row
                    primary_col[primary_idx] = col
                    mp_plus_m[primary_idx] = mp + mval
                    m_minus_mp[primary_idx] = mval - mp
                    diagonal_mask[primary_idx] = mp == mval
                    anti_diagonal_mask[primary_idx] = mp == -mval
                    special_2m[primary_idx] = 2 * mval
                    anti_diag_sign[primary_idx] = (-1) ** (ell - mval)
                    sqrt_factor = np.sqrt(
                        float(factorial[ell + mval] * factorial[ell - mval])
                        / float(factorial[ell + mp] * factorial[ell - mp])
                    )
                    WignerDCalculator._compute_case_coefficients(
                        case1,
                        primary_idx,
                        ell,
                        mp,
                        mval,
                        sqrt_factor,
                        factorial,
                        is_case1=True,
                    )
                    WignerDCalculator._compute_case_coefficients(
                        case2,
                        primary_idx,
                        ell,
                        mp,
                        mval,
                        sqrt_factor,
                        factorial,
                        is_case1=False,
                    )
                    primary_idx += 1
            block_start += block_size

        derived_idx = 0
        block_start = 0
        for ell in range(lmin, lmax + 1):
            block_size = 2 * ell + 1
            for mp_local in range(block_size):
                mp = mp_local - ell
                for m_local in range(block_size):
                    mval = m_local - ell
                    row = block_start + mp_local
                    col = block_start + m_local
                    is_primary = (mp + mval > 0) or (mp + mval == 0 and mp >= 0)
                    if is_primary:
                        continue
                    derived_row[derived_idx] = row
                    derived_col[derived_idx] = col
                    derived_primary_idx[derived_idx] = primary_map[
                        (block_start + (-mp + ell), block_start + (-mval + ell))
                    ]
                    derived_sign[derived_idx] = (-1) ** (mp - mval)
                    derived_idx += 1
            block_start += block_size

        return WignerPolynomialCoefficients(
            lmin=lmin,
            lmax=lmax,
            size=size,
            max_poly_len=max_poly_len,
            n_primary=n_primary,
            n_derived=n_derived,
            primary_row=primary_row,
            primary_col=primary_col,
            primary_flat=primary_row * size + primary_col,
            case1=WignerDCalculator._finalize_case_coefficients(case1, max_poly_len),
            case2=WignerDCalculator._finalize_case_coefficients(case2, max_poly_len),
            mp_plus_m=mp_plus_m,
            m_minus_mp=m_minus_mp,
            diagonal_mask=diagonal_mask,
            anti_diagonal_mask=anti_diagonal_mask,
            special_2m=special_2m,
            anti_diag_sign=anti_diag_sign,
            derived_row=derived_row,
            derived_col=derived_col,
            derived_flat=derived_row * size + derived_col,
            derived_primary_idx=derived_primary_idx,
            derived_sign=derived_sign,
        )

    @staticmethod
    def _build_complex_to_real_sh_block(ell: int) -> np.ndarray:
        size = 2 * ell + 1
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        U = np.zeros((size, size), dtype=np.complex128)
        for mval in range(-ell, ell + 1):
            row = mval + ell
            if mval == 0:
                U[row, ell] = 1.0
            elif mval > 0:
                U[row, mval + ell] = inv_sqrt2
                U[row, -mval + ell] = ((-1) ** mval) * inv_sqrt2
            else:
                U[row, -mval + ell] = -1j * inv_sqrt2
                U[row, mval + ell] = ((-1) ** mval) * 1j * inv_sqrt2
        return U

    @staticmethod
    def _precompute_real_basis_blocks(
        *,
        lmin: int,
        lmax: int,
        dtype: Any,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if lmin > lmax:
            return []
        blocks: list[tuple[np.ndarray, np.ndarray]] = []
        for ell in range(lmin, lmax + 1):
            U = WignerDCalculator._build_complex_to_real_sh_block(ell)
            blocks.append((U.real.astype(dtype), U.imag.astype(dtype)))
        return blocks

    @staticmethod
    def _assemble_block_diagonal_real_basis(
        U_blocks: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not U_blocks:
            empty = np.zeros((0, 0), dtype=np.float64)
            return empty, empty, empty, empty
        size = sum(U_re.shape[0] for U_re, _ in U_blocks)
        dtype = U_blocks[0][0].dtype
        U_re_full = np.zeros((size, size), dtype=dtype)
        U_im_full = np.zeros((size, size), dtype=dtype)
        offset = 0
        for U_re, U_im in U_blocks:
            block_size = U_re.shape[0]
            block_end = offset + block_size
            U_re_full[offset:block_end, offset:block_end] = U_re
            U_im_full[offset:block_end, offset:block_end] = U_im
            offset = block_end
        return (
            U_re_full,
            U_im_full,
            np.swapaxes(U_re_full, -1, -2).copy(),
            np.swapaxes(U_im_full, -1, -2).copy(),
        )

    @staticmethod
    def _quaternion_to_ra_rb_real(q: Array) -> tuple[Array, Array, Array, Array]:
        return q[..., 0], -q[..., 3], q[..., 2], -q[..., 1]

    @staticmethod
    def _vectorized_horner(
        ratio: Array,
        horner_coeffs: Array,
        horner_step_mask: Array,
    ) -> Array:
        xp = array_api_compat.array_namespace(ratio, horner_coeffs, horner_step_mask)
        n_batch = ratio.shape[0]
        n_elements = horner_coeffs.shape[0]
        result = xp.ones((n_batch, n_elements), dtype=ratio.dtype)
        if horner_step_mask.shape[1] == 0:
            return result
        ratio = ratio[:, None]
        for ii in range(horner_step_mask.shape[1]):
            new_result = 1.0 + result * (ratio * horner_coeffs[:, ii][None, :])
            result = xp.where(horner_step_mask[:, ii][None, :], new_result, result)
        return result

    @staticmethod
    def _compute_case_magnitude(
        log_ra: Array,
        log_rb: Array,
        ratio: Array,
        case: CaseCoefficients,
    ) -> Array:
        xp = array_api_compat.array_namespace(log_ra, log_rb, ratio)
        horner = _xp_asarray(xp, case.horner, dtype=log_ra.dtype)
        horner_mask = _xp_asarray(xp, case.horner_step_mask)
        horner_sum = WignerDCalculator._vectorized_horner(
            ratio,
            horner,
            horner_mask,
        )
        ra_exp = _xp_asarray(xp, case.ra_exp, dtype=log_ra.dtype)
        rb_exp = _xp_asarray(xp, case.rb_exp, dtype=log_ra.dtype)
        signed_coeff = _xp_asarray(xp, case.signed_coeff, dtype=log_ra.dtype)
        ra_powers = xp.exp(log_ra[:, None] * ra_exp[None, :])
        rb_powers = xp.exp(log_rb[:, None] * rb_exp[None, :])
        magnitude = signed_coeff[None, :] * ra_powers * rb_powers
        return magnitude * horner_sum

    @staticmethod
    def _scatter_flat(
        *,
        n_batch: int,
        size: int,
        flat_indices: Array,
        values: Array,
    ) -> Array:
        xp = array_api_compat.array_namespace(flat_indices, values)
        if array_api_compat.is_jax_namespace(xp):
            dense_idx = xp.arange(
                size * size,
                dtype=flat_indices.dtype,
                device=array_api_compat.device(flat_indices),
            )
            mask = xp.astype(
                flat_indices[:, None] == dense_idx[None, :],
                values.dtype,
            )
            out_flat = xp.matmul(values, mask)
            return xp.reshape(out_flat, (n_batch, size, size))
        batch_offsets = xp.arange(n_batch, dtype=xp.int64)[:, None] * (size * size)
        indices = xp.reshape(batch_offsets + flat_indices[None, :], (-1,))
        flat_values = xp.reshape(values, (-1,))
        out = xp.zeros((n_batch * size * size,), dtype=values.dtype)
        out = xp_add_at(out, indices, flat_values)
        return xp.reshape(out, (n_batch, size, size))

    @staticmethod
    def _wigner_d_matrix_realpair(
        ra_re: Array,
        ra_im: Array,
        rb_re: Array,
        rb_im: Array,
        coeffs: WignerPolynomialCoefficients,
        *,
        output_dtype: Any,
    ) -> tuple[Array, Array]:
        xp = array_api_compat.array_namespace(ra_re, ra_im, rb_re, rb_im)
        n_batch = ra_re.shape[0]
        if coeffs.size == 0:
            zeros = xp.zeros((n_batch, 0, 0), dtype=output_dtype)
            return zeros, zeros

        calc_dtype = xp.float64
        ra_re = xp.astype(ra_re, calc_dtype)
        ra_im = xp.astype(ra_im, calc_dtype)
        rb_re = xp.astype(rb_re, calc_dtype)
        rb_im = xp.astype(rb_im, calc_dtype)
        eps = np.finfo(np.float64).eps
        eps_sq = eps * eps
        ra_sq = ra_re * ra_re + ra_im * ra_im
        rb_sq = rb_re * rb_re + rb_im * rb_im
        ra_small = ra_sq <= eps_sq
        rb_small = rb_sq <= eps_sq
        ra = safe_for_sqrt(xp.maximum(ra_sq, eps_sq))
        rb = safe_for_sqrt(xp.maximum(rb_sq, eps_sq))
        general_mask = xp.logical_not(xp.logical_or(ra_small, rb_small))
        use_case1 = xp.logical_and(ra >= rb, general_mask)
        use_case2 = xp.logical_and(ra < rb, general_mask)

        safe_ra_re = xp.where(ra_small, xp.ones_like(ra_re), ra_re)
        safe_ra_im = xp.where(ra_small, xp.zeros_like(ra_im), ra_im)
        safe_rb_re = xp.where(rb_small, xp.ones_like(rb_re), rb_re)
        safe_rb_im = xp.where(rb_small, xp.zeros_like(rb_im), rb_im)
        phia = xp.atan2(safe_ra_im, safe_ra_re)
        phib = xp.atan2(safe_rb_im, safe_rb_re)

        mp_plus_m = _xp_asarray(xp, coeffs.mp_plus_m, dtype=calc_dtype)
        m_minus_mp = _xp_asarray(xp, coeffs.m_minus_mp, dtype=calc_dtype)
        phase = phia[:, None] * mp_plus_m[None, :] + phib[:, None] * m_minus_mp[None, :]
        exp_phase_re = xp.cos(phase)
        exp_phase_im = xp.sin(phase)

        safe_ra = xp.maximum(ra, eps)
        safe_rb = xp.maximum(rb, eps)
        log_ra = xp.log(safe_ra)
        log_rb = xp.log(safe_rb)

        result_re = xp.zeros((n_batch, coeffs.n_primary), dtype=calc_dtype)
        result_im = xp.zeros_like(result_re)

        special_2m = _xp_asarray(xp, coeffs.special_2m, dtype=calc_dtype)
        anti_diag_sign = _xp_asarray(xp, coeffs.anti_diag_sign, dtype=calc_dtype)
        anti_diagonal_mask = _xp_asarray(xp, coeffs.anti_diagonal_mask)
        anti_log_rb = xp.where(ra_small, log_rb, xp.zeros_like(log_rb))
        anti_phib = xp.where(ra_small, phib, xp.zeros_like(phib))
        rb_power_mag = xp.exp(anti_log_rb[:, None] * special_2m[None, :])
        rb_power_phase = anti_phib[:, None] * special_2m[None, :]
        anti_re = anti_diag_sign[None, :] * rb_power_mag * xp.cos(rb_power_phase)
        anti_im = anti_diag_sign[None, :] * rb_power_mag * xp.sin(rb_power_phase)
        anti_mask = xp.logical_and(ra_small[:, None], anti_diagonal_mask[None, :])
        result_re = xp.where(anti_mask, anti_re, result_re)
        result_im = xp.where(anti_mask, anti_im, result_im)

        diagonal_mask = _xp_asarray(xp, coeffs.diagonal_mask)
        diag_rows = xp.logical_and(rb_small, xp.logical_not(ra_small))
        diag_log_ra = xp.where(diag_rows, log_ra, xp.zeros_like(log_ra))
        diag_phia = xp.where(diag_rows, phia, xp.zeros_like(phia))
        ra_power_mag = xp.exp(diag_log_ra[:, None] * special_2m[None, :])
        ra_power_phase = diag_phia[:, None] * special_2m[None, :]
        diag_re = ra_power_mag * xp.cos(ra_power_phase)
        diag_im = ra_power_mag * xp.sin(ra_power_phase)
        diag_mask = xp.logical_and(diag_rows[:, None], diagonal_mask[None, :])
        result_re = xp.where(diag_mask, diag_re, result_re)
        result_im = xp.where(diag_mask, diag_im, result_im)

        valid1 = _xp_asarray(xp, coeffs.case1.valid_mask)
        ratio1 = -(rb * rb) / (safe_ra * safe_ra)
        magnitude1 = WignerDCalculator._compute_case_magnitude(
            xp.where(use_case1, log_ra, xp.zeros_like(log_ra)),
            xp.where(use_case1, log_rb, xp.zeros_like(log_rb)),
            xp.where(use_case1, ratio1, xp.zeros_like(ratio1)),
            coeffs.case1,
        )
        val1_re = magnitude1 * exp_phase_re
        val1_im = magnitude1 * exp_phase_im
        mask1 = xp.logical_and(use_case1[:, None], valid1[None, :])
        result_re = xp.where(mask1, val1_re, result_re)
        result_im = xp.where(mask1, val1_im, result_im)

        valid2 = _xp_asarray(xp, coeffs.case2.valid_mask)
        ratio2 = -(ra * ra) / (safe_rb * safe_rb)
        magnitude2 = WignerDCalculator._compute_case_magnitude(
            xp.where(use_case2, log_ra, xp.zeros_like(log_ra)),
            xp.where(use_case2, log_rb, xp.zeros_like(log_rb)),
            xp.where(use_case2, ratio2, xp.zeros_like(ratio2)),
            coeffs.case2,
        )
        val2_re = magnitude2 * exp_phase_re
        val2_im = magnitude2 * exp_phase_im
        mask2 = xp.logical_and(use_case2[:, None], valid2[None, :])
        result_re = xp.where(mask2, val2_re, result_re)
        result_im = xp.where(mask2, val2_im, result_im)

        primary_flat = _xp_asarray(xp, coeffs.primary_flat, dtype=xp.int64)
        D_re = WignerDCalculator._scatter_flat(
            n_batch=n_batch,
            size=coeffs.size,
            flat_indices=primary_flat,
            values=result_re,
        )
        D_im = WignerDCalculator._scatter_flat(
            n_batch=n_batch,
            size=coeffs.size,
            flat_indices=primary_flat,
            values=result_im,
        )

        if coeffs.n_derived > 0:
            derived_primary_idx = _xp_asarray(
                xp,
                coeffs.derived_primary_idx,
                dtype=xp.int64,
            )
            primary_re = xp.take(result_re, derived_primary_idx, axis=1)
            primary_im = xp.take(result_im, derived_primary_idx, axis=1)
            derived_sign = _xp_asarray(xp, coeffs.derived_sign, dtype=calc_dtype)
            derived_re = derived_sign[None, :] * primary_re
            derived_im = -derived_sign[None, :] * primary_im
            derived_flat = _xp_asarray(xp, coeffs.derived_flat, dtype=xp.int64)
            D_re = D_re + WignerDCalculator._scatter_flat(
                n_batch=n_batch,
                size=coeffs.size,
                flat_indices=derived_flat,
                values=derived_re,
            )
            D_im = D_im + WignerDCalculator._scatter_flat(
                n_batch=n_batch,
                size=coeffs.size,
                flat_indices=derived_flat,
                values=derived_im,
            )

        return xp.astype(D_re, output_dtype), xp.astype(D_im, output_dtype)

    @staticmethod
    def _wigner_d_pair_to_real(
        D_re: Array,
        D_im: Array,
        U_blocks: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> Array:
        xp = array_api_compat.array_namespace(D_re, D_im)
        U_re, U_im, U_re_t, U_im_t = (
            _xp_asarray(xp, arr, dtype=D_re.dtype) for arr in U_blocks
        )
        temp_re = xp.matmul(D_re, U_re_t) + xp.matmul(D_im, U_im_t)
        temp_im = xp.matmul(D_im, U_re_t) - xp.matmul(D_re, U_im_t)
        return xp.matmul(U_re, temp_re) - xp.matmul(U_im, temp_im)

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
