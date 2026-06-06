# SPDX-License-Identifier: LGPL-3.0-or-later
"""Equivariant feed-forward layers for the staged JAX/dpmodel SeZM port."""

from __future__ import annotations

from typing import (
    Any,
)

import math

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
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .sezm_so2 import (
    GatedActivation,
)
from .sezm_so3 import (
    FocusLinear,
    SO3Linear,
)
from .sezm_indexing import (
    build_l_major_index,
)
from .sezm_lebedev import (
    LEBEDEV_PRECISION_TO_NPOINTS,
    load_lebedev_rule,
)


def resolve_s2_grid_resolution(
    lmax: int,
    mmax: int,
    *,
    method: str = "lebedev",
) -> list[int]:
    method = str(method).lower()
    if method != "lebedev":
        raise NotImplementedError("JAX SeZM S2 activation supports only Lebedev.")
    required_precision = 3 * int(lmax)
    for precision, n_points in LEBEDEV_PRECISION_TO_NPOINTS.items():
        if precision >= required_precision:
            return [precision, n_points]
    raise ValueError(f"No packaged Lebedev rule has precision >= {required_precision}")


class S2GridProjector(NativeOP):
    """SO(3) coefficient <-> Lebedev S2 grid projector."""

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        precision: str = DEFAULT_PRECISION,
        grid_resolution_list: list[int] | None = None,
        coefficient_layout: str = "packed",
        grid_method: str = "lebedev",
    ) -> None:
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        self.precision = precision
        self.coefficient_layout = str(coefficient_layout).lower()
        if self.coefficient_layout != "packed":
            raise NotImplementedError("JAX S2GridProjector supports only packed layout.")
        self.grid_method = str(grid_method).lower()
        if self.grid_method != "lebedev":
            raise NotImplementedError("JAX S2GridProjector supports only Lebedev.")
        self.grid_resolution_list = resolve_s2_grid_resolution(
            self.lmax,
            self.mmax,
            method=self.grid_method,
        ) if grid_resolution_list is None else [int(x) for x in grid_resolution_list]
        self.lebedev_precision = self.grid_resolution_list[0]
        self.lebedev_npoints = self.grid_resolution_list[1]
        coeff_index = self._build_coefficient_index()
        self.coeff_dim = int(coeff_index.shape[0])
        self.to_grid_mat, self.from_grid_mat = self._build_projection_mats(coeff_index)

    def _build_coefficient_index(self) -> np.ndarray:
        if self.mmax == self.lmax:
            return np.arange((self.lmax + 1) ** 2, dtype=np.int64)
        return build_l_major_index(self.lmax, self.mmax)

    def _build_projection_mats(
        self,
        coeff_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        points, weights = load_lebedev_rule(
            self.lebedev_precision,
            float_precision="float64",
        )
        harmonics = _real_spherical_harmonics_np(self.lmax, points)
        scale = math.sqrt(float(self.lmax + 1))
        degree_factors = np.asarray(
            [
                float(2 * degree + 1)
                for degree in range(self.lmax + 1)
                for _ in range(2 * degree + 1)
            ],
            dtype=np.float64,
        )
        to_grid_mat = harmonics / scale
        from_grid_mat = harmonics * (
            weights[:, None].astype(np.float64) * scale * degree_factors[None, :]
        )
        dtype = PRECISION_DICT[self.precision.lower()]
        return (
            to_grid_mat[:, coeff_index].astype(dtype),
            from_grid_mat[:, coeff_index].T.astype(dtype),
        )

    def to_grid(self, embedding: Array) -> Array:
        xp = array_api_compat.array_namespace(embedding, self.to_grid_mat)
        return xp.einsum("aj,njc->nac", self.to_grid_mat[...], embedding)

    def call(self, embedding: Array) -> Array:
        return self.to_grid(embedding)

    def from_grid(self, grid: Array) -> Array:
        xp = array_api_compat.array_namespace(grid, self.from_grid_mat)
        return xp.einsum("ja,nac->njc", self.from_grid_mat[...], grid)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "S2GridProjector",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "precision": self.precision,
                "grid_resolution_list": self.grid_resolution_list,
                "coefficient_layout": self.coefficient_layout,
                "grid_method": self.grid_method,
            },
            "@variables": {
                "to_grid_mat": to_numpy_array(self.to_grid_mat[...]),
                "from_grid_mat": to_numpy_array(self.from_grid_mat[...]),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "S2GridProjector":
        data = data.copy()
        data_cls = data.pop("@class", None)
        if data_cls != "S2GridProjector":
            raise ValueError(f"Invalid class for S2GridProjector: {data_cls}")
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables", {})
        obj = cls(**config)
        if "to_grid_mat" in variables:
            obj.to_grid_mat = variables["to_grid_mat"]
        if "from_grid_mat" in variables:
            obj.from_grid_mat = variables["from_grid_mat"]
        return obj


class SwiGLUS2Activation(NativeOP):
    """Merged scalar/grid SwiGLU activation on a Lebedev S2 grid."""

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        precision: str = DEFAULT_PRECISION,
        n_focus: int = 1,
        layout: str = "ndfc",
        grid_resolution_list: list[int] | None = None,
        coefficient_layout: str = "packed",
        grid_method: str = "lebedev",
        mlp_bias: bool = False,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.precision = precision
        self.n_focus = int(n_focus)
        self.layout = str(layout).lower()
        if self.layout not in {"ndfc", "nfdc"}:
            raise ValueError("`layout` must be either 'ndfc' or 'nfdc'")
        self.grid_resolution_list = grid_resolution_list
        self.coefficient_layout = str(coefficient_layout)
        self.grid_method = str(grid_method)
        self.mlp_bias = bool(mlp_bias)
        self.trainable = bool(trainable)
        self.scalar_gate = FocusLinear(
            in_channels=2 * self.channels,
            out_channels=self.channels,
            n_focus=self.n_focus,
            precision=self.precision,
            bias=self.mlp_bias,
            trainable=self.trainable,
            seed=child_seed(seed, 0),
            init_std=0.01,
        )
        self.projector = (
            None
            if self.lmax == 0
            else S2GridProjector(
                lmax=self.lmax,
                mmax=self.lmax,
                precision=self.precision,
                grid_resolution_list=self.grid_resolution_list,
                coefficient_layout=self.coefficient_layout,
                grid_method=self.grid_method,
            )
        )
        if self.projector is not None:
            self.grid_resolution_list = self.projector.grid_resolution_list

    def call(self, x: Array) -> Array:
        scalar_inputs = self._extract_scalar_inputs(x)
        scalar_outputs = _swiglu(scalar_inputs)
        if self.projector is None:
            return self._restore_scalar_outputs(scalar_outputs)
        gate_scalars = xp_sigmoid(self.scalar_gate(scalar_inputs))
        x_flat, shape_info = self._flatten_inputs(x)
        x_grid = self.projector.to_grid(x_flat)
        x_grid_1 = x_grid[..., : self.channels]
        x_grid_2 = x_grid[..., self.channels :]
        out_flat = self.projector.from_grid(x_grid_1 * x_grid_2)
        outputs = self._restore_outputs(out_flat, shape_info)
        outputs = outputs * self._broadcast_scalar_gate(gate_scalars)
        return self._merge_scalar_outputs(outputs, scalar_outputs)

    def _extract_scalar_inputs(self, x: Array) -> Array:
        if self.layout == "ndfc":
            return x[:, 0, :, :]
        return x[:, :, 0, :]

    def _broadcast_scalar_gate(self, gate_scalars: Array) -> Array:
        xp = array_api_compat.array_namespace(gate_scalars)
        if self.layout == "ndfc":
            return xp.expand_dims(gate_scalars, axis=1)
        return xp.expand_dims(gate_scalars, axis=2)

    def _restore_scalar_outputs(self, scalar_outputs: Array) -> Array:
        xp = array_api_compat.array_namespace(scalar_outputs)
        if self.layout == "ndfc":
            return xp.expand_dims(scalar_outputs, axis=1)
        return xp.expand_dims(scalar_outputs, axis=2)

    def _flatten_inputs(self, x: Array) -> tuple[Array, tuple[int, int, int]]:
        xp = array_api_compat.array_namespace(x)
        if self.layout == "ndfc":
            n_batch, coeff_dim, n_focus, _ = x.shape
            return (
                xp.reshape(
                    xp.permute_dims(x, (0, 2, 1, 3)),
                    (n_batch * n_focus, coeff_dim, x.shape[-1]),
                ),
                (n_batch, coeff_dim, n_focus),
            )
        n_batch, n_focus, coeff_dim, _ = x.shape
        return (
            xp.reshape(x, (n_batch * n_focus, coeff_dim, x.shape[-1])),
            (n_batch, coeff_dim, n_focus),
        )

    def _restore_outputs(
        self,
        x: Array,
        shape_info: tuple[int, int, int],
    ) -> Array:
        xp = array_api_compat.array_namespace(x)
        n_batch, coeff_dim, n_focus = shape_info
        if self.layout == "ndfc":
            return xp.permute_dims(
                xp.reshape(x, (n_batch, n_focus, coeff_dim, self.channels)),
                (0, 2, 1, 3),
            )
        return xp.reshape(x, (n_batch, n_focus, coeff_dim, self.channels))

    def _merge_scalar_outputs(self, outputs: Array, scalar_outputs: Array) -> Array:
        if self.layout == "ndfc":
            if hasattr(outputs, "at"):
                return outputs.at[:, 0, :, :].add(scalar_outputs)
            outputs[:, 0, :, :] = outputs[:, 0, :, :] + scalar_outputs
            return outputs
        if hasattr(outputs, "at"):
            return outputs.at[:, :, 0, :].add(scalar_outputs)
        outputs[:, :, 0, :] = outputs[:, :, 0, :] + scalar_outputs
        return outputs

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SwiGLUS2Activation",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "channels": self.channels,
                "precision": self.precision,
                "n_focus": self.n_focus,
                "layout": self.layout,
                "grid_resolution_list": self.grid_resolution_list,
                "coefficient_layout": self.coefficient_layout,
                "grid_method": self.grid_method,
                "mlp_bias": self.mlp_bias,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": {
                "scalar_gate": self.scalar_gate.serialize(),
                "projector": (
                    None if self.projector is None else self.projector.serialize()
                ),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SwiGLUS2Activation":
        data = data.copy()
        data_cls = data.pop("@class", None)
        if data_cls != "SwiGLUS2Activation":
            raise ValueError(f"Invalid class for SwiGLUS2Activation: {data_cls}")
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj.scalar_gate = FocusLinear.deserialize(variables["scalar_gate"])
        obj.projector = (
            None
            if variables["projector"] is None
            else S2GridProjector.deserialize(variables["projector"])
        )
        return obj


def _swiglu(inputs: Array) -> Array:
    gate = inputs[..., : inputs.shape[-1] // 2]
    value = inputs[..., inputs.shape[-1] // 2 :]
    return (gate * xp_sigmoid(gate)) * value


def _real_spherical_harmonics_np(lmax: int, unit_vec: np.ndarray) -> np.ndarray:
    x = unit_vec[:, 0]
    y = unit_vec[:, 1]
    z = np.clip(unit_vec[:, 2], -1.0, 1.0)
    phi = np.arctan2(y, x)
    legendre = _associated_legendre_all_np(lmax, z)
    norm = _build_real_sh_norm_np(lmax)
    out = np.zeros((unit_vec.shape[0], (lmax + 1) ** 2), dtype=np.float64)
    sqrt_2 = math.sqrt(2.0)
    for degree in range(lmax + 1):
        for order in range(degree + 1):
            base = legendre[degree, order] * norm[degree, order] * math.sqrt(
                4.0 * math.pi
            )
            zero_idx = degree * degree + degree
            if order == 0:
                out[:, zero_idx] = base
                continue
            out[:, zero_idx - order] = sqrt_2 * base * np.sin(order * phi)
            out[:, zero_idx + order] = sqrt_2 * base * np.cos(order * phi)
    return out


def _build_real_sh_norm_np(lmax: int) -> np.ndarray:
    norm = np.zeros((lmax + 1, lmax + 1), dtype=np.float64)
    for degree in range(lmax + 1):
        for order in range(degree + 1):
            norm[degree, order] = math.sqrt(
                (2 * degree + 1)
                / (4.0 * math.pi)
                * math.exp(
                    math.lgamma(degree - order + 1)
                    - math.lgamma(degree + order + 1)
                )
            )
    return norm


def _associated_legendre_all_np(lmax: int, x: np.ndarray) -> np.ndarray:
    out = np.zeros((lmax + 1, lmax + 1, x.shape[0]), dtype=np.float64)
    out[0, 0] = 1.0
    if lmax == 0:
        return out
    sin_theta = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    for order in range(1, lmax + 1):
        out[order, order] = (
            -(2 * order - 1) * sin_theta * out[order - 1, order - 1]
        )
    for order in range(lmax):
        out[order + 1, order] = (2 * order + 1) * x * out[order, order]
    for order in range(lmax + 1):
        for degree in range(order + 2, lmax + 1):
            out[degree, order] = (
                (2 * degree - 1) * x * out[degree - 1, order]
                - (degree + order - 1) * out[degree - 2, order]
            ) / float(degree - order)
    return out


class EquivariantFFN(NativeOP):
    """Minimal SO(3)-equivariant FFN for SeZM interaction blocks."""

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        hidden_channels: int,
        grid_mlp: bool = False,
        precision: str = DEFAULT_PRECISION,
        s2_activation: bool = False,
        lebedev_quadrature: bool = False,
        activation_function: str = "silu",
        glu_activation: bool = True,
        mlp_bias: bool = False,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        if self.channels < 1:
            raise ValueError("`channels` must be positive")
        if self.hidden_channels < 1:
            raise ValueError("`hidden_channels` must be positive")
        self.use_grid_mlp = bool(grid_mlp)
        self.s2_activation = bool(s2_activation)
        self.lebedev_quadrature = bool(lebedev_quadrature)
        self.activation_function = str(activation_function)
        self.glu_activation = bool(glu_activation)
        self.mlp_bias = bool(mlp_bias)
        self.precision = precision
        self.trainable = bool(trainable)
        self._validate_minimal_path()
        seed_so3_in = child_seed(seed, 0)
        seed_act = child_seed(seed, 1)
        seed_so3_out = child_seed(seed, 2)

        linear1_out_channels = (
            2 * self.hidden_channels
            if self.glu_activation
            else self.hidden_channels
        )
        self.so3_linear_1 = SO3Linear(
            lmax=self.lmax,
            in_channels=self.channels,
            out_channels=linear1_out_channels,
            n_focus=1,
            precision=self.precision,
            mlp_bias=self.mlp_bias,
            trainable=self.trainable,
            seed=seed_so3_in,
        )
        self.act = (
            SwiGLUS2Activation(
                lmax=self.lmax,
                channels=self.hidden_channels,
                precision=self.precision,
                n_focus=1,
                layout="ndfc",
                coefficient_layout="packed",
                grid_method="lebedev",
                mlp_bias=self.mlp_bias,
                trainable=self.trainable,
                seed=seed_act,
            )
            if self.s2_activation
            else GatedActivation(
                lmax=self.lmax,
                channels=self.hidden_channels,
                n_focus=1,
                precision=self.precision,
                activation_function=self.activation_function,
                mlp_bias=self.mlp_bias,
                layout="ndfc",
                trainable=self.trainable,
                seed=seed_act,
            )
        )
        self.so3_linear_2 = SO3Linear(
            lmax=self.lmax,
            in_channels=self.hidden_channels,
            out_channels=self.channels,
            n_focus=1,
            precision=self.precision,
            mlp_bias=self.mlp_bias,
            trainable=self.trainable,
            seed=seed_so3_out,
            init_std=0.0,
        )

    def _validate_minimal_path(self) -> None:
        unsupported = {
            "grid_mlp": self.use_grid_mlp,
            "s2_activation_without_lebedev": (
                self.s2_activation and not self.lebedev_quadrature
            ),
        }
        enabled = [name for name, active in unsupported.items() if active]
        if enabled:
            raise NotImplementedError(
                "JAX EquivariantFFN minimal path does not support: "
                + ", ".join(enabled)
            )

    def call(self, x: Any) -> Any:
        x = self.so3_linear_1(x)
        if self.s2_activation:
            x = self.act(x)
        elif self.glu_activation:
            x_val = x[..., : self.hidden_channels]
            x_gate = x[..., self.hidden_channels :]
            x = self.act(x_val, gate=x_gate)
        else:
            x = self.act(x)
        return self.so3_linear_2(x)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "EquivariantFFN",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "channels": self.channels,
                "hidden_channels": self.hidden_channels,
                "grid_mlp": self.use_grid_mlp,
                "precision": self.precision,
                "s2_activation": self.s2_activation,
                "lebedev_quadrature": self.lebedev_quadrature,
                "activation_function": self.activation_function,
                "glu_activation": self.glu_activation,
                "mlp_bias": self.mlp_bias,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": {
                "so3_linear_1": self.so3_linear_1.serialize(),
                "act": self.act.serialize(),
                "so3_linear_2": self.so3_linear_2.serialize(),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "EquivariantFFN":
        data = data.copy()
        data_cls = data.pop("@class", None)
        if data_cls != "EquivariantFFN":
            raise ValueError(f"Invalid class for EquivariantFFN: {data_cls}")
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj.so3_linear_1 = SO3Linear.deserialize(variables["so3_linear_1"])
        act_cls = variables["act"].get("@class")
        obj.act = (
            SwiGLUS2Activation.deserialize(variables["act"])
            if act_cls == "SwiGLUS2Activation"
            else GatedActivation.deserialize(variables["act"])
        )
        obj.so3_linear_2 = SO3Linear.deserialize(variables["so3_linear_2"])
        return obj
