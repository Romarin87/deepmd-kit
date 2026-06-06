# SPDX-License-Identifier: LGPL-3.0-or-later
"""Equivariant feed-forward layers for the staged JAX/dpmodel SeZM port."""

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

from .sezm_so2 import (
    GatedActivation,
)
from .sezm_so3 import (
    SO3Linear,
)


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
        self.act = GatedActivation(
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
            "s2_activation": self.s2_activation,
        }
        enabled = [name for name, active in unsupported.items() if active]
        if enabled:
            raise NotImplementedError(
                "JAX EquivariantFFN minimal path does not support: "
                + ", ".join(enabled)
            )

    def call(self, x: Any) -> Any:
        x = self.so3_linear_1(x)
        if self.glu_activation:
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
        obj.act = GatedActivation.deserialize(variables["act"])
        obj.so3_linear_2 = SO3Linear.deserialize(variables["so3_linear_2"])
        return obj
