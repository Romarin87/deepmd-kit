# SPDX-License-Identifier: LGPL-3.0-or-later
"""JAX SeZM atomic model compatibility wrapper."""

from __future__ import annotations

from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.jax.atomic_model.energy_atomic_model import (
    DPAtomicModelEnergy,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


def _native_layer_payload(
    weight: Any,
    bias: Any | None,
    *,
    precision: str,
    trainable: bool,
) -> dict[str, Any]:
    return {
        "@class": "Layer",
        "@version": 2,
        "bias": bias is not None,
        "use_timestep": False,
        "activation_function": "none",
        "resnet": False,
        "precision": precision,
        "trainable": trainable,
        "@variables": {
            "w": weight,
            "b": bias,
            "idt": None,
        },
    }


def _normalize_pt_glu_network(data: dict[str, Any]) -> dict[str, Any]:
    if "hidden_layers" in data and "output_layer" in data:
        return data
    variables = data.get("@variables")
    if variables is None:
        return data
    if data.get("case_film_embd", False):
        raise NotImplementedError(
            "JAX SeZM currently does not implement case_film_embd."
        )

    normalized = {
        key: deepcopy(value) for key, value in data.items() if key != "@variables"
    }
    precision = str(normalized.get("precision", "float64"))
    trainable = bool(normalized.get("trainable", True))
    activation_function = str(normalized.get("activation_function", "silu"))
    neuron = [int(item) for item in normalized.get("neuron", [])]
    hidden_layers = []
    dim_in = int(normalized["in_dim"])
    for layer_idx, hidden_dim in enumerate(neuron):
        prefix = f"hidden_layers.{layer_idx}.linear"
        weight = variables[f"{prefix}.matrix"]
        bias = variables.get(f"{prefix}.bias")
        hidden_layers.append(
            {
                "@class": "GLULayer",
                "@version": 1,
                "num_in": dim_in,
                "num_out": hidden_dim,
                "activation_function": activation_function,
                "precision": precision,
                "trainable": trainable,
                "bias": bias is not None,
                "linear": _native_layer_payload(
                    weight,
                    bias,
                    precision=precision,
                    trainable=trainable,
                ),
            }
        )
        dim_in = hidden_dim

    normalized["hidden_layers"] = hidden_layers
    normalized["output_layer"] = _native_layer_payload(
        variables["output_layer.matrix"],
        variables.get("output_layer.bias"),
        precision=precision,
        trainable=trainable,
    )
    return normalized


def _normalize_pt_sezm_fitting_payload(data: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(data)
    nets = payload.get("nets")
    if (
        not isinstance(nets, dict)
        or nets.get("network_type") != "sezm_fitting_network"
    ):
        return payload
    networks = nets.get("networks")
    if not isinstance(networks, list):
        return payload
    nets["networks"] = [
        None if item is None else _normalize_pt_glu_network(item)
        for item in networks
    ]
    return payload


class SeZMAtomicModel(DPAtomicModelEnergy):
    """Energy-only JAX SeZM atomic model.

    PyTorch serializes DPA4/SeZM as an atomic payload with
    ``type="sezm_atomic"`` and version 3.  The current JAX implementation only
    covers the energy branch, so this class keeps that public payload shape
    while explicitly rejecting the optional ``dens`` branch.
    """

    def __init__(
        self,
        descriptor: Any,
        fitting: Any,
        type_map: list[str],
        dens_fitting: Any | None = None,
        active_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        if dens_fitting is not None:
            raise NotImplementedError(
                "JAX SeZM currently supports only the `ener` branch; "
                "`dens` is not implemented."
            )
        active_mode = "ener" if active_mode is None else str(active_mode).lower()
        if active_mode != "ener":
            raise NotImplementedError(
                "JAX SeZM currently supports only active_mode='ener'."
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)
        self.dens_fitting_net = None
        self._active_mode = "ener"

    def get_active_mode(self) -> str:
        return str(getattr(self, "_active_mode", "ener"))

    def set_active_mode(self, mode: str) -> None:
        normalized = str(mode).lower()
        if normalized != "ener":
            raise NotImplementedError(
                "JAX SeZM currently supports only active_mode='ener'."
            )
        self._active_mode = "ener"

    def serialize(self) -> dict[str, Any]:
        data = super().serialize()
        data.setdefault("@variables", {})["dens_force_rmsd"] = np.asarray(1.0)
        data.update(
            {
                "@version": 3,
                "type": "sezm_atomic",
                "dens_fitting": None,
                "active_mode": self.get_active_mode(),
            }
        )
        return data

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SeZMAtomicModel":
        payload = data.copy()
        version = int(payload.get("@version", 2))
        check_version_compatibility(version, 3, 2)

        dens_payload = payload.pop("dens_fitting", None)
        if dens_payload is not None:
            raise NotImplementedError(
                "JAX SeZM currently supports only the `ener` branch; "
                "`dens` is not implemented."
            )
        active_mode = payload.pop("active_mode", None)
        active_mode = "ener" if active_mode is None else str(active_mode).lower()
        if active_mode != "ener":
            raise NotImplementedError(
                "JAX SeZM currently supports only active_mode='ener'."
            )

        variables = payload.get("@variables")
        if variables is not None:
            variables = variables.copy()
            variables.pop("dens_force_rmsd", None)
            payload["@variables"] = variables
        if "fitting" in payload:
            payload["fitting"] = _normalize_pt_sezm_fitting_payload(payload["fitting"])
        payload["@version"] = 2
        payload["type"] = "standard"

        obj = super().deserialize(payload)
        obj._active_mode = "ener"
        obj.dens_fitting_net = None
        return obj
