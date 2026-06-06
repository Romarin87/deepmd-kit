# SPDX-License-Identifier: LGPL-3.0-or-later
"""JAX SeZM/DPA4 energy model."""

from __future__ import annotations

from typing import (
    Any,
)

from deepmd.dpmodel.model import EnergyModel as EnergyModelDP
from deepmd.jax.atomic_model.sezm_atomic_model import (
    SeZMAtomicModel,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.dp_model import (
    make_jax_dp_model_from_dpmodel,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


@BaseModel.register("dpa4")
@BaseModel.register("DPA4")
@BaseModel.register("sezm")
@BaseModel.register("SeZM")
class SeZMModel(make_jax_dp_model_from_dpmodel(EnergyModelDP, SeZMAtomicModel)):
    """Energy-only JAX SeZM/DPA4 model wrapper.

    The descriptor and fitting implementation remain JAX eager code.  This
    wrapper aligns the public model identity and serialization contract with
    the PyTorch SeZM model so future parity work can add optional branches
    without changing callers again.
    """

    model_type = "SeZM"

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "Model",
            "@version": 1,
            "type": self.model_type,
            "atomic_model": self.atomic_model.serialize(),
            "bridging_method": "NONE",
            "bridging_r_inner": 0.5,
            "bridging_r_outer": 0.8,
            "lora": None,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SeZMModel":
        payload = data.copy()
        if "atomic_model" not in payload:
            return cls(atomic_model_=SeZMAtomicModel.deserialize(payload))

        version = int(payload.pop("@version", 1))
        check_version_compatibility(version, 1, 1)
        payload.pop("@class", None)
        payload.pop("type", None)
        atomic_model = SeZMAtomicModel.deserialize(payload.pop("atomic_model"))

        bridging_method = str(payload.pop("bridging_method", "NONE")).upper()
        payload.pop("bridging_r_inner", None)
        payload.pop("bridging_r_outer", None)
        if bridging_method != "NONE":
            raise NotImplementedError(
                "JAX SeZM currently does not implement zone bridging."
            )
        if payload.pop("lora", None) is not None:
            raise NotImplementedError("JAX SeZM currently does not implement LoRA.")
        if payload:
            unexpected = ", ".join(sorted(payload))
            raise ValueError(f"Unexpected JAX SeZM model payload keys: {unexpected}")
        return cls(atomic_model_=atomic_model)
