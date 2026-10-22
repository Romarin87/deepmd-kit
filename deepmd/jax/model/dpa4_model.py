# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 energy-model dispatch for JAX."""

from copy import deepcopy

from deepmd.jax.model.base_model import BaseModel
from deepmd.jax.model.ener_model import EnergyModel
from deepmd.utils.version import check_version_compatibility


@BaseModel.register("dpa4")
@BaseModel.register("DPA4")
@BaseModel.register("sezm")
@BaseModel.register("SeZM")
@BaseModel.register("dpa4_ener")
@BaseModel.register("sezm_ener")
class DPA4EnergyModel(EnergyModel):
    @classmethod
    def deserialize(cls, data: dict) -> "DPA4EnergyModel":
        """Accept portable atomic records and PyTorch DPA4 energy envelopes."""
        data = deepcopy(data)
        if "atomic_model" in data:
            check_version_compatibility(data.get("@version", 1), 1, 1)
            if str(data.get("bridging_method", "none")).lower() not in ("none", ""):
                raise NotImplementedError(
                    "DPA4 JAX analytical bridging is not implemented."
                )
            if data.get("lora"):
                raise NotImplementedError(
                    "Merge LoRA weights before importing DPA4 into JAX."
                )
            data = data["atomic_model"]
            check_version_compatibility(data.get("@version", 2), 3, 2)
            if data.pop("active_mode", "ener") not in (None, "ener"):
                raise NotImplementedError("DPA4 JAX imports the energy mode only.")
            if data.pop("dens_fitting", None) is not None:
                raise NotImplementedError(
                    "Export the DPA4 energy head before importing into JAX."
                )
            data.get("@variables", {}).pop("dens_force_rmsd", None)
            data["type"] = "standard"
            data["@version"] = 2
        return super().deserialize(data)
