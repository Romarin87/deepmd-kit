# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import deepcopy

from deepmd.dpmodel.model.model_factory import (
    BackendModelFactory,
)
from deepmd.jax.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.jax.atomic_model.pairtab_atomic_model import (
    PairTabAtomicModel,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.dp_zbl_model import (
    DPZBLModel,
)

_model_factory = BackendModelFactory(
    descriptor_base=BaseDescriptor,
    fitting_base=BaseFitting,
    model_base=BaseModel,
    backend_name="JAX",
    atomic_model=DPAtomicModel,
    pairtab_model=PairTabAtomicModel,
    zbl_model=DPZBLModel,
)
get_standard_model = _model_factory.get_standard_model
get_zbl_model = _model_factory.get_zbl_model


def get_dpa4_model(data: dict) -> BaseModel:
    """Build the DPA4 energy path using its native GLU fitting defaults."""
    data = deepcopy(data)
    for key in ("spin", "lora", "shared_dict"):
        if data.get(key):
            raise NotImplementedError(f"DPA4 JAX does not yet support {key}.")
    if str(data.get("bridging_method", "none")).lower() not in ("none", ""):
        raise NotImplementedError("DPA4 JAX analytical bridging is not implemented.")
    descriptor = data.setdefault("descriptor", {})
    descriptor.setdefault("type", "dpa4")
    fitting = data.setdefault("fitting_net", {})
    fitting.setdefault("type", "dpa4_ener")
    if fitting["type"] not in ("dpa4_ener", "sezm_ener"):
        raise ValueError("DPA4 JAX requires dpa4_ener or sezm_ener fitting.")
    model_exclusions = data.get("pair_exclude_types")
    descriptor_exclusions = descriptor.get("exclude_types")
    if model_exclusions is not None and descriptor_exclusions is not None:

        def canonical(pairs: list[list[int]]) -> set[tuple[int, ...]]:
            return {tuple(sorted(pair)) for pair in pairs}

        if canonical(model_exclusions) != canonical(descriptor_exclusions):
            raise ValueError("Model and descriptor pair exclusions must agree.")
    exclusions = (
        model_exclusions if model_exclusions is not None else descriptor_exclusions
    )
    if exclusions is not None:
        data["pair_exclude_types"] = exclusions
        descriptor["exclude_types"] = exclusions
    return get_standard_model(data)


def get_model(data: dict) -> BaseModel:
    """Construct a JAX model, including the DPA4 model aliases."""
    return _model_factory.get_model(
        data,
        model_factories=dict.fromkeys(("dpa4", "DPA4", "sezm", "SeZM"), get_dpa4_model),
    )
