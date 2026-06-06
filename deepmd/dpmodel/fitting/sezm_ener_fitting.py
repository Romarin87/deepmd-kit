# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM/DPA4 GLU energy fitting for dpmodel and JAX backends."""

from __future__ import annotations

import math
from typing import (
    Any,
)

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
)
from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.fitting.invar_fitting import (
    InvarFitting,
)
from deepmd.dpmodel.utils import (
    GLUFittingNet,
    NetworkCollection,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


def _resolve_auto_neuron(
    neuron: list[int] | None,
    *,
    dim_descrpt: int,
    numb_fparam: int,
    numb_aparam: int,
    dim_case_embd: int,
    case_film_embd: bool,
    use_aparam_as_mask: bool,
) -> list[int]:
    """Resolve SeZM fitting hidden widths, using 0 as the auto-width marker."""
    resolved_neuron = [0] if neuron is None else [int(width) for width in neuron]
    if any(width < 0 for width in resolved_neuron):
        raise ValueError("`fitting_net.neuron` entries must be >= 0")
    if 0 not in resolved_neuron:
        return resolved_neuron
    case_dim = 0 if case_film_embd else int(dim_case_embd)
    dim_in = (
        int(dim_descrpt)
        + int(numb_fparam)
        + (0 if use_aparam_as_mask else int(numb_aparam))
        + case_dim
    )
    resolved_width = int(32 * math.ceil((8.0 * float(dim_in) / 3.0) / 32.0))
    return [resolved_width if width == 0 else width for width in resolved_neuron]


@InvarFitting.register("dpa4_ener")
@InvarFitting.register("sezm_ener")
class SeZMEnergyFittingNet(InvarFitting):
    """SeZM/DPA4 energy fitting with GLU hidden layers."""

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        neuron: list[int] | None = None,
        bias_atom_e: Array | None = None,
        resnet_dt: bool = False,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        case_film_embd: bool = False,
        rcond: float | None = None,
        tot_ener_zero: bool = False,
        trainable: list[bool] | bool | None = None,
        atom_ener: list[float] | None = None,
        activation_function: str = "silu",
        precision: str = DEFAULT_PRECISION,
        bias_out: bool = False,
        layer_name: list[str | None] | None = None,
        use_aparam_as_mask: bool = False,
        spin: Any = None,
        mixed_types: bool = True,
        exclude_types: list[int] = [],
        type_map: list[str] | None = None,
        seed: int | list[int] | None = None,
        default_fparam: list | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("_comment", None)
        if kwargs:
            raise TypeError(f"Unsupported SeZM energy fitting options: {kwargs}")
        self.seed = seed
        resolved_neuron = _resolve_auto_neuron(
            neuron,
            dim_descrpt=dim_descrpt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            case_film_embd=case_film_embd,
            use_aparam_as_mask=use_aparam_as_mask,
        )
        super().__init__(
            var_name="energy",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=1,
            neuron=resolved_neuron,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            bias_atom=bias_atom_e,
            rcond=rcond,
            tot_ener_zero=tot_ener_zero,
            trainable=trainable,
            atom_ener=atom_ener,
            activation_function=activation_function,
            precision=precision,
            layer_name=layer_name,
            use_aparam_as_mask=use_aparam_as_mask,
            spin=spin,
            mixed_types=mixed_types,
            exclude_types=exclude_types,
            type_map=type_map,
            seed=seed,
            default_fparam=default_fparam,
        )
        self.bias_out = bool(bias_out)
        self.case_film_embd = bool(case_film_embd and self.dim_case_embd > 0)
        self._build_glu_fitting_layers()

    def _build_glu_fitting_layers(self) -> None:
        case_dim = 0 if self.case_film_embd else self.dim_case_embd
        in_dim = (
            self.dim_descrpt
            + self.numb_fparam
            + (0 if self.use_aparam_as_mask else self.numb_aparam)
            + case_dim
        )
        net_dim_out = self._net_out_dim()
        n_networks = self.ntypes if not self.mixed_types else 1
        self.nets = NetworkCollection(
            1 if not self.mixed_types else 0,
            self.ntypes,
            network_type="sezm_fitting_network",
            networks=[
                GLUFittingNet(
                    in_dim,
                    net_dim_out,
                    self.neuron,
                    activation_function=self.activation_function,
                    resnet_dt=self.resnet_dt,
                    precision=self.precision,
                    bias_out=self.bias_out,
                    seed=child_seed(self.seed, idx),
                    trainable=self.trainable,
                    descriptor_dim=self.dim_descrpt,
                    dim_case_embd=self.dim_case_embd,
                    case_film_embd=self.case_film_embd,
                )
                for idx in range(n_networks)
            ],
        )

    @classmethod
    def deserialize(cls, data: dict) -> "SeZMEnergyFittingNet":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 4, 1)
        data.pop("var_name")
        data.pop("dim_out")
        return super().deserialize(data)

    def serialize(self) -> dict:
        return {
            **super().serialize(),
            "type": "sezm_ener",
            "bias_out": self.bias_out,
            "case_film_embd": self.case_film_embd,
        }
