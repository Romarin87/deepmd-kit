# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.sezm_norm import (
    EquivariantRMSNorm as EquivariantRMSNormDP,
    ScalarRMSNorm as ScalarRMSNormDP,
)
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    to_jax_array,
)
from deepmd.jax.utils.network import (
    ArrayAPIParam,
)


@flax_module
class EquivariantRMSNorm(EquivariantRMSNormDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"adam_scale", "bias"}:
            value = to_jax_array(value)
            if value is not None:
                if getattr(self, "trainable", True):
                    value = ArrayAPIParam(value)
                else:
                    value = ArrayAPIVariable(value)
        elif name in {"expand_index", "balance_weight"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
        return super().__setattr__(name, value)


@flax_module
class ScalarRMSNorm(ScalarRMSNormDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"adam_scale"}:
            value = to_jax_array(value)
            if value is not None:
                if getattr(self, "trainable", True):
                    value = ArrayAPIParam(value)
                else:
                    value = ArrayAPIVariable(value)
        return super().__setattr__(name, value)
