# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from packaging.version import (
    Version,
)

from deepmd.dpmodel.descriptor.sezm import (
    DescrptSeZM as DescrptSeZMDP,
)
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    to_jax_array,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.env import (
    flax_version,
    nnx,
)


@BaseDescriptor.register("SeZM")
@BaseDescriptor.register("sezm")
@BaseDescriptor.register("DPA4")
@BaseDescriptor.register("dpa4")
@flax_module
class DescrptSeZM(DescrptSeZMDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"mean", "stddev"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
            elif Version(flax_version) >= Version("0.12.0"):
                value = nnx.data(value)
        return super().__setattr__(name, value)
