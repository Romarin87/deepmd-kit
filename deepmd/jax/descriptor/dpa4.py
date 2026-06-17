# SPDX-License-Identifier: LGPL-3.0-or-later
import deepmd.jax.utils.exclude_mask as _jax_exclude_mask  # noqa: F401
import deepmd.jax.utils.network as _jax_network  # noqa: F401
from deepmd.dpmodel.descriptor.dpa4 import (
    DescrptDPA4 as DescrptDPA4DP,
)
from deepmd.jax.common import (
    flax_module,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("SeZM")
@BaseDescriptor.register("sezm")
@BaseDescriptor.register("DPA4")
@BaseDescriptor.register("dpa4")
@flax_module
class DescrptDPA4(DescrptDPA4DP):
    pass
