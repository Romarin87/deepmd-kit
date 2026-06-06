# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from packaging.version import (
    Version,
)

from deepmd.dpmodel.descriptor.sezm import (
    C3CutoffEnvelope as C3CutoffEnvelopeDP,
    DescrptSeZM as DescrptSeZMDP,
    RadialBasis as RadialBasisDP,
    RadialMLP as RadialMLPDP,
    RMSNorm as RMSNormDP,
    SeZMTypeEmbedding as SeZMTypeEmbeddingDP,
)
from deepmd.dpmodel.descriptor.sezm_wignerd import (
    WignerDCalculator as WignerDCalculatorDP,
    build_edge_quaternion,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_rotation_matrix,
    quaternion_z_rotation,
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
from deepmd.jax.utils.network import (
    ArrayAPIParam,
    NativeLayer,
)


def _maybe_nnx_list(value: list[Any]) -> Any:
    if Version(flax_version) >= Version("0.12.0"):
        return nnx.List(value)
    return value


@flax_module
class SeZMTypeEmbedding(SeZMTypeEmbeddingDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"embedding"}:
            value = to_jax_array(value)
            if value is not None:
                if getattr(self, "trainable", True):
                    value = ArrayAPIParam(value)
                else:
                    value = ArrayAPIVariable(value)
        return super().__setattr__(name, value)


@flax_module
class C3CutoffEnvelope(C3CutoffEnvelopeDP):
    pass


@flax_module
class RadialBasis(RadialBasisDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"freqs"}:
            value = to_jax_array(value)
            if value is not None:
                if getattr(self, "trainable", True):
                    value = ArrayAPIParam(value)
                else:
                    value = ArrayAPIVariable(value)
        elif name in {"envelope"}:
            value = C3CutoffEnvelope.deserialize(value.serialize())
        return super().__setattr__(name, value)


@flax_module
class RMSNorm(RMSNormDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"scale"}:
            value = to_jax_array(value)
            if value is not None:
                if getattr(self, "trainable", True):
                    value = ArrayAPIParam(value)
                else:
                    value = ArrayAPIVariable(value)
        return super().__setattr__(name, value)


@flax_module
class RadialMLP(RadialMLPDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"layers"}:
            value = [
                layer
                if isinstance(layer, NativeLayer)
                else NativeLayer.deserialize(layer.serialize())
                for layer in value
            ]
            value = _maybe_nnx_list(value)
        elif name in {"norms"}:
            value = [
                norm
                if isinstance(norm, RMSNorm)
                else RMSNorm.deserialize(norm.serialize())
                for norm in value
            ]
            value = _maybe_nnx_list(value)
        return super().__setattr__(name, value)


@flax_module
class WignerDCalculator(WignerDCalculatorDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"l1_perm", "l1_sign_outer"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
        return super().__setattr__(name, value)


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
        elif name in {"type_embedding"}:
            if not isinstance(value, SeZMTypeEmbedding):
                value = SeZMTypeEmbedding.deserialize(value.serialize())
        elif name in {"radial_basis"}:
            if not isinstance(value, RadialBasis):
                value = RadialBasis.deserialize(value.serialize())
        elif name in {"edge_envelope"}:
            if not isinstance(value, C3CutoffEnvelope):
                value = C3CutoffEnvelope.deserialize(value.serialize())
        elif name in {"radial_embedding"}:
            if not isinstance(value, RadialMLP):
                value = RadialMLP.deserialize(value.serialize())
        elif name in {"wigner_calc"}:
            if not isinstance(value, WignerDCalculator):
                value = WignerDCalculator.deserialize(value.serialize())
        return super().__setattr__(name, value)
