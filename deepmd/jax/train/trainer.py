#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Local training utilities for the JAX backend."""

import logging
import os
import platform
import shutil
import time
from contextlib import (
    contextmanager,
)
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
    Iterator,
    TextIO,
)

import numpy as np
import optax
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from packaging.version import (
    Version,
)

from deepmd.dpmodel.loss.ener import (
    EnergyHessianLoss,
    EnergyLoss,
)
from deepmd.dpmodel.model.transform_output import (
    communicate_extended_output,
)
from deepmd.dpmodel.utils.learning_rate import (
    BaseLR,
)
from deepmd.dpmodel.utils.training_utils import (
    compute_total_numb_batch,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.jax.env import (
    flax_version,
    jax,
    jnp,
    nnx,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.model import (
    get_model,
)
from deepmd.jax.optimizer import (
    hybrid_muon,
)
from deepmd.jax.utils.serialization import (
    pack_zero_size_arrays_for_orbax,
    serialize_from_file,
)
from deepmd.loggers.training import (
    format_training_message,
    format_training_message_per_task,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
    prob_sys_size_ext,
    process_sys_probs,
)
from deepmd.utils.model_stat import (
    make_stat_input,
)

log = logging.getLogger(__name__)

EMA_DECAY_KEY = "decay"
EMA_MODEL_STATE_KEY = "model"
_ACTIVE_JAX_MESH_CONTEXT: list[Any] = []


def _clear_jax_mesh_for_host_ops() -> None:
    _set_nnx_eager_sharding(False)
    _set_jax_mesh(Mesh(np.empty((), dtype=object), ()))


def _set_nnx_eager_sharding(enabled: bool) -> None:
    use_eager_sharding = getattr(nnx, "use_eager_sharding", None)
    if use_eager_sharding is not None:
        use_eager_sharding(enabled)


def _set_jax_mesh(mesh: Mesh) -> None:
    """Set the global mesh across JAX versions used by users."""
    if _ACTIVE_JAX_MESH_CONTEXT:
        _ACTIVE_JAX_MESH_CONTEXT.pop().__exit__(None, None, None)

    set_mesh = getattr(jax, "set_mesh", None)
    if set_mesh is not None:
        set_mesh(mesh)
        return

    enter = getattr(mesh, "__enter__", None)
    if enter is None or getattr(mesh, "__exit__", None) is None:
        raise AttributeError("This JAX version cannot set a global mesh.")

    enter()
    _ACTIVE_JAX_MESH_CONTEXT.append(mesh)


def _normalize_jax_parallel_mode(parallel_mode: str | None) -> str:
    """Normalize JAX local training parallel mode aliases."""
    mode = str(parallel_mode or "auto").lower().replace("-", "_")
    alias = {
        "": "auto",
        "default": "auto",
        "dp": "data",
        "data_parallel": "data",
        "batch": "data",
        "atom": "natoms",
        "atoms": "natoms",
        "natoms_parallel": "natoms",
    }
    mode = alias.get(mode, mode)
    if mode not in {"auto", "data", "natoms"}:
        raise ValueError(
            "JAX training parallel mode must be 'auto', 'data', or 'natoms', "
            f"got {parallel_mode!r}."
        )
    return mode


def _model_prefers_data_parallel(
    model_def_script: dict[str, Any],
    has_hessian: bool,
) -> bool:
    """Return whether the model should avoid atom-axis SPMD by default."""
    model_type = str(model_def_script.get("type", "")).lower()
    descriptor = model_def_script.get("descriptor", {})
    descriptor_type = (
        str(descriptor.get("type", "")).lower() if isinstance(descriptor, dict) else ""
    )
    return has_hessian or model_type in {"sezm", "dpa4"} or descriptor_type in {
        "sezm",
        "dpa4",
    }


def _resolve_jax_parallel_mode(
    requested_mode: str,
    model_def_script: dict[str, Any],
    has_hessian: bool,
) -> str:
    """Resolve 'auto' to the concrete local JAX sharding strategy."""
    mode = _normalize_jax_parallel_mode(requested_mode)
    if mode != "auto":
        return mode
    if _model_prefers_data_parallel(model_def_script, has_hessian):
        return "data"
    return "natoms"


def _make_training_mesh(parallel_mode: str) -> Mesh:
    """Create the global JAX mesh for the selected training parallel mode."""
    if parallel_mode == "data":
        return jax.make_mesh((jax.device_count(),), ("data",))
    if parallel_mode == "natoms":
        return jax.make_mesh(
            (jax.process_count(), jax.local_device_count()),
            ("data", "natoms"),
        )
    raise ValueError(f"Unsupported JAX parallel mode {parallel_mode!r}.")


def _mesh_axis_size(mesh: Mesh, axis_name: str) -> int:
    """Return the size of a mesh axis if present, otherwise one."""
    return int(getattr(mesh, "shape", {}).get(axis_name, 1))


def _data_partition_spec(
    key: str,
    value: jnp.ndarray,
    *,
    data_axis_size: int = 1,
    natoms_axis_size: int = 1,
) -> P:
    """Choose the input sharding spec for a JAX training array."""
    data_axis = None
    if (
        data_axis_size > 1
        and value.ndim >= 1
        and value.shape[0] % data_axis_size == 0
    ):
        data_axis = "data"
    if key in {"energy", "box", "numb_copy", "virial", "real_natoms_vec"}:
        return P(data_axis) if data_axis is not None else P()
    natoms_axis = None
    if (
        natoms_axis_size > 1
        and value.ndim >= 2
        and value.shape[1] % natoms_axis_size == 0
    ):
        natoms_axis = "natoms"
    if data_axis is None and natoms_axis is None:
        return P()
    if natoms_axis is None:
        return P(data_axis)
    return P(data_axis, natoms_axis)


def _loss_uses_hessian(loss_param: dict) -> bool:
    return loss_param.get("type", "ener") == "ener" and loss_param.get(
        "start_pref_h", 0.0
    ) > 0.0


def _enable_hessian_output(model: BaseModel) -> None:
    if not hasattr(model, "enable_hessian"):
        raise NotImplementedError(
            f"JAX model {type(model).__name__} does not support Hessian output."
        )
    model.enable_hessian()


def _scatter_extended_to_local(
    value: jnp.ndarray,
    mapping: jnp.ndarray | None,
    nloc: int,
) -> jnp.ndarray:
    if mapping is None or mapping.shape[0] == nloc:
        return value[:nloc]
    return jnp.zeros((nloc, *value.shape[1:]), dtype=value.dtype).at[mapping].add(value)


def _communicate_or_passthrough(
    model_dict_lower: dict[str, jnp.ndarray],
    model_output_def: Any,
    mapping: jnp.ndarray | None,
    nloc: int,
) -> dict[str, jnp.ndarray]:
    if mapping is None or mapping.shape[1] == nloc:
        return model_dict_lower
    return communicate_extended_output(
        model_dict_lower,
        model_output_def,
        mapping,
        do_atomic_virial=False,
    )


def _call_hessian_local_block(
    model: BaseModel,
    extended_coord: jnp.ndarray,
    extended_atype: jnp.ndarray,
    nlist: jnp.ndarray,
    mapping: jnp.ndarray | None,
    fparam: jnp.ndarray | None,
    aparam: jnp.ndarray | None,
    block_index: jnp.ndarray,
    chunk_size: int,
) -> jnp.ndarray:
    """Return local Hessian rows [nf, chunk_size, nloc * 3]."""
    block_index = jnp.asarray(block_index, dtype=jnp.int32)

    def energy_one(
        coord_one: jnp.ndarray,
        atype_one: jnp.ndarray,
        nlist_one: jnp.ndarray,
        mapping_one: jnp.ndarray | None,
        fparam_one: jnp.ndarray | None,
        aparam_one: jnp.ndarray | None,
    ) -> jnp.ndarray:
        output = model.call_common_lower(
            coord_one[None, ...],
            atype_one[None, ...],
            nlist_one[None, ...],
            mapping=mapping_one[None, ...] if mapping_one is not None else None,
            fparam=fparam_one[None, ...] if fparam_one is not None else None,
            aparam=aparam_one[None, ...] if aparam_one is not None else None,
            do_atomic_virial=False,
        )
        return jnp.sum(output["energy_redu"])

    energy_grad_one = jax.grad(energy_one, argnums=0)

    def hessian_block_one(
        coord_one: jnp.ndarray,
        atype_one: jnp.ndarray,
        nlist_one: jnp.ndarray,
        mapping_one: jnp.ndarray | None,
        fparam_one: jnp.ndarray | None,
        aparam_one: jnp.ndarray | None,
    ) -> jnp.ndarray:
        nloc = nlist_one.shape[0]
        local_dim = nloc * 3
        row_ids = block_index * chunk_size + jnp.arange(
            chunk_size,
            dtype=block_index.dtype,
        )
        safe_row_ids = jnp.minimum(row_ids, local_dim - 1)

        def local_energy_grad(coord_ext: jnp.ndarray) -> jnp.ndarray:
            grad_ext = energy_grad_one(
                coord_ext,
                atype_one,
                nlist_one,
                mapping_one,
                fparam_one,
                aparam_one,
            )
            return _scatter_extended_to_local(grad_ext, mapping_one, nloc)

        def hessian_row(row_id: jnp.ndarray) -> jnp.ndarray:
            row_ext = jax.grad(
                lambda coord_ext: local_energy_grad(coord_ext).reshape(-1)[row_id]
            )(coord_one)
            return _scatter_extended_to_local(row_ext, mapping_one, nloc).reshape(-1)

        rows = jax.vmap(hessian_row)(safe_row_ids)
        return jnp.where(row_ids[:, None] < local_dim, rows, 0.0)

    in_axes = (
        0,
        0,
        0,
        None if mapping is None else 0,
        None if fparam is None else 0,
        None if aparam is None else 0,
    )
    return jax.vmap(hessian_block_one, in_axes=in_axes)(
        extended_coord,
        extended_atype,
        nlist,
        mapping,
        fparam,
        aparam,
    )


def _format_energy_loss_outputs(
    model_dict: dict[str, jnp.ndarray],
    label_dict: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    model_dict["atom_energy"] = model_dict["energy"]
    model_dict["energy"] = model_dict["energy_redu"]
    force = model_dict["energy_derv_r"].squeeze(-2)
    if "force" in label_dict and force.shape != label_dict["force"].shape:
        force = jnp.reshape(force, label_dict["force"].shape)
    model_dict["force"] = force
    model_dict["virial"] = model_dict["energy_derv_c_redu"].squeeze(-2)
    return model_dict


def _append_suffix(path_like: str | Path, suffix: str) -> Path:
    """Append a suffix before the final file suffix when present."""
    path = Path(path_like)
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")
    return path.with_name(f"{path.name}{suffix}")


def _get_ema_checkpoint_prefix(save_ckpt: str | Path) -> str:
    """Derive the EMA checkpoint prefix from the regular checkpoint prefix."""
    return str(_append_suffix(save_ckpt, "_ema"))


def _param_state_dict(model: BaseModel) -> dict[str, Any]:
    """Return a pure dict containing trainable NNX parameters only."""
    return nnx.state(model, nnx.Param).to_pure_dict()


def _copy_tree(tree: Any) -> Any:
    """Copy a pytree of JAX arrays without moving it off device."""
    return jax.tree_util.tree_map(lambda value: jnp.array(value), tree)


def _nonfinite_gradient_count(grads: Any) -> jnp.ndarray:
    """Count non-finite floating gradient entries."""
    counts = []
    for value in jax.tree_util.tree_leaves(grads):
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.inexact):
            counts.append(jnp.sum(~jnp.isfinite(value)))
    if not counts:
        return jnp.asarray(0, dtype=jnp.int64)
    return sum(counts, jnp.asarray(0, dtype=counts[0].dtype))


def _iter_named_gradient_leaves(
    tree: Any,
    path: tuple[str, ...] = (),
) -> Iterator[tuple[str, Any]]:
    """Yield named leaves from an NNX/state pytree for debug messages."""
    if hasattr(tree, "to_pure_dict"):
        tree = tree.to_pure_dict()
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _iter_named_gradient_leaves(value, (*path, str(key)))
        return
    if isinstance(tree, (list, tuple)):
        for idx, value in enumerate(tree):
            yield from _iter_named_gradient_leaves(value, (*path, str(idx)))
        return
    yield ".".join(path) if path else "<root>", tree


def _nonfinite_gradient_summary(
    grads: Any,
    *,
    limit: int = 20,
) -> tuple[int, list[str]]:
    """Return non-finite gradient count and a compact per-leaf summary."""
    total = 0
    summaries = []
    for name, value in _iter_named_gradient_leaves(grads):
        dtype = getattr(value, "dtype", None)
        if dtype is None or not jnp.issubdtype(dtype, jnp.inexact):
            continue
        nan_count = int(jax.device_get(jnp.sum(jnp.isnan(value))))
        inf_count = int(jax.device_get(jnp.sum(jnp.isinf(value))))
        bad_count = nan_count + inf_count
        total += bad_count
        if bad_count and len(summaries) < limit:
            summaries.append(
                f"{name}: bad={bad_count}, nan={nan_count}, "
                f"inf={inf_count}, shape={getattr(value, 'shape', None)}"
            )
    return total, summaries


def _map_named_gradient_leaves(
    tree: Any,
    fn: Any,
    path: tuple[str, ...] = (),
) -> Any:
    if hasattr(tree, "to_pure_dict") and hasattr(tree, "replace_by_pure_dict"):
        pure_tree = tree.to_pure_dict()
        mapped = _map_named_gradient_leaves(pure_tree, fn, path)
        tree.replace_by_pure_dict(mapped)
        return tree
    if isinstance(tree, dict):
        return {
            key: _map_named_gradient_leaves(value, fn, (*path, str(key)))
            for key, value in tree.items()
        }
    if isinstance(tree, list):
        return [
            _map_named_gradient_leaves(value, fn, (*path, str(idx)))
            for idx, value in enumerate(tree)
        ]
    if isinstance(tree, tuple):
        return tuple(
            _map_named_gradient_leaves(value, fn, (*path, str(idx)))
            for idx, value in enumerate(tree)
        )
    return fn(".".join(path) if path else "<root>", tree)


def _is_sezm_equivariant_norm_bias(name: str) -> bool:
    if not name.startswith("atomic_model.descriptor.blocks."):
        return False
    return (
        name.endswith(".pre_so2_norm.bias")
        or name.endswith(".post_so2_norm.bias")
        or (".pre_ffn_norms." in name and name.endswith(".bias"))
        or (".post_ffn_norms." in name and name.endswith(".bias"))
    )


def _drop_sezm_norm_bias_hessian_grads(grads: Any) -> Any:
    def drop_if_needed(name: str, value: Any) -> Any:
        if _is_sezm_equivariant_norm_bias(name):
            return jnp.zeros_like(value)
        return value

    return _map_named_gradient_leaves(grads, drop_if_needed)


def _block_until_ready_tree(tree: Any) -> None:
    """Synchronize a pytree of JAX arrays for debug timing."""
    for value in jax.tree_util.tree_leaves(tree):
        block_until_ready = getattr(value, "block_until_ready", None)
        if block_until_ready is not None:
            block_until_ready()


def _scale_by_extra_arg_learning_rate() -> optax.GradientTransformationExtraArgs:
    """Scale updates by the host-provided learning rate extra arg."""

    def init_fn(_params: Any) -> optax.EmptyState:
        return optax.EmptyState()

    def update_fn(
        updates: Any,
        state: optax.EmptyState,
        params: Any | None = None,
        *,
        learning_rate: jnp.ndarray | float | None = None,
        **_extra_args: Any,
    ) -> tuple[Any, optax.EmptyState]:
        del params
        if learning_rate is None:
            raise ValueError(
                "JAX optimizer update requires a host-provided learning_rate."
            )
        lr = jnp.asarray(learning_rate)
        return jax.tree.map(lambda update: -lr * update, updates), state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _validate_param_tree(reference: Any, loaded: Any, path: str = "") -> None:
    """Validate that two parameter pytrees have matching keys and leaf shapes."""
    if isinstance(reference, dict):
        if not isinstance(loaded, dict):
            raise TypeError(f"EMA checkpoint field {path or '<root>'} must be a dict.")
        reference_keys = set(reference)
        loaded_keys = set(loaded)
        missing = sorted(reference_keys - loaded_keys, key=str)
        unexpected = sorted(loaded_keys - reference_keys, key=str)
        if missing or unexpected:
            raise KeyError(
                "EMA checkpoint parameter keys do not match the current model. "
                f"At {path or '<root>'}: missing {missing[:5]}, "
                f"unexpected {unexpected[:5]}."
            )
        for key in reference:
            next_path = f"{path}.{key}" if path else str(key)
            _validate_param_tree(reference[key], loaded[key], next_path)
        return

    if getattr(reference, "shape", None) != getattr(loaded, "shape", None):
        raise ValueError(
            "EMA checkpoint parameter shape does not match the current model "
            f"for {path!r}: expected {getattr(reference, 'shape', None)}, "
            f"got {getattr(loaded, 'shape', None)}."
        )


def _pt_style_epoch_nbatches(train_data: DeepmdDataSystem) -> list[int]:
    """Return epoch batch counts matching PyTorch DataLoader(drop_last=False)."""
    if not hasattr(train_data, "data_systems") or not hasattr(train_data, "batch_size"):
        return list(train_data.nbatches)

    nbatches = []
    for data_system, batch_size in zip(
        train_data.data_systems,
        train_data.batch_size,
        strict=True,
    ):
        nframes = int(getattr(data_system, "nframes"))
        batch_size = int(batch_size)
        nbatches.append(max(1, int(np.ceil(nframes / batch_size))))
    return nbatches


def _resolve_epoch_sys_probs(
    train_data_param: dict[str, Any],
    nbatches: list[int],
) -> np.ndarray:
    """Resolve system sampling probabilities with PyTorch sampler semantics."""
    sys_probs = train_data_param.get("sys_probs", None)
    if sys_probs is not None:
        return process_sys_probs(sys_probs, np.asarray(nbatches, dtype=np.float64))

    auto_prob_style = train_data_param.get("auto_prob", "prob_sys_size")
    if auto_prob_style == "prob_uniform":
        return np.full(len(nbatches), 1.0 / float(len(nbatches)), dtype=np.float64)
    if auto_prob_style == "prob_sys_size":
        auto_prob_style = f"prob_sys_size;0:{len(nbatches)}:1.0"
    if str(auto_prob_style).startswith("prob_sys_size"):
        return np.asarray(
            prob_sys_size_ext(auto_prob_style, len(nbatches), nbatches),
            dtype=np.float64,
        )
    raise RuntimeError("Unknown auto prob style: " + str(auto_prob_style))


class JAXModelEMA:
    """Maintain an exponential moving average of JAX/NNX model parameters."""

    def __init__(
        self,
        model: BaseModel,
        decay: float,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.decay = float(decay)
        self.shadow_params = _copy_tree(_param_state_dict(model))
        if state is not None:
            self.load_state_dict(model, state)

    @staticmethod
    def _update_leaf(decay: float, shadow: Any, param: Any) -> Any:
        if hasattr(param, "dtype") and jnp.issubdtype(param.dtype, jnp.inexact):
            return decay * shadow + (1.0 - decay) * param
        return param

    def update(self, model: BaseModel) -> None:
        """Update EMA shadow parameters from the current model parameters."""
        params = _param_state_dict(model)
        self.shadow_params = jax.tree_util.tree_map(
            lambda shadow, param: self._update_leaf(self.decay, shadow, param),
            self.shadow_params,
            params,
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize EMA state for restart."""
        return {
            EMA_DECAY_KEY: self.decay,
            EMA_MODEL_STATE_KEY: _copy_tree(self.shadow_params),
        }

    def load_state_dict(self, model: BaseModel, state: dict[str, Any]) -> None:
        """Restore EMA shadow parameters."""
        if EMA_DECAY_KEY in state:
            checkpoint_decay = float(state[EMA_DECAY_KEY])
            if checkpoint_decay != self.decay:
                log.warning(
                    "Ignoring EMA checkpoint decay=%s because training.ema_decay=%s "
                    "is configured.",
                    checkpoint_decay,
                    self.decay,
                )
        model_state = state.get(EMA_MODEL_STATE_KEY, {})
        if not isinstance(model_state, dict):
            raise TypeError("EMA checkpoint field `model` must be a dict.")
        current_params = _param_state_dict(model)
        _validate_param_tree(current_params, model_state)
        self.shadow_params = _copy_tree(model_state)

    @contextmanager
    def apply_shadow(self, model: BaseModel) -> Iterator[None]:
        """Temporarily replace model parameters with the EMA shadow state."""
        param_state = nnx.state(model, nnx.Param)
        backup = _copy_tree(param_state.to_pure_dict())
        try:
            param_state.replace_by_pure_dict(self.shadow_params)
            nnx.update(model, param_state)
            yield
        finally:
            restore_state = nnx.state(model, nnx.Param)
            restore_state.replace_by_pure_dict(backup)
            nnx.update(model, restore_state)


class DPTrainer:
    """Train JAX DeePMD models on local devices."""

    def __init__(
        self,
        jdata: dict,
        init_model: str | None = None,
        restart: str | None = None,
    ) -> None:
        """Initialize the trainer from input data and optional checkpoints."""
        self.init_model = init_model
        self.restart = restart
        self.model_def_script = deepcopy(jdata["model"])
        self.training_param = jdata["training"]
        self.start_step = 0
        ema_state_dict = None
        if self.init_model is not None:
            model_dict = serialize_from_file(self.init_model)
            self.model = BaseModel.deserialize(model_dict["model"])
        elif self.restart is not None:
            model_dict = serialize_from_file(self.restart)
            self.model = BaseModel.deserialize(model_dict["model"])
            ema_state_dict = model_dict.get("ema")
            self.start_step = model_dict.get("model_def_script", {}).get(
                "current_step",
                model_dict.get("@variables", {}).get("current_step", 0),
            )
        else:
            # from scratch
            self.model = get_model(jdata["model"])
        self.num_steps = self.training_param.get("numb_steps")
        self.num_epoch = self.training_param.get("numb_epoch")
        if self.num_epoch is None:
            self.num_epoch = self.training_param.get("num_epoch")
        if self.num_epoch is None:
            self.num_epoch = self.training_param.get("num_epochs")

        learning_rate_param = jdata["learning_rate"]
        self.learning_rate_param = learning_rate_param
        self.lr = (
            BaseLR(
                **self.learning_rate_param,
                num_steps=self.num_steps,
            )
            if self.num_steps is not None
            else None
        )
        self.optimizer_param = dict(jdata.get("optimizer", {}))
        self.opt_type = self.optimizer_param.get("type", "Adam")
        if self.opt_type not in ("Adam", "AdamW", "HybridMuon"):
            raise ValueError(
                f"JAX training does not support optimizer type '{self.opt_type}'."
            )
        self.optimizer_param.pop("type", None)
        tr_data = self.training_param
        self.hessian_train_chunk_size = int(
            tr_data.get(
                "hessian_chunk_size",
                os.environ.get(
                    "DP_JAX_HESSIAN_TRAIN_CHUNK_SIZE",
                    4,
                ),
            )
        )
        self.trim_mixed_padding = bool(
            int(os.environ.get("DP_JAX_TRIM_MIXED_PADDING", "1"))
        )
        self.hessian_sync_blocks = bool(
            int(os.environ.get("DP_JAX_HESSIAN_SYNC_BLOCKS", "1"))
        )
        self.sync_train_step = bool(
            int(os.environ.get("DP_JAX_SYNC_TRAIN_STEP", "1"))
        )
        loss_param = deepcopy(jdata.get("loss", {}))
        loss_param["starter_learning_rate"] = learning_rate_param["start_lr"]
        self.loss_param = deepcopy(loss_param)
        self.has_hessian = _loss_uses_hessian(loss_param)

        loss_type = loss_param.get("type", "ener")
        if self.has_hessian:
            self.loss = EnergyHessianLoss.get_loss(loss_param)
            if self.hessian_train_chunk_size <= 0:
                _enable_hessian_output(self.model)
            self.model_def_script["hessian_mode"] = True
        elif loss_type == "ener":
            self.loss = EnergyLoss.get_loss(loss_param)
        else:
            raise RuntimeError("unknown loss type " + loss_type)

        # training
        self.disp_file = tr_data.get("disp_file", "lcurve.out")
        self.disp_freq = tr_data.get("disp_freq", 1000)
        self.save_freq = tr_data.get("save_freq", 1000)
        self.save_ckpt = tr_data.get("save_ckpt", "model.ckpt")
        self.max_ckpt_keep = tr_data.get("max_ckpt_keep", 5)
        self.gradient_max_norm = tr_data.get("gradient_max_norm", None)
        self.enable_ema = bool(tr_data.get("enable_ema", False))
        self.ema_decay = float(tr_data.get("ema_decay", 0.999))
        self.ema_ckpt_keep = int(tr_data.get("ema_ckpt_keep", 3))
        self.ema_save_ckpt = _get_ema_checkpoint_prefix(self.save_ckpt)
        self.zero_stage = int(tr_data.get("zero_stage", 0))
        if self.zero_stage not in (0, 1, 2, 3):
            raise ValueError(
                f"training.zero_stage must be 0, 1, 2, or 3, got {self.zero_stage}"
            )
        if self.enable_ema and self.zero_stage >= 2:
            raise ValueError(
                "training.enable_ema currently only supports training.zero_stage < 2."
            )
        if self.zero_stage > 0:
            log.info(
                "JAX local trainer does not shard optimizer state; treating "
                "training.zero_stage=%d as zero_stage=0.",
                self.zero_stage,
            )
            self.zero_stage = 0
        self.model_ema = (
            JAXModelEMA(self.model, decay=self.ema_decay, state=ema_state_dict)
            if self.enable_ema
            else None
        )
        self.display_in_training = tr_data.get("disp_training", True)
        self.timing_in_training = tr_data.get("time_training", True)
        self.profiling = tr_data.get("profiling", False)
        self.profiling_file = tr_data.get("profiling_file", "timeline.json")
        self.enable_profiler = tr_data.get("enable_profiler", False)
        self.tensorboard = tr_data.get("tensorboard", False)
        self.tensorboard_log_dir = tr_data.get("tensorboard_log_dir", "log")
        self.tensorboard_freq = tr_data.get("tensorboard_freq", 1)
        self.mixed_prec = tr_data.get("mixed_precision", None)
        self.change_bias_after_training = tr_data.get(
            "change_bias_after_training", False
        )
        self.jax_parallel_mode = _normalize_jax_parallel_mode(
            tr_data.get(
                "jax_parallel_mode",
                os.environ.get("DP_JAX_PARALLEL_MODE", "auto"),
            )
        )
        self.numb_fparam = self.model.get_dim_fparam()

        if tr_data.get("validation_data", None) is not None:
            self.valid_numb_batch = max(
                tr_data["validation_data"].get("numb_btch", 1),
                1,
            )
        else:
            self.valid_numb_batch = 1

        # if init the graph with the frozen model
        self.frz_model = None
        self.ckpt_meta = None
        self.model_type = None

    @property
    def data_requirements(self) -> list[DataRequirementItem]:
        """Labels required by the configured loss."""
        return self.loss.label_requirement

    def train(
        self, train_data: DeepmdDataSystem, valid_data: DeepmdDataSystem | None = None
    ) -> None:
        """Run the training loop with optional validation data."""
        self._resolve_num_steps(train_data)
        assert self.lr is not None
        model = self.model
        _clear_jax_mesh_for_host_ops()

        # data stat
        if self.init_model is None and self.restart is None:
            data_stat_nbatch = self.model_def_script.get("data_stat_nbatch", 10)
            stat_data = make_stat_input(train_data, data_stat_nbatch)
            stat_data_jax = [
                {
                    kk: jnp.asarray(vv) if isinstance(vv, np.ndarray) else vv
                    for kk, vv in single_data.items()
                }
                for single_data in stat_data
            ]
            model.atomic_model.compute_or_load_stat(lambda: stat_data_jax)

        parallel_mode = _resolve_jax_parallel_mode(
            self.jax_parallel_mode,
            self.model_def_script,
            self.has_hessian,
        )
        auto_mesh = _make_training_mesh(parallel_mode)
        data_axis_size = _mesh_axis_size(auto_mesh, "data")
        natoms_axis_size = _mesh_axis_size(auto_mesh, "natoms")
        _set_nnx_eager_sharding(True)
        _set_jax_mesh(auto_mesh)
        sharding = (
            NamedSharding(auto_mesh, P("data"))
            if int(os.environ.get("DP_JAX_MULTI_NPROC", "0")) > 1
            else None
        )
        log.info(
            "JAX training parallel_mode=%s, mesh shape=%s, "
            "local_device_count=%d, process_count=%d.",
            parallel_mode,
            auto_mesh.shape,
            jax.local_device_count(),
            jax.process_count(),
        )
        if parallel_mode == "data" and data_axis_size > 1:
            log.info(
                "JAX data-parallel mode shards only the batch axis; atom and "
                "Hessian axes are replicated on each device."
            )
        if parallel_mode == "natoms" and self.has_hessian:
            log.warning(
                "JAX natoms parallel mode is experimental for Hessian training "
                "and may fail in XLA SPMD on gather/scatter-heavy models."
            )
        model = BaseModel.deserialize(model.serialize())
        if self.has_hessian and self.hessian_train_chunk_size <= 0:
            _enable_hessian_output(model)
        if self.has_hessian and self.hessian_train_chunk_size > 0:
            log.info(
                "JAX Hessian training uses row/block loss accumulation with "
                "chunk_size=%d.",
                self.hessian_train_chunk_size,
            )
        tx = self._build_optimizer_tx(model)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        def _call_energy_force_loss_model(
            model: BaseModel,
            label_dict: dict[str, jnp.ndarray],
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            mapping: jnp.ndarray | None,
            fp: jnp.ndarray | None,
            ap: jnp.ndarray | None,
        ) -> dict[str, jnp.ndarray]:
            model_dict_lower = model.call_common_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
            )
            model_dict = _communicate_or_passthrough(
                model_dict_lower,
                model.model_output_def(),
                mapping,
                nlist.shape[1],
            )
            return _format_energy_loss_outputs(model_dict, label_dict)

        def hessian_coord_mask(
            label_dict: dict[str, jnp.ndarray],
            dtype: jnp.dtype,
        ) -> jnp.ndarray:
            atom_mask = jnp.asarray(label_dict["type"] >= 0, dtype=dtype)
            return jnp.repeat(atom_mask, 3, axis=1)

        def hessian_total_count(
            label_dict: dict[str, jnp.ndarray],
        ) -> jnp.ndarray:
            coord_mask = hessian_coord_mask(label_dict, label_dict["hessian"].dtype)
            weight = coord_mask[:, :, None] * coord_mask[:, None, :]
            return jnp.sum(weight)

        def hessian_block_sum_count(
            model: BaseModel,
            label_dict: dict[str, jnp.ndarray],
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            mapping: jnp.ndarray | None,
            fp: jnp.ndarray | None,
            ap: jnp.ndarray | None,
            block_index: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            natoms = label_dict["type"].shape[1]
            dim = natoms * 3
            chunk_size = min(self.hessian_train_chunk_size, dim)
            nblocks = (dim + chunk_size - 1) // chunk_size
            padded_dim = nblocks * chunk_size
            hessian_hat = jnp.reshape(
                label_dict["hessian"],
                (label_dict["hessian"].shape[0], dim, dim),
            )
            pad_rows = padded_dim - dim
            if pad_rows:
                hessian_hat = jnp.pad(hessian_hat, ((0, 0), (0, pad_rows), (0, 0)))
            coord_mask = hessian_coord_mask(label_dict, hessian_hat.dtype)
            if pad_rows:
                coord_mask = jnp.pad(coord_mask, ((0, 0), (0, pad_rows)))

            block = _call_hessian_local_block(
                model,
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
                block_index,
                chunk_size,
            )
            row_start = block_index * chunk_size
            slice_zero = jnp.asarray(0, dtype=row_start.dtype)
            hessian_block_hat = jax.lax.dynamic_slice(
                hessian_hat,
                (slice_zero, row_start, slice_zero),
                (hessian_hat.shape[0], chunk_size, dim),
            )
            row_ids = row_start + jnp.arange(chunk_size, dtype=row_start.dtype)
            row_weight = jnp.asarray(row_ids < dim, dtype=block.dtype)
            row_coord_mask = jax.lax.dynamic_slice(
                coord_mask,
                (slice_zero, row_start),
                (coord_mask.shape[0], chunk_size),
            )
            weight = (
                row_weight[None, :, None]
                * row_coord_mask[:, :, None]
                * coord_mask[:, None, :dim]
            )
            values = hessian_block_hat - block
            if self.loss.loss_func == "mse":
                values = values * values
            elif self.loss.loss_func == "mae":
                values = jnp.abs(values)
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss.loss_func} is not implemented "
                    "for Hessian loss."
                )
            return jnp.sum(values * weight), jnp.sum(jnp.ones_like(values) * weight)

        def hessian_block_loss(
            model: BaseModel,
            lr: float,
            label_dict: dict[str, jnp.ndarray],
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            mapping: jnp.ndarray | None,
            fp: jnp.ndarray | None,
            ap: jnp.ndarray | None,
            block_index: jnp.ndarray,
            total_count: jnp.ndarray,
        ) -> jnp.ndarray:
            block_sum, _ = hessian_block_sum_count(
                model,
                label_dict,
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
                block_index,
            )
            lr_ratio = lr / self.loss.starter_learning_rate
            pref_h = self.loss.limit_pref_h + (
                self.loss.start_pref_h - self.loss.limit_pref_h
            ) * lr_ratio
            find_hessian = label_dict.get("find_hessian", True)
            return pref_h * find_hessian * block_sum / jnp.maximum(total_count, 1.0)

        def loss_fn(
            model: BaseModel,
            lr: float,
            label_dict: dict[str, jnp.ndarray],
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            mapping: jnp.ndarray | None,
            fp: jnp.ndarray | None,
            ap: jnp.ndarray | None,
        ) -> jnp.ndarray:
            model_dict = _call_energy_force_loss_model(
                model,
                label_dict,
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
            )
            loss, _ = self.loss(
                learning_rate=lr,
                natoms=label_dict["type"].shape[1],
                model_dict=model_dict,
                label_dict=label_dict,
            )
            return loss

        @nnx.jit
        def loss_fn_more_loss(
            model: BaseModel,
            lr: float,
            label_dict: dict[str, jnp.ndarray],
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            mapping: jnp.ndarray | None,
            fp: jnp.ndarray | None,
            ap: jnp.ndarray | None,
        ) -> dict[str, jnp.ndarray]:
            model_dict = _call_energy_force_loss_model(
                model,
                label_dict,
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
            )
            loss, more_loss = self.loss(
                learning_rate=lr,
                natoms=label_dict["type"].shape[1],
                model_dict=model_dict,
                label_dict=label_dict,
            )
            return more_loss

        ef_grad_fn = nnx.jit(nnx.grad(loss_fn))
        hessian_block_grad_fn = nnx.jit(nnx.grad(hessian_block_loss))
        hessian_block_sum_count_fn = nnx.jit(hessian_block_sum_count)
        drop_sezm_norm_bias_hessian_grads = bool(
            int(os.environ.get("DP_JAX_HESSIAN_DROP_SEZM_NORM_BIAS_GRADS", "1"))
        )

        @nnx.jit
        def apply_grads(
            model: BaseModel,
            optimizer: nnx.Optimizer,
            grads: Any,
            learning_rate: float,
        ) -> None:
            if Version(flax_version) >= Version("0.11.0"):
                optimizer.update(model, grads, learning_rate=learning_rate)
            else:
                optimizer.update(grads, learning_rate=learning_rate)

        def hessian_nblocks(label_dict: dict[str, jnp.ndarray]) -> int:
            dim = label_dict["type"].shape[1] * 3
            chunk_size = min(self.hessian_train_chunk_size, dim)
            return (dim + chunk_size - 1) // chunk_size

        def add_hessian_more_loss(
            more_loss: dict[str, jnp.ndarray],
            model: BaseModel,
            lr: float,
            label_dict: dict[str, jnp.ndarray],
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            mapping: jnp.ndarray | None,
            fp: jnp.ndarray | None,
            ap: jnp.ndarray | None,
        ) -> dict[str, jnp.ndarray]:
            if not (
                self.has_hessian
                and self.hessian_train_chunk_size > 0
                and isinstance(self.loss, EnergyHessianLoss)
                and "hessian" in label_dict
            ):
                return more_loss
            total_sum = jnp.asarray(0.0, dtype=label_dict["hessian"].dtype)
            total_count = jnp.asarray(0.0, dtype=label_dict["hessian"].dtype)
            for block_index in range(hessian_nblocks(label_dict)):
                block_sum, block_count = hessian_block_sum_count_fn(
                    model,
                    label_dict,
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping,
                    fp,
                    ap,
                    jnp.asarray(block_index, dtype=jnp.int32),
                )
                total_sum = total_sum + block_sum
                total_count = total_count + block_count
            hessian_metric = total_sum / jnp.maximum(total_count, 1.0)
            find_hessian = label_dict.get("find_hessian", True)
            if self.loss.loss_func == "mse":
                more_loss["rmse_h"] = self.loss.display_if_exist(
                    jnp.sqrt(hessian_metric),
                    find_hessian,
                )
            else:
                more_loss["mae_h"] = self.loss.display_if_exist(
                    hessian_metric,
                    find_hessian,
                )
            return more_loss

        def train_step(
            model: BaseModel,
            optimizer: nnx.Optimizer,
            lr: float,
            label_dict: dict[str, jnp.ndarray],
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            mapping: jnp.ndarray | None,
            fp: jnp.ndarray | None,
            ap: jnp.ndarray | None,
            step_index: int,
        ) -> None:
            debug_nonfinite = bool(
                int(os.environ.get("DP_JAX_DEBUG_NONFINITE_GRADS", "0"))
            )
            debug_timing = bool(
                int(os.environ.get("DP_JAX_DEBUG_STEP_TIMING", "0"))
            )
            if debug_timing:
                log.info("JAX debug step %d: start energy/force grad", step_index)
            grads = ef_grad_fn(
                model,
                lr,
                label_dict,
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
            )
            if debug_timing:
                _block_until_ready_tree(grads)
                log.info("JAX debug step %d: energy/force grad ready", step_index)
            if debug_nonfinite:
                bad_count, bad_summary = _nonfinite_gradient_summary(grads)
                if bad_count:
                    raise FloatingPointError(
                        "Non-finite JAX energy/force gradient entries detected "
                        f"before Hessian blocks: {bad_count}.\n"
                        + "\n".join(bad_summary)
                    )
            if (
                self.has_hessian
                and self.hessian_train_chunk_size > 0
                and "hessian" in label_dict
            ):
                total_count = hessian_total_count(label_dict)
                nblocks = hessian_nblocks(label_dict)
                for block_index in range(nblocks):
                    if debug_timing:
                        log.info(
                            "JAX debug step %d: start Hessian block %d/%d",
                            step_index,
                            block_index + 1,
                            nblocks,
                        )
                    block_grads = hessian_block_grad_fn(
                        model,
                        lr,
                        label_dict,
                        extended_coord,
                        extended_atype,
                        nlist,
                        mapping,
                        fp,
                        ap,
                        jnp.asarray(block_index, dtype=jnp.int32),
                        total_count,
                    )
                    if debug_timing or self.hessian_sync_blocks:
                        _block_until_ready_tree(block_grads)
                    if debug_timing:
                        log.info(
                            "JAX debug step %d: Hessian block %d/%d ready",
                            step_index,
                            block_index + 1,
                            nblocks,
                        )
                    if drop_sezm_norm_bias_hessian_grads:
                        block_grads = _drop_sezm_norm_bias_hessian_grads(block_grads)
                    if debug_nonfinite:
                        bad_count, bad_summary = _nonfinite_gradient_summary(
                            block_grads
                        )
                        if bad_count:
                            chunk_size = min(
                                self.hessian_train_chunk_size,
                                label_dict["type"].shape[1] * 3,
                            )
                            row_start = block_index * chunk_size
                            row_stop = min(
                                row_start + chunk_size,
                                label_dict["type"].shape[1] * 3,
                            )
                            raise FloatingPointError(
                                "Non-finite JAX Hessian block gradient entries "
                                f"detected at block_index={block_index}, "
                                f"rows=[{row_start}, {row_stop}): {bad_count}.\n"
                                + "\n".join(bad_summary)
                            )
                    grads = jax.tree.map(
                        lambda left, right: left + right,
                        grads,
                        block_grads,
                    )
            if debug_timing:
                log.info("JAX debug step %d: start nonfinite check", step_index)
            nonfinite_count = int(jax.device_get(_nonfinite_gradient_count(grads)))
            if debug_timing:
                log.info("JAX debug step %d: nonfinite check ready", step_index)
            if nonfinite_count:
                _, bad_summary = _nonfinite_gradient_summary(grads)
                raise FloatingPointError(
                    "Non-finite JAX gradient entries detected before optimizer "
                    f"update: {nonfinite_count}.\n" + "\n".join(bad_summary)
                )
            if debug_timing:
                log.info("JAX debug step %d: start optimizer update", step_index)
            apply_grads(model, optimizer, grads, lr)
            if debug_timing or self.sync_train_step:
                _block_until_ready_tree(nnx.state(model, nnx.Param))
            if debug_timing:
                log.info("JAX debug step %d: optimizer update ready", step_index)

        if (
            self.has_hessian
            and self.hessian_train_chunk_size > 0
            and drop_sezm_norm_bias_hessian_grads
        ):
            log.info(
                "JAX Hessian training drops SeZM EquivariantRMSNorm.bias "
                "Hessian-loss gradients; energy/force gradients for these "
                "biases are still applied. Set "
                "DP_JAX_HESSIAN_DROP_SEZM_NORM_BIAS_GRADS=0 to debug the full "
                "third-order path."
            )

        start_time = time.time()
        disp_path = Path(self.disp_file)
        disp_mode = "a" if self.start_step > 0 and disp_path.exists() else "w"
        with open(disp_path, disp_mode) as disp_file_fp:
            for step in range(self.start_step, self.num_steps):
                batch_data = trim_mixed_padding_batch(
                    train_data.get_batch(),
                    enabled=self.trim_mixed_padding,
                )
                # numpy to jax
                jax_data = convert_numpy_data_to_jax_data(
                    batch_data,
                    sharding,
                    data_axis_size=data_axis_size,
                    natoms_axis_size=natoms_axis_size,
                )
                extended_coord, extended_atype, nlist, mapping, fp, ap = prepare_input(
                    rcut=model.get_rcut(),
                    sel=model.get_sel(),
                    coord=jax_data["coord"],
                    atype=jax_data["type"],
                    box=jax_data["box"] if jax_data["find_box"] else None,
                    fparam=jax_data.get("fparam", None),
                    aparam=jax_data.get("aparam", None),
                )
                train_step(
                    model,
                    optimizer,
                    self.lr.value(step),
                    jax_data,
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping,
                    fp,
                    ap,
                    step + 1,
                )
                if self.model_ema is not None:
                    self.model_ema.update(model)
                if self.display_in_training and (
                    step == 0 or (step + 1) % self.disp_freq == 0
                ):
                    wall_time = time.time() - start_time
                    log.info(
                        format_training_message(
                            batch=step + 1,
                            wall_time=wall_time,
                        )
                    )
                    more_loss = loss_fn_more_loss(
                        model,
                        self.lr.value(step),
                        jax_data,
                        extended_coord,
                        extended_atype,
                        nlist,
                        mapping,
                        fp,
                        ap,
                    )
                    more_loss = add_hessian_more_loss(
                        more_loss,
                        model,
                        self.lr.value(step),
                        jax_data,
                        extended_coord,
                        extended_atype,
                        nlist,
                        mapping,
                        fp,
                        ap,
                    )
                    if valid_data is not None:
                        valid_more_loss_list = []
                        for _ in range(self.valid_numb_batch):
                            valid_batch_data = trim_mixed_padding_batch(
                                valid_data.get_batch(),
                                enabled=self.trim_mixed_padding,
                            )
                            jax_valid_data = convert_numpy_data_to_jax_data(
                                valid_batch_data,
                                sharding,
                                data_axis_size=data_axis_size,
                                natoms_axis_size=natoms_axis_size,
                            )
                            extended_coord, extended_atype, nlist, mapping, fp, ap = (
                                prepare_input(
                                    rcut=model.get_rcut(),
                                    sel=model.get_sel(),
                                    coord=jax_valid_data["coord"],
                                    atype=jax_valid_data["type"],
                                    box=jax_valid_data["box"]
                                    if jax_valid_data["find_box"]
                                    else None,
                                    fparam=jax_valid_data.get("fparam", None),
                                    aparam=jax_valid_data.get("aparam", None),
                                )
                            )
                            valid_more_loss_item = loss_fn_more_loss(
                                model,
                                self.lr.value(step),
                                jax_valid_data,
                                extended_coord,
                                extended_atype,
                                nlist,
                                mapping,
                                fp,
                                ap,
                            )
                            valid_more_loss_item = add_hessian_more_loss(
                                valid_more_loss_item,
                                model,
                                self.lr.value(step),
                                jax_valid_data,
                                extended_coord,
                                extended_atype,
                                nlist,
                                mapping,
                                fp,
                                ap,
                            )
                            valid_more_loss_list.append(valid_more_loss_item)
                        valid_more_loss = {
                            key: sum(loss[key] for loss in valid_more_loss_list)
                            / len(valid_more_loss_list)
                            for key in valid_more_loss_list[0]
                        }
                    else:
                        valid_more_loss = None
                    if disp_file_fp.tell() == 0:
                        self.print_header(
                            disp_file_fp,
                            train_results=more_loss,
                            valid_results=valid_more_loss,
                        )
                    self.print_on_training(
                        disp_file_fp,
                        train_results=more_loss,
                        valid_results=valid_more_loss,
                        cur_batch=step + 1,
                        cur_lr=self.lr.value(step),
                    )
                    start_time = time.time()
                if (step + 1) % self.save_freq == 0:
                    self._save_checkpoint(model, step + 1)
                    self._save_ema_checkpoint(model, step + 1)
        if self.num_steps > self.start_step and self.num_steps % self.save_freq != 0:
            self._save_checkpoint(model, self.num_steps)
            self._save_ema_checkpoint(model, self.num_steps)

    def _save_checkpoint(
        self,
        model: BaseModel,
        step: int,
        *,
        ckpt_prefix: str | None = None,
        max_ckpt_keep: int | None = None,
        include_ema_state: bool = True,
    ) -> None:
        """Save a JAX checkpoint and update the stable checkpoint pointer."""
        ckpt_prefix = self.save_ckpt if ckpt_prefix is None else ckpt_prefix
        max_ckpt_keep = self.max_ckpt_keep if max_ckpt_keep is None else max_ckpt_keep
        _, state = nnx.split(model)
        ckpt_path = Path(f"{ckpt_prefix}-{step}.jax")
        if ckpt_path.is_dir():
            # remove old checkpoint if it exists
            shutil.rmtree(ckpt_path)
        model_def_script_cpy = self.model_def_script.copy()
        model_def_script_cpy["current_step"] = step
        handler_keys = ["state", "model_def_script"]
        save_items = {
            "state": ocp.args.StandardSave(
                pack_zero_size_arrays_for_orbax(state.to_pure_dict()),
            ),
            "model_def_script": ocp.args.JsonSave(model_def_script_cpy),
        }
        if include_ema_state and self.model_ema is not None:
            handler_keys.append("ema")
            save_items["ema"] = ocp.args.StandardSave(
                pack_zero_size_arrays_for_orbax(self.model_ema.state_dict()),
            )
        with ocp.Checkpointer(
            ocp.CompositeCheckpointHandler(*handler_keys)
        ) as checkpointer:
            checkpointer.save(
                ckpt_path.absolute(),
                ocp.args.Composite(**save_items),
            )
        log.info(f"Trained model has been saved to: {ckpt_path!s}")
        _link_checkpoint(ckpt_path, Path(f"{ckpt_prefix}.jax"))
        self._cleanup_old_checkpoints(ckpt_prefix, max_ckpt_keep)
        if ckpt_prefix == self.save_ckpt:
            with open("checkpoint", "w") as fp:
                fp.write(f"{self.save_ckpt}.jax")

    def _save_ema_checkpoint(self, model: BaseModel, step: int) -> None:
        """Save an EMA-weight JAX checkpoint when EMA is enabled."""
        if self.model_ema is None:
            return
        with self.model_ema.apply_shadow(model):
            self._save_checkpoint(
                model,
                step,
                ckpt_prefix=self.ema_save_ckpt,
                max_ckpt_keep=self.ema_ckpt_keep,
                include_ema_state=False,
            )

    def _resolve_num_steps(self, train_data: DeepmdDataSystem) -> None:
        """Resolve step-based training length from epoch-based input if needed."""
        if self.num_steps is not None:
            return
        if self.num_epoch is None:
            raise ValueError(
                "Either training.numb_steps or training.num_epoch must be set."
            )
        if self.num_epoch <= 0:
            raise ValueError("training.num_epoch must be positive.")
        if hasattr(train_data, "data_systems"):
            train_data_param = self.training_param.get("training_data", {})
            epoch_nbatches = _pt_style_epoch_nbatches(train_data)
            epoch_sys_probs = _resolve_epoch_sys_probs(train_data_param, epoch_nbatches)
            train_data.nbatches = epoch_nbatches
            train_data.sys_probs = epoch_sys_probs
        else:
            epoch_nbatches = list(train_data.nbatches)
            epoch_sys_probs = train_data.sys_probs
        total_numb_batch = compute_total_numb_batch(
            epoch_nbatches,
            epoch_sys_probs,
        )
        if total_numb_batch <= 0:
            raise ValueError("Total number of training batches must be positive.")
        self.num_steps = int(np.ceil(self.num_epoch * total_numb_batch))
        log.info(
            "Computed numb_steps=%d from num_epoch=%s and total_numb_batch=%d.",
            self.num_steps,
            self.num_epoch,
            total_numb_batch,
        )
        self.lr = BaseLR(
            **self.learning_rate_param,
            num_steps=self.num_steps,
        )

    def _build_optimizer_tx(self, model: BaseModel) -> optax.GradientTransformation:
        """Build the configured optax optimizer transformation."""
        assert self.lr is not None
        adam_betas = (
            float(self.optimizer_param.get("adam_beta1", 0.9)),
            float(self.optimizer_param.get("adam_beta2", 0.999)),
        )
        weight_decay = float(self.optimizer_param.get("weight_decay", 0.0))

        if self.opt_type == "Adam":
            if weight_decay == 0.0:
                tx = optax.chain(
                    optax.scale_by_adam(
                        b1=adam_betas[0],
                        b2=adam_betas[1],
                    ),
                    _scale_by_extra_arg_learning_rate(),
                )
            else:
                tx = optax.chain(
                    optax.add_decayed_weights(weight_decay),
                    optax.scale_by_adam(
                        b1=adam_betas[0],
                        b2=adam_betas[1],
                    ),
                    _scale_by_extra_arg_learning_rate(),
                )
            return self._maybe_clip_optimizer_tx(tx)

        if self.opt_type == "AdamW":
            return self._maybe_clip_optimizer_tx(
                optax.chain(
                    optax.scale_by_adam(
                        b1=adam_betas[0],
                        b2=adam_betas[1],
                    ),
                    optax.add_decayed_weights(weight_decay),
                    _scale_by_extra_arg_learning_rate(),
                )
            )

        if self.opt_type == "HybridMuon":
            return self._maybe_clip_optimizer_tx(
                hybrid_muon(
                    learning_rate=1.0,
                    params=nnx.state(model, nnx.Param),
                    momentum=float(self.optimizer_param.get("momentum", 0.95)),
                    weight_decay=weight_decay,
                    adam_betas=(
                        float(self.optimizer_param.get("adam_beta1", 0.9)),
                        float(self.optimizer_param.get("adam_beta2", 0.95)),
                    ),
                    lr_adjust=float(self.optimizer_param.get("lr_adjust", 0.0)),
                    lr_adjust_coeff=float(
                        self.optimizer_param.get("lr_adjust_coeff", 0.18)
                    ),
                    muon_mode=str(self.optimizer_param.get("muon_mode", "slice")),
                    enable_gram=bool(self.optimizer_param.get("enable_gram", True)),
                    flash_muon=bool(self.optimizer_param.get("flash_muon", True)),
                    magma_muon=bool(self.optimizer_param.get("magma_muon", True)),
                )
            )

        raise ValueError(
            f"JAX training does not support optimizer type '{self.opt_type}'."
        )

    def _maybe_clip_optimizer_tx(
        self,
        tx: optax.GradientTransformation,
    ) -> optax.GradientTransformation:
        """Apply PyTorch-aligned global gradient clipping when configured."""
        if self.gradient_max_norm is None:
            return tx
        max_norm = float(self.gradient_max_norm)
        if max_norm <= 0.0:
            return tx
        return optax.chain(optax.clip_by_global_norm(max_norm), tx)

    def _cleanup_old_checkpoints(
        self,
        ckpt_prefix: str | None = None,
        max_ckpt_keep: int | None = None,
    ) -> None:
        """Remove old checkpoint directories beyond the retention limit."""
        ckpt_prefix = self.save_ckpt if ckpt_prefix is None else ckpt_prefix
        max_ckpt_keep = self.max_ckpt_keep if max_ckpt_keep is None else max_ckpt_keep
        if max_ckpt_keep <= 0:
            return
        ckpt_parent = Path(ckpt_prefix).parent
        ckpt_prefix_name = Path(ckpt_prefix).name
        checkpoints = []
        for path in ckpt_parent.glob(f"{ckpt_prefix_name}-*.jax"):
            if not path.is_dir() or path.is_symlink():
                continue
            step_text = path.name.removeprefix(f"{ckpt_prefix_name}-").removesuffix(
                ".jax"
            )
            if step_text.isdigit():
                checkpoints.append((int(step_text), path))
        for _, path in sorted(checkpoints)[: -max_ckpt_keep]:
            shutil.rmtree(path)

    @staticmethod
    def print_on_training(
        fp: TextIO,
        train_results: dict[str, float],
        valid_results: dict[str, float] | None,
        cur_batch: int,
        cur_lr: float,
    ) -> None:
        """Append one training/validation loss row to the learning-curve file."""
        print_str = ""
        print_str += f"{cur_batch:7d}"
        if valid_results is not None:
            prop_fmt = "   %11.2e %11.2e"
            for k in valid_results.keys():
                # assert k in train_results.keys()
                print_str += prop_fmt % (valid_results[k], train_results[k])
        else:
            prop_fmt = "   %11.2e"
            for k in train_results.keys():
                print_str += prop_fmt % (train_results[k])
        print_str += f"   {cur_lr:8.1e}\n"
        log.info(
            format_training_message_per_task(
                batch=cur_batch,
                task_name="trn",
                rmse=train_results,
                learning_rate=cur_lr,
            )
        )
        if valid_results is not None:
            log.info(
                format_training_message_per_task(
                    batch=cur_batch,
                    task_name="val",
                    rmse=valid_results,
                    learning_rate=None,
                )
            )
        fp.write(print_str)
        fp.flush()

    @staticmethod
    def print_header(
        fp: TextIO,
        train_results: dict[str, float],
        valid_results: dict[str, float] | None,
    ) -> None:
        """Write the learning-curve header for the configured loss terms."""
        print_str = ""
        print_str += "# {:5s}".format("step")
        if valid_results is not None:
            prop_fmt = "   %11s %11s"
            for k in train_results.keys():
                print_str += prop_fmt % (k + "_val", k + "_trn")
        else:
            prop_fmt = "   %11s"
            for k in train_results.keys():
                print_str += prop_fmt % (k + "_trn")
        print_str += "   {:8s}\n".format("lr")
        print_str += "# If there is no available reference data, metric_*_{val,trn} will print nan\n"
        fp.write(print_str)
        fp.flush()


def _link_checkpoint(source: Path, target: Path) -> None:
    """Point the stable checkpoint path to the latest checkpoint directory."""
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    if platform.system() != "Windows":
        os.symlink(os.path.relpath(source, target.parent), target)
    else:
        shutil.copytree(source, target)


def prepare_input(
    *,  # enforce keyword-only arguments
    rcut: float,
    sel: list[int],
    coord: np.ndarray,
    atype: np.ndarray,
    box: np.ndarray | None = None,
    fparam: np.ndarray | None = None,
    aparam: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Build extended coordinates and neighbor lists for a training batch."""
    nframes, nloc = atype.shape[:2]
    cc, bb, fp, ap = coord, box, fparam, aparam
    del coord, box, fparam, aparam
    if bb is not None:
        coord_normalized = normalize_coord(
            cc.reshape(nframes, nloc, 3),
            bb.reshape(nframes, 3, 3),
        )
    else:
        coord_normalized = cc.reshape(nframes, nloc, 3).copy()
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
        coord_normalized, atype, bb, rcut
    )
    nlist = build_neighbor_list(
        extended_coord,
        extended_atype,
        nloc,
        rcut,
        sel,
        # types will be distinguished in the lower interface,
        # so it doesn't need to be distinguished here
        distinguish_types=False,
    )
    extended_coord = extended_coord.reshape(nframes, -1, 3)
    return extended_coord, extended_atype, nlist, mapping, fp, ap


def convert_numpy_data_to_jax_data(
    numpy_data: dict[str, np.ndarray | np.floating],
    sharding: Any | None = None,
    data_axis_size: int = 1,
    natoms_axis_size: int = 1,
) -> dict[str, jnp.ndarray | bool]:
    """Convert NumPy data to JAX data.

    Parameters
    ----------
    numpy_data : dict[str, np.ndarray | np.floating]
        NumPy data

    Returns
    -------
    jax_data
        JAX data
    """
    # numpy to jax
    jax_data = {
        kk: jnp.asarray(vv) if not kk.startswith("find_") else bool(vv.item())
        for kk, vv in numpy_data.items()
    }
    if sharding is not None:
        jax_data = {
            kk: jax.make_array_from_process_local_data(sharding, vv)
            if not kk.startswith("find_")
            and vv is not None
            and kk not in {"natoms_vec", "default_mesh"}
            else vv
            for kk, vv in jax_data.items()
        }

    def _label_sharding(key: str, value: jnp.ndarray) -> Any:
        spec = _data_partition_spec(
            key,
            value,
            data_axis_size=data_axis_size,
            natoms_axis_size=natoms_axis_size,
        )
        if sharding is not None and hasattr(sharding, "mesh"):
            return NamedSharding(sharding.mesh, spec)
        if getattr(jax, "set_mesh", None) is None:
            return None
        return spec

    jax_data = {
        kk: jax.device_put(
            vv,
            _label_sharding(kk, vv),
        )
        if not kk.startswith("find_")
        and vv is not None
        and kk not in {"natoms_vec", "default_mesh"}
        else vv
        for kk, vv in jax_data.items()
    }
    return jax_data


def trim_mixed_padding_batch(
    numpy_data: dict[str, np.ndarray | np.floating],
    enabled: bool = True,
) -> dict[str, np.ndarray | np.floating]:
    """Trim mixed-type padding atoms before JAX tracing."""
    if not enabled:
        return numpy_data
    atype = numpy_data.get("type")
    if not isinstance(atype, np.ndarray) or atype.ndim != 2 or not np.any(atype < 0):
        return numpy_data
    nf, old_nloc = atype.shape
    nloc = int(np.max(np.sum(atype >= 0, axis=1)))
    if nloc <= 0 or nloc >= old_nloc:
        return numpy_data

    trimmed = dict(numpy_data)
    trimmed_type = atype[:, :nloc].copy()
    trimmed["type"] = trimmed_type

    for key in ("coord", "force"):
        value = trimmed.get(key)
        if (
            isinstance(value, np.ndarray)
            and value.ndim == 2
            and value.shape == (nf, old_nloc * 3)
        ):
            trimmed[key] = value.reshape(nf, old_nloc, 3)[:, :nloc, :].reshape(
                nf, nloc * 3
            )

    hessian = trimmed.get("hessian")
    if (
        isinstance(hessian, np.ndarray)
        and hessian.ndim == 3
        and hessian.shape == (nf, old_nloc * 3, old_nloc * 3)
    ):
        ncoord = nloc * 3
        trimmed["hessian"] = hessian[:, :ncoord, :ncoord]
    elif (
        isinstance(hessian, np.ndarray)
        and hessian.ndim == 2
        and hessian.shape == (nf, (old_nloc * 3) * (old_nloc * 3))
    ):
        ncoord = nloc * 3
        trimmed["hessian"] = hessian.reshape(
            nf, old_nloc * 3, old_nloc * 3
        )[:, :ncoord, :ncoord].reshape(nf, ncoord * ncoord)

    natoms_vec = trimmed.get("natoms_vec")
    if isinstance(natoms_vec, np.ndarray) and natoms_vec.ndim == 1:
        ntypes = max(int(natoms_vec.shape[0]) - 2, 0)
        counts = np.bincount(
            trimmed_type[trimmed_type >= 0].reshape(-1), minlength=ntypes
        )[:ntypes].astype(natoms_vec.dtype, copy=False)
        trimmed["natoms_vec"] = np.concatenate(
            (np.asarray([nloc, nloc], dtype=natoms_vec.dtype), counts)
        )

    real_natoms_vec = trimmed.get("real_natoms_vec")
    if isinstance(real_natoms_vec, np.ndarray) and real_natoms_vec.ndim == 2:
        ntypes = max(int(real_natoms_vec.shape[1]) - 2, 0)
        counts = np.zeros((nf, ntypes), dtype=real_natoms_vec.dtype)
        for iframe in range(nf):
            counts[iframe] = np.bincount(
                trimmed_type[iframe][trimmed_type[iframe] >= 0], minlength=ntypes
            )[:ntypes].astype(real_natoms_vec.dtype, copy=False)
        trimmed["real_natoms_vec"] = np.concatenate(
            (
                np.full((nf, 2), nloc, dtype=real_natoms_vec.dtype),
                counts,
            ),
            axis=1,
        )

    return trimmed
