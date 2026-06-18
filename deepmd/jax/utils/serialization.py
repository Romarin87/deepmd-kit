# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)
import os
from pathlib import (
    Path,
)

import numpy as np
import orbax.checkpoint as ocp

from deepmd.dpmodel.output_def import (
    get_deriv_name,
    get_reduce_name,
)
from deepmd.dpmodel.utils.serialization import (
    load_dp_model,
    save_dp_model,
)
from deepmd.jax.env import (
    jax,
    jax_export,
    jnp,
    nnx,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.model import (
    get_model_for_wrapper,
)
from deepmd.jax.model.multitask import (
    ModelWrapper,
)
from deepmd.jax.utils.multi_task import (
    get_case_embd_config,
)
from deepmd.utils.model_branch_dict import (
    get_model_dict,
)


def _is_topology_mismatch_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "Ranks do not match" in message
        or "Topology mismatch detected" in message
        or "available devices are different from the devices used to save the checkpoint"
        in message
        or "was not found in jax.local_devices()" in message
    )


def select_model_branch(
    data: dict,
    model_branch: str | None,
) -> dict:
    model_def_script = data["model_def_script"]
    if "model_dict" not in model_def_script:
        return data
    if not model_branch:
        raise ValueError(
            "Freezing a multitask JAX checkpoint to a single model requires "
            "selecting a branch with --head/--model-branch."
        )
    model_alias_dict, _ = get_model_dict(model_def_script["model_dict"])
    if model_branch not in model_alias_dict:
        raise ValueError(
            f"No model branch or alias named '{model_branch}'. "
            f"Available branches are: {list(model_def_script['model_dict'].keys())}"
        )
    model_branch = model_alias_dict[model_branch]
    return {
        **data,
        "model_def_script": deepcopy(model_def_script["model_dict"][model_branch]),
        "model": deepcopy(data["model"]["model_dict"][model_branch]),
    }


def _call_common_lower_ef(
    model: BaseModel,
    extended_coord: jnp.ndarray,
    extended_atype: jnp.ndarray,
    nlist: jnp.ndarray,
    mapping: jnp.ndarray,
    fparam: jnp.ndarray,
    aparam: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Lower JAX HLO path that computes only energy and force."""
    nframes = extended_atype.shape[0]
    extended_coord = extended_coord.reshape(nframes, -1, 3)
    nlist = model.format_nlist(
        extended_coord,
        extended_atype,
        nlist,
        extra_nlist_sort=model.need_sorted_nlist_for_lower(),
    )
    cc_ext, _, fp, ap, input_prec = model._input_type_cast(
        extended_coord, fparam=fparam, aparam=aparam
    )
    atomic_ret = model.atomic_model.forward_common_atomic(
        cc_ext,
        extended_atype,
        nlist,
        mapping=mapping,
        fparam=fp,
        aparam=ap,
    )
    atomic_output_def = model.atomic_output_def()
    model_predict = {}
    if "mask" in atomic_ret:
        model_predict["mask"] = atomic_ret["mask"]
    for kk, vv in atomic_ret.items():
        vdef = atomic_output_def[kk]
        if not vdef.reducible or not vdef.r_differentiable:
            continue
        shap = vdef.shape
        atom_axis = -(len(shap) + 1)
        kk_redu = get_reduce_name(kk)
        kk_derv_r = get_deriv_name(kk)[0]
        model_predict[kk] = vv
        if vdef.intensive:
            mask = atomic_ret["mask"] if "mask" in atomic_ret else None
            if mask is not None:
                model_predict[kk_redu] = jnp.sum(vv, axis=atom_axis) / jnp.sum(
                    mask, axis=-1, keepdims=True
                )
            else:
                model_predict[kk_redu] = jnp.mean(vv, axis=atom_axis)
        else:
            model_predict[kk_redu] = jnp.sum(vv, axis=atom_axis)

        def eval_output(
            cc_ext_one: jnp.ndarray,
            extended_atype_one: jnp.ndarray,
            nlist_one: jnp.ndarray,
            mapping_one: jnp.ndarray,
            fparam_one: jnp.ndarray,
            aparam_one: jnp.ndarray,
            *,
            _kk: str = kk,
            _atom_axis: int = atom_axis,
        ) -> jnp.ndarray:
            atomic_ret_one = model.atomic_model.forward_common_atomic(
                cc_ext_one[None, ...],
                extended_atype_one[None, ...],
                nlist_one[None, ...],
                mapping=mapping_one[None, ...],
                fparam=fparam_one[None, ...] if fparam_one is not None else None,
                aparam=aparam_one[None, ...] if aparam_one is not None else None,
            )
            return jnp.sum(atomic_ret_one[_kk][0], axis=_atom_axis)

        ff = -jax.vmap(
            jax.jacrev(eval_output, argnums=0),
            in_axes=(
                0,
                0,
                0,
                0,
                0 if fp is not None else None,
                0 if ap is not None else None,
            ),
        )(
            cc_ext,
            extended_atype,
            nlist,
            mapping,
            fp,
            ap,
        )
        def_ndim = len(vdef.shape)
        model_predict[kk_derv_r] = jnp.transpose(
            ff, [0, def_ndim + 1, *range(1, def_ndim + 1), def_ndim + 2]
        )
    return model._output_type_cast(model_predict, input_prec)


def deserialize_to_file(model_file: str, data: dict, hessian: bool = False) -> None:
    """Deserialize the dictionary to a model file."""
    if model_file.endswith(".jax"):
        model_def_script = data["model_def_script"].copy()
        shared_links = model_def_script.get("shared_links")
        if "model_dict" in model_def_script:
            _, case_embd_index = get_case_embd_config(model_def_script)
            model = ModelWrapper.deserialize(
                data["model"],
                shared_links=shared_links,
                case_embd_index=case_embd_index,
            )
            if hessian:
                raise ValueError(
                    "Freezing Hessian into a multitask .jax checkpoint is not supported. "
                    "Please select a single branch first."
                )
        else:
            model = BaseModel.deserialize(data["model"])
            if hessian:
                model.enable_hessian()
                model_def_script["hessian_mode"] = True
        _, state = nnx.split(model)
        with ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "model_def_script")
        ) as checkpointer:
            checkpointer.save(
                Path(model_file).absolute(),
                ocp.args.Composite(
                    state=ocp.args.StandardSave(state.to_pure_dict()),
                    model_def_script=ocp.args.JsonSave(model_def_script),
                ),
            )
    elif model_file.endswith(".hlo"):
        if "model_dict" in data["model_def_script"]:
            raise ValueError(
                "Freezing a multitask JAX checkpoint to .hlo requires selecting a single branch with --head/--model-branch."
            )
        model = BaseModel.deserialize(data["model"])
        model_def_script = data["model_def_script"]
        hessian_chunk_size = _get_hessian_chunk_size() if hessian else 0
        if hessian:
            if hessian_chunk_size > 0:
                model_def_script["hessian_chunk_size"] = hessian_chunk_size
            else:
                model.enable_hessian()
                model_def_script["hessian_mode"] = True
        call_lower = model.call_common_lower

        nf, nloc, nghost = jax_export.symbolic_shape("nf, nloc, nghost")

        def exported_whether_do_atomic_virial(
            do_atomic_virial: bool, has_ghost_atoms: bool, ef_only: bool = False
        ) -> "jax_export.Exported":
            def call_lower_with_fixed_do_atomic_virial(
                coord: jnp.ndarray,
                atype: jnp.ndarray,
                nlist: jnp.ndarray,
                mapping: jnp.ndarray,
                fparam: jnp.ndarray,
                aparam: jnp.ndarray,
            ) -> dict[str, jnp.ndarray]:
                if ef_only:
                    return _call_common_lower_ef(
                        model,
                        coord,
                        atype,
                        nlist,
                        mapping,
                        fparam,
                        aparam,
                    )
                output = call_lower(
                    coord,
                    atype,
                    nlist,
                    mapping,
                    fparam,
                    aparam,
                    do_atomic_virial=do_atomic_virial,
                )
                return output

            if has_ghost_atoms:
                nghost_ = nghost
            else:
                nghost_ = 0

            return jax_export.export(jax.jit(call_lower_with_fixed_do_atomic_virial))(
                jax.ShapeDtypeStruct((nf, nloc + nghost_, 3), jnp.float64),
                jax.ShapeDtypeStruct((nf, nloc + nghost_), jnp.int32),
                jax.ShapeDtypeStruct((nf, nloc, model.get_nnei()), jnp.int64),
                jax.ShapeDtypeStruct((nf, nloc + nghost_), jnp.int64),
                jax.ShapeDtypeStruct((nf, model.get_dim_fparam()), jnp.float64)
                if model.get_dim_fparam()
                else None,
                jax.ShapeDtypeStruct((nf, nloc, model.get_dim_aparam()), jnp.float64)
                if model.get_dim_aparam()
                else None,
            )

        exported = exported_whether_do_atomic_virial(
            do_atomic_virial=False, has_ghost_atoms=True
        )
        exported_ef = exported_whether_do_atomic_virial(
            do_atomic_virial=False, has_ghost_atoms=True, ef_only=True
        )
        exported_atomic_virial = exported_whether_do_atomic_virial(
            do_atomic_virial=True, has_ghost_atoms=True
        )
        serialized: bytearray = exported.serialize()
        serialized_ef: bytearray = exported_ef.serialize()
        serialized_atomic_virial = exported_atomic_virial.serialize()

        exported_no_ghost = exported_whether_do_atomic_virial(
            do_atomic_virial=False, has_ghost_atoms=False
        )
        exported_ef_no_ghost = exported_whether_do_atomic_virial(
            do_atomic_virial=False, has_ghost_atoms=False, ef_only=True
        )
        exported_atomic_virial_no_ghost = exported_whether_do_atomic_virial(
            do_atomic_virial=True, has_ghost_atoms=False
        )
        serialized_no_ghost: bytearray = exported_no_ghost.serialize()
        serialized_ef_no_ghost: bytearray = exported_ef_no_ghost.serialize()
        serialized_atomic_virial_no_ghost = exported_atomic_virial_no_ghost.serialize()
        if hessian_chunk_size > 0:
            serialized_hessian_block = _export_hessian_block(
                model,
                hessian_chunk_size,
            ).serialize()

        data = data.copy()
        data.setdefault("@variables", {})
        data["@variables"]["stablehlo"] = np.void(serialized)
        data["@variables"]["stablehlo_ef"] = np.void(serialized_ef)
        data["@variables"]["stablehlo_atomic_virial"] = np.void(
            serialized_atomic_virial
        )
        data["@variables"]["stablehlo_no_ghost"] = np.void(serialized_no_ghost)
        data["@variables"]["stablehlo_ef_no_ghost"] = np.void(
            serialized_ef_no_ghost
        )
        data["@variables"]["stablehlo_atomic_virial_no_ghost"] = np.void(
            serialized_atomic_virial_no_ghost
        )
        if hessian_chunk_size > 0:
            data["@variables"]["stablehlo_hessian_block"] = np.void(
                serialized_hessian_block
            )
        data["constants"] = {
            "type_map": model.get_type_map(),
            "rcut": model.get_rcut(),
            "dim_fparam": model.get_dim_fparam(),
            "dim_aparam": model.get_dim_aparam(),
            "sel_type": model.get_sel_type(),
            "is_aparam_nall": model.is_aparam_nall(),
            "model_output_type": model.model_output_type(),
            "mixed_types": model.mixed_types(),
            "min_nbor_dist": model.get_min_nbor_dist(),
            "sel": model.get_sel(),
            "has_default_fparam": model.has_default_fparam(),
            "default_fparam": model.get_default_fparam(),
            "hessian_chunk_size": hessian_chunk_size,
        }
        save_dp_model(filename=model_file, model_dict=data)
    elif model_file.endswith(".savedmodel"):
        from deepmd.jax.jax2tf.serialization import (
            deserialize_to_file as deserialize_to_savedmodel,
        )

        return deserialize_to_savedmodel(model_file, data)
    else:
        raise ValueError("Unsupported file extension")


def serialize_from_file(model_file: str) -> dict:
    """Serialize the model file to a dictionary."""
    if model_file.endswith(".jax"):
        with ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "model_def_script")
        ) as checkpointer:
            try:
                data = checkpointer.restore(
                    Path(model_file).absolute(),
                    ocp.args.Composite(
                        state=ocp.args.StandardRestore(),
                        model_def_script=ocp.args.JsonRestore(),
                    ),
                )
            except ValueError as exc:
                if not _is_topology_mismatch_error(exc):
                    raise
                model_def_script = checkpointer.restore(
                    Path(model_file).absolute(),
                    ocp.args.Composite(model_def_script=ocp.args.JsonRestore()),
                ).model_def_script
                shared_links = model_def_script.get("shared_links")
                abstract_model = get_model_for_wrapper(
                    model_def_script,
                    shared_links=shared_links,
                )
                if "model_dict" in model_def_script:
                    for model_key in model_def_script["model_dict"]:
                        if model_def_script["model_dict"][model_key].get(
                            "hessian_mode", False
                        ):
                            abstract_model[model_key].enable_hessian()
                elif model_def_script.get("hessian_mode", False):
                    abstract_model.enable_hessian()
                _, abstract_state = nnx.split(abstract_model)
                data = checkpointer.restore(
                    Path(model_file).absolute(),
                    ocp.args.Composite(
                        state=ocp.args.StandardRestore(
                            item=abstract_state.to_pure_dict(),
                            strict=False,
                        ),
                        model_def_script=ocp.args.JsonRestore(),
                    ),
                )
        state = data.state

        def convert_str_to_int_key(item: dict) -> None:
            for key, value in item.copy().items():
                if isinstance(value, dict):
                    convert_str_to_int_key(value)
                if isinstance(key, str) and key.isdigit():
                    item[int(key)] = item.pop(key)

        convert_str_to_int_key(state)

        model_def_script = data.model_def_script
        current_step = model_def_script.pop("current_step", 0)
        shared_links = model_def_script.get("shared_links")
        abstract_model = get_model_for_wrapper(
            model_def_script,
            shared_links=shared_links,
        )
        if "model_dict" in model_def_script:
            for model_key in model_def_script["model_dict"]:
                if model_def_script["model_dict"][model_key].get(
                    "hessian_mode", False
                ):
                    abstract_model[model_key].enable_hessian()
        elif model_def_script.get("hessian_mode", False):
            abstract_model.enable_hessian()
        graphdef, abstract_state = nnx.split(abstract_model)
        state = _normalize_checkpoint_state_for_model(
            state,
            abstract_state.to_pure_dict(),
        )
        abstract_state.replace_by_pure_dict(state)
        model = nnx.merge(graphdef, abstract_state)
        return {
            "backend": "JAX",
            "jax_version": jax.__version__,
            "model": model.serialize(),
            "model_def_script": model_def_script,
            "@variables": {
                "current_step": current_step,
            },
        }
    elif model_file.endswith(".hlo"):
        data = load_dp_model(model_file)
        data.pop("constants")
        data["@variables"].pop("stablehlo")
        return data
    else:
        raise ValueError("JAX backend only supports converting .jax directory")


def _normalize_checkpoint_state_for_model(state: dict, target: dict) -> dict:
    """Normalize legacy checkpoint state to the current model state tree.

    Older JAX checkpoints may use ``atomic_model.fitting_net`` where current
    models expose ``atomic_model.fitting``. Some checkpoints also carry
    auxiliary fitting keys that are no longer present in the active model
    state. Keep only keys accepted by the target state so checkpoint freeze can
    restore the matching parameters while leaving newly initialized keys at
    their model defaults.
    """
    _rename_legacy_fitting_net(state)
    _drop_unknown_state_keys(state, target)
    return state


def _rename_legacy_fitting_net(item: dict) -> None:
    for value in item.values():
        if isinstance(value, dict):
            atomic_model = value.get("atomic_model")
            if (
                isinstance(atomic_model, dict)
                and "fitting_net" in atomic_model
                and "fitting" not in atomic_model
            ):
                atomic_model["fitting"] = atomic_model.pop("fitting_net")
            _rename_legacy_fitting_net(value)


def _drop_unknown_state_keys(item: dict, target: dict) -> None:
    for key in list(item.keys()):
        if key not in target:
            item.pop(key)
            continue
        value = item[key]
        target_value = target[key]
        if isinstance(value, dict) and isinstance(target_value, dict):
            _drop_unknown_state_keys(value, target_value)


def _get_hessian_chunk_size() -> int:
    raw_value = os.environ.get("DP_JAX_HESSIAN_CHUNK_SIZE")
    if raw_value is None or raw_value == "":
        return 0
    try:
        chunk_size = int(raw_value)
    except ValueError as err:
        raise ValueError(
            "DP_JAX_HESSIAN_CHUNK_SIZE must be a positive integer"
        ) from err
    if chunk_size <= 0:
        raise ValueError("DP_JAX_HESSIAN_CHUNK_SIZE must be a positive integer")
    return chunk_size


def _export_hessian_block(
    model: BaseModel,
    chunk_size: int,
) -> "jax_export.Exported":
    nf, nloc, nghost = jax_export.symbolic_shape(
        "nf, nloc, nghost",
        constraints=("nghost >= 0",),
    )
    nall = nloc + nghost

    def call_hessian_block(
        extended_coord: jnp.ndarray,
        extended_atype: jnp.ndarray,
        nlist: jnp.ndarray,
        mapping: jnp.ndarray,
        fparam: jnp.ndarray | None,
        aparam: jnp.ndarray | None,
        block_index: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        block_index = jnp.asarray(block_index, dtype=jnp.int32)

        def energy_one(
            extended_coord_one: jnp.ndarray,
            extended_atype_one: jnp.ndarray,
            nlist_one: jnp.ndarray,
            mapping_one: jnp.ndarray,
            fparam_one: jnp.ndarray | None,
            aparam_one: jnp.ndarray | None,
        ) -> jnp.ndarray:
            output = model.call_common_lower(
                extended_coord_one[None, ...],
                extended_atype_one[None, ...],
                nlist_one[None, ...],
                mapping=mapping_one[None, ...],
                fparam=None if fparam_one is None else fparam_one[None, ...],
                aparam=None if aparam_one is None else aparam_one[None, ...],
                do_atomic_virial=False,
            )
            return jnp.sum(output["energy_redu"])

        grad_one = jax.grad(energy_one, argnums=0)

        def hessian_block_one(
            extended_coord_one: jnp.ndarray,
            extended_atype_one: jnp.ndarray,
            nlist_one: jnp.ndarray,
            mapping_one: jnp.ndarray,
            fparam_one: jnp.ndarray | None,
            aparam_one: jnp.ndarray | None,
        ) -> jnp.ndarray:
            local_natoms = nlist_one.shape[0]
            local_dim = local_natoms * 3
            row_ids = block_index * chunk_size + jnp.arange(chunk_size)
            safe_row_ids = jnp.minimum(row_ids, local_dim - 1)

            def local_grad(coord_ext: jnp.ndarray) -> jnp.ndarray:
                grad_ext = grad_one(
                    coord_ext,
                    extended_atype_one,
                    nlist_one,
                    mapping_one,
                    fparam_one,
                    aparam_one,
                )
                return jnp.zeros(
                    (local_natoms, 3),
                    dtype=grad_ext.dtype,
                ).at[mapping_one].add(grad_ext)

            def hessian_row(row_id: jnp.ndarray) -> jnp.ndarray:
                row_ext = jax.grad(
                    lambda coord_ext: local_grad(coord_ext).reshape(-1)[row_id]
                )(extended_coord_one)
                return jnp.zeros(
                    (local_natoms, 3),
                    dtype=row_ext.dtype,
                ).at[mapping_one].add(row_ext).reshape(-1)

            rows = jax.vmap(hessian_row)(safe_row_ids)
            return jnp.where(row_ids[:, None] < local_dim, rows, 0.0)

        return {
            "energy_derv_r_derv_r_block": jax.vmap(hessian_block_one)(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fparam,
                aparam,
            )
        }

    exported = jax_export.export(jax.jit(call_hessian_block))(
        jax.ShapeDtypeStruct((nf, nall, 3), jnp.float64),
        jax.ShapeDtypeStruct((nf, nall), jnp.int32),
        jax.ShapeDtypeStruct((nf, nloc, model.get_nnei()), jnp.int64),
        jax.ShapeDtypeStruct((nf, nall), jnp.int64),
        jax.ShapeDtypeStruct((nf, model.get_dim_fparam()), jnp.float64)
        if model.get_dim_fparam()
        else None,
        jax.ShapeDtypeStruct((nf, nloc, model.get_dim_aparam()), jnp.float64)
        if model.get_dim_aparam()
        else None,
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    return exported
