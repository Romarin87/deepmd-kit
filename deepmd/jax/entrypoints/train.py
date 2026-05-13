# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeePMD training entrypoint script.

Can handle local or distributed training.
"""

import inspect
import json
import logging
import os
import pickle
import subprocess
import sys
import tempfile
import time
from typing import (
    Any,
    Optional,
)

from deepmd.common import (
    j_loader,
)
from deepmd.jax.env import (
    jax,
    jax_export,
)
from deepmd.jax.train.trainer import (
    DPTrainer,
)
from deepmd.jax.utils.distributed import (
    broadcast_object_from_process0,
)
from deepmd.jax.utils.finetune import (
    get_finetune_rules,
)
from deepmd.jax.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.jax.utils.serialization import (
    serialize_from_file,
)
from deepmd.utils import random as dp_random
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data_system import (
    get_data,
)
from deepmd.utils.summary import SummaryPrinter as BaseSummaryPrinter

__all__ = ["train"]

log = logging.getLogger(__name__)


def _serialize_finetune_in_subprocess(finetune: str) -> dict[str, Any]:
    """Read a .jax checkpoint outside the distributed parent process."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fp:
        output = fp.name
    code = (
        "import pickle, sys\n"
        "from deepmd.jax.utils.serialization import serialize_from_file\n"
        "data = serialize_from_file(sys.argv[1])\n"
        "with open(sys.argv[2], 'wb') as fp:\n"
        "    pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)\n"
    )
    try:
        subprocess.run(
            [sys.executable, "-c", code, finetune, output],
            check=True,
            env=os.environ.copy(),
        )
        with open(output, "rb") as fp:
            return pickle.load(fp)
    finally:
        try:
            os.remove(output)
        except OSError:
            pass


def _load_finetune_data(finetune: str) -> dict[str, Any]:
    if jax.process_count() <= 1:
        return serialize_from_file(finetune)

    finetune_data = None
    status: dict[str, Any] = {"ok": True}
    if jax.process_index() == 0:
        try:
            finetune_data = _serialize_finetune_in_subprocess(finetune)
            status["model_def_script"] = finetune_data["model_def_script"]
        except Exception as exc:
            status = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    status = broadcast_object_from_process0(status, purpose="finetune metadata")
    if not status["ok"]:
        raise RuntimeError(
            "JAX process 0 failed to load finetune checkpoint: "
            f"{status['error_type']}: {status['error']}"
        )
    if finetune_data is None:
        finetune_data = {"model_def_script": status["model_def_script"]}
    return finetune_data


def _get_jax_distributed_config() -> Optional[tuple[str, int, int]]:
    """Get explicit JAX distributed config from DP or PET environment variables."""
    multi_nproc = os.environ.get("DP_JAX_MULTI_NPROC") or os.environ.get("PET_NNODES")
    if not multi_nproc or int(multi_nproc) <= 1:
        return None
    multi_nproc_int = int(multi_nproc)

    multi_iproc = os.environ.get("DP_JAX_MULTI_IPROC") or os.environ.get("PET_NODE_RANK")
    if multi_iproc is None or int(multi_iproc) < 0:
        raise ValueError(
            "DP_JAX_MULTI_IPROC/PET_NODE_RANK is not given or is less than 0"
        )
    multi_iproc_int = int(multi_iproc)

    multi_host = os.environ.get("DP_JAX_MULTI_HOST")
    if not multi_host:
        master_addr = os.environ.get("PET_MASTER_ADDR")
        master_port = os.environ.get("PET_MASTER_PORT")
        if master_addr and master_port:
            multi_host = f"{master_addr}:{master_port}"
    if not multi_host:
        raise ValueError(
            "DP_JAX_MULTI_HOST or PET_MASTER_ADDR/PET_MASTER_PORT is not given"
        )

    return multi_host, multi_nproc_int, multi_iproc_int


def _get_jax_local_device_ids() -> Optional[list[int]]:
    """Get local device ids for one JAX process using all visible local GPUs."""
    local_device_ids = os.environ.get("DP_JAX_LOCAL_DEVICE_IDS")
    if local_device_ids:
        return [int(ii) for ii in local_device_ids.split(",") if ii.strip()]

    local_device_count = os.environ.get("DP_JAX_LOCAL_DEVICE_COUNT")
    if local_device_count:
        return list(range(int(local_device_count)))

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        visible_device_count = len(
            [ii for ii in visible_devices.split(",") if ii.strip()]
        )
        return list(range(visible_device_count))

    pet_nproc_per_node = os.environ.get("PET_NPROC_PER_NODE")
    if pet_nproc_per_node:
        return list(range(int(pet_nproc_per_node)))

    return None


class SummaryPrinter(BaseSummaryPrinter):
    """Summary printer for JAX."""

    def is_built_with_cuda(self) -> bool:
        return jax_export.default_export_platform() == "cuda"

    def is_built_with_rocm(self) -> bool:
        return jax_export.default_export_platform() == "rocm"

    def get_compute_device(self) -> str:
        return jax.default_backend()

    def get_ngpus(self) -> int:
        return jax.device_count()

    def get_backend_info(self) -> dict:
        return {
            "Backend": "JAX",
            "JAX ver": jax.__version__,
        }

    def get_device_name(self) -> str:
        devices = jax.devices()
        if devices:
            return devices[0].device_kind
        return "Unknown"


def train(
    *,
    INPUT: str,
    init_model: Optional[str],
    restart: Optional[str],
    output: str,
    init_frz_model: str,
    mpi_log: str,
    log_level: int,
    log_path: Optional[str],
    skip_neighbor_stat: bool = False,
    finetune: Optional[str] = None,
    use_pretrain_script: bool = False,
    force_load: bool = False,
    model_branch: str = "",
    **kwargs: Any,
) -> None:
    distributed_config = _get_jax_distributed_config()
    if distributed_config is not None:
        multi_host, multi_nproc, multi_iproc = distributed_config
        init_kwargs = dict(
            coordinator_address=multi_host,
            num_processes=multi_nproc,
            process_id=multi_iproc,
        )
        local_device_ids = _get_jax_local_device_ids()
        if local_device_ids is not None:
            init_kwargs["local_device_ids"] = local_device_ids
        if "cluster_detection_method" in inspect.signature(
            jax.distributed.initialize
        ).parameters:
            init_kwargs["cluster_detection_method"] = "deactivate"
        print(
            "JAX distributed initialize: "
            f"coordinator_address={multi_host}, "
            f"num_processes={multi_nproc}, "
            f"process_id={multi_iproc}, "
            f"local_device_ids={local_device_ids}",
            flush=True,
        )
        jax.distributed.initialize(**init_kwargs)

    jdata = j_loader(INPUT)

    multi_task = "model_dict" in jdata["model"]
    shared_links = None
    if multi_task:
        jdata["model"], shared_links = preprocess_shared_params(jdata["model"])
        if "RANDOM" in jdata["model"]["model_dict"]:
            raise ValueError("Model name can not be 'RANDOM' in multi-task mode!")

    finetune_links = None
    finetune_data = None
    if finetune is not None:
        finetune_data = _load_finetune_data(finetune)
        jdata["model"], finetune_links, finetune_data = get_finetune_rules(
            finetune,
            jdata["model"],
            model_branch=model_branch,
            change_model_params=use_pretrain_script,
            finetune_data=finetune_data,
        )
    if (init_model is not None or init_frz_model) and use_pretrain_script:
        source_model = init_model if init_model is not None else init_frz_model
        source_model_data = serialize_from_file(source_model)
        jdata["model"] = source_model_data["model_def_script"]

    jdata = update_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")
    jdata = normalize(jdata, multi_task=multi_task)
    jdata = update_sel(jdata, multi_task=multi_task)

    with open(output, "w") as fp:
        json.dump(jdata, fp, indent=4)
    SummaryPrinter()()

    model = DPTrainer(
        jdata,
        init_model=init_model,
        restart=restart,
        init_frz_model=init_frz_model or None,
        finetune_model=finetune,
        force_load=force_load,
        shared_links=shared_links,
        finetune_links=finetune_links,
        finetune_model_data=finetune_data,
    )

    seed = jdata["training"].get("seed", None)
    if seed is not None:
        seed += jax.process_index()
        seed = seed % (2**32)
    dp_random.seed(seed)

    if not multi_task:
        rcut = model.model.get_rcut()
        type_map = model.model.get_type_map()
        ipt_type_map = None if len(type_map) == 0 else type_map
        train_data = get_data(
            jdata["training"]["training_data"], rcut, ipt_type_map, None
        )
        train_data.add_data_requirements(model.data_requirements)
        train_data.print_summary("training")
        if jdata["training"].get("validation_data", None) is not None:
            valid_data = get_data(
                jdata["training"]["validation_data"],
                rcut,
                train_data.type_map,
                None,
            )
            valid_data.add_data_requirements(model.data_requirements)
            valid_data.print_summary("validation")
        else:
            valid_data = None
    else:
        train_data = {}
        valid_data = {}
        for model_key in model.model_keys:
            branch_model = model.model[model_key]
            rcut = branch_model.get_rcut()
            type_map = branch_model.get_type_map()
            ipt_type_map = None if len(type_map) == 0 else type_map
            branch_train = get_data(
                jdata["training"]["data_dict"][model_key]["training_data"],
                rcut,
                ipt_type_map,
                None,
            )
            branch_train.add_data_requirements(model.data_requirements[model_key])
            branch_train.print_summary(f"training in {model_key}")
            train_data[model_key] = branch_train
            if (
                jdata["training"]["data_dict"][model_key].get("validation_data", None)
                is not None
            ):
                branch_valid = get_data(
                    jdata["training"]["data_dict"][model_key]["validation_data"],
                    rcut,
                    branch_train.type_map,
                    None,
                )
                branch_valid.add_data_requirements(model.data_requirements[model_key])
                branch_valid.print_summary(f"validation in {model_key}")
                valid_data[model_key] = branch_valid
            else:
                valid_data[model_key] = None

    start_time = time.time()
    model.train(train_data, valid_data)
    end_time = time.time()
    log.info("finished training")
    log.info(f"wall time: {(end_time - start_time):.3f} s")


def update_sel(jdata: dict, *, multi_task: bool = False) -> dict:
    log.info(
        "Calculate neighbor statistics... (add --skip-neighbor-stat to skip this step)"
    )
    jdata_cpy = jdata.copy()
    if not multi_task:
        type_map = jdata["model"].get("type_map")
        train_data = get_data(
            jdata["training"]["training_data"],
            0,
            type_map,
            None,
        )
        del train_data
    else:
        for model_key in jdata["model"]["model_dict"]:
            type_map = jdata["model"]["model_dict"][model_key].get("type_map")
            train_data = get_data(
                jdata["training"]["data_dict"][model_key]["training_data"],
                0,
                type_map,
                None,
            )
            del train_data
    return jdata_cpy
