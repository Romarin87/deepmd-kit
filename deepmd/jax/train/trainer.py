#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Local training utilities for the JAX backend."""

import logging
import os
import platform
import shutil
import time
from pathlib import (
    Path,
)
from typing import (
    TextIO,
)

import numpy as np
import optax
import orbax.checkpoint as ocp
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
)
from deepmd.utils.model_stat import (
    make_stat_input,
)

log = logging.getLogger(__name__)


def whether_hessian(loss_params: dict) -> bool:
    """Return whether the loss configuration requests Hessian training."""
    loss_type = loss_params.get("type", "ener")
    return loss_type == "ener" and loss_params.get("start_pref_h", 0.0) > 0.0


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
        self.model_def_script = jdata["model"]
        self.start_step = 0
        if self.init_model is not None:
            model_dict = serialize_from_file(self.init_model)
            self.model = BaseModel.deserialize(model_dict["model"])
        elif self.restart is not None:
            model_dict = serialize_from_file(self.restart)
            self.model = BaseModel.deserialize(model_dict["model"])
            self.start_step = model_dict.get("model_def_script", {}).get(
                "current_step",
                model_dict.get("@variables", {}).get("current_step", 0),
            )
        else:
            # from scratch
            self.model = get_model(jdata["model"])
        self.training_param = jdata["training"]
        self.num_steps = self.training_param.get("numb_steps")
        self.num_epoch = self.training_param.get("numb_epoch")
        if self.num_epoch is None:
            self.num_epoch = self.training_param.get("num_epoch")
        if self.num_epoch is None:
            self.num_epoch = self.training_param.get("num_epochs")

        learning_rate_param = jdata["learning_rate"]
        self.learning_rate_param = learning_rate_param
        self.lr = None
        self.optimizer_param = dict(jdata.get("optimizer", {}))
        self.opt_type = self.optimizer_param.pop("type", "Adam")
        self.gradient_max_norm = self.training_param.get("gradient_max_norm")
        loss_param = jdata.get("loss", {})
        loss_param["starter_learning_rate"] = learning_rate_param["start_lr"]

        loss_type = loss_param.get("type", "ener")
        if loss_type == "ener":
            self.has_hessian = whether_hessian(loss_param)
            if self.has_hessian:
                self.model.enable_hessian()
                self.model_def_script["hessian_mode"] = True
                self.loss = EnergyHessianLoss.get_loss(loss_param)
            else:
                self.loss = EnergyLoss.get_loss(loss_param)
        else:
            raise RuntimeError("unknown loss type " + loss_type)

        # training
        tr_data = jdata["training"]
        self.disp_file = tr_data.get("disp_file", "lcurve.out")
        self.disp_freq = tr_data.get("disp_freq", 1000)
        self.save_freq = tr_data.get("save_freq", 1000)
        self.save_ckpt = tr_data.get("save_ckpt", "model.ckpt")
        self.max_ckpt_keep = tr_data.get("max_ckpt_keep", 5)
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

    def _build_optimizer_tx(self) -> optax.GradientTransformation:
        """Build the configured optax optimizer transformation."""
        assert self.lr is not None
        opt_type = str(self.opt_type)
        lr_schedule = lambda step: self.lr.value(self.start_step + step)
        weight_decay = float(self.optimizer_param.get("weight_decay", 0.0))
        b1 = float(self.optimizer_param.get("adam_beta1", 0.9))
        b2 = float(self.optimizer_param.get("adam_beta2", 0.999))
        eps = float(self.optimizer_param.get("eps", 1e-8))
        if opt_type == "Adam":
            tx = optax.adam(learning_rate=lr_schedule, b1=b1, b2=b2, eps=eps)
        elif opt_type == "AdamW":
            tx = optax.adamw(
                learning_rate=lr_schedule,
                b1=b1,
                b2=b2,
                eps=eps,
                weight_decay=weight_decay,
            )
        elif opt_type == "HybridMuon":
            tx = hybrid_muon(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
                momentum=float(self.optimizer_param.get("momentum", 0.95)),
                adam_betas=(
                    float(self.optimizer_param.get("adam_beta1", 0.9)),
                    float(self.optimizer_param.get("adam_beta2", 0.95)),
                ),
                adam_eps=float(self.optimizer_param.get("adam_eps", 1e-20)),
                lr_adjust=float(self.optimizer_param.get("lr_adjust", 0.0)),
                lr_adjust_coeff=float(
                    self.optimizer_param.get("lr_adjust_coeff", 0.18)
                ),
                muon_mode=str(self.optimizer_param.get("muon_mode", "slice")),
                enable_gram=bool(self.optimizer_param.get("enable_gram", True)),
                flash_muon=bool(self.optimizer_param.get("flash_muon", True)),
                magma_muon=bool(self.optimizer_param.get("magma_muon", True)),
            )
        else:
            raise RuntimeError(f"unknown optimizer type {opt_type}")
        if self.gradient_max_norm is not None:
            tx = optax.chain(optax.clip_by_global_norm(float(self.gradient_max_norm)), tx)
        return tx

    def _ensure_training_length(self, train_data: DeepmdDataSystem) -> None:
        """Resolve step-based training length and construct the LR schedule."""
        if self.num_steps is None:
            if self.num_epoch is None:
                raise ValueError("Either training.numb_steps or training.num_epochs must be set.")
            if self.num_epoch <= 0:
                raise ValueError("training.num_epochs must be positive.")
            total_numb_batch = compute_total_numb_batch(
                train_data.nbatches,
                train_data.sys_probs,
            )
            self.num_steps = int(np.ceil(float(self.num_epoch) * total_numb_batch))
            log.info(
                "Computed numb_steps=%d from num_epochs=%s and total_numb_batch=%d.",
                self.num_steps,
                self.num_epoch,
                total_numb_batch,
            )
        self.lr = BaseLR(
            **self.learning_rate_param,
            num_steps=int(self.num_steps),
        )

    @property
    def data_requirements(self) -> list[DataRequirementItem]:
        """Labels required by the configured loss."""
        return self.loss.label_requirement

    def train(
        self, train_data: DeepmdDataSystem, valid_data: DeepmdDataSystem | None = None
    ) -> None:
        """Run the training loop with optional validation data."""
        self._ensure_training_length(train_data)
        model = self.model
        tx = self._build_optimizer_tx()
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

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
            model_dict_lower = model.call_common_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
            )
            model_dict = communicate_extended_output(
                model_dict_lower,
                model.model_output_def(),
                mapping,
                do_atomic_virial=False,
            )
            model_dict["atom_energy"] = model_dict["energy"]
            model_dict["energy"] = model_dict["energy_redu"]
            model_dict["force"] = model_dict["energy_derv_r"].squeeze(-2)
            model_dict["virial"] = model_dict["energy_derv_c_redu"].squeeze(-2)
            if self.has_hessian and model_dict.get("energy_derv_r_derv_r") is not None:
                model_dict["hessian"] = model_dict["energy_derv_r_derv_r"].squeeze(-3)
            loss, more_loss = self.loss(
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
            model_dict_lower = model.call_common_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
            )
            model_dict = communicate_extended_output(
                model_dict_lower,
                model.model_output_def(),
                mapping,
                do_atomic_virial=False,
            )
            model_dict["atom_energy"] = model_dict["energy"]
            model_dict["energy"] = model_dict["energy_redu"]
            model_dict["force"] = model_dict["energy_derv_r"].squeeze(-2)
            model_dict["virial"] = model_dict["energy_derv_c_redu"].squeeze(-2)
            if self.has_hessian and model_dict.get("energy_derv_r_derv_r") is not None:
                model_dict["hessian"] = model_dict["energy_derv_r_derv_r"].squeeze(-3)
            loss, more_loss = self.loss(
                learning_rate=lr,
                natoms=label_dict["type"].shape[1],
                model_dict=model_dict,
                label_dict=label_dict,
            )
            return more_loss

        @nnx.jit
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
        ) -> None:
            grads = nnx.grad(loss_fn)(
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
            if Version(flax_version) >= Version("0.11.0"):
                optimizer.update(model, grads)
            else:
                optimizer.update(grads)

        start_time = time.time()
        disp_path = Path(self.disp_file)
        disp_mode = "a" if self.start_step > 0 and disp_path.exists() else "w"
        with open(disp_path, disp_mode) as disp_file_fp:
            for step in range(self.start_step, self.num_steps):
                batch_data = train_data.get_batch()
                # numpy to jax
                jax_data = convert_numpy_data_to_jax_data(batch_data)
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
                )
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
                    if valid_data is not None:
                        valid_more_loss_list = []
                        for _ in range(self.valid_numb_batch):
                            valid_batch_data = valid_data.get_batch()
                            jax_valid_data = convert_numpy_data_to_jax_data(
                                valid_batch_data
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
                            valid_more_loss_list.append(
                                loss_fn_more_loss(
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
                            )
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
        if self.num_steps > self.start_step and self.num_steps % self.save_freq != 0:
            self._save_checkpoint(model, self.num_steps)

    def _save_checkpoint(self, model: BaseModel, step: int) -> None:
        """Save a JAX checkpoint and update the stable checkpoint pointer."""
        _, state = nnx.split(model)
        ckpt_path = Path(f"{self.save_ckpt}-{step}.jax")
        if ckpt_path.is_dir():
            # remove old checkpoint if it exists
            shutil.rmtree(ckpt_path)
        model_def_script_cpy = self.model_def_script.copy()
        model_def_script_cpy["current_step"] = step
        with ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "model_def_script")
        ) as checkpointer:
            checkpointer.save(
                ckpt_path.absolute(),
                ocp.args.Composite(
                    state=ocp.args.StandardSave(state.to_pure_dict()),
                    model_def_script=ocp.args.JsonSave(model_def_script_cpy),
                ),
            )
        log.info(f"Trained model has been saved to: {ckpt_path!s}")
        _link_checkpoint(ckpt_path, Path(f"{self.save_ckpt}.jax"))
        self._cleanup_old_checkpoints()
        with open("checkpoint", "w") as fp:
            fp.write(f"{self.save_ckpt}.jax")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoint directories beyond the retention limit."""
        if self.max_ckpt_keep <= 0:
            return
        ckpt_parent = Path(self.save_ckpt).parent
        ckpt_prefix = Path(self.save_ckpt).name
        checkpoints = []
        for path in ckpt_parent.glob(f"{ckpt_prefix}-*.jax"):
            if not path.is_dir() or path.is_symlink():
                continue
            step_text = path.name.removeprefix(f"{ckpt_prefix}-").removesuffix(".jax")
            if step_text.isdigit():
                checkpoints.append((int(step_text), path))
        for _, path in sorted(checkpoints)[: -self.max_ckpt_keep]:
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
        print_str += "# If there is no available reference data, rmse_*_{val,trn} will print nan\n"
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
    return jax_data
