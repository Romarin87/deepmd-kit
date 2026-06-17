# SPDX-License-Identifier: LGPL-3.0-or-later
import json
from collections.abc import (
    Callable,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.output_def import (
    ModelOutputDef,
    OutputVariableCategory,
    OutputVariableDef,
)
from deepmd.dpmodel.utils.serialization import (
    load_dp_model,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.infer.deep_dipole import (
    DeepDipole,
)
from deepmd.infer.deep_dos import (
    DeepDOS,
)
from deepmd.infer.deep_eval import DeepEval as DeepEvalWrapper
from deepmd.infer.deep_eval import (
    DeepEvalBackend,
)
from deepmd.infer.deep_polar import (
    DeepPolar,
)
from deepmd.infer.deep_pot import (
    DeepPot,
)
from deepmd.infer.deep_wfc import (
    DeepWFC,
)
from deepmd.jax.common import (
    to_jax_array,
)
from deepmd.jax.model.hlo import (
    HLO,
)
from deepmd.jax.utils.auto_batch_size import (
    AutoBatchSize,
)

if TYPE_CHECKING:
    import ase.neighborlist


class DeepEval(DeepEvalBackend):
    """NumPy backend implementation of DeepEval.

    Parameters
    ----------
    model_file : str
        The name of the frozen model file.
    output_def : ModelOutputDef
        The output definition of the model.
    *args : list
        Positional arguments.
    auto_batch_size : bool or int or AutoBatchSize, default: True
        If True, automatic batch size will be used. If int, it will be used
        as the initial batch size.
    neighbor_list : ase.neighborlist.NewPrimitiveNeighborList, optional
        The ASE neighbor list class to produce the neighbor list. If None, the
        neighbor list will be built natively in the model.
    **kwargs : dict
        Keyword arguments.
    """

    def __init__(
        self,
        model_file: str,
        output_def: ModelOutputDef,
        *args: Any,
        auto_batch_size: bool | int | AutoBatchSize = True,
        neighbor_list: Optional["ase.neighborlist.NewPrimitiveNeighborList"] = None,
        **kwargs: Any,
    ) -> None:
        self.output_def = output_def
        self.model_path = model_file

        if model_file.endswith(".hlo"):
            model_data = load_dp_model(model_file)
            self.dp = HLO(
                stablehlo=model_data["@variables"]["stablehlo"].tobytes(),
                stablehlo_atomic_virial=model_data["@variables"][
                    "stablehlo_atomic_virial"
                ].tobytes(),
                stablehlo_no_ghost=model_data["@variables"][
                    "stablehlo_no_ghost"
                ].tobytes(),
                stablehlo_atomic_virial_no_ghost=model_data["@variables"][
                    "stablehlo_atomic_virial_no_ghost"
                ].tobytes(),
                stablehlo_hessian_block=(
                    model_data["@variables"]["stablehlo_hessian_block"].tobytes()
                    if "stablehlo_hessian_block" in model_data["@variables"]
                    else None
                ),
                stablehlo_ef=(
                    model_data["@variables"]["stablehlo_ef"].tobytes()
                    if "stablehlo_ef" in model_data["@variables"]
                    else None
                ),
                stablehlo_ef_no_ghost=(
                    model_data["@variables"]["stablehlo_ef_no_ghost"].tobytes()
                    if "stablehlo_ef_no_ghost" in model_data["@variables"]
                    else None
                ),
                model_def_script=json.dumps(model_data["model_def_script"]),
                **model_data["constants"],
            )
        elif model_file.endswith(".savedmodel"):
            from deepmd.jax.jax2tf.tfmodel import (
                TFModelWrapper,
            )

            self.dp = TFModelWrapper(model_file)
        else:
            raise ValueError("Unsupported file extension")
        self.rcut = self.dp.get_rcut()
        self.type_map = self.dp.get_type_map()
        if isinstance(auto_batch_size, bool):
            if auto_batch_size:
                self.auto_batch_size = AutoBatchSize()
            else:
                self.auto_batch_size = None
        elif isinstance(auto_batch_size, int):
            self.auto_batch_size = AutoBatchSize(auto_batch_size)
        elif isinstance(auto_batch_size, AutoBatchSize):
            self.auto_batch_size = auto_batch_size
        else:
            raise TypeError("auto_batch_size should be bool, int, or AutoBatchSize")

    def get_rcut(self) -> float:
        """Get the cutoff radius of this model."""
        return self.rcut

    def get_ntypes(self) -> int:
        """Get the number of atom types of this model."""
        return len(self.type_map)

    def get_type_map(self) -> list[str]:
        """Get the type map (element name of the atom types) of this model."""
        return self.type_map

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this DP."""
        return self.dp.get_dim_fparam()

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this DP."""
        return self.dp.get_dim_aparam()

    @property
    def model_type(self) -> type["DeepEvalWrapper"]:
        """The evaluator of the model type."""
        model_output_type = self.dp.model_output_type()
        if "energy" in model_output_type:
            return DeepPot
        elif "dos" in model_output_type:
            return DeepDOS
        elif "dipole" in model_output_type:
            return DeepDipole
        elif "polar" in model_output_type or "polarizability" in model_output_type:
            return DeepPolar
        elif "wfc" in model_output_type:
            return DeepWFC
        else:
            raise RuntimeError("Unknown model type")

    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.dp.get_sel_type()

    def get_numb_dos(self) -> int:
        """Get the number of DOS."""
        return 0

    def get_has_efield(self) -> bool:
        """Check if the model has efield."""
        return False

    def get_ntypes_spin(self) -> int:
        """Get the number of spin atom types of this model."""
        return 0

    def eval(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        atomic: bool = False,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Evaluate the energy, force and virial by using this DP.

        Parameters
        ----------
        coords
            The coordinates of atoms.
            The array should be of size nframes x natoms x 3
        cells
            The cell of the region.
            If None then non-PBC is assumed, otherwise using PBC.
            The array should be of size nframes x 3 x 3
        atom_types
            The atom types
            The list should contain natoms ints
        atomic
            Calculate the atomic energy and virial
        fparam
            The frame parameter.
            The array can be of size :
            - nframes x dim_fparam.
            - dim_fparam. Then all frames are assumed to be provided with the same fparam.
        aparam
            The atomic parameter
            The array can be of size :
            - nframes x natoms x dim_aparam.
            - natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
            - dim_aparam. Then all frames and atoms are provided with the same aparam.
        **kwargs
            Other parameters

        Returns
        -------
        output_dict : dict
            The output of the evaluation. The keys are the names of the output
            variables, and the values are the corresponding output arrays.
        """
        # convert all of the input to numpy array
        atom_types = np.array(atom_types, dtype=np.int32)
        coords = np.array(coords)
        if cells is not None:
            cells = np.array(cells)
        if fparam is not None:
            fparam = np.array(fparam)
        if aparam is not None:
            aparam = np.array(aparam)
        natoms, numb_test = self._get_natoms_and_nframes(
            coords, atom_types, len(atom_types.shape) > 1
        )
        request_defs = self._get_request_defs(atomic)
        out = self._eval_func(self._eval_model, numb_test, natoms)(
            coords, cells, atom_types, fparam, aparam, request_defs
        )
        return dict(
            zip(
                [x.name for x in request_defs],
                out,
                strict=True,
            )
        )

    def eval_energy_force(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Evaluate energy and force without requesting virial or Hessian."""
        return self._eval_categories(
            coords,
            cells,
            atom_types,
            fparam,
            aparam,
            (
                OutputVariableCategory.REDU,
                OutputVariableCategory.DERV_R,
            ),
        )

    def eval_energy_force_virial(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Evaluate energy, force, and reduced virial."""
        return self._eval_categories(
            coords,
            cells,
            atom_types,
            fparam,
            aparam,
            (
                OutputVariableCategory.REDU,
                OutputVariableCategory.DERV_R,
                OutputVariableCategory.DERV_C_REDU,
            ),
        )

    def eval_energy_force_hessian(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Evaluate energy, force, and Hessian without requesting virial."""
        return self._eval_categories(
            coords,
            cells,
            atom_types,
            fparam,
            aparam,
            (
                OutputVariableCategory.REDU,
                OutputVariableCategory.DERV_R,
                OutputVariableCategory.DERV_R_DERV_R,
            ),
        )

    def _eval_categories(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        categories: tuple[OutputVariableCategory, ...],
    ) -> dict[str, np.ndarray]:
        atom_types = np.array(atom_types, dtype=np.int32)
        coords = np.array(coords)
        if cells is not None:
            cells = np.array(cells)
        if fparam is not None:
            fparam = np.array(fparam)
        if aparam is not None:
            aparam = np.array(aparam)
        natoms, numb_test = self._get_natoms_and_nframes(
            coords, atom_types, len(atom_types.shape) > 1
        )
        request_defs = self._get_request_defs_by_categories(categories)
        out = self._eval_func(self._eval_model, numb_test, natoms)(
            coords, cells, atom_types, fparam, aparam, request_defs
        )
        return dict(
            zip(
                [x.name for x in request_defs],
                out,
                strict=True,
            )
        )

    def _get_request_defs(self, atomic: bool) -> list[OutputVariableDef]:
        """Get the requested output definitions.

        When atomic is True, all output_def are requested.
        When atomic is False, only energy (tensor), force, and virial
        are requested.

        Parameters
        ----------
        atomic : bool
            Whether to request the atomic output.

        Returns
        -------
        list[OutputVariableDef]
            The requested output definitions.
        """
        if atomic:
            return list(self.output_def.var_defs.values())
        else:
            return [
                x
                for x in self.output_def.var_defs.values()
                if x.category
                in (
                    OutputVariableCategory.REDU,
                    OutputVariableCategory.DERV_R,
                    OutputVariableCategory.DERV_C_REDU,
                    OutputVariableCategory.DERV_R_DERV_R,
                )
            ]

    def _get_request_defs_by_categories(
        self, categories: tuple[OutputVariableCategory, ...]
    ) -> list[OutputVariableDef]:
        return [
            x for x in self.output_def.var_defs.values() if x.category in categories
        ]

    def _eval_func(self, inner_func: Callable, numb_test: int, natoms: int) -> Callable:
        """Wrapper method with auto batch size.

        Parameters
        ----------
        inner_func : Callable
            the method to be wrapped
        numb_test : int
            number of tests
        natoms : int
            number of atoms

        Returns
        -------
        Callable
            the wrapper
        """
        if self.auto_batch_size is not None:

            def eval_func(*args: Any, **kwargs: Any) -> Any:
                return self.auto_batch_size.execute_all(
                    inner_func, numb_test, natoms, *args, **kwargs
                )

        else:
            eval_func = inner_func
        return eval_func

    def _get_natoms_and_nframes(
        self,
        coords: np.ndarray,
        atom_types: np.ndarray,
        mixed_type: bool = False,
    ) -> tuple[int, int]:
        if mixed_type:
            natoms = len(atom_types[0])
        else:
            natoms = len(atom_types)
        if natoms == 0:
            assert coords.size == 0
        else:
            coords = np.reshape(np.array(coords), [-1, natoms * 3])
        nframes = coords.shape[0]
        return natoms, nframes

    def _eval_model(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: np.ndarray,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        request_defs: list[OutputVariableDef],
    ) -> tuple[np.ndarray, ...]:
        model = self.dp

        nframes = coords.shape[0]
        if len(atom_types.shape) == 1:
            natoms = len(atom_types)
            atom_types = np.tile(atom_types, nframes).reshape(nframes, -1)
        else:
            natoms = len(atom_types[0])

        coord_input = coords.reshape([-1, natoms, 3])
        type_input = atom_types
        if cells is not None:
            box_input = cells.reshape([-1, 3, 3])
        else:
            box_input = None
        if fparam is not None:
            fparam_input = fparam.reshape(nframes, self.get_dim_fparam())
        elif self.dp.has_default_fparam():
            # JAX (XLA) requires static shapes, so default must be implemented here
            default_fparam = self.dp.get_default_fparam()
            assert default_fparam is not None
            fparam_input = np.tile(
                np.array(default_fparam, dtype=GLOBAL_NP_FLOAT_PRECISION),
                (nframes, 1),
            )
        else:
            fparam_input = None
        if aparam is not None:
            aparam_input = aparam.reshape(nframes, natoms, self.get_dim_aparam())
        else:
            aparam_input = None

        padding_counts = self._get_padding_valid_counts(type_input)
        if padding_counts is not None and len(np.unique(padding_counts)) > 1:
            grouped_results: list[np.ndarray] | None = None
            for count in np.unique(padding_counts):
                frame_idx = np.nonzero(padding_counts == count)[0]
                group_results = self._eval_model(
                    coord_input[frame_idx].reshape(len(frame_idx), -1),
                    box_input[frame_idx] if box_input is not None else None,
                    type_input[frame_idx],
                    fparam_input[frame_idx] if fparam_input is not None else None,
                    aparam_input[frame_idx] if aparam_input is not None else None,
                    request_defs,
                )
                if grouped_results is None:
                    grouped_results = [
                        np.zeros((nframes, *result.shape[1:]), dtype=result.dtype)
                        for result in group_results
                    ]
                for target, result in zip(grouped_results, group_results, strict=True):
                    target[frame_idx] = result
            assert grouped_results is not None
            return tuple(grouped_results)

        eval_natoms = self._get_padding_trim_natoms(type_input)
        if eval_natoms < natoms:
            eval_coord_input = coord_input[:, :eval_natoms, :]
            eval_type_input = type_input[:, :eval_natoms]
            eval_aparam_input = (
                aparam_input[:, :eval_natoms, :] if aparam_input is not None else None
            )
        else:
            eval_coord_input = coord_input
            eval_type_input = type_input
            eval_aparam_input = aparam_input

        do_atomic_virial = any(
            x.category == OutputVariableCategory.DERV_C_REDU for x in request_defs
        )
        needs_virial = any(
            x.category
            in (
                OutputVariableCategory.DERV_C,
                OutputVariableCategory.DERV_C_REDU,
            )
            for x in request_defs
        )
        needs_hessian = any(
            x.category == OutputVariableCategory.DERV_R_DERV_R for x in request_defs
        )
        use_ef_only = (
            isinstance(model, HLO)
            and model.has_ef_only()
            and not needs_virial
            and (not needs_hessian or self._has_hessian_block())
        )
        if use_ef_only:
            batch_output = model.call_ef(
                to_jax_array(eval_coord_input),
                to_jax_array(eval_type_input),
                box=to_jax_array(box_input),
                fparam=to_jax_array(fparam_input),
                aparam=to_jax_array(eval_aparam_input),
            )
        else:
            batch_output = model(
                to_jax_array(eval_coord_input),
                to_jax_array(eval_type_input),
                box=to_jax_array(box_input),
                fparam=to_jax_array(fparam_input),
                aparam=to_jax_array(eval_aparam_input),
                do_atomic_virial=do_atomic_virial,
            )
        if isinstance(batch_output, tuple):
            batch_output = batch_output[0]
        for kk, vv in batch_output.items():
            batch_output[kk] = to_numpy_array(vv)
        if self._has_hessian_block() and needs_hessian:
            batch_output["energy_derv_r_derv_r"] = self._eval_hessian_block_model(
                model,
                eval_coord_input,
                eval_type_input,
                box_input,
                fparam_input,
                eval_aparam_input,
            )
        if eval_natoms < natoms:
            batch_output = self._pad_trimmed_outputs(
                batch_output,
                request_defs,
                nframes,
                eval_natoms,
                natoms,
            )

        results = []
        for odef in request_defs:
            # HLO and TFModelWrapper return raw internal keys (not translated),
            # so no key mapping is needed here.
            dp_name = odef.name
            if dp_name in batch_output:
                shape = self._get_output_shape(odef, nframes, natoms)
                if batch_output[dp_name] is not None:
                    out = batch_output[dp_name].reshape(shape)
                else:
                    out = np.full(shape, np.nan, dtype=GLOBAL_NP_FLOAT_PRECISION)
                results.append(out)
            else:
                shape = self._get_output_shape(odef, nframes, natoms)
                results.append(
                    np.full(np.abs(shape), np.nan, dtype=GLOBAL_NP_FLOAT_PRECISION)
                )  # this is kinda hacky
        return tuple(results)

    def _get_padding_trim_natoms(self, type_input: np.ndarray) -> int:
        """Trim suffix padding atoms marked by negative per-frame atom types."""
        natoms = type_input.shape[1]
        valid_counts = self._get_padding_valid_counts(type_input)
        if valid_counts is None:
            return natoms
        trim_natoms = int(np.max(valid_counts))
        if not np.all(valid_counts == trim_natoms):
            return natoms
        if trim_natoms >= natoms:
            return natoms
        return trim_natoms

    def _get_padding_valid_counts(self, type_input: np.ndarray) -> np.ndarray | None:
        valid = type_input >= 0
        if np.all(valid):
            return None
        valid_counts = np.sum(valid, axis=1)
        if np.any(valid_counts == 0):
            raise ValueError("JAX HLO inference does not support all-padding frames.")
        for row, count in zip(valid, valid_counts, strict=True):
            if not np.all(row[:count]) or np.any(row[count:]):
                raise ValueError(
                    "JAX HLO inference only supports suffix padding atoms marked by "
                    "negative atom types."
                )
        return valid_counts

    def _pad_trimmed_outputs(
        self,
        batch_output: dict[str, np.ndarray],
        request_defs: list[OutputVariableDef],
        nframes: int,
        trim_natoms: int,
        natoms: int,
    ) -> dict[str, np.ndarray]:
        padded_output: dict[str, np.ndarray] = {}
        trim_dim = 3 * trim_natoms
        full_dim = 3 * natoms
        output_defs = {odef.name: odef for odef in request_defs}
        for kk, vv in batch_output.items():
            arr = np.asarray(vv)
            odef = output_defs.get(kk)
            category = None if odef is None else odef.category
            if category == OutputVariableCategory.DERV_R_DERV_R:
                padded = np.zeros(
                    (*arr.shape[:-2], full_dim, full_dim),
                    dtype=arr.dtype,
                )
                padded[..., :trim_dim, :trim_dim] = arr
                padded_output[kk] = padded
            elif category in (
                OutputVariableCategory.DERV_R,
                OutputVariableCategory.DERV_C,
            ):
                assert odef is not None
                component_dim = 3 if category == OutputVariableCategory.DERV_R else 9
                prefix_shape = tuple(odef.shape[:-1])
                if arr.shape[-1] != component_dim:
                    raise ValueError(
                        f"Unexpected shape for {kk}: {arr.shape}; expected suffix "
                        f"component dimension {component_dim}."
                    )
                interior_shape = arr.shape[1:-1]
                if interior_shape == (*prefix_shape, trim_natoms):
                    normalized = arr
                elif interior_shape == (trim_natoms, *prefix_shape):
                    normalized = np.moveaxis(arr, 1, -2)
                elif interior_shape == (trim_natoms,) and np.prod(
                    prefix_shape, dtype=int
                ) == 1:
                    normalized = arr.reshape(
                        (arr.shape[0], *prefix_shape, trim_natoms, component_dim)
                    )
                else:
                    raise ValueError(
                        f"Unexpected shape for {kk}: {arr.shape}; expected atom axis "
                        f"with {trim_natoms} atoms and prefix {prefix_shape}."
                    )
                padded = np.zeros(
                    (normalized.shape[0], *prefix_shape, natoms, component_dim),
                    dtype=normalized.dtype,
                )
                padded[..., :trim_natoms, :] = normalized
                padded_output[kk] = padded
            elif category == OutputVariableCategory.OUT:
                padded = np.zeros(
                    (arr.shape[0], natoms, *arr.shape[2:]),
                    dtype=arr.dtype,
                )
                padded[:, :trim_natoms, ...] = arr
                padded_output[kk] = padded
            else:
                padded_output[kk] = arr
        return padded_output

    def _get_output_shape(
        self, odef: OutputVariableDef, nframes: int, natoms: int
    ) -> list[int]:
        if odef.category == OutputVariableCategory.DERV_C_REDU:
            # virial
            return [nframes, *odef.shape[:-1], 9]
        elif odef.category == OutputVariableCategory.REDU:
            # energy
            return [nframes, *odef.shape, 1]
        elif odef.category == OutputVariableCategory.DERV_C:
            # atom_virial
            return [nframes, *odef.shape[:-1], natoms, 9]
        elif odef.category == OutputVariableCategory.DERV_R:
            # force
            return [nframes, *odef.shape[:-1], natoms, 3]
        elif odef.category == OutputVariableCategory.OUT:
            # atom_energy, atom_tensor
            return [nframes, natoms, *odef.shape, 1]
        elif odef.category == OutputVariableCategory.DERV_R_DERV_R:
            # hessian
            return [nframes, 3 * natoms, 3 * natoms]
        else:
            raise RuntimeError("unknown category")

    def get_model_def_script(self) -> dict:
        """Get model definition script."""
        return json.loads(self.dp.get_model_def_script())

    def get_has_hessian(self) -> bool:
        model_def_script = self.get_model_def_script()
        return model_def_script.get("hessian_mode", False) or self._has_hessian_block()

    def _has_hessian_block(self) -> bool:
        return (
            isinstance(self.dp, HLO)
            and hasattr(self.dp, "get_hessian_chunk_size")
            and self.dp.get_hessian_chunk_size() > 0
        )

    def _eval_hessian_block_model(
        self,
        model: HLO,
        coord_input: np.ndarray,
        type_input: np.ndarray,
        box_input: np.ndarray | None,
        fparam_input: np.ndarray | None,
        aparam_input: np.ndarray | None,
    ) -> np.ndarray:
        nframes, natoms = coord_input.shape[:2]
        dim = natoms * 3
        chunk_size = model.get_hessian_chunk_size()
        hessian = np.zeros(
            (nframes, dim, dim),
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        )
        for start in range(0, dim, chunk_size):
            block_index = start // chunk_size
            block = model.call_hessian_block(
                to_jax_array(coord_input),
                to_jax_array(type_input),
                box=to_jax_array(box_input),
                fparam=to_jax_array(fparam_input),
                aparam=to_jax_array(aparam_input),
                block_index=block_index,
            )["energy_derv_r_derv_r_block"]
            block_np = to_numpy_array(block)
            row_count = min(chunk_size, dim - start)
            hessian[:, start : start + row_count, :] = block_np[:, :row_count, :]
        return hessian

    def get_model(self) -> Any:
        """Get the JAX model as BaseModel.

        Returns
        -------
        BaseModel
            The JAX model as BaseModel instance.
        """
        return self.dp

    def has_default_fparam(self) -> bool:
        """Check if the model has default frame parameters."""
        return self.dp.has_default_fparam()
