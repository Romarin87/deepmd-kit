# SPDX-License-Identifier: LGPL-3.0-or-later
"""ASE calculator interface optimized for JAX HLO models."""

from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Optional,
    Union,
)

import numpy as np
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)

from deepmd.infer import (
    DeepPot,
)
from deepmd.jax.infer.deep_eval import (
    DeepEval as JaxDeepEval,
)

if TYPE_CHECKING:
    from ase import (
        Atoms,
    )

__all__ = ["DP"]


class DP(Calculator):
    """ASE calculator that uses the JAX HLO inference backend directly.

    Unlike the generic :mod:`deepmd.calculator` interface, this calculator
    requests only the properties needed by ASE/MAPLE. Energy and force calls do
    not request virial or Hessian; stress and Hessian are opt-in.
    """

    name = "DP-JAX"
    implemented_properties: ClassVar[list[str]] = [
        "energy",
        "free_energy",
        "forces",
        "virial",
        "stress",
        "hessian",
    ]

    def __init__(
        self,
        model: Union[str, "Path"],
        label: str = "DP-JAX",
        type_dict: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        model_path = Path(model).resolve()
        if model_path.suffix != ".hlo":
            raise ValueError("deepmd.jax.calculator.DP only supports JAX .hlo models.")
        Calculator.__init__(self, label=label, **kwargs)
        self.dp = DeepPot(str(model_path))
        if not isinstance(self.dp.deep_eval, JaxDeepEval):
            raise ValueError("deepmd.jax.calculator.DP requires the JAX backend.")
        self._backend = self.dp.deep_eval
        if type_dict:
            self.type_dict = type_dict
        else:
            self.type_dict = dict(
                zip(self.dp.get_type_map(), range(self.dp.get_ntypes()), strict=True)
            )

    def calculate(
        self,
        atoms: Optional["Atoms"] = None,
        properties: list[str] = ["energy", "forces"],
        system_changes: list[str] = all_changes,
    ) -> None:
        Calculator.calculate(self, atoms, properties, system_changes)
        assert self.atoms is not None

        needs_virial = "virial" in properties or "stress" in properties
        needs_hessian = "hessian" in properties
        if "stress" in properties and not np.any(self.atoms.get_pbc()):
            raise PropertyNotImplementedError(
                "Stress is not available for non-periodic systems."
            )

        output = self._eval_outputs(
            self.atoms,
            virial=needs_virial,
            hessian=needs_hessian and not needs_virial,
        )
        self._set_energy_force_results(output)
        if needs_virial:
            self._set_virial_stress_results(output)
        if needs_hessian:
            if "energy_derv_r_derv_r" in output:
                natoms = len(self.atoms)
                self.results["hessian"] = output["energy_derv_r_derv_r"].reshape(
                    1, 3 * natoms, 3 * natoms
                )[0]
            else:
                self.results["hessian"] = self.get_hessian(self.atoms)

    def get_hessian(self, atoms: Optional["Atoms"] = None) -> np.ndarray:
        """Return the analytical Hessian for one ASE ``Atoms`` object."""
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("atoms must be provided before evaluating Hessian.")
        if not self.dp.has_hessian:
            raise RuntimeError("This JAX HLO model does not provide Hessian output.")
        output = self._eval_outputs(atoms, hessian=True)
        natoms = len(atoms)
        hessian = output["energy_derv_r_derv_r"].reshape(1, 3 * natoms, 3 * natoms)
        return hessian[0]

    def _eval_outputs(
        self,
        atoms: "Atoms",
        *,
        virial: bool = False,
        hessian: bool = False,
    ) -> dict[str, np.ndarray]:
        coord, cell, atype, fparam, aparam = self._atoms_to_inputs(atoms)
        if hessian:
            method = getattr(self._backend, "eval_energy_force_hessian", None)
            if method is not None:
                return method(
                    coord,
                    cell,
                    atype,
                    fparam=fparam,
                    aparam=aparam,
                )
            output = self._backend.eval(
                coord,
                cell,
                atype,
                False,
                fparam=fparam,
                aparam=aparam,
            )
            if "energy_derv_r_derv_r" not in output:
                raise RuntimeError("This JAX HLO model did not return Hessian output.")
            return output
        if virial:
            method = getattr(self._backend, "eval_energy_force_virial", None)
            if method is not None:
                return method(
                    coord,
                    cell,
                    atype,
                    fparam=fparam,
                    aparam=aparam,
                )
            return self._backend.eval(
                coord,
                cell,
                atype,
                False,
                fparam=fparam,
                aparam=aparam,
            )
        method = getattr(self._backend, "eval_energy_force", None)
        if method is not None:
            return method(
                coord,
                cell,
                atype,
                fparam=fparam,
                aparam=aparam,
            )
        return self._backend.eval(
            coord,
            cell,
            atype,
            False,
            fparam=fparam,
            aparam=aparam,
        )

    def _atoms_to_inputs(
        self, atoms: "Atoms"
    ) -> tuple[np.ndarray, np.ndarray | None, list[int], Any, Any]:
        coord = atoms.get_positions().reshape(1, -1)
        cell = atoms.get_cell().reshape(1, -1) if np.any(atoms.get_pbc()) else None
        symbols = atoms.get_chemical_symbols()
        try:
            atype = [self.type_dict[symbol] for symbol in symbols]
        except KeyError as exc:
            raise KeyError(
                f"Element {exc.args[0]!r} is not present in the model type map."
            ) from exc
        fparam = atoms.info.get("fparam", None)
        aparam = atoms.info.get("aparam", None)
        return coord, cell, atype, fparam, aparam

    def _set_energy_force_results(self, output: dict[str, np.ndarray]) -> None:
        energy = output["energy_redu"].reshape(1, -1)[0, 0]
        forces = output["energy_derv_r"].reshape(1, -1, 3)[0]
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces

    def _set_virial_stress_results(self, output: dict[str, np.ndarray]) -> None:
        assert self.atoms is not None
        virial = output["energy_derv_c_redu"].reshape(1, 9)[0].reshape(3, 3)
        self.results["virial"] = virial
        if np.any(self.atoms.get_pbc()):
            stress = -0.5 * (virial.copy() + virial.copy().T) / self.atoms.get_volume()
            self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]
