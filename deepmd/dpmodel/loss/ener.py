# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.loss.loss import (
    Loss,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.loss import (
    resolve_huber_deltas,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


def _masked_mean(xp: Any, values: Array, mask: Array | None = None) -> Array:
    if mask is None:
        return xp.mean(values)
    mask_float = xp.astype(mask, values.dtype)
    while mask_float.ndim < values.ndim:
        mask_float = xp.expand_dims(mask_float, axis=-1)
    weight = xp.ones_like(values) * mask_float
    return xp.sum(values * mask_float) / xp.sum(weight)


def _atom_mask(
    model_dict: dict[str, Array],
    label_dict: dict[str, Array],
) -> Array | None:
    mask = model_dict.get("mask")
    if mask is None and "type" in label_dict:
        mask = label_dict["type"] >= 0
    return mask


def _atom_normalizer(
    xp: Any,
    mask: Array | None,
    natoms: int,
    dtype: Any,
) -> Array | float:
    if mask is None:
        return 1.0 / natoms
    mask_float = xp.astype(mask, dtype)
    real_natoms = xp.sum(mask_float, axis=-1, keepdims=True)
    return 1.0 / real_natoms


def _coord_mask_from_atom_mask(xp: Any, mask: Array | None) -> Array | None:
    if mask is None:
        return None
    mask3 = xp.broadcast_to(mask[..., :, None], (*mask.shape, 3))
    return xp.reshape(mask3, (*mask.shape[:-1], mask.shape[-1] * 3))


def _hessian_mask_from_atom_mask(xp: Any, mask: Array | None) -> Array | None:
    coord_mask = _coord_mask_from_atom_mask(xp, mask)
    if coord_mask is None:
        return None
    return xp.logical_and(coord_mask[..., :, None], coord_mask[..., None, :])


def _reshape_force_like_atoms(xp: Any, value: Array, mask: Array | None) -> Array:
    if value.ndim >= 3 and value.shape[-1] == 3:
        return value
    if mask is not None:
        return xp.reshape(value, (*mask.shape, 3))
    return xp.reshape(value, (*value.shape[:-1], value.shape[-1] // 3, 3))


def _reshape_hessian_like_atoms(xp: Any, value: Array, mask: Array | None) -> Array:
    if value.ndim >= 3:
        return value
    if mask is None:
        return value
    ncoord = mask.shape[-1] * 3
    return xp.reshape(value, (*mask.shape[:-1], ncoord, ncoord))


def custom_huber_loss(
    predictions: Array,
    targets: Array,
    delta: float = 1.0,
    mask: Array | None = None,
) -> Array:
    xp = array_api_compat.array_namespace(predictions, targets)
    error = targets - predictions
    abs_error = xp.abs(error)
    quadratic_loss = 0.5 * error**2
    linear_loss = delta * (abs_error - 0.5 * delta)
    loss = xp.where(abs_error <= delta, quadratic_loss, linear_loss)
    return _masked_mean(xp, loss, mask)


class EnergyLoss(Loss):
    r"""Construct a layer to compute loss on energy, force and virial.

    Parameters
    ----------
    starter_learning_rate : float
        The learning rate at the start of the training.
    start_pref_e : float
        The prefactor of energy loss at the start of the training.
    limit_pref_e : float
        The prefactor of energy loss at the end of the training.
    start_pref_f : float
        The prefactor of force loss at the start of the training.
    limit_pref_f : float
        The prefactor of force loss at the end of the training.
    start_pref_v : float
        The prefactor of virial loss at the start of the training.
    limit_pref_v : float
        The prefactor of virial loss at the end of the training.
    start_pref_ae : float
        The prefactor of atomic energy loss at the start of the training.
    limit_pref_ae : float
        The prefactor of atomic energy loss at the end of the training.
    start_pref_pf : float
        The prefactor of atomic prefactor force loss at the start of the training.
    limit_pref_pf : float
        The prefactor of atomic prefactor force loss at the end of the training.
    relative_f : float
        If provided, relative force error will be used in the loss. The difference
        of force will be normalized by the magnitude of the force in the label with
        a shift given by relative_f
    enable_atom_ener_coeff : bool
        if true, the energy will be computed as \sum_i c_i E_i
    start_pref_gf : float
        The prefactor of generalized force loss at the start of the training.
    limit_pref_gf : float
        The prefactor of generalized force loss at the end of the training.
    numb_generalized_coord : int
        The dimension of generalized coordinates.
    use_huber : bool
        Enables Huber loss calculation for energy/force/virial terms with user-defined threshold delta (D).
        The loss function smoothly transitions between L2 and L1 loss:
        - For absolute prediction errors within D: quadratic loss (0.5 * (error**2))
        - For absolute errors exceeding D: linear loss (D * |error| - 0.5 * D)
        Formula: loss = 0.5 * (error**2) if |error| <= D else D * (|error| - 0.5 * D).
    huber_delta : float | list[float]
        The threshold delta (D) used for Huber loss, controlling transition between
        L2 and L1 loss. It can be either one float shared by all terms or a list of
        three values ordered as [energy, force, virial].
    loss_func : str
        Loss function type for energy, force, and virial terms.
        Options: 'mse' (Mean Squared Error, L2 loss, default) or 'mae' (Mean Absolute Error, L1 loss).
        MAE loss is less sensitive to outliers compared to MSE loss.
        Future extensions may support additional loss types.
    f_use_norm : bool
        If true, use L2 norm of force vectors for loss calculation when loss_func='mae' or use_huber is True.
        Instead of computing loss on force components, computes loss on ||F_pred - F_label||_2.
        This treats the force vector as a whole rather than three independent components.
    intensive_ener_virial : bool
        If true, the non-Huber MSE energy and virial losses use intensive normalization,
        i.e. a 1/N^2 factor instead of the legacy 1/N scaling. This matches per-atom
        RMSE-style normalization for those terms. MAE and Huber modes use different
        scaling and are not affected in the same way by this flag.
        If false (default), the legacy normalization is used for the affected terms.
        The default is false for backward compatibility with models trained using
        deepmd-kit <= 3.1.3.
    **kwargs
        Other keyword arguments.
    """

    def __init__(
        self,
        starter_learning_rate: float,
        start_pref_e: float = 0.02,
        limit_pref_e: float = 1.00,
        start_pref_f: float = 1000,
        limit_pref_f: float = 1.00,
        start_pref_v: float = 0.0,
        limit_pref_v: float = 0.0,
        start_pref_ae: float = 0.0,
        limit_pref_ae: float = 0.0,
        start_pref_pf: float = 0.0,
        limit_pref_pf: float = 0.0,
        relative_f: float | None = None,
        enable_atom_ener_coeff: bool = False,
        start_pref_gf: float = 0.0,
        limit_pref_gf: float = 0.0,
        numb_generalized_coord: int = 0,
        use_huber: bool = False,
        huber_delta: float | list[float] = 0.01,
        loss_func: str = "mse",
        f_use_norm: bool = False,
        intensive_ener_virial: bool = False,
        **kwargs: Any,
    ) -> None:
        # Validate loss_func
        valid_loss_funcs = ["mse", "mae"]
        if loss_func not in valid_loss_funcs:
            raise ValueError(
                f"Invalid loss_func '{loss_func}'. Must be one of {valid_loss_funcs}."
            )

        self.loss_func = loss_func
        self.starter_learning_rate = starter_learning_rate
        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_f = start_pref_f
        self.limit_pref_f = limit_pref_f
        self.start_pref_v = start_pref_v
        self.limit_pref_v = limit_pref_v
        self.start_pref_ae = start_pref_ae
        self.limit_pref_ae = limit_pref_ae
        self.start_pref_pf = start_pref_pf
        self.limit_pref_pf = limit_pref_pf
        self.relative_f = relative_f
        self.enable_atom_ener_coeff = enable_atom_ener_coeff
        self.start_pref_gf = start_pref_gf
        self.limit_pref_gf = limit_pref_gf
        self.numb_generalized_coord = numb_generalized_coord
        self.has_e = self.start_pref_e != 0.0 or self.limit_pref_e != 0.0
        self.has_f = self.start_pref_f != 0.0 or self.limit_pref_f != 0.0
        self.has_v = self.start_pref_v != 0.0 or self.limit_pref_v != 0.0
        self.has_ae = self.start_pref_ae != 0.0 or self.limit_pref_ae != 0.0
        self.has_pf = self.start_pref_pf != 0.0 or self.limit_pref_pf != 0.0
        self.has_gf = self.start_pref_gf != 0.0 or self.limit_pref_gf != 0.0
        if self.has_gf and self.numb_generalized_coord < 1:
            raise RuntimeError(
                "When generalized force loss is used, the dimension of generalized coordinates should be larger than 0"
            )
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.f_use_norm = f_use_norm
        self.intensive_ener_virial = intensive_ener_virial
        if self.f_use_norm and not (self.use_huber or self.loss_func == "mae"):
            raise RuntimeError(
                "f_use_norm can only be True when use_huber or loss_func='mae'."
            )
        (
            self._huber_delta_energy,
            self._huber_delta_force,
            self._huber_delta_virial,
        ) = resolve_huber_deltas(huber_delta)
        if self.use_huber and (
            self.has_pf or self.has_gf or self.relative_f is not None
        ):
            raise RuntimeError(
                "Huber loss is not implemented for force with atom_pref, generalized force and relative force. "
            )

    def call(
        self,
        learning_rate: float,
        natoms: int,
        model_dict: dict[str, Array],
        label_dict: dict[str, Array],
        mae: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        """Calculate loss from model results and labeled results."""
        energy = model_dict["energy"]
        force = model_dict["force"]
        virial = model_dict["virial"]
        atom_ener = model_dict["atom_energy"]
        energy_hat = label_dict["energy"]
        force_hat = label_dict["force"]
        virial_hat = label_dict["virial"]
        atom_ener_hat = label_dict["atom_ener"]
        atom_pref = label_dict["atom_pref"]
        find_energy = label_dict["find_energy"]
        find_force = label_dict["find_force"]
        find_virial = label_dict["find_virial"]
        find_atom_ener = label_dict["find_atom_ener"]
        find_atom_pref = label_dict["find_atom_pref"]
        xp = array_api_compat.array_namespace(
            energy,
            force,
            virial,
            atom_ener,
            energy_hat,
            force_hat,
            virial_hat,
            atom_ener_hat,
            atom_pref,
        )

        if self.enable_atom_ener_coeff:
            # when ener_coeff (\nu) is defined, the energy is defined as
            # E = \sum_i \nu_i E_i
            # instead of the sum of atomic energies.
            #
            # A case is that we want to train reaction energy
            # A + B -> C + D
            # E = - E(A) - E(B) + E(C) + E(D)
            # A, B, C, D could be put far away from each other
            atom_ener_coeff = label_dict["atom_ener_coeff"]
            atom_ener_coeff = xp.reshape(atom_ener_coeff, atom_ener.shape)
            energy = xp.sum(atom_ener_coeff * atom_ener, axis=1)
        atom_mask = _atom_mask(model_dict, label_dict)
        if self.has_f or self.has_pf or self.relative_f or self.has_gf:
            force = _reshape_force_like_atoms(xp, force, atom_mask)
            force_hat = _reshape_force_like_atoms(xp, force_hat, atom_mask)
            diff_f_full = force_hat - force
        else:
            diff_f_full = None

        if self.relative_f is not None:
            norm_f = xp.linalg.vector_norm(force_hat, axis=-1, keepdims=True)
            diff_f_full = diff_f_full / (norm_f + self.relative_f)

        atom_norm = _atom_normalizer(xp, atom_mask, natoms, energy.dtype)
        atom_norm_ener = atom_norm
        lr_ratio = learning_rate / self.starter_learning_rate
        pref_e = find_energy * (
            self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * lr_ratio
        )
        pref_f = find_force * (
            self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * lr_ratio
        )
        pref_v = find_virial * (
            self.limit_pref_v + (self.start_pref_v - self.limit_pref_v) * lr_ratio
        )
        pref_ae = find_atom_ener * (
            self.limit_pref_ae + (self.start_pref_ae - self.limit_pref_ae) * lr_ratio
        )
        pref_pf = find_atom_pref * (
            self.limit_pref_pf + (self.start_pref_pf - self.limit_pref_pf) * lr_ratio
        )

        loss = 0
        more_loss = {}
        # Normalization exponent controls loss scaling with system size:
        # - norm_exp=2 (intensive_ener_virial=True): loss uses 1/N² scaling, making it independent of system size
        # - norm_exp=1 (intensive_ener_virial=False, legacy): loss uses 1/N scaling, which varies with system size
        norm_exp = 2 if self.intensive_ener_virial else 1
        if self.has_e:
            diff_e = energy - energy_hat
            if self.loss_func == "mse":
                l2_ener_loss = xp.mean(xp.square(diff_e))
                if not self.use_huber:
                    if norm_exp == 2:
                        l2_ener_loss_scaled = xp.mean(
                            xp.square(diff_e * atom_norm_ener)
                        )
                    else:
                        l2_ener_loss_scaled = xp.mean(
                            xp.square(diff_e) * atom_norm_ener
                        )
                    loss += pref_e * l2_ener_loss_scaled
                else:
                    l_huber_loss = custom_huber_loss(
                        atom_norm_ener * energy,
                        atom_norm_ener * energy_hat,
                        delta=self._huber_delta_energy,
                    )
                    loss += pref_e * l_huber_loss
                more_loss["rmse_e"] = self.display_if_exist(
                    xp.sqrt(xp.mean(xp.square(diff_e * atom_norm_ener))), find_energy
                )
            elif self.loss_func == "mae":
                l1_ener_loss_scaled = xp.mean(xp.abs(diff_e) * atom_norm_ener)
                loss += pref_e * l1_ener_loss_scaled
                more_loss["mae_e"] = self.display_if_exist(
                    l1_ener_loss_scaled, find_energy
                )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for energy loss."
                )
            if mae:
                mae_e = xp.mean(xp.abs(diff_e) * atom_norm_ener)
                more_loss["mae_e"] = self.display_if_exist(mae_e, find_energy)
                mae_e_all = xp.mean(xp.abs(diff_e))
                more_loss["mae_e_all"] = self.display_if_exist(mae_e_all, find_energy)
        if self.has_f:
            if self.loss_func == "mse":
                l2_force_loss = _masked_mean(xp, xp.square(diff_f_full), atom_mask)
                if not self.use_huber:
                    loss += pref_f * l2_force_loss
                else:
                    if not self.f_use_norm:
                        l_huber_loss = custom_huber_loss(
                            force,
                            force_hat,
                            delta=self._huber_delta_force,
                            mask=atom_mask,
                        )
                    else:
                        force_diff_norm = xp.expand_dims(
                            xp.linalg.vector_norm(force_hat - force, axis=-1),
                            axis=-1,
                        )
                        l_huber_loss = custom_huber_loss(
                            force_diff_norm,
                            xp.zeros_like(force_diff_norm),
                            delta=self._huber_delta_force,
                            mask=atom_mask,
                        )
                    loss += pref_f * l_huber_loss
                more_loss["rmse_f"] = self.display_if_exist(
                    xp.sqrt(l2_force_loss), find_force
                )
            elif self.loss_func == "mae":
                if not self.f_use_norm:
                    l1_force_loss = _masked_mean(xp, xp.abs(diff_f_full), atom_mask)
                else:
                    force_diff_norm = xp.linalg.vector_norm(diff_f_full, axis=-1)
                    l1_force_loss = _masked_mean(xp, force_diff_norm, atom_mask)
                loss += pref_f * l1_force_loss
                more_loss["mae_f"] = self.display_if_exist(l1_force_loss, find_force)
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for force loss."
                )
            if mae:
                mae_f = _masked_mean(xp, xp.abs(diff_f_full), atom_mask)
                more_loss["mae_f"] = self.display_if_exist(mae_f, find_force)
        if self.has_v:
            diff_v = virial_hat - virial
            if self.loss_func == "mse":
                l2_virial_loss = xp.mean(xp.square(diff_v))
                if not self.use_huber:
                    if norm_exp == 2:
                        l2_virial_loss_scaled = xp.mean(xp.square(diff_v * atom_norm))
                    else:
                        l2_virial_loss_scaled = xp.mean(
                            xp.square(diff_v) * atom_norm
                        )
                    loss += pref_v * l2_virial_loss_scaled
                else:
                    l_huber_loss = custom_huber_loss(
                        atom_norm * virial,
                        atom_norm * virial_hat,
                        delta=self._huber_delta_virial,
                    )
                    loss += pref_v * l_huber_loss
                more_loss["rmse_v"] = self.display_if_exist(
                    xp.sqrt(xp.mean(xp.square(diff_v * atom_norm))), find_virial
                )
            elif self.loss_func == "mae":
                l1_virial_loss = xp.mean(xp.abs(diff_v) * atom_norm)
                loss += pref_v * l1_virial_loss
                more_loss["mae_v"] = self.display_if_exist(
                    l1_virial_loss, find_virial
                )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for virial loss."
                )
            if mae:
                mae_v = xp.mean(xp.abs(diff_v) * atom_norm)
                more_loss["mae_v"] = self.display_if_exist(mae_v, find_virial)
        if self.has_ae:
            if self.loss_func == "mse":
                l2_atom_ener_loss = _masked_mean(
                    xp, xp.square(atom_ener_hat - atom_ener), atom_mask
                )
                if not self.use_huber:
                    loss += pref_ae * l2_atom_ener_loss
                else:
                    l_huber_loss = custom_huber_loss(
                        atom_ener,
                        atom_ener_hat,
                        delta=self._huber_delta_energy,
                        mask=atom_mask,
                    )
                    loss += pref_ae * l_huber_loss
                more_loss["rmse_ae"] = self.display_if_exist(
                    xp.sqrt(l2_atom_ener_loss), find_atom_ener
                )
            elif self.loss_func == "mae":
                l1_atom_ener_loss = _masked_mean(
                    xp, xp.abs(atom_ener_hat - atom_ener), atom_mask
                )
                loss += pref_ae * l1_atom_ener_loss
                more_loss["mae_ae"] = self.display_if_exist(
                    l1_atom_ener_loss, find_atom_ener
                )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for atomic energy loss."
                )
        if self.has_pf:
            atom_pref_reshape = xp.reshape(atom_pref, diff_f_full.shape)

            if self.loss_func == "mse":
                l2_pref_force_loss = _masked_mean(
                    xp,
                    xp.multiply(xp.square(diff_f_full), atom_pref_reshape),
                    atom_mask,
                )
                loss += pref_pf * l2_pref_force_loss
                more_loss["rmse_pf"] = self.display_if_exist(
                    xp.sqrt(l2_pref_force_loss), find_atom_pref
                )
            elif self.loss_func == "mae":
                l1_pref_force_loss = _masked_mean(
                    xp,
                    xp.multiply(xp.abs(diff_f_full), atom_pref_reshape),
                    atom_mask,
                )
                loss += pref_pf * l1_pref_force_loss
                more_loss["mae_pf"] = self.display_if_exist(
                    l1_pref_force_loss, find_atom_pref
                )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for atom prefactor force loss."
                )
        if self.has_gf:
            find_drdq = label_dict["find_drdq"]
            drdq = label_dict["drdq"]
            force_reshape_nframes = xp.reshape(force, (-1, natoms * 3))
            force_hat_reshape_nframes = xp.reshape(force_hat, (-1, natoms * 3))
            drdq_reshape = xp.reshape(
                drdq, (-1, natoms * 3, self.numb_generalized_coord)
            )
            # "bij,bi->bj" einsum replaced with array-API-compatible ops
            gen_force_hat = xp.sum(
                drdq_reshape * force_hat_reshape_nframes[:, :, None], axis=1
            )
            gen_force = xp.sum(drdq_reshape * force_reshape_nframes[:, :, None], axis=1)
            diff_gen_force = gen_force_hat - gen_force
            l2_gen_force_loss = xp.mean(xp.square(diff_gen_force))
            pref_gf = find_drdq * (
                self.limit_pref_gf
                + (self.start_pref_gf - self.limit_pref_gf) * lr_ratio
            )
            loss += pref_gf * l2_gen_force_loss
            more_loss["rmse_gf"] = self.display_if_exist(
                xp.sqrt(l2_gen_force_loss), find_drdq
            )

        self.l2_l = loss
        more_loss["rmse"] = xp.sqrt(loss)
        self.l2_more = more_loss
        return loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        label_requirement.append(
            DataRequirementItem(
                "energy",
                ndof=1,
                atomic=False,
                must=False,
                high_prec=True,
            )
        )
        label_requirement.append(
            DataRequirementItem(
                "force",
                ndof=3,
                atomic=True,
                must=False,
                high_prec=False,
            )
        )
        label_requirement.append(
            DataRequirementItem(
                "virial",
                ndof=9,
                atomic=False,
                must=False,
                high_prec=False,
            )
        )
        label_requirement.append(
            DataRequirementItem(
                "atom_ener",
                ndof=1,
                atomic=True,
                must=False,
                high_prec=False,
            )
        )
        label_requirement.append(
            DataRequirementItem(
                "atom_pref",
                ndof=1,
                atomic=True,
                must=False,
                high_prec=False,
                repeat=3,
            )
        )
        if self.has_gf > 0:
            label_requirement.append(
                DataRequirementItem(
                    "drdq",
                    ndof=self.numb_generalized_coord * 3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.enable_atom_ener_coeff:
            label_requirement.append(
                DataRequirementItem(
                    "atom_ener_coeff",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                    default=1.0,
                )
            )
        return label_requirement

    def serialize(self) -> dict:
        """Serialize the loss module.

        Returns
        -------
        dict
            The serialized loss module
        """
        return {
            "@class": "EnergyLoss",
            "@version": 3,
            "starter_learning_rate": self.starter_learning_rate,
            "start_pref_e": self.start_pref_e,
            "limit_pref_e": self.limit_pref_e,
            "start_pref_f": self.start_pref_f,
            "limit_pref_f": self.limit_pref_f,
            "start_pref_v": self.start_pref_v,
            "limit_pref_v": self.limit_pref_v,
            "start_pref_ae": self.start_pref_ae,
            "limit_pref_ae": self.limit_pref_ae,
            "start_pref_pf": self.start_pref_pf,
            "limit_pref_pf": self.limit_pref_pf,
            "relative_f": self.relative_f,
            "enable_atom_ener_coeff": self.enable_atom_ener_coeff,
            "start_pref_gf": self.start_pref_gf,
            "limit_pref_gf": self.limit_pref_gf,
            "numb_generalized_coord": self.numb_generalized_coord,
            "use_huber": self.use_huber,
            "huber_delta": self.huber_delta,
            "loss_func": self.loss_func,
            "f_use_norm": self.f_use_norm,
            "intensive_ener_virial": self.intensive_ener_virial,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Loss":
        """Deserialize the loss module.

        Parameters
        ----------
        data : dict
            The serialized loss module

        Returns
        -------
        Loss
            The deserialized loss module
        """
        data = data.copy()
        version = data.pop("@version")
        check_version_compatibility(version, 3, 1)
        data.pop("@class")
        # Backward compatibility: version 1-2 used legacy normalization
        if version < 3:
            data.setdefault("intensive_ener_virial", False)
        return cls(**data)


class EnergyHessianLoss(EnergyLoss):
    def __init__(
        self,
        start_pref_h: float = 0.0,
        limit_pref_h: float = 0.0,
        **kwargs: Any,
    ) -> None:
        r"""Enable the layer to compute loss on hessian.

        Parameters
        ----------
        start_pref_h : float
            The prefactor of hessian loss at the start of the training.
        limit_pref_h : float
            The prefactor of hessian loss at the end of the training.
        **kwargs
            Other keyword arguments.
        """
        EnergyLoss.__init__(self, **kwargs)
        self.has_h = start_pref_h != 0.0 and limit_pref_h != 0.0

        self.start_pref_h = start_pref_h
        self.limit_pref_h = limit_pref_h

    def call(
        self,
        learning_rate: float,
        natoms: int,
        model_dict: dict[str, Array],
        label_dict: dict[str, Array],
        mae: bool = False,
    ) -> dict[str, Array]:
        """Calculate loss from model results and labeled results."""
        loss, more_loss = EnergyLoss.call(
            self, learning_rate, natoms, model_dict, label_dict, mae=mae
        )
        xp = array_api_compat.array_namespace(model_dict["energy"])
        coef = learning_rate / self.starter_learning_rate
        pref_h = self.limit_pref_h + (self.start_pref_h - self.limit_pref_h) * coef

        if (
            self.has_h
            and "energy_derv_r_derv_r" in model_dict
            and "hessian" in label_dict
        ):
            find_hessian = label_dict.get("find_hessian", 0.0)
            pref_h = pref_h * find_hessian
            atom_mask = _atom_mask(model_dict, label_dict)
            hessian_mask = _hessian_mask_from_atom_mask(xp, atom_mask)
            pred_hessian = _reshape_hessian_like_atoms(
                xp, model_dict["energy_derv_r_derv_r"], atom_mask
            )
            label_hessian = _reshape_hessian_like_atoms(
                xp, label_dict["hessian"], atom_mask
            )
            if hessian_mask is None:
                pred_hessian = xp.reshape(pred_hessian, label_hessian.shape)
            diff_h = label_hessian - pred_hessian
            l2_hessian_loss = _masked_mean(xp, xp.square(diff_h), hessian_mask)
            loss += pref_h * l2_hessian_loss
            rmse_h = xp.sqrt(l2_hessian_loss)
            more_loss["rmse_h"] = self.display_if_exist(rmse_h, find_hessian)
            if mae:
                mae_h = _masked_mean(xp, xp.abs(diff_h), hessian_mask)
                more_loss["mae_h"] = self.display_if_exist(mae_h, find_hessian)

        more_loss["rmse"] = xp.sqrt(loss)
        return loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Add hessian label requirement needed for this loss calculation."""
        label_requirement = super().label_requirement
        if self.has_h:
            label_requirement.append(
                DataRequirementItem(
                    "hessian",
                    ndof=1,  # 9=3*3 --> 3N*3N=ndof*natoms*natoms
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        return label_requirement
