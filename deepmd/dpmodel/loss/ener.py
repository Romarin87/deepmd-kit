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


def custom_huber_loss(predictions: Array, targets: Array, delta: float = 1.0) -> Array:
    xp = array_api_compat.array_namespace(predictions, targets)
    error = targets - predictions
    abs_error = xp.abs(error)
    quadratic_loss = 0.5 * error**2
    linear_loss = delta * (abs_error - 0.5 * delta)
    loss = xp.where(abs_error <= delta, quadratic_loss, linear_loss)
    return xp.mean(loss)


def safe_vector_norm(
    values: Array,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Return an L2 norm with finite gradients at exactly zero."""
    xp = array_api_compat.array_namespace(values)
    sq_norm = xp.sum(values * values, axis=axis, keepdims=keepdims)
    eps = xp.asarray(1e-30, dtype=values.dtype)
    norm = xp.sqrt(xp.where(sq_norm > 0, sq_norm, eps))
    return xp.where(sq_norm > 0, norm, xp.zeros_like(norm))


def _masked_mean(values: Array, mask: Array | None) -> Array:
    """Mean over valid mask entries, preserving the unmasked mean path."""
    xp = array_api_compat.array_namespace(values)
    if mask is None:
        return xp.mean(values)
    weight = xp.astype(mask, values.dtype)
    while len(weight.shape) < len(values.shape):
        weight = weight[..., None]
    weight = xp.ones_like(values) * weight
    denom = xp.sum(weight)
    denom = xp.where(denom > 0, denom, xp.asarray(1.0, dtype=values.dtype))
    return xp.sum(values * weight) / denom


def _get_atom_mask(
    model_dict: dict[str, Array],
    label_dict: dict[str, Array],
    natoms: int,
    ref: Array,
    xp: Any,
) -> Array | None:
    """Return a float atom mask with shape ``[nframes, natoms]``."""
    mask = model_dict.get("mask")
    if mask is None and "type" in label_dict:
        mask = label_dict["type"] >= 0
    if mask is None:
        return None
    nframes = ref.shape[0]
    mask = xp.reshape(mask, (nframes, natoms))
    return xp.astype(mask, ref.dtype)


def _frame_atom_norm(atom_mask: Array | None, natoms: int, ref: Array, xp: Any) -> Array:
    """Return 1 / real_natoms per frame, or the legacy scalar 1 / natoms."""
    if atom_mask is None:
        return xp.asarray(1.0 / natoms, dtype=ref.dtype)
    atom_count = xp.sum(atom_mask, axis=-1)
    atom_count = xp.where(
        atom_count > 0,
        atom_count,
        xp.asarray(float(natoms), dtype=ref.dtype),
    )
    return 1.0 / atom_count


def _apply_frame_weight(values: Array, weight: Array, xp: Any) -> Array:
    """Broadcast a per-frame weight to an output tensor."""
    if len(weight.shape) == 0:
        return values * weight
    target_shape = (weight.shape[0],) + (1,) * (len(values.shape) - 1)
    return values * xp.reshape(weight, target_shape)


def _coordinate_mask(atom_mask: Array | None, xp: Any, dtype: Any) -> Array | None:
    """Expand an atom mask to a flattened 3N coordinate mask."""
    if atom_mask is None:
        return None
    coord_mask = atom_mask[..., None] * xp.ones((1, 1, 3), dtype=dtype)
    return xp.reshape(coord_mask, (atom_mask.shape[0], atom_mask.shape[1] * 3))


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
    use_default_pf : bool
        If true, use default atom_pref of 1.0 for all atoms when atom_pref data is not provided.
        This allows using the prefactor force loss (pf) without requiring atom_pref.npy files.
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
        use_default_pf: bool = False,
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
        self.use_default_pf = use_default_pf
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
        find_atom_pref = (
            label_dict["find_atom_pref"] if not self.use_default_pf else 1.0
        )
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
        if self.has_f or self.has_pf or self.relative_f or self.has_gf:
            force_atom_shape = (force_hat.shape[0], natoms, 3)
            force = xp.reshape(force, force_atom_shape)
            force_hat = xp.reshape(force_hat, force_atom_shape)
        atom_mask = _get_atom_mask(model_dict, label_dict, natoms, energy, xp)
        atom_norm = _frame_atom_norm(atom_mask, natoms, energy, xp)
        atom_norm_ener = atom_norm

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
        if self.has_f or self.has_pf or self.relative_f or self.has_gf:
            diff_f_atom = force_hat - force
        else:
            diff_f_atom = None
            diff_f = None

        if self.relative_f is not None:
            force_hat_3 = xp.reshape(force_hat, (-1, 3))
            norm_f = (
                xp.reshape(xp.linalg.vector_norm(force_hat_3, axis=1), (-1, 1))
                + self.relative_f
            )
            diff_f_3 = xp.reshape(diff_f_atom, (-1, 3))
            diff_f_3 = diff_f_3 / norm_f
            diff_f_atom = xp.reshape(diff_f_3, force_hat.shape)
        if diff_f_atom is not None:
            diff_f = xp.reshape(diff_f_atom, (-1,))

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
                if atom_mask is None:
                    l2_ener_loss = xp.mean(xp.square(diff_e))
                    l2_ener_loss_scaled = atom_norm_ener**norm_exp * l2_ener_loss
                    rmse_e = xp.sqrt(l2_ener_loss) * atom_norm_ener
                else:
                    if self.intensive_ener_virial:
                        normed_diff_e = _apply_frame_weight(diff_e, atom_norm_ener, xp)
                        l2_ener_loss = xp.mean(xp.square(normed_diff_e))
                    else:
                        l2_ener_loss = xp.mean(
                            xp.square(diff_e)
                            * xp.reshape(atom_norm_ener, diff_e.shape)
                        )
                    l2_ener_loss_scaled = l2_ener_loss
                    rmse_e = xp.sqrt(
                        xp.mean(
                            xp.square(
                                _apply_frame_weight(diff_e, atom_norm_ener, xp)
                            )
                        )
                    )
                if not self.use_huber:
                    loss += pref_e * l2_ener_loss_scaled
                else:
                    energy_norm = _apply_frame_weight(energy, atom_norm_ener, xp)
                    energy_hat_norm = _apply_frame_weight(energy_hat, atom_norm_ener, xp)
                    l_huber_loss = custom_huber_loss(
                        energy_norm,
                        energy_hat_norm,
                        delta=self._huber_delta_energy,
                    )
                    loss += pref_e * l_huber_loss
                more_loss["rmse_e"] = self.display_if_exist(
                    rmse_e, find_energy
                )
            elif self.loss_func == "mae":
                if atom_mask is None:
                    l1_ener_loss = xp.mean(xp.abs(diff_e))
                    loss += atom_norm_ener * (pref_e * l1_ener_loss)
                    mae_e = l1_ener_loss * atom_norm_ener
                else:
                    mae_e = xp.mean(
                        xp.abs(_apply_frame_weight(diff_e, atom_norm_ener, xp))
                    )
                    l1_ener_loss = mae_e
                    loss += pref_e * l1_ener_loss
                more_loss["mae_e"] = self.display_if_exist(
                    mae_e, find_energy
                )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for energy loss."
                )
            if mae:
                if atom_mask is None:
                    mae_e = xp.mean(xp.abs(diff_e)) * atom_norm_ener
                else:
                    mae_e = xp.mean(
                        xp.abs(_apply_frame_weight(diff_e, atom_norm_ener, xp))
                    )
                more_loss["mae_e"] = self.display_if_exist(mae_e, find_energy)
                mae_e_all = xp.mean(xp.abs(diff_e))
                more_loss["mae_e_all"] = self.display_if_exist(mae_e_all, find_energy)
        if self.has_f:
            if self.loss_func == "mse":
                l2_force_loss = _masked_mean(xp.square(diff_f_atom), atom_mask)
                if not self.use_huber:
                    loss += pref_f * l2_force_loss
                else:
                    if not self.f_use_norm:
                        diff_for_huber = xp.abs(force_hat - force)
                        abs_error = diff_for_huber
                        quadratic_loss = 0.5 * diff_for_huber**2
                        linear_loss = self._huber_delta_force * (
                            abs_error - 0.5 * self._huber_delta_force
                        )
                        huber_values = xp.where(
                            abs_error <= self._huber_delta_force,
                            quadratic_loss,
                            linear_loss,
                        )
                        l_huber_loss = _masked_mean(huber_values, atom_mask)
                    else:
                        force_diff_norm = safe_vector_norm(diff_f_atom, axis=-1)
                        abs_error = xp.abs(force_diff_norm)
                        quadratic_loss = 0.5 * force_diff_norm**2
                        linear_loss = self._huber_delta_force * (
                            abs_error - 0.5 * self._huber_delta_force
                        )
                        huber_values = xp.where(
                            abs_error <= self._huber_delta_force,
                            quadratic_loss,
                            linear_loss,
                        )
                        l_huber_loss = _masked_mean(huber_values, atom_mask)
                    loss += pref_f * l_huber_loss
                more_loss["rmse_f"] = self.display_if_exist(
                    xp.sqrt(l2_force_loss), find_force
                )
            elif self.loss_func == "mae":
                if not self.f_use_norm:
                    l1_force_loss = _masked_mean(xp.abs(diff_f_atom), atom_mask)
                else:
                    l1_force_loss = _masked_mean(
                        safe_vector_norm(diff_f_atom, axis=-1), atom_mask
                    )
                loss += pref_f * l1_force_loss
                more_loss["mae_f"] = self.display_if_exist(l1_force_loss, find_force)
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for force loss."
                )
            if mae:
                mae_f = _masked_mean(xp.abs(diff_f_atom), atom_mask)
                more_loss["mae_f"] = self.display_if_exist(mae_f, find_force)
        if self.has_v:
            virial_reshape = xp.reshape(virial, (-1,))
            virial_hat_reshape = xp.reshape(virial_hat, (-1,))
            diff_v = virial_hat - virial
            if self.loss_func == "mse":
                if atom_mask is None:
                    l2_virial_loss = xp.mean(
                        xp.square(virial_hat_reshape - virial_reshape),
                    )
                    l2_virial_loss_scaled = atom_norm**norm_exp * l2_virial_loss
                    rmse_v = xp.sqrt(l2_virial_loss) * atom_norm
                else:
                    if self.intensive_ener_virial:
                        normed_diff_v = _apply_frame_weight(diff_v, atom_norm, xp)
                        l2_virial_loss = xp.mean(xp.square(normed_diff_v))
                    else:
                        l2_virial_loss = xp.mean(
                            xp.square(diff_v)
                            * xp.reshape(atom_norm, (atom_norm.shape[0], 1))
                        )
                    l2_virial_loss_scaled = l2_virial_loss
                    rmse_v = xp.sqrt(
                        xp.mean(
                            xp.square(_apply_frame_weight(diff_v, atom_norm, xp))
                        )
                    )
                if not self.use_huber:
                    loss += pref_v * l2_virial_loss_scaled
                else:
                    virial_norm = _apply_frame_weight(virial, atom_norm, xp)
                    virial_hat_norm = _apply_frame_weight(virial_hat, atom_norm, xp)
                    l_huber_loss = custom_huber_loss(
                        virial_norm,
                        virial_hat_norm,
                        delta=self._huber_delta_virial,
                    )
                    loss += pref_v * l_huber_loss
                more_loss["rmse_v"] = self.display_if_exist(
                    rmse_v, find_virial
                )
            elif self.loss_func == "mae":
                if atom_mask is None:
                    l1_virial_loss = xp.mean(
                        xp.abs(virial_hat_reshape - virial_reshape)
                    )
                    loss += atom_norm * (pref_v * l1_virial_loss)
                    mae_v = l1_virial_loss * atom_norm
                else:
                    mae_v = xp.mean(
                        xp.abs(_apply_frame_weight(diff_v, atom_norm, xp))
                    )
                    l1_virial_loss = mae_v
                    loss += pref_v * l1_virial_loss
                more_loss["mae_v"] = self.display_if_exist(
                    mae_v, find_virial
                )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for virial loss."
                )
            if mae:
                if atom_mask is None:
                    mae_v = (
                        xp.mean(xp.abs(virial_hat_reshape - virial_reshape))
                        * atom_norm
                    )
                else:
                    mae_v = xp.mean(
                        xp.abs(_apply_frame_weight(diff_v, atom_norm, xp))
                    )
                more_loss["mae_v"] = self.display_if_exist(mae_v, find_virial)
        if self.has_ae:
            atom_ener = xp.reshape(atom_ener, atom_ener_hat.shape)
            diff_ae = atom_ener_hat - atom_ener
            if self.loss_func == "mse":
                l2_atom_ener_loss = _masked_mean(xp.square(diff_ae), atom_mask)
                if not self.use_huber:
                    loss += pref_ae * l2_atom_ener_loss
                else:
                    l_huber_loss = custom_huber_loss(
                        xp.reshape(atom_ener, (-1,)),
                        xp.reshape(atom_ener_hat, (-1,)),
                        delta=self._huber_delta_energy,
                    )
                    loss += pref_ae * l_huber_loss
                more_loss["rmse_ae"] = self.display_if_exist(
                    xp.sqrt(l2_atom_ener_loss), find_atom_ener
                )
            elif self.loss_func == "mae":
                l1_atom_ener_loss = _masked_mean(xp.abs(diff_ae), atom_mask)
                loss += pref_ae * l1_atom_ener_loss
                more_loss["mae_ae"] = self.display_if_exist(
                    l1_atom_ener_loss, find_atom_ener
                )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for atomic energy loss."
                )
        if self.has_pf:
            atom_pref = xp.reshape(atom_pref, diff_f_atom.shape)

            if self.loss_func == "mse":
                l2_pref_force_loss = _masked_mean(
                    xp.multiply(xp.square(diff_f_atom), atom_pref),
                    atom_mask,
                )
                loss += pref_pf * l2_pref_force_loss
                more_loss["rmse_pf"] = self.display_if_exist(
                    xp.sqrt(l2_pref_force_loss), find_atom_pref
                )
            elif self.loss_func == "mae":
                l1_pref_force_loss = _masked_mean(
                    xp.multiply(xp.abs(diff_f_atom), atom_pref),
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
                default=1.0,
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
            "@version": 4,
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
            "use_default_pf": self.use_default_pf,
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
        check_version_compatibility(version, 4, 1)
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
        r"""Enable the layer to compute loss on Hessian labels.

        Parameters
        ----------
        start_pref_h : float
            The prefactor of Hessian loss at the start of the training.
        limit_pref_h : float
            The prefactor of Hessian loss at the end of the training.
        **kwargs
            Other keyword arguments.
        """
        super().__init__(**kwargs)
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
    ) -> tuple[Array, dict[str, Array]]:
        """Calculate energy/force/virial and Hessian losses."""
        loss, more_loss = super().call(
            learning_rate,
            natoms,
            model_dict,
            label_dict,
            mae=mae,
        )
        xp = array_api_compat.array_namespace(model_dict["energy"])
        lr_ratio = learning_rate / self.starter_learning_rate
        pref_h = self.limit_pref_h + (self.start_pref_h - self.limit_pref_h) * lr_ratio

        hessian = model_dict.get("hessian", model_dict.get("energy_derv_r_derv_r"))
        if self.has_h and hessian is not None and "hessian" in label_dict:
            find_hessian = label_dict.get("find_hessian", 0.0)
            hessian_shape = (label_dict["hessian"].shape[0], natoms * 3, natoms * 3)
            hessian = xp.reshape(hessian, hessian_shape)
            hessian_hat = xp.reshape(label_dict["hessian"], hessian_shape)
            diff_h = hessian_hat - hessian
            force_for_mask = xp.reshape(
                label_dict["force"],
                (label_dict["force"].shape[0], natoms, 3),
            )
            atom_mask = _get_atom_mask(
                model_dict,
                label_dict,
                natoms,
                force_for_mask,
                xp,
            )
            coord_mask = _coordinate_mask(atom_mask, xp, diff_h.dtype)
            hessian_mask = (
                None
                if coord_mask is None
                else coord_mask[:, :, None] * coord_mask[:, None, :]
            )
            if self.loss_func == "mse":
                l2_hessian_loss = _masked_mean(xp.square(diff_h), hessian_mask)
                loss += pref_h * find_hessian * l2_hessian_loss
                more_loss["rmse_h"] = self.display_if_exist(
                    xp.sqrt(l2_hessian_loss),
                    find_hessian,
                )
            elif self.loss_func == "mae":
                l1_hessian_loss = _masked_mean(xp.abs(diff_h), hessian_mask)
                loss += pref_h * find_hessian * l1_hessian_loss
                more_loss["mae_h"] = self.display_if_exist(
                    l1_hessian_loss,
                    find_hessian,
                )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for Hessian loss."
                )
            if mae:
                mae_h = _masked_mean(xp.abs(diff_h), hessian_mask)
                more_loss["mae_h"] = self.display_if_exist(mae_h, find_hessian)

        more_loss.pop("rmse", None)
        more_loss["rmse"] = xp.sqrt(loss)
        return loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for Hessian loss calculation."""
        label_requirement = super().label_requirement
        if self.has_h:
            label_requirement.append(
                DataRequirementItem(
                    "hessian",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        return label_requirement

    def serialize(self) -> dict:
        """Serialize the loss module."""
        data = super().serialize()
        data.update(
            {
                "@class": "EnergyHessianLoss",
                "start_pref_h": self.start_pref_h,
                "limit_pref_h": self.limit_pref_h,
            }
        )
        return data
