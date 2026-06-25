# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.loss.ener import (
    EnergyHessianLoss,
    EnergyLoss,
)

from ...seed import (
    GLOBAL_SEED,
)


class TestEnergyLossBase(unittest.TestCase):
    """Base class providing common setup for dpmodel EnergyLoss tests."""

    def _make_data(self, natoms=5, nframes=2, numb_generalized_coord=0):
        """Generate fake model predictions and labels."""
        rng = np.random.default_rng(GLOBAL_SEED)
        model_dict = {
            "energy": rng.random((nframes, 1)),
            "force": rng.random((nframes, natoms, 3)),
            "virial": rng.random((nframes, 9)),
            "atom_energy": rng.random((nframes, natoms, 1)),
        }
        label_dict = {
            "energy": rng.random((nframes, 1)),
            "force": rng.random((nframes, natoms, 3)),
            "virial": rng.random((nframes, 9)),
            "atom_ener": rng.random((nframes, natoms, 1)),
            "atom_pref": rng.random((nframes, natoms * 3)),
            "find_energy": 1.0,
            "find_force": 1.0,
            "find_virial": 1.0,
            "find_atom_ener": 1.0,
            "find_atom_pref": 1.0,
        }
        if numb_generalized_coord > 0:
            label_dict["drdq"] = rng.random(
                (nframes, natoms * 3 * numb_generalized_coord)
            )
            label_dict["find_drdq"] = 1.0
        if hasattr(self, "enable_atom_ener_coeff") and self.enable_atom_ener_coeff:
            label_dict["atom_ener_coeff"] = rng.random((nframes, natoms, 1))
        return model_dict, label_dict, natoms


class TestEnergyLossBasic(TestEnergyLossBase):
    """Test basic energy loss (e, f, v, ae)."""

    def test_forward(self) -> None:
        loss_fn = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=1.0,
            limit_pref_e=0.5,
            start_pref_f=1.0,
            limit_pref_f=0.5,
            start_pref_v=1.0,
            limit_pref_v=0.5,
            start_pref_ae=1.0,
            limit_pref_ae=0.5,
        )
        model_dict, label_dict, natoms = self._make_data()
        loss, more_loss = loss_fn.call(1.0, natoms, model_dict, label_dict)
        self.assertIsNotNone(loss)
        self.assertIn("rmse_e", more_loss)
        self.assertIn("rmse_f", more_loss)
        self.assertIn("rmse_v", more_loss)
        self.assertIn("rmse_ae", more_loss)

    def test_force_padding_mask(self) -> None:
        loss_fn = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_f=1.0,
            limit_pref_f=1.0,
        )
        nframes, nreal, nmax = 1, 2, 4
        model_dict = {
            "energy": np.zeros((nframes, 1)),
            "force": np.zeros((nframes, nmax, 3)),
            "virial": np.zeros((nframes, 9)),
            "atom_energy": np.zeros((nframes, nmax, 1)),
            "mask": np.array([[1, 1, 0, 0]], dtype=np.int32),
        }
        label_force = np.zeros((nframes, nmax, 3))
        label_force[:, :nreal, :] = 1.0
        label_dict = {
            "energy": np.zeros((nframes, 1)),
            "force": label_force,
            "virial": np.zeros((nframes, 9)),
            "atom_ener": np.zeros((nframes, nmax, 1)),
            "atom_pref": np.zeros((nframes, nmax * 3)),
            "type": np.array([[0, 0, -1, -1]], dtype=np.int32),
            "find_energy": 0.0,
            "find_force": 1.0,
            "find_virial": 0.0,
            "find_atom_ener": 0.0,
            "find_atom_pref": 0.0,
        }
        _loss, more_loss = loss_fn.call(1.0, nmax, model_dict, label_dict)
        np.testing.assert_allclose(more_loss["rmse_f"], 1.0)

    def test_force_padding_mask_with_flat_label(self) -> None:
        loss_fn = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_f=1.0,
            limit_pref_f=1.0,
        )
        nframes, nreal, nmax = 1, 2, 4
        model_dict = {
            "energy": np.zeros((nframes, 1)),
            "force": np.zeros((nframes, nmax, 3)),
            "virial": np.zeros((nframes, 9)),
            "atom_energy": np.zeros((nframes, nmax, 1)),
            "mask": np.array([[1, 1, 0, 0]], dtype=np.int32),
        }
        label_force = np.zeros((nframes, nmax, 3))
        label_force[:, :nreal, :] = 1.0
        label_dict = {
            "energy": np.zeros((nframes, 1)),
            "force": label_force.reshape(nframes, nmax * 3),
            "virial": np.zeros((nframes, 9)),
            "atom_ener": np.zeros((nframes, nmax, 1)),
            "atom_pref": np.zeros((nframes, nmax * 3)),
            "type": np.array([[0, 0, -1, -1]], dtype=np.int32),
            "find_energy": 0.0,
            "find_force": 1.0,
            "find_virial": 0.0,
            "find_atom_ener": 0.0,
            "find_atom_pref": 0.0,
        }
        _loss, more_loss = loss_fn.call(1.0, nmax, model_dict, label_dict)
        np.testing.assert_allclose(more_loss["rmse_f"], 1.0)

    def test_energy_padding_uses_real_natoms(self) -> None:
        loss_fn = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=1.0,
            limit_pref_e=1.0,
        )
        nframes, nmax = 1, 4
        model_dict = {
            "energy": np.zeros((nframes, 1)),
            "force": np.zeros((nframes, nmax, 3)),
            "virial": np.zeros((nframes, 9)),
            "atom_energy": np.zeros((nframes, nmax, 1)),
            "mask": np.array([[1, 1, 0, 0]], dtype=np.int32),
        }
        label_dict = {
            "energy": np.array([[2.0]]),
            "force": np.zeros((nframes, nmax, 3)),
            "virial": np.zeros((nframes, 9)),
            "atom_ener": np.zeros((nframes, nmax, 1)),
            "atom_pref": np.zeros((nframes, nmax * 3)),
            "type": np.array([[0, 0, -1, -1]], dtype=np.int32),
            "find_energy": 1.0,
            "find_force": 0.0,
            "find_virial": 0.0,
            "find_atom_ener": 0.0,
            "find_atom_pref": 0.0,
        }
        _loss, more_loss = loss_fn.call(1.0, nmax, model_dict, label_dict)
        np.testing.assert_allclose(more_loss["rmse_e"], 1.0)


class TestEnergyLossAecoeff(TestEnergyLossBase):
    """Test energy loss with atom_ener_coeff."""

    enable_atom_ener_coeff = True

    def test_forward(self) -> None:
        loss_fn = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=1.0,
            limit_pref_e=0.5,
            start_pref_f=1.0,
            limit_pref_f=0.5,
            start_pref_v=1.0,
            limit_pref_v=0.5,
            enable_atom_ener_coeff=True,
        )
        model_dict, label_dict, natoms = self._make_data()
        loss, more_loss = loss_fn.call(1.0, natoms, model_dict, label_dict)
        self.assertIsNotNone(loss)


class TestEnergyLossGeneralizedForce(TestEnergyLossBase):
    """Test energy loss with generalized force (numb_generalized_coord > 0).

    This exercises the code path with natoms used as int scalar
    (not array), which previously had a natoms[0] bug.
    """

    def test_forward(self) -> None:
        numb_generalized_coord = 2
        loss_fn = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=1.0,
            limit_pref_e=0.5,
            start_pref_f=1.0,
            limit_pref_f=0.5,
            start_pref_v=1.0,
            limit_pref_v=0.5,
            start_pref_ae=1.0,
            limit_pref_ae=0.5,
            start_pref_pf=1.0,
            limit_pref_pf=0.5,
            start_pref_gf=1.0,
            limit_pref_gf=0.5,
            numb_generalized_coord=numb_generalized_coord,
        )
        model_dict, label_dict, natoms = self._make_data(
            numb_generalized_coord=numb_generalized_coord,
        )
        loss, more_loss = loss_fn.call(1.0, natoms, model_dict, label_dict)
        self.assertIsNotNone(loss)
        self.assertIn("rmse_gf", more_loss)
        self.assertIn("rmse_pf", more_loss)


class TestEnergyLossHuber(TestEnergyLossBase):
    """Test energy loss with Huber loss."""

    def test_forward(self) -> None:
        loss_fn = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=1.0,
            limit_pref_e=0.5,
            start_pref_f=1.0,
            limit_pref_f=0.5,
            start_pref_v=1.0,
            limit_pref_v=0.5,
            use_huber=True,
            huber_delta=0.01,
        )
        model_dict, label_dict, natoms = self._make_data()
        loss, more_loss = loss_fn.call(1.0, natoms, model_dict, label_dict)
        self.assertIsNotNone(loss)


class TestEnergyHessianLossPadding(TestEnergyLossBase):
    """Test Hessian loss ignores padded atoms in both Hessian axes."""

    def test_hessian_padding_mask(self) -> None:
        loss_fn = EnergyHessianLoss(
            starter_learning_rate=1.0,
            start_pref_h=1.0,
            limit_pref_h=1.0,
        )
        nframes, nreal, nmax = 1, 2, 4
        hdim = nmax * 3
        model_dict = {
            "energy": np.zeros((nframes, 1)),
            "force": np.zeros((nframes, nmax, 3)),
            "virial": np.zeros((nframes, 9)),
            "atom_energy": np.zeros((nframes, nmax, 1)),
            "energy_derv_r_derv_r": np.zeros((nframes, hdim, hdim)),
            "mask": np.array([[1, 1, 0, 0]], dtype=np.int32),
        }
        label_hessian = np.zeros((nframes, hdim, hdim))
        label_hessian[:, : nreal * 3, : nreal * 3] = 1.0
        label_dict = {
            "energy": np.zeros((nframes, 1)),
            "force": np.zeros((nframes, nmax, 3)),
            "virial": np.zeros((nframes, 9)),
            "atom_ener": np.zeros((nframes, nmax, 1)),
            "atom_pref": np.zeros((nframes, nmax * 3)),
            "hessian": label_hessian,
            "type": np.array([[0, 0, -1, -1]], dtype=np.int32),
            "find_energy": 0.0,
            "find_force": 0.0,
            "find_virial": 0.0,
            "find_atom_ener": 0.0,
            "find_atom_pref": 0.0,
            "find_hessian": 1.0,
        }
        _loss, more_loss = loss_fn.call(1.0, nmax, model_dict, label_dict)
        np.testing.assert_allclose(more_loss["rmse_h"], 1.0)

    def test_hessian_padding_mask_with_flat_label(self) -> None:
        loss_fn = EnergyHessianLoss(
            starter_learning_rate=1.0,
            start_pref_h=1.0,
            limit_pref_h=1.0,
        )
        nframes, nreal, nmax = 1, 2, 4
        hdim = nmax * 3
        model_dict = {
            "energy": np.zeros((nframes, 1)),
            "force": np.zeros((nframes, nmax, 3)),
            "virial": np.zeros((nframes, 9)),
            "atom_energy": np.zeros((nframes, nmax, 1)),
            "energy_derv_r_derv_r": np.zeros((nframes, hdim, hdim)),
            "mask": np.array([[1, 1, 0, 0]], dtype=np.int32),
        }
        label_hessian = np.zeros((nframes, hdim, hdim))
        label_hessian[:, : nreal * 3, : nreal * 3] = 1.0
        label_dict = {
            "energy": np.zeros((nframes, 1)),
            "force": np.zeros((nframes, nmax, 3)),
            "virial": np.zeros((nframes, 9)),
            "atom_ener": np.zeros((nframes, nmax, 1)),
            "atom_pref": np.zeros((nframes, nmax * 3)),
            "hessian": label_hessian.reshape(nframes, hdim * hdim),
            "type": np.array([[0, 0, -1, -1]], dtype=np.int32),
            "find_energy": 0.0,
            "find_force": 0.0,
            "find_virial": 0.0,
            "find_atom_ener": 0.0,
            "find_atom_pref": 0.0,
            "find_hessian": 1.0,
        }
        _loss, more_loss = loss_fn.call(1.0, nmax, model_dict, label_dict)
        np.testing.assert_allclose(more_loss["rmse_h"], 1.0)


class TestEnergyLossSerialize(TestEnergyLossBase):
    """Test serialize/deserialize round-trip."""

    def test_serialize_deserialize(self) -> None:
        loss_fn = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=1.0,
            limit_pref_e=0.5,
            start_pref_f=1.0,
            limit_pref_f=0.5,
            start_pref_v=1.0,
            limit_pref_v=0.5,
            start_pref_gf=1.0,
            limit_pref_gf=0.5,
            numb_generalized_coord=2,
        )
        data = loss_fn.serialize()
        loss_fn2 = EnergyLoss.deserialize(data)
        model_dict, label_dict, natoms = self._make_data(numb_generalized_coord=2)
        loss1, more1 = loss_fn.call(1.0, natoms, model_dict, label_dict)
        loss2, more2 = loss_fn2.call(1.0, natoms, model_dict, label_dict)
        np.testing.assert_allclose(loss1, loss2)
        for key in more1:
            np.testing.assert_allclose(more1[key], more2[key])


if __name__ == "__main__":
    unittest.main()
