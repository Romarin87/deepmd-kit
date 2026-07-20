# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.entrypoints.test import (
    _compute_energy_type_metrics_with_atom_mask,
    _compute_hessian_error_stat_with_atom_mask,
    _compute_weighted_force_error_stat_with_atom_mask,
    _get_real_atom_mask,
)


class TestDPTestPaddingMask(unittest.TestCase):
    def test_padding_atoms_do_not_affect_efh_metrics(self) -> None:
        energy_ref = np.array([[0.0]])
        energy_pred = np.array([[2.0]])

        type_unpadded = np.array([[0, 1]], dtype=np.int32)
        mask_unpadded = _get_real_atom_mask(
            type_unpadded,
            numb_test=1,
            natoms=2,
        )
        force_ref_unpadded = np.zeros((1, 6))
        force_pred_unpadded = np.ones((1, 6))
        hessian_ref_unpadded = np.zeros((1, 6, 6))
        hessian_pred_unpadded = np.full((1, 6, 6), 2.0)

        type_padded = np.array([[0, 1, -1, -1]], dtype=np.int32)
        mask_padded = _get_real_atom_mask(
            type_padded,
            numb_test=1,
            natoms=4,
        )
        force_ref_padded = np.zeros((1, 12))
        force_pred_padded = np.full((1, 12), 99.0)
        force_pred_padded[:, :6] = 1.0
        hessian_ref_padded = np.zeros((1, 12, 12))
        hessian_pred_padded = np.full((1, 12, 12), 99.0)
        hessian_pred_padded[:, :6, :6] = 2.0

        metrics_unpadded = _compute_energy_type_metrics_with_atom_mask(
            prediction={
                "energy": energy_pred,
                "force": force_pred_unpadded,
            },
            test_data={
                "find_energy": 1.0,
                "find_force": 1.0,
                "energy": energy_ref,
                "force": force_ref_unpadded,
            },
            natoms=2,
            has_pbc=False,
            atom_mask=mask_unpadded,
        )
        metrics_padded = _compute_energy_type_metrics_with_atom_mask(
            prediction={
                "energy": energy_pred,
                "force": force_pred_padded,
            },
            test_data={
                "find_energy": 1.0,
                "find_force": 1.0,
                "energy": energy_ref,
                "force": force_ref_padded,
            },
            natoms=4,
            has_pbc=False,
            atom_mask=mask_padded,
        )

        self.assertIsNotNone(metrics_unpadded.energy)
        self.assertIsNotNone(metrics_unpadded.energy_per_atom)
        self.assertIsNotNone(metrics_unpadded.force)
        self.assertIsNotNone(metrics_padded.energy)
        self.assertIsNotNone(metrics_padded.energy_per_atom)
        self.assertIsNotNone(metrics_padded.force)

        np.testing.assert_allclose(
            metrics_padded.energy.mae,
            metrics_unpadded.energy.mae,
        )
        np.testing.assert_allclose(
            metrics_padded.energy_per_atom.mae,
            metrics_unpadded.energy_per_atom.mae,
        )
        np.testing.assert_allclose(
            metrics_padded.force.mae,
            metrics_unpadded.force.mae,
        )
        np.testing.assert_allclose(metrics_padded.force.rmse, 1.0)
        self.assertEqual(metrics_padded.force.weight, 6.0)

        hessian_unpadded = _compute_hessian_error_stat_with_atom_mask(
            hessian_pred_unpadded,
            hessian_ref_unpadded,
            natoms=2,
            atom_mask=mask_unpadded,
        )
        hessian_padded = _compute_hessian_error_stat_with_atom_mask(
            hessian_pred_padded,
            hessian_ref_padded,
            natoms=4,
            atom_mask=mask_padded,
        )
        np.testing.assert_allclose(hessian_padded.mae, hessian_unpadded.mae)
        np.testing.assert_allclose(hessian_padded.rmse, 2.0)
        self.assertEqual(hessian_padded.weight, 36.0)

        weighted_force_unpadded = _compute_weighted_force_error_stat_with_atom_mask(
            force_pred_unpadded,
            force_ref_unpadded,
            np.ones((1, 2)),
            natoms=2,
            atom_mask=mask_unpadded,
        )
        weighted_force_padded = _compute_weighted_force_error_stat_with_atom_mask(
            force_pred_padded,
            force_ref_padded,
            np.array([[1.0, 1.0, 999.0, 999.0]]),
            natoms=4,
            atom_mask=mask_padded,
        )
        np.testing.assert_allclose(
            weighted_force_padded.mae,
            weighted_force_unpadded.mae,
        )
        np.testing.assert_allclose(weighted_force_padded.rmse, 1.0)
        self.assertEqual(weighted_force_padded.weight, 6.0)


if __name__ == "__main__":
    unittest.main()
