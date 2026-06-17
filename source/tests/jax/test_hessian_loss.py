# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.loss import (
    EnergyLoss,
    EnergyHessianLoss,
)
from deepmd.jax.env import (
    jax,
    jnp,
)


@unittest.skipIf(
    sys.version_info < (3, 10),
    "JAX requires Python 3.10 or later",
)
class TestEnergyHessianLoss(unittest.TestCase):
    def test_energy_force_loss_ignores_padding_atoms(self) -> None:
        loss_fn = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_f=1.0,
            limit_pref_f=1.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            loss_func="mae",
            f_use_norm=True,
        )
        nframes, natoms = 1, 4
        mask = jnp.asarray([[1, 1, 0, 0]], dtype=jnp.int32)
        model_dict = {
            "energy": jnp.asarray([0.0]),
            "force": jnp.zeros((nframes, natoms, 3)),
            "virial": jnp.zeros((nframes, 9)),
            "atom_energy": jnp.zeros((nframes, natoms)),
            "mask": mask,
        }
        label_dict = {
            "energy": jnp.asarray([2.0]),
            "force": jnp.asarray(
                [
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ]
            ),
            "virial": jnp.zeros((nframes, 9)),
            "atom_ener": jnp.zeros((nframes, natoms)),
            "atom_pref": jnp.ones((nframes, natoms, 3)),
            "type": jnp.asarray([[0, 0, -1, -1]], dtype=jnp.int32),
            "find_energy": 1.0,
            "find_force": 1.0,
            "find_virial": 0.0,
            "find_atom_ener": 0.0,
            "find_atom_pref": 0.0,
        }

        loss, more_loss = loss_fn(1.0, natoms, model_dict, label_dict)

        np.testing.assert_allclose(to_numpy_array(more_loss["mae_e"]), 1.0)
        np.testing.assert_allclose(to_numpy_array(more_loss["mae_f"]), 1.5)
        np.testing.assert_allclose(to_numpy_array(loss), 2.5)

    def test_hessian_loss_ignores_padding_axes(self) -> None:
        loss_fn = EnergyHessianLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_h=1.0,
            limit_pref_h=1.0,
            loss_func="mae",
        )
        nframes, natoms, nreal = 1, 4, 2
        hessian = np.zeros((nframes, natoms * 3, natoms * 3))
        hessian[:, : nreal * 3, : nreal * 3] = 1.0
        model_dict = {
            "energy": jnp.zeros((nframes,)),
            "force": jnp.zeros((nframes, natoms, 3)),
            "virial": jnp.zeros((nframes, 9)),
            "atom_energy": jnp.zeros((nframes, natoms)),
            "hessian": jnp.zeros((nframes, natoms * 3, natoms * 3)),
            "mask": jnp.asarray([[1, 1, 0, 0]], dtype=jnp.int32),
        }
        label_dict = {
            "energy": jnp.zeros((nframes,)),
            "force": jnp.zeros((nframes, natoms, 3)),
            "virial": jnp.zeros((nframes, 9)),
            "atom_ener": jnp.zeros((nframes, natoms)),
            "atom_pref": jnp.ones((nframes, natoms, 3)),
            "hessian": jnp.asarray(hessian),
            "type": jnp.asarray([[0, 0, -1, -1]], dtype=jnp.int32),
            "find_energy": 0.0,
            "find_force": 0.0,
            "find_virial": 0.0,
            "find_atom_ener": 0.0,
            "find_atom_pref": 0.0,
            "find_hessian": 1.0,
        }

        loss, more_loss = loss_fn(1.0, natoms, model_dict, label_dict)

        np.testing.assert_allclose(to_numpy_array(more_loss["mae_h"]), 1.0)
        np.testing.assert_allclose(to_numpy_array(loss), 1.0)

    def test_mae_force_norm_has_finite_zero_gradient(self) -> None:
        loss_fn = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=1.0,
            limit_pref_f=1.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            loss_func="mae",
            f_use_norm=True,
        )
        nframes, natoms = 1, 2
        label_dict = {
            "energy": jnp.zeros((nframes,)),
            "force": jnp.zeros((nframes, natoms, 3)),
            "virial": jnp.zeros((nframes, 9)),
            "atom_ener": jnp.zeros((nframes, natoms)),
            "atom_pref": jnp.ones((nframes, natoms, 3)),
            "find_energy": 0.0,
            "find_force": 1.0,
            "find_virial": 0.0,
            "find_atom_ener": 0.0,
            "find_atom_pref": 0.0,
        }

        def eval_loss(force: jnp.ndarray) -> jnp.ndarray:
            model_dict = {
                "energy": jnp.zeros((nframes,)),
                "force": force,
                "virial": jnp.zeros((nframes, 9)),
                "atom_energy": jnp.zeros((nframes, natoms)),
            }
            loss, _ = loss_fn(1.0, natoms, model_dict, label_dict)
            return loss

        grad = jax.grad(eval_loss)(jnp.zeros((nframes, natoms, 3)))

        np.testing.assert_allclose(
            to_numpy_array(grad),
            np.zeros((nframes, natoms, 3)),
        )

    def test_hessian_loss_and_label_requirement(self) -> None:
        loss_fn = EnergyHessianLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_h=1.0,
            limit_pref_h=1.0,
        )
        nframes, natoms = 1, 2
        hessian_shape = (nframes, natoms * 3, natoms * 3)
        model_dict = {
            "energy": jnp.zeros((nframes,)),
            "force": jnp.zeros((nframes, natoms, 3)),
            "virial": jnp.zeros((nframes, 9)),
            "atom_energy": jnp.zeros((nframes, natoms)),
            "energy_derv_r_derv_r": jnp.zeros((nframes, 1, natoms * 3, natoms * 3)),
        }
        label_dict = {
            "energy": jnp.zeros((nframes,)),
            "force": jnp.zeros((nframes, natoms, 3)),
            "virial": jnp.zeros((nframes, 9)),
            "atom_ener": jnp.zeros((nframes, natoms)),
            "atom_pref": jnp.ones((nframes, natoms, 3)),
            "hessian": jnp.ones(hessian_shape),
            "find_energy": 0.0,
            "find_force": 0.0,
            "find_virial": 0.0,
            "find_atom_ener": 0.0,
            "find_atom_pref": 0.0,
            "find_hessian": 1.0,
        }

        loss, more_loss = loss_fn(1.0, natoms, model_dict, label_dict)
        requirement_keys = {item.key for item in loss_fn.label_requirement}

        self.assertIn("hessian", requirement_keys)
        self.assertIn("rmse_h", more_loss)
        self.assertLess(list(more_loss).index("rmse_h"), list(more_loss).index("rmse"))
        np.testing.assert_allclose(to_numpy_array(loss), 1.0)
        np.testing.assert_allclose(to_numpy_array(more_loss["rmse_h"]), 1.0)

    def test_mae_hessian_loss_reports_mae_h(self) -> None:
        loss_fn = EnergyHessianLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_h=1.0,
            limit_pref_h=1.0,
            loss_func="mae",
        )
        nframes, natoms = 1, 2
        hessian_shape = (nframes, natoms * 3, natoms * 3)
        model_dict = {
            "energy": jnp.zeros((nframes,)),
            "force": jnp.zeros((nframes, natoms, 3)),
            "virial": jnp.zeros((nframes, 9)),
            "atom_energy": jnp.zeros((nframes, natoms)),
            "hessian": jnp.zeros(hessian_shape),
        }
        label_dict = {
            "energy": jnp.zeros((nframes,)),
            "force": jnp.zeros((nframes, natoms, 3)),
            "virial": jnp.zeros((nframes, 9)),
            "atom_ener": jnp.zeros((nframes, natoms)),
            "atom_pref": jnp.ones((nframes, natoms, 3)),
            "hessian": jnp.ones(hessian_shape),
            "find_energy": 0.0,
            "find_force": 0.0,
            "find_virial": 0.0,
            "find_atom_ener": 0.0,
            "find_atom_pref": 0.0,
            "find_hessian": 1.0,
        }

        loss, more_loss = loss_fn(1.0, natoms, model_dict, label_dict)

        self.assertIn("mae_h", more_loss)
        self.assertNotIn("rmse_h", more_loss)
        self.assertLess(list(more_loss).index("mae_h"), list(more_loss).index("rmse"))
        np.testing.assert_allclose(to_numpy_array(loss), 1.0)
        np.testing.assert_allclose(to_numpy_array(more_loss["mae_h"]), 1.0)


if __name__ == "__main__":
    unittest.main()
