# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.loss import (
    EnergyHessianLoss,
)
from deepmd.jax.env import (
    jnp,
)


@unittest.skipIf(
    sys.version_info < (3, 10),
    "JAX requires Python 3.10 or later",
)
class TestEnergyHessianLoss(unittest.TestCase):
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
