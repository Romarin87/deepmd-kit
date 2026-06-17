# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import torch

from deepmd.pt.loss import (
    EnergyHessianStdLoss,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.training import (
    get_model_for_wrapper,
)
from deepmd.pt.utils import (
    env,
)


def _sezm_model_params(**overrides):
    params = {
        "type": "SeZM",
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "SeZM",
            "sel": [2, 2],
            "rcut": 3.0,
            "channels": 4,
            "n_focus": 1,
            "n_radial": 3,
            "radial_mlp": [6],
            "use_env_seed": True,
            "l_schedule": [1, 0],
            "mmax": 1,
            "so2_norm": False,
            "so2_layers": 1,
            "n_atten_head": 1,
            "sandwich_norm": [True, False, True, False],
            "ffn_neurons": 8,
            "ffn_blocks": 1,
            "s2_activation": [False, True],
            "mlp_bias": False,
            "layer_scale": False,
            "use_amp": False,
            "activation_function": "silu",
            "glu_activation": True,
            "precision": "float32",
            "seed": 7,
        },
        "fitting_net": {
            "neuron": [8],
            "activation_function": "silu",
            "precision": "float32",
            "seed": 7,
        },
        "use_compile": False,
    }
    params.update(overrides)
    return params


def _tiny_inputs():
    coord = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.1, 0.3, 0.0],
                [0.2, 1.5, 0.4],
                [1.7, 1.2, 0.2],
                [2.3, 0.1, 1.0],
            ]
        ],
        device=env.DEVICE,
        dtype=torch.float32,
    )
    atype = torch.tensor([[0, 1, 0, 1, 0]], device=env.DEVICE, dtype=torch.int64)
    box = torch.tensor(
        [[8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0]],
        device=env.DEVICE,
        dtype=torch.float32,
    )
    return coord, atype, box


class TestSeZMHessian(unittest.TestCase):
    def test_forward_outputs_finite_hessian(self):
        model = get_model(
            copy.deepcopy(_sezm_model_params(hessian_mode=True))
        ).to(env.DEVICE)
        coord, atype, box = _tiny_inputs()

        self.assertTrue(model.atomic_output_def()["energy"].r_hessian)
        out = model(coord, atype, box=box)

        self.assertIn("hessian", out)
        self.assertEqual(
            out["hessian"].shape, (1, coord.shape[1] * 3, coord.shape[1] * 3)
        )
        self.assertTrue(torch.isfinite(out["hessian"]).all())

    def test_hessian_loss_backward_has_finite_gradients(self):
        model = get_model(
            copy.deepcopy(_sezm_model_params(hessian_mode=True))
        ).to(env.DEVICE)
        model.train()
        coord, atype, box = _tiny_inputs()
        label = {
            "hessian": torch.zeros(
                1,
                coord.shape[1] * 3,
                coord.shape[1] * 3,
                device=env.DEVICE,
                dtype=torch.float32,
            ),
            "find_hessian": 1.0,
        }
        loss_fn = EnergyHessianStdLoss(
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

        _, loss, more_loss = loss_fn(
            {"coord": coord, "atype": atype, "box": box},
            model,
            label,
            coord.shape[1],
            1.0,
        )
        loss.backward()
        finite_grads = [
            torch.isfinite(param.grad).all()
            for param in model.parameters()
            if param.grad is not None
        ]

        self.assertIn("rmse_h", more_loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(finite_grads)
        self.assertTrue(all(finite_grads))

    def test_hessian_mae_loss_reports_mae_h(self):
        model = get_model(
            copy.deepcopy(_sezm_model_params(hessian_mode=True))
        ).to(env.DEVICE)
        coord, atype, box = _tiny_inputs()
        label = {
            "hessian": torch.zeros(
                1,
                coord.shape[1] * 3,
                coord.shape[1] * 3,
                device=env.DEVICE,
                dtype=torch.float32,
            ),
            "find_hessian": 1.0,
        }
        loss_fn = EnergyHessianStdLoss(
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

        _, loss, more_loss = loss_fn(
            {"coord": coord, "atype": atype, "box": box},
            model,
            label,
            coord.shape[1],
            1.0,
        )

        self.assertIn("mae_h", more_loss)
        self.assertNotIn("rmse_h", more_loss)
        self.assertNotIn("l2_hessian_loss", more_loss)
        self.assertLess(list(more_loss).index("mae_h"), list(more_loss).index("rmse"))
        self.assertTrue(torch.isfinite(loss))

    def test_multitask_hessian_flags_are_branch_local(self):
        branch_h = _sezm_model_params()
        branch_plain = _sezm_model_params()
        model_params = {"model_dict": {"h": branch_h, "plain": branch_plain}}
        models = get_model_for_wrapper(
            model_params,
            _loss_params={
                "h": {"type": "ener", "start_pref_h": 1.0, "limit_pref_h": 1.0},
                "plain": {"type": "ener"},
            },
        )

        self.assertTrue(model_params["model_dict"]["h"]["hessian_mode"])
        self.assertNotIn("hessian_mode", model_params["model_dict"]["plain"])
        self.assertTrue(models["h"].atomic_output_def()["energy"].r_hessian)
        self.assertFalse(models["plain"].atomic_output_def()["energy"].r_hessian)

    def test_single_task_hessian_flag_marks_model_params(self):
        model_params = _sezm_model_params()
        model = get_model_for_wrapper(
            model_params,
            _loss_params={"type": "ener", "start_pref_h": 1.0, "limit_pref_h": 1.0},
        )

        self.assertTrue(model_params["hessian_mode"])
        self.assertTrue(model.atomic_output_def()["energy"].r_hessian)

    def test_compile_is_disabled_in_hessian_mode(self):
        model = get_model(
            copy.deepcopy(_sezm_model_params(hessian_mode=True, use_compile=True))
        ).to(env.DEVICE)

        self.assertFalse(model.use_compile)
        self.assertFalse(model.should_use_compile())


if __name__ == "__main__":
    unittest.main()
