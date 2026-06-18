# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import os
import tempfile
import unittest

import numpy as np
import torch

from deepmd.infer import (
    DeepPot,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils import (
    env,
)


def _model_params() -> dict:
    return {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [4, 8],
            "rcut_smth": 0.5,
            "rcut": 4.0,
            "neuron": [4, 8],
            "axis_neuron": 4,
            "seed": 1,
        },
        "fitting_net": {
            "neuron": [8, 8],
            "resnet_dt": True,
            "seed": 1,
        },
    }


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.array(
        [[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.8, 0.0]]],
        dtype=np.float64,
    )
    cells = np.eye(3, dtype=np.float64).reshape(1, 9) * 8.0
    atom_types = np.array([0, 1, 1], dtype=np.int32)
    return coords, cells, atom_types


def _save_multitask_checkpoint(model_params: dict, wrapper_params: dict) -> str:
    model = get_model(copy.deepcopy(model_params)).to(env.DEVICE).eval()
    wrapper = ModelWrapper(
        {"head_a": model},
        model_params=wrapper_params,
    )
    path = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
    torch.save({"model": wrapper.state_dict()}, path)
    return path


class TestPtHessianCheckpointInfer(unittest.TestCase):
    def tearDown(self) -> None:
        path = getattr(self, "pt_path", None)
        if path and os.path.exists(path):
            os.unlink(path)

    def _assert_deeppot_hessian_is_finite(self, path: str) -> None:
        coords, cells, atom_types = _inputs()
        dp = DeepPot(path, head="head_a", auto_batch_size=False)
        self.assertTrue(dp.has_hessian)
        result = dp.eval(coords, cells, atom_types, atomic=False)
        self.assertEqual(len(result), 4)
        hessian = result[-1]
        self.assertEqual(hessian.shape, (1, 9, 9))
        self.assertTrue(np.isfinite(hessian).all())

    def test_multitask_head_hessian_mode_returns_finite_hessian(self) -> None:
        params = _model_params()
        params["hessian_mode"] = True
        self.pt_path = _save_multitask_checkpoint(
            params,
            {"model_dict": {"head_a": params}},
        )
        self._assert_deeppot_hessian_is_finite(self.pt_path)

    def test_multitask_global_hessian_mode_enables_selected_head(self) -> None:
        params = _model_params()
        self.pt_path = _save_multitask_checkpoint(
            params,
            {"hessian_mode": True, "model_dict": {"head_a": params}},
        )
        self._assert_deeppot_hessian_is_finite(self.pt_path)


if __name__ == "__main__":
    unittest.main()
