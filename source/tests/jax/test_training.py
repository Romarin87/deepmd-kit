# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end tests for the local JAX training entrypoint."""

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import (
    Path,
)
from unittest.mock import (
    patch,
)

import numpy as np
from jax.sharding import (
    PartitionSpec as P,
)

from deepmd.jax.entrypoints.freeze import (
    freeze,
)
from deepmd.jax.entrypoints.main import (
    main,
)
from deepmd.jax.model.model import (
    get_model,
)
from deepmd.jax.train.trainer import (
    DPTrainer,
    _data_partition_spec,
    trim_mixed_padding_batch,
)
from deepmd.jax.utils.serialization import (
    serialize_from_file,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    convert_optimizer_v31_to_v32,
    update_deepmd_input,
)

MODEL_SE_E2_A = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [6, 12, 1],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [2, 4, 8],
        "resnet_dt": False,
        "axis_neuron": 2,
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [4, 4, 4],
        "resnet_dt": True,
        "seed": 1,
    },
    "data_stat_nbatch": 1,
}


TRAINING_SCRIPT = """
from pathlib import Path
from unittest.mock import patch

from deepmd.main import main

with patch("deepmd.jax.entrypoints.train.SummaryPrinter.__call__"):
    main(["--jax", "train", "input.json", "--log-level", "2"])

for path in ["out.json", "lcurve.out", "checkpoint", "model-1.jax"]:
    if not Path(path).exists():
        raise FileNotFoundError(path)
"""


_LCURVE_STEP_RE = re.compile(r"^\s*(\d+)\b")


def _lcurve_steps(path: Path) -> set[int]:
    """Return integer step numbers written in an lcurve.out file."""
    steps: set[int] = set()
    for line in path.read_text().splitlines():
        match = _LCURVE_STEP_RE.match(line)
        if match:
            steps.add(int(match.group(1)))
    return steps


class TestJAXTraining(unittest.TestCase):
    """Regression tests for complete JAX training runs."""

    def setUp(self) -> None:
        """Create a temporary work directory with a one-step training input."""
        self.work_dir = Path(tempfile.mkdtemp())
        self.cwd = Path.cwd()
        os.chdir(self.work_dir)

        source_dir = Path(__file__).resolve().parents[1] / "pt" / "water"
        shutil.copytree(source_dir, self.work_dir / "water")
        data_file = [str(self.work_dir / "water" / "data" / "single")]

        with (self.work_dir / "water" / "se_atten.json").open() as f:
            self.config = json.load(f)
        self.config = convert_optimizer_v31_to_v32(self.config, warning=False)
        self.config["model"] = MODEL_SE_E2_A
        self.config["model"]["data_stat_nbatch"] = 1
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["disp_freq"] = 1
        self.config["training"]["save_freq"] = 1
        self.config["training"]["save_ckpt"] = "model"

        self.input_file = self.work_dir / "input.json"
        with self.input_file.open("w") as f:
            json.dump(self.config, f)

    def tearDown(self) -> None:
        """Remove temporary training outputs."""
        os.chdir(self.cwd)
        shutil.rmtree(self.work_dir)

    def test_trim_mixed_padding_batch_slices_hessian_labels(self) -> None:
        """Trim mixed padding atoms before tracing JAX Hessian training."""
        batch = {
            "type": np.asarray([[0, 1, -1, -1]], dtype=np.int32),
            "coord": np.arange(12, dtype=np.float32).reshape(1, 12),
            "force": np.arange(12, dtype=np.float32).reshape(1, 12) + 100.0,
            "hessian": np.arange(144, dtype=np.float32).reshape(1, 144),
            "natoms_vec": np.asarray([4, 4, 1, 1, 0], dtype=np.int32),
            "real_natoms_vec": np.asarray([[4, 4, 1, 1, 0]], dtype=np.int32),
        }
        trimmed = trim_mixed_padding_batch(batch)

        self.assertEqual(trimmed["type"].shape, (1, 2))
        self.assertEqual(trimmed["coord"].shape, (1, 6))
        self.assertEqual(trimmed["force"].shape, (1, 6))
        self.assertEqual(trimmed["hessian"].shape, (1, 36))
        np.testing.assert_array_equal(trimmed["natoms_vec"], [2, 2, 1, 1, 0])
        np.testing.assert_array_equal(trimmed["real_natoms_vec"], [[2, 2, 1, 1, 0]])

    def test_train_entrypoint_runs_one_step_from_scratch(self) -> None:
        """Run local JAX training in a child process and check artifacts."""
        if os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ):
            # TODO: Re-enable this in GitHub CUDA CI once the hosted/self-hosted
            # runner JAX/PJRT abort is understood. The same test passes on a
            # local GPU, but the GitHub Actions CUDA job can terminate with
            # CUDA_ERROR_LAUNCH_FAILED while PJRT releases device buffers.
            self.skipTest(
                "JAX training is temporarily skipped on GitHub Actions CUDA runners"
            )

        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(TRAINING_SCRIPT)],
            cwd=self.work_dir,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn(1, _lcurve_steps(self.work_dir / "lcurve.out"))

    def test_hessian_loss_enables_model_and_label_requirement(self) -> None:
        """Hessian loss turns on model Hessian output before data loading."""
        config = copy.deepcopy(self.config)
        config["loss"]["start_pref_h"] = 1.0
        config["loss"]["limit_pref_h"] = 1.0
        jdata = update_deepmd_input(config, warning=False)
        jdata = normalize(jdata)

        trainer = DPTrainer(jdata)
        requirement_keys = {item.key for item in trainer.data_requirements}

        self.assertTrue(trainer.has_hessian)
        self.assertTrue(trainer.model.atomic_output_def()["energy"].r_hessian)
        self.assertTrue(trainer.model_def_script["hessian_mode"])
        self.assertIn("hessian", requirement_keys)

    def test_positive_start_pref_h_enables_hessian_mode(self) -> None:
        """Match the DPA3 JAX Hessian switch: start_pref_h controls the mode."""
        config = copy.deepcopy(self.config)
        config["loss"]["start_pref_h"] = 1.0
        config["loss"]["limit_pref_h"] = 0.0
        jdata = update_deepmd_input(config, warning=False)
        jdata = normalize(jdata)

        trainer = DPTrainer(jdata)

        self.assertTrue(trainer.has_hessian)
        self.assertTrue(trainer.model.atomic_output_def()["energy"].r_hessian)
        self.assertTrue(trainer.model_def_script["hessian_mode"])

    def test_wsd_learning_rate_is_supported(self) -> None:
        """JAX trainer accepts the shared WSD learning rate schedule."""
        config = copy.deepcopy(self.config)
        config["training"]["numb_steps"] = 10
        config["learning_rate"] = {
            "type": "wsd",
            "start_lr": 1.0,
            "stop_lr": 0.1,
            "decay_phase_ratio": 0.2,
            "decay_type": "linear",
        }
        jdata = update_deepmd_input(config, warning=False)
        jdata = normalize(jdata)

        trainer = DPTrainer(jdata)

        self.assertEqual(trainer.lr.__class__.__name__, "LearningRateWSD")
        self.assertAlmostEqual(trainer.lr.value(0), 1.0)
        self.assertAlmostEqual(trainer.lr.value(8), 1.0)
        self.assertAlmostEqual(trainer.lr.value(9), 0.55)
        self.assertAlmostEqual(trainer.lr.value(10), 0.1)

    def test_ema_and_zero_stage_config_are_supported(self) -> None:
        """JAX trainer consumes shared EMA/zero_stage training options."""
        config = copy.deepcopy(self.config)
        config["training"]["enable_ema"] = True
        config["training"]["ema_decay"] = 0.9
        config["training"]["ema_ckpt_keep"] = 2
        config["training"]["zero_stage"] = 1
        jdata = update_deepmd_input(config, warning=False)
        jdata = normalize(jdata)

        trainer = DPTrainer(jdata)

        self.assertTrue(trainer.enable_ema)
        self.assertEqual(trainer.ema_decay, 0.9)
        self.assertEqual(trainer.ema_ckpt_keep, 2)
        self.assertIsNotNone(trainer.model_ema)
        self.assertEqual(trainer.zero_stage, 0)

    def test_jax_parallel_mode_config_is_supported(self) -> None:
        """JAX trainer consumes the local parallel-mode selector."""
        config = copy.deepcopy(self.config)
        config["training"]["jax_parallel_mode"] = "data_parallel"
        jdata = update_deepmd_input(config, warning=False)
        jdata = normalize(jdata)

        trainer = DPTrainer(jdata)

        self.assertEqual(trainer.jax_parallel_mode, "data")

    def test_jax_parallel_partition_specs(self) -> None:
        """Data mode does not shard atom/Hessian axes."""
        force = np.zeros((2, 48, 3))
        hessian = np.zeros((2, 144, 144))

        self.assertEqual(
            _data_partition_spec("force", force, data_axis_size=2),
            P("data"),
        )
        self.assertEqual(
            _data_partition_spec("hessian", hessian, data_axis_size=2),
            P("data"),
        )
        self.assertEqual(
            _data_partition_spec("force", force, natoms_axis_size=2),
            P(None, "natoms"),
        )
        self.assertEqual(
            _data_partition_spec("force", force[:1], data_axis_size=2),
            P(),
        )

    def test_hybrid_muon_optimizer_config_is_supported(self) -> None:
        """JAX trainer consumes the shared HybridMuon optimizer config."""
        config = copy.deepcopy(self.config)
        config["optimizer"] = {
            "type": "HybridMuon",
            "muon_mode": "slice",
            "magma_muon": True,
            "lr_adjust": 0.0,
            "weight_decay": 0.001,
        }
        jdata = update_deepmd_input(config, warning=False)
        jdata = normalize(jdata)

        trainer = DPTrainer(jdata)
        dummy_train_data = type(
            "DummyTrainData",
            (),
            {"nbatches": [1], "sys_probs": [1.0]},
        )()
        trainer._resolve_num_steps(dummy_train_data)
        tx = trainer._build_optimizer_tx(trainer.model)

        self.assertEqual(trainer.opt_type, "HybridMuon")
        self.assertEqual(trainer.optimizer_param["muon_mode"], "slice")
        self.assertEqual(tx.__class__.__name__, "GradientTransformation")

    def test_ema_checkpoint_save_and_restart(self) -> None:
        """JAX regular checkpoints carry EMA state and save EMA-weight ckpts."""
        config = copy.deepcopy(self.config)
        config["training"]["enable_ema"] = True
        config["training"]["ema_decay"] = 0.9
        jdata = update_deepmd_input(config, warning=False)
        jdata = normalize(jdata)

        trainer = DPTrainer(jdata)
        trainer._save_checkpoint(trainer.model, 1)
        trainer._save_ema_checkpoint(trainer.model, 1)

        for path in [
            "model-1.jax",
            "model.jax",
            "model_ema-1.jax",
            "model_ema.jax",
        ]:
            self.assertTrue(Path(path).exists(), path)
        self.assertEqual(Path("checkpoint").read_text(), "model.jax")

        saved_data = serialize_from_file("model-1.jax")
        self.assertIn("ema", saved_data)

        restarted = DPTrainer(jdata, restart="model-1.jax")
        self.assertIsNotNone(restarted.model_ema)
        self.assertEqual(restarted.start_step, 1)

    def test_num_epoch_resolves_training_steps(self) -> None:
        """JAX trainer resolves epoch-based input after data is available."""

        class DummyTrainData:
            nbatches = [3, 7]
            sys_probs = [0.5, 0.5]

        for epoch_key in ("numb_epoch", "num_epoch", "num_epochs"):
            with self.subTest(epoch_key=epoch_key):
                config = copy.deepcopy(self.config)
                config["training"].pop("numb_steps")
                config["training"][epoch_key] = 1.5
                config["learning_rate"] = {
                    "type": "wsd",
                    "start_lr": 1.0,
                    "stop_lr": 0.1,
                    "decay_phase_ratio": 0.2,
                    "decay_type": "linear",
                }
                jdata = update_deepmd_input(config, warning=False)
                jdata = normalize(jdata)

                trainer = DPTrainer(jdata)

                self.assertIsNone(trainer.lr)
                trainer._resolve_num_steps(DummyTrainData())
                self.assertEqual(trainer.num_steps, 21)
                self.assertEqual(trainer.lr.__class__.__name__, "LearningRateWSD")

    def test_num_epoch_uses_pytorch_partial_batches(self) -> None:
        """JAX epoch steps match PyTorch DataLoader(drop_last=False)."""

        class DummySystem:
            def __init__(self, nframes: int) -> None:
                self.nframes = nframes

        class DummyTrainData:
            nbatches = [90, 1005]
            sys_probs = [90 / 1095, 1005 / 1095]
            batch_size = [2, 2]
            data_systems = [DummySystem(180), DummySystem(2011)]

        config = copy.deepcopy(self.config)
        config["training"].pop("numb_steps")
        config["training"]["num_epochs"] = 1
        jdata = update_deepmd_input(config, warning=False)
        jdata = normalize(jdata)

        trainer = DPTrainer(jdata)
        train_data = DummyTrainData()
        trainer._resolve_num_steps(train_data)

        self.assertEqual(train_data.nbatches, [90, 1006])
        self.assertEqual(trainer.num_steps, 1096)

    def test_model_factory_restores_hessian_mode(self) -> None:
        """Checkpoint model definitions keep Hessian output mode."""
        model_params = copy.deepcopy(MODEL_SE_E2_A)
        model_params["hessian_mode"] = True

        model = get_model(model_params)

        self.assertTrue(model.atomic_output_def()["energy"].r_hessian)
        self.assertTrue(model.model_output_def()["energy"].r_hessian)

    @patch("deepmd.jax.entrypoints.freeze.deserialize_to_file")
    @patch("deepmd.jax.entrypoints.freeze.serialize_from_file")
    def test_freeze_entrypoint_uses_checkpoint_pointer(
        self, serialize_from_file, deserialize_to_file
    ) -> None:
        """Freeze resolves the stable checkpoint pointer without Hessian options."""
        checkpoint_dir = self.work_dir / "ckpt"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "checkpoint").write_text("model-1.jax")
        serialize_from_file.return_value = {"model": {}, "model_def_script": {}}

        freeze(checkpoint_folder=str(checkpoint_dir), output="frozen_model")

        serialize_from_file.assert_called_once_with(str(checkpoint_dir / "model-1.jax"))
        deserialize_to_file.assert_called_once_with(
            "frozen_model.hlo", serialize_from_file.return_value
        )

    @patch("deepmd.jax.entrypoints.main.freeze")
    def test_main_dispatches_freeze(self, freeze_entrypoint) -> None:
        """JAX CLI main imports and dispatches the freeze command."""
        args = argparse.Namespace(
            command="freeze",
            log_level=2,
            log_path=None,
            checkpoint_folder=".",
            output="frozen_model",
        )

        main(args)

        freeze_entrypoint.assert_called_once()
