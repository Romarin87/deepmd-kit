# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest

from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.descriptor.sezm import (
    DescrptSeZM,
)
from deepmd.jax.fitting.fitting import (
    SeZMEnergyFittingNet,
)
from deepmd.jax.model.model import (
    get_model,
)


@unittest.skipIf(
    sys.version_info < (3, 10),
    "JAX requires Python 3.10 or later",
)
class TestSeZMDescriptor(unittest.TestCase):
    def test_aliases_and_roundtrip(self) -> None:
        for alias in ("SeZM", "sezm", "DPA4", "dpa4"):
            self.assertIs(BaseDescriptor.get_class_by_type(alias), DescrptSeZM)

        descriptor = DescrptSeZM(
            ntypes=3,
            sel=416,
            rcut=6.0,
            env_exp=[7, 5],
            channels=32,
            n_radial=16,
            radial_mlp=[0],
            use_env_seed=True,
            random_gamma=True,
            lmax=3,
            mmax=1,
            n_blocks=2,
            so2_layers=3,
            radial_so2_mode="degree_channel",
            radial_so2_rank=1,
            n_focus=2,
            focus_dim=0,
            n_atten_head=1,
            atten_f_mix=False,
            atten_v_proj=False,
            atten_o_proj=False,
            ffn_neurons=0,
            grid_mlp=False,
            ffn_blocks=2,
            sandwich_norm=[False, True, True, False],
            mlp_bias=False,
            layer_scale=False,
            s2_activation=[False, True],
            lebedev_quadrature=True,
            activation_function="silu",
            glu_activation=True,
            use_amp=True,
            precision="float32",
            seed=42,
            type_map=["H", "C", "O"],
        )
        self.assertEqual(descriptor.get_dim_out(), 32)
        self.assertEqual(descriptor.get_dim_emb(), 32)
        self.assertEqual(descriptor.get_rcut_smth(), 6.0)
        self.assertEqual(descriptor.get_sel(), [416])
        self.assertTrue(descriptor.mixed_types())

        restored = DescrptSeZM.deserialize(descriptor.serialize())
        self.assertEqual(restored.get_dim_out(), 32)
        self.assertEqual(restored.get_type_map(), ["H", "C", "O"])

    def test_model_type_defaults_to_sezm_energy_fitting(self) -> None:
        model = get_model(
            {
                "type": "SeZM",
                "type_map": ["H", "C", "O"],
                "descriptor": {
                    "type": "SeZM",
                    "sel": 416,
                    "rcut": 6.0,
                    "channels": 32,
                    "n_blocks": 2,
                    "so2_layers": 3,
                    "n_focus": 2,
                    "ffn_blocks": 2,
                    "precision": "float32",
                    "seed": 42,
                },
                "fitting_net": {
                    "neuron": [0],
                    "activation_function": "silu",
                    "precision": "float32",
                    "seed": 42,
                },
            }
        )
        self.assertIsInstance(model.atomic_model.descriptor, DescrptSeZM)
        self.assertIsInstance(model.atomic_model.fitting_net, SeZMEnergyFittingNet)

    def test_unsupported_paths_are_explicit(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "grid_mlp"):
            DescrptSeZM(ntypes=1, sel=16, grid_mlp=True)


if __name__ == "__main__":
    unittest.main()
