# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest

import numpy as np

from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.descriptor.sezm import (
    DescrptSeZM,
)
from deepmd.jax.env import (
    jnp,
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
        self.assertEqual(descriptor.radial_mlp, [32])

        restored = DescrptSeZM.deserialize(descriptor.serialize())
        self.assertEqual(restored.get_dim_out(), 32)
        self.assertEqual(restored.get_type_map(), ["H", "C", "O"])
        self.assertEqual(restored.radial_embedding.mlp_layers, [16, 32, 128])

    def test_base_embedding_and_edge_cache(self) -> None:
        descriptor = DescrptSeZM(
            ntypes=2,
            sel=2,
            rcut=6.0,
            channels=8,
            n_radial=4,
            radial_mlp=[0],
            n_blocks=2,
            so2_layers=3,
            precision="float32",
            seed=42,
        )
        distances = jnp.asarray([[0.5], [6.0]], dtype=jnp.float32)
        radial = descriptor.radial_basis(distances)
        self.assertEqual(radial.shape, (2, 4))
        self.assertTrue(bool(jnp.all(jnp.isfinite(radial))))
        np.testing.assert_allclose(np.asarray(radial[1]), np.zeros(4), atol=1e-6)

        atype = jnp.asarray([[0, -1]], dtype=jnp.int64)
        type_feat = descriptor.type_embedding(atype)
        self.assertEqual(type_feat.shape, (1, 2, 8))
        np.testing.assert_allclose(np.asarray(type_feat[0, 1]), np.zeros(8), atol=1e-6)

        coord_ext = jnp.asarray(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            dtype=jnp.float32,
        )
        atype_ext = jnp.asarray([[0, 1]], dtype=jnp.int64)
        nlist = jnp.asarray([[[1, -1], [0, -1]]], dtype=jnp.int64)
        mapping = jnp.asarray([[0, 1]], dtype=jnp.int64)
        cache = descriptor._build_edge_cache(coord_ext, atype_ext, nlist, mapping)
        self.assertEqual(cache.edge_vec.shape, (2, 3))
        self.assertEqual(cache.edge_rbf.shape, (2, 4))
        self.assertEqual(cache.edge_type_feat.shape, (2, 8))
        self.assertEqual(cache.inv_sqrt_deg.shape, (2, 1, 1))
        self.assertTrue(bool(jnp.all(jnp.isfinite(cache.edge_rbf))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(cache.inv_sqrt_deg))))

    def test_model_type_defaults_to_sezm_energy_fitting(self) -> None:
        model = get_model(
            {
                "type": "SeZM",
                "type_map": ["H", "C", "O"],
                "descriptor": {
                    "type": "SeZM",
                    "_comment": "ignored config note",
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
                    "_comment": "ignored config note",
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
