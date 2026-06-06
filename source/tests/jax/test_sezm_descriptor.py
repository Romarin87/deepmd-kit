# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest

import numpy as np

from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.descriptor.sezm import (
    DescrptSeZM,
    WignerDCalculator,
    build_edge_quaternion,
    quaternion_to_rotation_matrix,
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
from deepmd.dpmodel.descriptor.sezm_lebedev import (
    load_lebedev_rule,
)
from deepmd.dpmodel.descriptor.sezm import (
    EdgeFeatureCache,
)
from deepmd.dpmodel.descriptor.sezm_indexing import (
    build_l_major_index,
    build_m_major_index,
    build_m_major_l_index,
    build_rotate_inv_rescale,
    get_so3_dim_of_lmax,
    map_degree_idx,
)
from deepmd.jax.descriptor.sezm_so3 import (
    ChannelLinear,
    FocusLinear,
    SO3Linear,
)
from deepmd.jax.descriptor.sezm_so2 import (
    DynamicRadialDegreeMixer,
    SO2Convolution,
    SO2Linear,
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

    def test_edge_frame_wigner_l1_and_lebedev(self) -> None:
        points, weights = load_lebedev_rule(3, float_precision="float32")
        self.assertEqual(points.shape, (6, 3))
        self.assertEqual(weights.shape, (6,))
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-6)

        edge_vec = jnp.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float32,
        )
        edge_quat = build_edge_quaternion(edge_vec, eps=1e-7)
        rot = quaternion_to_rotation_matrix(edge_quat)
        edge_unit = edge_vec / jnp.sqrt(jnp.sum(edge_vec * edge_vec, axis=-1, keepdims=True))
        aligned = jnp.einsum("nij,nj->ni", rot, edge_unit)
        expected = jnp.asarray(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float32,
        )
        np.testing.assert_allclose(np.asarray(aligned), np.asarray(expected), atol=2e-6)

        calc = WignerDCalculator(lmax=3, precision="float32")
        D_full, Dt_full = calc(edge_quat)
        self.assertEqual(D_full.shape, (2, 16, 16))
        ident = jnp.matmul(D_full, Dt_full)
        np.testing.assert_allclose(
            np.asarray(ident),
            np.asarray(jnp.broadcast_to(jnp.eye(16, dtype=jnp.float32), (2, 16, 16))),
            atol=1e-3,
        )

        edge_quat64 = build_edge_quaternion(edge_vec.astype(jnp.float64), eps=1e-7)
        D64, Dt64 = WignerDCalculator(lmax=3, precision="float64")(edge_quat64)
        ident64 = jnp.matmul(D64, Dt64)
        np.testing.assert_allclose(
            np.asarray(ident64),
            np.asarray(jnp.broadcast_to(jnp.eye(16, dtype=jnp.float64), (2, 16, 16))),
            atol=1e-10,
        )

        descriptor = DescrptSeZM(
            ntypes=2,
            sel=2,
            rcut=6.0,
            channels=8,
            n_radial=4,
            radial_mlp=[0],
            random_gamma=False,
            lmax=3,
            n_blocks=2,
            so2_layers=3,
            precision="float32",
            seed=42,
        )
        coord_ext = jnp.asarray(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            dtype=jnp.float32,
        )
        atype_ext = jnp.asarray([[0, 1]], dtype=jnp.int64)
        nlist = jnp.asarray([[[1, -1], [0, -1]]], dtype=jnp.int64)
        mapping = jnp.asarray([[0, 1]], dtype=jnp.int64)
        cache = descriptor._build_edge_cache(
            coord_ext,
            atype_ext,
            nlist,
            mapping,
            include_wigner=True,
        )
        self.assertEqual(cache.D_full.shape, (2, 16, 16))
        self.assertEqual(cache.Dt_full.shape, (2, 16, 16))

    def test_so3_indexing_and_linear_layers(self) -> None:
        self.assertEqual(get_so3_dim_of_lmax(3), 16)
        np.testing.assert_array_equal(
            map_degree_idx(3),
            np.asarray([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]),
        )
        np.testing.assert_array_equal(
            build_l_major_index(3, 1),
            np.asarray([0, 1, 2, 3, 5, 6, 7, 11, 12, 13]),
        )
        m_major = build_m_major_index(3, 1)
        np.testing.assert_array_equal(
            m_major,
            np.asarray([0, 2, 6, 12, 1, 5, 11, 3, 7, 13]),
        )
        degree_m = build_m_major_l_index(3, 1)
        np.testing.assert_array_equal(
            degree_m,
            np.asarray([0, 1, 2, 3, 1, 2, 3, 1, 2, 3]),
        )
        rescale = build_rotate_inv_rescale(
            3,
            1,
            jnp.asarray(degree_m, dtype=jnp.int64),
            dtype=jnp.float32,
        )
        expected_rescale = np.asarray(
            [
                1.0,
                1.0,
                np.sqrt(5.0 / 3.0),
                np.sqrt(7.0 / 3.0),
                1.0,
                np.sqrt(5.0 / 3.0),
                np.sqrt(7.0 / 3.0),
                1.0,
                np.sqrt(5.0 / 3.0),
                np.sqrt(7.0 / 3.0),
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(np.asarray(rescale), expected_rescale, atol=1e-6)

        chan = ChannelLinear(
            in_channels=2,
            out_channels=3,
            precision="float32",
            seed=7,
        )
        chan_out = chan(jnp.ones((4, 2), dtype=jnp.float32))
        self.assertEqual(chan_out.shape, (4, 3))
        self.assertTrue(bool(jnp.all(jnp.isfinite(chan_out))))

        focus = FocusLinear(
            in_channels=2,
            out_channels=3,
            n_focus=2,
            precision="float32",
            seed=7,
        )
        focus_out = focus(jnp.ones((4, 2, 2), dtype=jnp.float32))
        self.assertEqual(focus_out.shape, (4, 2, 3))
        self.assertTrue(bool(jnp.all(jnp.isfinite(focus_out))))

        so3 = SO3Linear(
            lmax=2,
            in_channels=2,
            out_channels=3,
            n_focus=2,
            precision="float32",
            mlp_bias=True,
            seed=7,
        )
        so3.weight = jnp.zeros_like(so3.weight[...])
        so3.bias = jnp.arange(6, dtype=jnp.float32)
        so3_out = so3(jnp.ones((4, 9, 2, 2), dtype=jnp.float32))
        self.assertEqual(so3_out.shape, (4, 9, 2, 3))
        expected_bias = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.testing.assert_allclose(
            np.asarray(so3_out[:, 0, :, :]),
            np.broadcast_to(expected_bias, (4, 2, 3)),
        )
        np.testing.assert_allclose(np.asarray(so3_out[:, 1:, :, :]), 0.0)

        restored = SO3Linear.deserialize(so3.serialize())
        self.assertEqual(restored.expand_index.shape, (9,))
        np.testing.assert_allclose(
            np.asarray(restored(jnp.ones((1, 9, 2, 2), dtype=jnp.float32))[:, 0]),
            np.broadcast_to(expected_bias, (1, 2, 3)),
        )

    def test_so2_linear_layout_and_block_coupling(self) -> None:
        so2 = SO2Linear(
            lmax=2,
            mmax=1,
            in_channels=1,
            out_channels=1,
            n_focus=1,
            precision="float32",
            mlp_bias=True,
            seed=7,
        )
        self.assertEqual(so2.reduced_dim, 7)
        so2.weight_m0 = jnp.eye(3, dtype=jnp.float32)
        so2.bias0 = jnp.asarray([10.0], dtype=jnp.float32)
        so2.weight_m = [
            jnp.concatenate(
                [
                    jnp.eye(2, dtype=jnp.float32),
                    2.0 * jnp.eye(2, dtype=jnp.float32),
                ],
                axis=1,
            )
        ]
        x = jnp.asarray(
            [[[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]]],
            dtype=jnp.float32,
        )
        y = so2(x)
        self.assertEqual(y.shape, (1, 1, 7, 1))
        expected = np.asarray(
            [[[[11.0], [2.0], [3.0], [-8.0], [-9.0], [14.0], [17.0]]]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(np.asarray(y), expected, atol=1e-6)

        restored = SO2Linear.deserialize(so2.serialize())
        self.assertEqual(restored.reduced_dim, 7)
        np.testing.assert_allclose(np.asarray(restored(x)), expected, atol=1e-6)

    def test_dynamic_radial_degree_mixer(self) -> None:
        mixer = DynamicRadialDegreeMixer(
            lmax=2,
            mmax=1,
            channels=1,
            mode="degree",
            precision="float32",
            seed=7,
        )
        self.assertEqual(mixer.reduced_dim, 7)
        self.assertEqual(mixer.degree_kernel_size, 13)
        compact_identity = jnp.asarray(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            dtype=jnp.float32,
        )
        mixer.weight = jnp.zeros((3, 13), dtype=jnp.float32)
        mixer.weight = mixer.weight[...].at[0].set(compact_identity)
        x = jnp.arange(1, 8, dtype=jnp.float32).reshape(1, 7, 1)
        radial = jnp.zeros_like(x).at[0, 0, 0].set(1.0)
        np.testing.assert_allclose(np.asarray(mixer(x, radial)), np.asarray(x))

        restored = DynamicRadialDegreeMixer.deserialize(mixer.serialize())
        np.testing.assert_allclose(np.asarray(restored(x, radial)), np.asarray(x))

        rank_mixer = DynamicRadialDegreeMixer(
            lmax=1,
            mmax=1,
            channels=2,
            mode="degree_channel",
            rank=1,
            precision="float32",
            seed=7,
        )
        rank_mixer.weight = jnp.zeros((4, 5), dtype=jnp.float32)
        rank_mixer.weight = rank_mixer.weight[...].at[0].set(
            jnp.asarray([1.0, 0.0, 0.0, 1.0, 1.0], dtype=jnp.float32)
        )
        rank_mixer.channel_basis = jnp.asarray([[2.0, 3.0]], dtype=jnp.float32)
        x_rank = jnp.arange(1, 9, dtype=jnp.float32).reshape(1, 4, 2)
        radial_rank = jnp.zeros_like(x_rank).at[0, 0, 0].set(1.0)
        expected_rank = x_rank * jnp.asarray([2.0, 3.0], dtype=jnp.float32)
        np.testing.assert_allclose(
            np.asarray(rank_mixer(x_rank, radial_rank)),
            np.asarray(expected_rank),
        )

    def test_so2_convolution_minimal_message_path(self) -> None:
        conv = SO2Convolution(
            lmax=1,
            mmax=1,
            channels=1,
            n_focus=1,
            so2_layers=1,
            n_atten_head=0,
            radial_so2_mode="none",
            precision="float32",
            seed=7,
        )
        conv.pre_focus_mix.weight = jnp.ones((2, 1, 1), dtype=jnp.float32)
        conv.post_focus_mix.weight = jnp.ones((2, 1, 1), dtype=jnp.float32)
        conv.so2_linears[0].weight_m0 = jnp.zeros((2, 2), dtype=jnp.float32)
        conv.so2_linears[0].weight_m = [jnp.zeros((1, 2), dtype=jnp.float32)]

        eye = jnp.eye(4, dtype=jnp.float32).reshape(1, 4, 4)
        edge_cache = EdgeFeatureCache(
            src=jnp.asarray([1], dtype=jnp.int64),
            dst=jnp.asarray([0], dtype=jnp.int64),
            edge_type_feat=jnp.zeros((1, 1), dtype=jnp.float32),
            edge_vec=jnp.zeros((1, 3), dtype=jnp.float32),
            edge_len=jnp.ones((1, 1), dtype=jnp.float32),
            edge_rbf=jnp.ones((1, 1), dtype=jnp.float32),
            edge_env=jnp.ones((1, 1), dtype=jnp.float32),
            deg=jnp.ones((2,), dtype=jnp.float32),
            inv_sqrt_deg=jnp.ones((2, 1, 1), dtype=jnp.float32),
            D_full=eye,
            Dt_full=eye,
        )
        x = jnp.asarray(
            [
                [[0.0], [0.0], [0.0], [0.0]],
                [[1.0], [2.0], [3.0], [4.0]],
            ],
            dtype=jnp.float32,
        )
        radial_feat = jnp.ones((1, 2, 1), dtype=jnp.float32)
        y = conv(x, edge_cache, radial_feat)
        self.assertEqual(y.shape, (2, 4, 1))
        np.testing.assert_allclose(np.asarray(y[0]), np.asarray(x[1]), atol=1e-6)
        np.testing.assert_allclose(np.asarray(y[1]), 0.0, atol=1e-6)

        restored = SO2Convolution.deserialize(conv.serialize())
        y_restored = restored(x, edge_cache, radial_feat)
        np.testing.assert_allclose(np.asarray(y_restored), np.asarray(y), atol=1e-6)

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
