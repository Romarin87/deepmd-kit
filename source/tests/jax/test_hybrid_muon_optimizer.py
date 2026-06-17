# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the JAX HybridMuon optimizer."""

import importlib
import unittest

import optax

from deepmd.jax.env import (
    jax,
    jnp,
    nnx,
)
from deepmd.jax.optimizer.hybrid_muon import (
    HybridMuonRoute,
    _newton_schulz_standard,
    build_hybrid_muon_routes,
    hybrid_muon,
)
from deepmd.jax.utils.network import (
    ArrayAPIParam,
)

hybrid_muon_module = importlib.import_module("deepmd.jax.optimizer.hybrid_muon")


class TestJAXHybridMuonOptimizer(unittest.TestCase):
    """Check routing and finite updates for JAX HybridMuon."""

    def test_route_tree_matches_pytorch_rules(self) -> None:
        """Name and shape routing follows the PyTorch HybridMuon rules."""
        params = {
            "dense": {
                "kernel": jnp.ones((2, 3), dtype=jnp.float32),
                "bias": jnp.ones((3,), dtype=jnp.float32),
                "adamw_gate": jnp.ones((2, 2), dtype=jnp.float32),
                "tensor3": jnp.ones((2, 3, 4), dtype=jnp.float32),
            }
        }

        routes = build_hybrid_muon_routes(params, muon_mode="slice")

        self.assertEqual(routes["dense"]["kernel"].kind, "muon")
        self.assertEqual(routes["dense"]["kernel"].batch_size, 1)
        self.assertEqual(routes["dense"]["kernel"].rows, 2)
        self.assertEqual(routes["dense"]["kernel"].cols, 3)
        self.assertEqual(routes["dense"]["bias"].kind, "adam")
        self.assertEqual(routes["dense"]["adamw_gate"].kind, "adamw")
        self.assertEqual(routes["dense"]["tensor3"].kind, "muon")
        self.assertEqual(routes["dense"]["tensor3"].batch_size, 2)
        self.assertEqual(routes["dense"]["tensor3"].rows, 3)
        self.assertEqual(routes["dense"]["tensor3"].cols, 4)

        routes_2d = build_hybrid_muon_routes(params, muon_mode="2d")
        self.assertEqual(routes_2d["dense"]["tensor3"].kind, "adamw")

    def test_route_tree_preserves_nnx_state_structure(self) -> None:
        """Route trees match the NNX State treedef used by nnx.Optimizer."""
        params = nnx.State(
            {
                "kernel": ArrayAPIParam(jnp.ones((2, 3), dtype=jnp.float32)),
                "bias": nnx.Param(jnp.ones((3,), dtype=jnp.float32)),
            }
        )

        routes = build_hybrid_muon_routes(params, muon_mode="slice")
        route_leaves = jax.tree_util.tree_leaves(routes)

        self.assertEqual(
            jax.tree_util.tree_structure(params),
            jax.tree_util.tree_structure(routes),
        )
        self.assertTrue(all(isinstance(item, HybridMuonRoute) for item in route_leaves))
        self.assertEqual(routes["bias"].get_value().kind, "adam")
        self.assertEqual(routes["kernel"].get_value().kind, "muon")

    def test_update_is_finite_and_updates_magma_state(self) -> None:
        """One optimizer step produces finite updates for all routes."""
        params = {
            "dense": {
                "kernel": jnp.ones((2, 3), dtype=jnp.float32),
                "bias": jnp.ones((3,), dtype=jnp.float32),
                "adamw_gate": jnp.ones((2, 2), dtype=jnp.float32),
            }
        }
        grads = {
            "dense": {
                "kernel": jnp.full((2, 3), 0.25, dtype=jnp.float32),
                "bias": jnp.full((3,), 0.5, dtype=jnp.float32),
                "adamw_gate": jnp.full((2, 2), 0.75, dtype=jnp.float32),
            }
        }
        tx = hybrid_muon(
            learning_rate=0.1,
            params=params,
            weight_decay=0.01,
            muon_mode="slice",
            enable_gram=False,
            magma_muon=True,
        )

        state = tx.init(params)
        updates, state = tx.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)

        self.assertEqual(int(state.count), 1)
        self.assertTrue(jnp.all(jnp.isfinite(new_params["dense"]["kernel"])))
        self.assertTrue(jnp.all(jnp.isfinite(new_params["dense"]["bias"])))
        self.assertTrue(jnp.all(jnp.isfinite(new_params["dense"]["adamw_gate"])))
        self.assertEqual(state.magma_score["dense"]["kernel"].shape, (1,))
        self.assertTrue(jnp.all(jnp.isfinite(state.magma_score["dense"]["kernel"])))

    def test_extra_arg_learning_rate_overrides_schedule(self) -> None:
        """Trainer-provided learning rates scale HybridMuon updates."""
        params = {
            "dense": {
                "kernel": jnp.ones((2, 3), dtype=jnp.float32),
                "bias": jnp.ones((3,), dtype=jnp.float32),
            }
        }
        grads = {
            "dense": {
                "kernel": jnp.full((2, 3), 0.25, dtype=jnp.float32),
                "bias": jnp.full((3,), 0.5, dtype=jnp.float32),
            }
        }
        tx = hybrid_muon(
            learning_rate=1.0,
            params=params,
            weight_decay=0.0,
            muon_mode="slice",
            enable_gram=False,
            magma_muon=False,
        )

        state = tx.init(params)
        updates_large, _ = tx.update(
            grads,
            state,
            params,
            learning_rate=jnp.asarray(0.1, dtype=jnp.float32),
        )
        updates_small, _ = tx.update(
            grads,
            state,
            params,
            learning_rate=jnp.asarray(0.01, dtype=jnp.float32),
        )

        kernel_ratio = jnp.linalg.norm(
            updates_large["dense"]["kernel"]
        ) / jnp.linalg.norm(
            updates_small["dense"]["kernel"]
        )
        bias_ratio = jnp.linalg.norm(
            updates_large["dense"]["bias"]
        ) / jnp.linalg.norm(
            updates_small["dense"]["bias"]
        )
        self.assertAlmostEqual(float(kernel_ratio), 10.0, places=5)
        self.assertAlmostEqual(float(bias_ratio), 10.0, places=5)

    @unittest.skipUnless(
        hybrid_muon_module.PALLAS_AVAILABLE and jax.default_backend() == "gpu",
        "Pallas/Triton flash path requires a GPU backend",
    )
    def test_flash_newton_schulz_matches_standard(self) -> None:
        """Pallas flash path matches the ordinary Newton-Schulz path."""
        update = jnp.linspace(
            -0.5,
            0.5,
            16 * 24,
            dtype=jnp.float32,
        ).reshape(1, 16, 24)

        reference = _newton_schulz_standard(update, flash_muon=False)
        actual = _newton_schulz_standard(
            update,
            flash_muon=True,
            flash_min_dim=1,
        )

        self.assertTrue(jnp.all(jnp.isfinite(actual)))
        self.assertTrue(
            jnp.allclose(
                actual.astype(jnp.float32),
                reference.astype(jnp.float32),
                rtol=2e-2,
                atol=2e-2,
            )
        )


if __name__ == "__main__":
    unittest.main()
