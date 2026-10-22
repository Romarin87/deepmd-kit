# SPDX-License-Identifier: LGPL-3.0-or-later
"""Optimizer numerics and explicit multi-device placement regressions."""

import numpy as np
import optax
import pytest

from deepmd.jax.env import jax, jnp, nnx
from deepmd.jax.train.trainer import make_optimizer


@pytest.mark.parametrize("kind", ["Adam", "AdamW"])
@pytest.mark.parametrize("clip", [0.0, 0.2])
def test_optimizer_matches_reference(kind, clip):
    config = {
        "type": kind,
        "adam_beta1": 0.8,
        "adam_beta2": 0.95,
        "adam_eps": 1e-6,
        "weight_decay": 0.07,
    }
    actual = make_optimizer(config, gradient_max_norm=clip)
    p = {"w": jnp.array([1.0, -2.0])}
    state = actual.init(p)
    # Independently update the moments, including coupled/decoupled decay.
    expected = np.array([1.0, -2.0])
    m, v = np.zeros(2), np.zeros(2)
    for step, lr in enumerate([0.03, 0.01, 0.002], 1):
        grad = np.array([0.4, -0.7]) * step
        updates, state = actual.update(
            {"w": jnp.array(grad)}, state, p, learning_rate=lr
        )
        p = optax.apply_updates(p, updates)
        if clip:
            grad *= min(1.0, clip / np.linalg.norm(grad))
        if kind == "Adam":
            grad += 0.07 * expected
        m = 0.8 * m + 0.2 * grad
        v = 0.95 * v + 0.05 * grad**2
        update = (m / (1 - 0.8**step)) / (np.sqrt(v / (1 - 0.95**step)) + 1e-6)
        if kind == "AdamW":
            update += 0.07 * expected
        expected -= lr * update
        np.testing.assert_allclose(p["w"], expected, atol=1e-12)


def test_unsupported_optimizer():
    with pytest.raises(ValueError, match="unsupported"):
        make_optimizer({"type": "LKF"})


@pytest.mark.skipif(
    jax.local_device_count() < 2,
    reason="run with XLA_FLAGS=--xla_force_host_platform_device_count=2",
)
def test_atom_sharded_hessian_update_matches_single_device():
    from .test_dpa4_energy_hessian import make_model, COORD, ATYPE
    from deepmd.jax.train.trainer import _evaluate_model_dict, prepare_input
    from deepmd.jax.utils.parallel import (
        make_training_mesh,
        replicate_model,
        shard_prepared_batch,
    )

    coord = jnp.concatenate((COORD, jnp.zeros((1, 1, 3))), axis=1)
    atype = jnp.concatenate((ATYPE, jnp.array([[-1]])), axis=1)
    single = make_model()
    single.enable_hessian()
    parallel = nnx.clone(single)
    mesh = make_training_mesh()
    replicate_model(parallel, mesh)
    inputs = prepare_input(
        rcut=single.get_rcut(),
        sel=single.get_sel(),
        coord=coord,
        atype=atype,
        box=None,
        fparam=None,
        aparam=None,
    )
    labels = {"type": atype, "charge_spin": jnp.array([[0.0, 1.0]]), "find_force": True}
    prepared = shard_prepared_batch((labels, *inputs), mesh)
    assert prepared[1].sharding.spec[1] == "natoms"
    assert len(prepared[0]["charge_spin"].sharding.spec) == 1

    @nnx.jit
    def step(model, optimizer, inputs):
        def loss(m):
            result = _evaluate_model_dict(m, *inputs)
            return jnp.mean((result["hessian"] - jnp.eye(12)) ** 2) + jnp.mean(
                result["force"] ** 2
            )

        value, grad = nnx.value_and_grad(loss)(model)
        optimizer.update(model, grad, learning_rate=0.001)
        return value

    tx = make_optimizer({"type": "AdamW"})
    a = nnx.Optimizer(single, tx, wrt=nnx.Param)
    b = nnx.Optimizer(parallel, tx, wrt=nnx.Param)
    np.testing.assert_allclose(
        step(single, a, inputs), step(parallel, b, prepared[1:]), atol=1e-12
    )
    for x, y in zip(
        jax.tree.leaves(nnx.state(single)),
        jax.tree.leaves(nnx.state(parallel)),
        strict=True,
    ):
        np.testing.assert_allclose(x, y, atol=2e-8)
    # Odd atom counts must remain legal: use frame-only placement.
    odd = prepare_input(
        rcut=single.get_rcut(),
        sel=single.get_sel(),
        coord=COORD,
        atype=ATYPE,
        box=None,
        fparam=None,
        aparam=None,
    )
    odd_prepared = shard_prepared_batch(({"type": ATYPE}, *odd), mesh)
    assert len(odd_prepared[1].sharding.spec) == 1
