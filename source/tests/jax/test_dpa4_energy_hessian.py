# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 model, higher derivatives, and optimizer integration regressions."""

import numpy as np
import optax
import pytest

from deepmd.jax.env import jax, jnp, nnx
from deepmd.jax.model.base_model import BaseModel
from deepmd.jax.model.model import get_model


def model_config():
    return {
        "type": "dpa4",
        "type_map": ["H", "O"],
        "descriptor": {
            "sel": 3,
            "rcut": 3.0,
            "channels": 2,
            "n_radial": 2,
            "lmax": 1,
            "mmax": 1,
            "n_blocks": 1,
            "mixing_layers": 1,
            "edge_norm": False,
            "sandwich_norm": [False] * 4,
            "precision": "float64",
            "seed": 1,
        },
        "fitting_net": {"neuron": [4], "precision": "float64", "seed": 2},
    }


def make_model(config=None):
    model = get_model(model_config() if config is None else config)
    # Residual output projections start at zero. Move off that initialization
    # so nonzero forces/Hessians exercise every geometry-dependent branch.
    state = nnx.state(model, nnx.Param)
    rng = np.random.default_rng(42)
    for _, variable in nnx.to_flat_state(state):
        variable.value = variable.value + jnp.asarray(
            rng.normal(scale=0.03, size=variable.value.shape)
        )
    nnx.update(model, state)
    return model


COORD = jnp.array([[[0.2, 0.1, 0.3], [1.1, 0.4, 0.2], [0.4, 1.2, 0.5]]])
ATYPE = jnp.array([[0, 1, 0]])


def test_energy_force_hessian_and_serialization():
    model = make_model()
    restored = BaseModel.deserialize(model.serialize())
    evaluate = nnx.jit(lambda m, x: m(x, ATYPE))
    force_result = evaluate(model, COORD)
    assert np.linalg.norm(force_result["force"]) > 1e-6
    for key in ("energy", "force"):
        np.testing.assert_allclose(
            force_result[key],
            evaluate(restored, COORD)[key],
            atol=1e-12,
        )
    model.enable_hessian()
    result = evaluate(model, COORD)
    hessian = np.asarray(result["hessian"])[0]
    assert np.linalg.norm(hessian) > 1e-6
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-10)
    np.testing.assert_allclose(hessian.reshape(3, 3, 3, 3).sum(axis=2), 0, atol=1e-10)
    delta = 1e-4
    finite_difference = []
    for index in range(9):
        shift = (
            jnp.zeros_like(COORD).reshape(-1).at[index].set(delta).reshape(COORD.shape)
        )
        plus, minus = (
            evaluate(restored, COORD + shift),
            evaluate(restored, COORD - shift),
        )
        finite_difference.append(
            -(plus["force"] - minus["force"]).reshape(-1) / (2 * delta)
        )
        np.testing.assert_allclose(
            (plus["energy"] - minus["energy"]) / (2 * delta),
            -force_result["force"].reshape(-1)[index],
            atol=2e-8,
        )
    np.testing.assert_allclose(
        hessian, np.stack(finite_difference, axis=1), atol=2e-7, rtol=2e-4
    )


def test_hessian_loss_parameter_gradient_and_update():
    model = make_model()
    model.enable_hessian()

    def loss(m):
        result = m(COORD, ATYPE)
        return jnp.mean((100 * result["hessian"] - jnp.eye(9)) ** 2)

    value_and_grad = nnx.jit(nnx.value_and_grad(loss))
    before, grads = value_and_grad(model)
    leaves = jax.tree.leaves(grads)
    assert all(np.isfinite(leaf).all() for leaf in leaves)
    assert sum(float(jnp.sum(leaf**2)) for leaf in leaves) > 1e-14
    state = nnx.state(model, nnx.Param)
    direction = jax.tree.map(lambda v: jnp.ones_like(v) * 0.01, state)
    analytic = sum(
        float(jnp.vdot(g, d))
        for g, d in zip(jax.tree.leaves(grads), jax.tree.leaves(direction), strict=True)
    )
    delta = 1e-4
    values = []
    for sign in (1, -1):
        shifted = nnx.clone(model)
        nnx.update(
            shifted, jax.tree.map(lambda p, d: p + sign * delta * d, state, direction)
        )
        values.append(float(value_and_grad(shifted)[0]))
    np.testing.assert_allclose(
        (values[0] - values[1]) / (2 * delta), analytic, atol=1e-9, rtol=2e-3
    )
    optimizer = nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param)
    optimizer.update(model, grads)
    assert float(value_and_grad(model)[0]) < float(before)


@pytest.mark.parametrize("alias", ["dpa4", "DPA4", "sezm", "SeZM"])
def test_model_aliases(alias):
    config = model_config()
    config["type"] = alias
    assert get_model(config).atomic_model.fitting_net.serialize()["type"] == "sezm_ener"


@pytest.mark.parametrize("option", ["random_gamma", "use_amp"])
def test_cli_rejects_explicit_unsupported_dpa4_options(option):
    from deepmd.jax.entrypoints.train import JAXTrainEntrypoint

    config = {"model": model_config()}
    config["model"]["descriptor"][option] = True
    with pytest.raises(NotImplementedError, match=f"{option}=True"):
        JAXTrainEntrypoint().normalize_config(config, multi_task=False)


def test_multitask_cli_dpa4_defaults():
    from deepmd.jax.entrypoints.train import JAXTrainEntrypoint

    models = {alias: {**model_config(), "type": alias} for alias in ("dpa4", "sezm")}
    config = {
        "model": {"model_dict": models},
        "learning_rate": {"type": "exp", "start_lr": 0.001, "decay_steps": 10},
        "loss_dict": {key: {"type": "ener"} for key in models},
        "training": {
            "numb_steps": 1,
            "data_dict": {
                key: {"training_data": {"systems": ["data"], "batch_size": 1}}
                for key in models
            },
        },
    }
    normalized = JAXTrainEntrypoint().normalize_config(config, multi_task=True)
    for model in normalized["model"]["model_dict"].values():
        assert model["descriptor"]["random_gamma"] is False
        assert model["descriptor"]["use_amp"] is False
        get_model(model)
    assert "random_gamma" not in models["dpa4"]["descriptor"]


def test_training_checkpoint_restart(tmp_path, monkeypatch):
    from deepmd.jax.entrypoints.train import JAXTrainEntrypoint
    from deepmd.jax.train.trainer import DPTrainer
    from deepmd.utils.data_system import get_data

    monkeypatch.chdir(tmp_path)
    system = tmp_path / "data"
    dataset = system / "set.000"
    dataset.mkdir(parents=True)
    (system / "nopbc").touch()
    np.savetxt(system / "type.raw", [0, 1, 0], fmt="%d")
    (system / "type_map.raw").write_text("H\nO\n")
    np.save(dataset / "coord.npy", np.asarray(COORD).reshape(1, -1))
    np.save(dataset / "energy.npy", np.array([[0.5]]))
    np.save(dataset / "force.npy", np.zeros((1, 9)))
    np.save(dataset / "hessian.npy", np.eye(9)[None])
    np.save(dataset / "charge_spin.npy", np.array([[0.0, 1.0]]))
    config = {
        "model": model_config(),
        "training": {
            "numb_steps": 1,
            "disp_freq": 1,
            "save_freq": 1,
            "save_ckpt": "model",
        },
        "learning_rate": {
            "type": "exp",
            "start_lr": 0.001,
            "stop_lr": 0.0001,
            "decay_steps": 10,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 1.0,
            "limit_pref_e": 1.0,
            "start_pref_f": 1.0,
            "limit_pref_f": 1.0,
            "start_pref_h": 1.0,
            "limit_pref_h": 1.0,
        },
    }
    config["model"]["data_stat_nbatch"] = 1
    config["model"]["descriptor"]["add_chg_spin_ebd"] = True
    data_config = {"systems": [str(system)], "batch_size": 1}
    config["training"]["training_data"] = data_config
    config = JAXTrainEntrypoint().normalize_config(config, multi_task=False)
    assert config["model"]["descriptor"]["random_gamma"] is False
    assert config["model"]["descriptor"]["use_amp"] is False
    trainer = DPTrainer(config)
    trainer.train(get_data(data_config, 3.0, ["H", "O"], None))
    assert (tmp_path / "model-1.jax").is_dir()
    restored = DPTrainer(config, restart=str(tmp_path / "model-1.jax"))
    assert restored.start_step == 1
    from deepmd.dpmodel.train import DEFAULT_TASK_KEY

    resumed_data = get_data(data_config, 3.0, ["H", "O"], None)
    restored._setup_training({DEFAULT_TASK_KEY: resumed_data}, {DEFAULT_TASK_KEY: None})
    for original, resumed in zip(
        jax.tree.leaves(nnx.state(trainer.optimizer)),
        jax.tree.leaves(nnx.state(restored.optimizer)),
        strict=True,
    ):
        np.testing.assert_array_equal(original, resumed)
    batch = restored._prepare_batch(resumed_data.get_batch(), DEFAULT_TASK_KEY)
    for current in (trainer, restored):
        current._train_step_impls[DEFAULT_TASK_KEY](
            current.model, current.optimizer, current.lr.value(1), *batch
        )
    from deepmd.jax.train.trainer import _evaluate_model_dict

    evaluate = nnx.jit(_evaluate_model_dict)
    expected = evaluate(trainer.model, *batch[1:], charge_spin=batch[0]["charge_spin"])
    actual = evaluate(restored.model, *batch[1:], charge_spin=batch[0]["charge_spin"])
    for key in ("energy", "force", "hessian"):
        assert np.isfinite(actual[key]).all()
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-12)
    assert "rmse_h" in (tmp_path / "lcurve.out").read_text()


def test_pytorch_energy_force_and_hessian_parity(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    pytest.importorskip("vesin")
    from deepmd.pt.model.model import get_model as get_pt_model

    config = model_config()
    config["descriptor"].update(random_gamma=False, use_amp=False)
    config["use_compile"] = False
    pt_model = get_pt_model(config).double().eval()
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for param in pt_model.parameters():
            param.add_(
                torch.randn(param.shape, generator=generator, dtype=param.dtype) * 0.03
            )
    record = pt_model.serialize()
    from deepmd.jax.utils.serialization import deserialize_to_file, serialize_from_file

    checkpoint = str(tmp_path / "imported.jax")
    deserialize_to_file(
        checkpoint, {"model": record, "model_def_script": config}, hessian=True
    )
    model = BaseModel.deserialize(serialize_from_file(checkpoint)["model"])
    model.enable_hessian()
    result = nnx.jit(lambda m: m(COORD, ATYPE))(model)
    coord = torch.tensor(np.asarray(COORD), dtype=torch.float64)
    atype = torch.tensor(np.asarray(ATYPE), dtype=torch.int64)
    expected = pt_model(coord, atype)
    for key in ("energy", "force"):
        np.testing.assert_allclose(
            result[key], expected[key].detach().numpy(), atol=1e-8, rtol=1e-6
        )
    delta = 1e-4
    columns = []
    for index in range(9):
        shift = torch.zeros_like(coord)
        shift.reshape(-1)[index] = delta
        plus = pt_model(coord + shift, atype)["force"].detach().numpy()
        minus = pt_model(coord - shift, atype)["force"].detach().numpy()
        columns.append(-(plus - minus).reshape(-1) / (2 * delta))
    np.testing.assert_allclose(
        np.asarray(result["hessian"])[0],
        np.stack(columns, axis=1),
        atol=2e-7,
        rtol=2e-4,
    )


def test_periodic_ghost_hessian_mapping():
    from deepmd.jax.train.trainer import _evaluate_model_dict

    model = make_model()
    model.enable_hessian()
    mapping = jnp.array([[0, 1, 2, 0, 1]])
    atype = ATYPE[:, mapping[0]]
    nlist = jnp.array([[[4, 2, -1], [3, 2, -1], [0, 1, -1]]])
    shifts = jnp.array([[[0.0, 0.0, 0.0]] * 3 + [[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]])
    coord = COORD.at[0, 1, 0].set(2.8)

    @nnx.jit
    def evaluate(m, x):
        extended = x[:, mapping[0], :] + shifts
        return _evaluate_model_dict(m, extended, atype, nlist, mapping, None, None)

    result = evaluate(model, coord)
    hessian = np.asarray(result["hessian"]).reshape(9, 9)
    assert np.linalg.norm(hessian) > 1e-6
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-10)
    delta = 1e-4
    shift = jnp.zeros_like(coord).at[0, 1, 0].set(delta)
    plus, minus = evaluate(model, coord + shift), evaluate(model, coord - shift)
    np.testing.assert_allclose(
        hessian[:, 3],
        -(plus["force"] - minus["force"]).reshape(-1) / (2 * delta),
        atol=2e-7,
        rtol=2e-4,
    )


def test_padded_atom_hessian():
    model = make_model()
    model.enable_hessian()
    evaluate = nnx.jit(lambda m, x, t: m(x, t))
    expected = evaluate(model, COORD, ATYPE)
    coord = jnp.concatenate((COORD, jnp.zeros((1, 1, 3))), axis=1)
    atype = jnp.concatenate((ATYPE, jnp.array([[-1]])), axis=1)
    actual = evaluate(model, coord, atype)
    assert np.isfinite(actual["hessian"]).all()
    np.testing.assert_allclose(actual["energy"], expected["energy"], atol=1e-12)
    np.testing.assert_allclose(actual["force"][:, :3], expected["force"], atol=1e-12)
    np.testing.assert_allclose(
        actual["hessian"][:, :9, :9], expected["hessian"], atol=1e-12
    )
    np.testing.assert_allclose(actual["hessian"][:, 9:, :], 0, atol=1e-12)
    np.testing.assert_allclose(actual["hessian"][:, :, 9:], 0, atol=1e-12)


def test_cloned_interaction_block_parameter_gradient():
    model = make_model()
    cloned = nnx.clone(model)
    block = cloned.atomic_model.descriptor.blocks[0]
    assert block._forward_impl.__self__ is block

    def energy(m):
        return m(COORD, ATYPE)["energy"].sum()

    derivative = nnx.jit(nnx.grad(energy))(cloned)
    path = (
        "atomic_model",
        "descriptor",
        "blocks",
        0,
        "so2_conv",
        "pre_focus_mix",
        "weight",
    )
    grad = dict(nnx.to_flat_state(derivative))[path].value
    index = np.unravel_index(np.argmax(np.abs(grad)), grad.shape)
    assert abs(float(grad[index])) > 1e-8
    evaluate = nnx.jit(energy)
    values = []
    delta = 1e-4
    for sign in (1, -1):
        shifted = nnx.clone(model)
        weight = shifted.atomic_model.descriptor.blocks[0].so2_conv.pre_focus_mix.weight
        weight.value = weight.value.at[index].add(sign * delta)
        values.append(float(evaluate(shifted)))
    np.testing.assert_allclose(
        (values[0] - values[1]) / (2 * delta), grad[index], atol=1e-9, rtol=1e-4
    )
    np.testing.assert_allclose(evaluate(model), evaluate(cloned), atol=1e-12)
