# SPDX-License-Identifier: LGPL-3.0-or-later
"""Exact Hessian block export and host assembly, including ghosts and padding."""

import numpy as np
import pytest
from deepmd.jax.env import nnx, jnp
from deepmd.jax.utils.serialization import deserialize_to_file, _get_hessian_chunk_size
from deepmd.infer import DeepPot
from .test_dpa4_energy_hessian import make_model, model_config, COORD, ATYPE


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_bad_chunk_size(monkeypatch, value):
    monkeypatch.setenv("DP_JAX_HESSIAN_CHUNK_SIZE", value)
    with pytest.raises(ValueError, match="positive integer"):
        _get_hessian_chunk_size()


@pytest.mark.parametrize("descriptor", ["dpa4", "dpa3"])
def test_dpa4_block_export(tmp_path, monkeypatch, descriptor):
    config = model_config()
    if descriptor == "dpa3":
        config = {
            "type_map": ["H", "O"],
            "descriptor": {
                "type": "dpa3",
                "repflow": {
                    "n_dim": 4,
                    "e_dim": 4,
                    "a_dim": 4,
                    "nlayers": 1,
                    "e_rcut": 3.0,
                    "e_rcut_smth": 2.5,
                    "e_sel": 3,
                    "a_rcut": 2.5,
                    "a_rcut_smth": 2.0,
                    "a_sel": 2,
                    "axis_neuron": 2,
                },
                "seed": 1,
            },
            "fitting_net": {"neuron": [4], "seed": 2},
        }
    else:
        config["pair_exclude_types"] = [[0, 1]]
    model = make_model(config)
    record = {"model": model.serialize(), "model_def_script": config}
    monkeypatch.setenv("DP_JAX_HESSIAN_CHUNK_SIZE", "4")
    filename = str(tmp_path / "blocks.hlo")
    deserialize_to_file(filename, record, hessian=True)
    evaluator = DeepPot(filename, auto_batch_size=False)
    assert evaluator.deep_eval.dp.get_hessian_chunk_size() == 4
    model.enable_hessian()
    from deepmd.jax.train.trainer import _evaluate_model_dict, prepare_input

    evaluate = nnx.jit(_evaluate_model_dict)
    for box, padded in [(None, False), (np.eye(3)[None] * 4.0, False), (None, True)]:
        coord, atype = COORD, ATYPE
        if padded:
            coord = jnp.concatenate((coord, jnp.zeros((1, 1, 3))), axis=1)
            atype = jnp.concatenate((atype, jnp.array([[-1]])), axis=1)
        inputs = prepare_input(
            rcut=model.get_rcut(),
            sel=model.get_sel(),
            coord=coord,
            atype=atype,
            box=None if box is None else jnp.array(box),
            fparam=None,
            aparam=None,
        )
        expected = evaluate(model, *inputs)
        result = evaluator.eval(np.asarray(coord), box, np.asarray(atype[0]))
        np.testing.assert_allclose(result[0], expected["energy"], atol=1e-10)
        np.testing.assert_allclose(result[1], expected["force"], atol=1e-10)
        np.testing.assert_allclose(
            result[-1], expected["hessian"].reshape(result[-1].shape), atol=1e-10
        )
        assert np.linalg.norm(result[-1]) > 1e-6
    # A final partial block contains one valid row and three zero rows (3 atoms).
    tail = evaluator.deep_eval.dp.call_hessian_block(COORD, ATYPE, block_index=2)[
        "energy_derv_r_derv_r_block"
    ]
    np.testing.assert_allclose(tail[:, 1:], 0, atol=1e-12)


def test_conditioned_blocks_and_legacy_hlo(tmp_path, monkeypatch):
    from deepmd.jax.model.model import get_model
    from deepmd.dpmodel.utils.serialization import load_dp_model, save_dp_model

    config = model_config()
    config["descriptor"]["add_chg_spin_ebd"] = True
    config["descriptor"]["default_chg_spin"] = [0.0, 1.0]
    model = get_model(config)
    state = nnx.state(model, nnx.Param)
    rng = np.random.default_rng(15)
    for _, variable in nnx.to_flat_state(state):
        variable.value += jnp.asarray(rng.normal(scale=0.03, size=variable.value.shape))
    nnx.update(model, state)
    monkeypatch.setenv("DP_JAX_HESSIAN_CHUNK_SIZE", "16")
    path = str(tmp_path / "conditioned.hlo")
    deserialize_to_file(
        path, {"model": model.serialize(), "model_def_script": config}, hessian=True
    )
    evaluator = DeepPot(path, auto_batch_size=False)
    model.enable_hessian()
    evaluate = nnx.jit(lambda m, cs: m(COORD, ATYPE, charge_spin=cs))
    for cs in [None, np.array([[2.0, 3.0]])]:
        result = evaluator.eval(
            np.asarray(COORD), None, np.asarray(ATYPE[0]), charge_spin=cs
        )
        expected = evaluate(model, None if cs is None else jnp.asarray(cs))
        for actual, key in zip(
            [result[0], result[1], result[-1]],
            ["energy", "force", "hessian"],
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected[key], atol=1e-10)
    # Old six-input HLO files have neither chunk nor charge metadata.
    monkeypatch.delenv("DP_JAX_HESSIAN_CHUNK_SIZE")
    plain = make_model()
    legacy_path = str(tmp_path / "legacy.hlo")
    deserialize_to_file(
        legacy_path,
        {"model": plain.serialize(), "model_def_script": model_config()},
        hessian=True,
    )
    record = load_dp_model(legacy_path)
    for key in ("hessian_chunk_size", "dim_chg_spin", "default_chg_spin"):
        record["constants"].pop(key)
    save_dp_model(legacy_path, record)
    legacy = DeepPot(legacy_path, auto_batch_size=False)
    plain.enable_hessian()
    expected = nnx.jit(lambda m: m(COORD, ATYPE))(plain)
    result = legacy.eval(np.asarray(COORD), None, np.asarray(ATYPE[0]))
    np.testing.assert_allclose(result[-1], expected["hessian"], atol=1e-10)
