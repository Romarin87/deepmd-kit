# SPDX-License-Identifier: LGPL-3.0-or-later
"""Explicit device placement for frame and atom parallel JAX training."""

import os
from typing import Any

import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from deepmd.jax.env import jax, nnx


def initialize_distributed() -> None:
    """Initialize the production launcher protocol before any device queries."""
    count = int(os.environ.get("DP_JAX_MULTI_NPROC", "1"))
    if count <= 1:
        return
    rank = int(os.environ.get("DP_JAX_MULTI_IPROC", "-1"))
    host = os.environ.get("DP_JAX_MULTI_HOST")
    if not 0 <= rank < count or not host:
        raise ValueError(
            "Set DP_JAX_MULTI_IPROC in [0, NPROC) and DP_JAX_MULTI_HOST=host:port."
        )
    if not jax.distributed.is_initialized():
        jax.distributed.initialize(
            coordinator_address=host, num_processes=count, process_id=rank
        )


def make_training_mesh() -> Mesh:
    """Group each process's local devices along the atom axis."""
    devices = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
    return Mesh(
        np.asarray(devices).reshape(jax.process_count(), jax.local_device_count()),
        ("data", "natoms"),
    )


def replicate_model(model: Any, mesh: Mesh) -> None:
    """Place shared model variables on a replicated mesh without cloning modules."""
    state = nnx.state(model)
    nnx.update(
        model,
        jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P())), state),
    )


def localize_model(model: Any) -> None:
    """Place replicated variables on one local device for chief-only evaluation."""
    state = nnx.state(model)
    nnx.update(
        model,
        jax.tree.map(
            lambda value: jax.device_put(np.asarray(value), jax.local_devices()[0]),
            state,
        ),
    )


def shard_prepared_batch(prepared: tuple, mesh: Mesh | None) -> tuple:
    """Shard locally built neighbors and labels; metadata stays on the host.

    Neighbor construction precedes global placement because periodic ghost
    counts depend on the local cell. All processes must supply equal shapes.
    Frame properties must never be partitioned as if they were atomic labels.
    """
    if mesh is None or mesh.size == 1:
        return prepared
    labels, coord, atype, nlist, mapping, fp, ap = prepared
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        import hashlib

        signature = repr(
            jax.tree.map(
                lambda x: (x.shape, str(x.dtype)) if hasattr(x, "shape") else x,
                prepared,
            )
        )
        digest = np.frombuffer(
            hashlib.sha256(signature.encode()).digest(), dtype=np.uint8
        )
        digests = multihost_utils.process_allgather(digest, tiled=False)
        if not np.all(digests == digests[0]):
            raise ValueError(
                "JAX ranks require matching batch shapes, ghost counts, and label availability."
            )

    def place(value: Any, atomic: bool = False) -> Any:
        if value is None:
            return None
        spec = (
            P("data", "natoms")
            if atomic and value.ndim >= 2 and value.shape[1] % mesh.shape["natoms"] == 0
            else P("data")
        )
        sharding = NamedSharding(mesh, spec)
        if jax.process_count() > 1:
            return jax.make_array_from_process_local_data(sharding, np.asarray(value))
        return jax.device_put(value, sharding)

    atom_labels = {
        "coord",
        "type",
        "force",
        "atom_ener",
        "atom_pref",
        "atom_virial",
        "aparam",
        "hessian",
    }
    labels = {
        key: value
        if key.startswith("find_")
        or key in {"natoms_vec", "natoms", "default_mesh"}
        or getattr(value, "ndim", 0) == 0
        else place(value, key in atom_labels)
        for key, value in labels.items()
    }
    return (
        labels,
        place(coord, True),
        place(atype, True),
        place(nlist, True),
        place(mapping, True),
        place(fp),
        place(ap, True),
    )
