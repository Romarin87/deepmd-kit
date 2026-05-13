# SPDX-License-Identifier: LGPL-3.0-or-later
"""Small helpers for host-level JAX distributed coordination."""

import pickle
import traceback
from typing import (
    Any,
    Callable,
)

import numpy as np

from deepmd.jax.env import (
    jax,
)


def is_process0_or_single_process() -> bool:
    return jax.process_count() <= 1 or jax.process_index() == 0


def broadcast_object_from_process0(data: Any, *, purpose: str) -> Any:
    if jax.process_count() <= 1:
        return data
    from jax.experimental import multihost_utils

    payload = (
        np.array(
            np.frombuffer(
                pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL),
                dtype=np.uint8,
            ),
            copy=True,
        )
        if jax.process_index() == 0
        else np.zeros(0, dtype=np.uint8)
    )
    payload_size = np.array([payload.size], dtype=np.int64)
    shared_size = multihost_utils.broadcast_one_to_all(
        payload_size,
        is_source=(jax.process_index() == 0),
    )
    shared_size = int(np.asarray(shared_size)[0])
    if jax.process_index() != 0:
        payload = np.zeros(shared_size, dtype=np.uint8)
    elif payload.size != shared_size:
        raise ValueError(
            f"Unexpected {purpose} payload size mismatch: "
            f"{payload.size} != {shared_size}"
        )
    shared_payload = multihost_utils.broadcast_one_to_all(
        payload,
        is_source=(jax.process_index() == 0),
    )
    return pickle.loads(np.asarray(shared_payload, dtype=np.uint8).tobytes())


def run_on_process0_or_raise(name: str, func: Callable[[], Any]) -> Any:
    """Run host-only setup on process 0 and propagate failures to all processes."""
    if jax.process_count() <= 1:
        return func()

    result = None
    source_exc: Exception | None = None
    status: dict[str, Any] = {"ok": True, "name": name}
    if jax.process_index() == 0:
        try:
            result = func()
        except Exception as exc:
            source_exc = exc
            status = {
                "ok": False,
                "name": name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

    status = broadcast_object_from_process0(status, purpose=f"{name} status")
    if not status["ok"]:
        message = (
            f"JAX process 0 failed during {status['name']}: "
            f"{status['error_type']}: {status['error']}\n"
            f"{status['traceback']}"
        )
        if source_exc is not None:
            raise RuntimeError(message) from source_exc
        raise RuntimeError(message)
    return result
