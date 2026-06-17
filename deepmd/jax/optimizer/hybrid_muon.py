# SPDX-License-Identifier: LGPL-3.0-or-later
"""HybridMuon optimizer entrypoints for the JAX backend.

This module preserves the public JAX trainer interface for DPA4 inputs that use
``optimizer.type = "HybridMuon"``.  The full PyTorch optimizer routes parameters
between Adam, AdamW and Muon matrix updates by name; the first JAX rewrite keeps
the AdamW route so inputs remain runnable while the Muon route is reintroduced
and tested against the old validated branch.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
    Callable,
)

import optax


def get_adam_route(param_name: str | None) -> str:
    """Return the name-based HybridMuon route used by PyTorch DPA4."""
    if param_name is None:
        return "muon"
    param_name_lower = param_name.lower()
    name_segments = param_name_lower.split(".")
    leaf_name_idx = len(name_segments) - 1
    while leaf_name_idx > 0 and name_segments[leaf_name_idx].isdigit():
        leaf_name_idx -= 1
    leaf_name = name_segments[leaf_name_idx]
    if "bias" in leaf_name:
        return "adam"
    if leaf_name.startswith("adam_"):
        return "adam"
    if leaf_name.startswith("adamw_"):
        return "adamw"
    return "muon"


def build_hybrid_muon_routes(params: Any, muon_mode: str = "slice") -> Any:
    """Build a static route tree matching a parameter pytree."""
    del muon_mode

    def walk(value: Any, path: tuple[str, ...]) -> Any:
        if hasattr(value, "items"):
            return {key: walk(child, (*path, str(key))) for key, child in value.items()}
        if isinstance(value, dict):
            return {key: walk(child, (*path, str(key))) for key, child in value.items()}
        return get_adam_route(".".join(path))

    return walk(params, ())


def hybrid_muon(
    learning_rate: float | Callable[[Any], Any],
    *,
    weight_decay: float = 0.0,
    momentum: float = 0.95,
    adam_betas: tuple[float, float] = (0.9, 0.95),
    adam_eps: float = 1e-20,
    lr_adjust: float = 0.0,
    lr_adjust_coeff: float = 0.18,
    muon_mode: str = "slice",
    enable_gram: bool = True,
    flash_muon: bool = True,
    magma_muon: bool = True,
) -> optax.GradientTransformation:
    """Create the current JAX HybridMuon compatibility optimizer.

    The compatibility path intentionally maps to AdamW while preserving accepted
    HybridMuon configuration keys.  It is conservative for correctness and lets
    Neo.json run before the Muon-specific matrix update is re-landed.
    """
    del momentum
    del lr_adjust
    del lr_adjust_coeff
    del muon_mode
    del enable_gram
    del flash_muon
    del magma_muon
    return optax.adamw(
        learning_rate=learning_rate,
        b1=adam_betas[0],
        b2=adam_betas[1],
        eps=adam_eps,
        weight_decay=weight_decay,
    )
