# SPDX-License-Identifier: LGPL-3.0-or-later
"""Lebedev quadrature data loader for the JAX/dpmodel SeZM port."""

from __future__ import annotations

from pathlib import (
    Path,
)

import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)


LEBEDEV_PRECISION_TO_NPOINTS = {
    3: 6,
    5: 14,
    7: 26,
    9: 38,
    11: 50,
    13: 74,
    15: 86,
    17: 110,
    19: 146,
    21: 170,
    23: 194,
    25: 230,
    27: 266,
    29: 302,
    31: 350,
    35: 434,
    41: 590,
    47: 770,
    53: 974,
    59: 1202,
    65: 1454,
    71: 1730,
    77: 2030,
    83: 2354,
    89: 2702,
    95: 3074,
    101: 3470,
    107: 3890,
    113: 4334,
    119: 4802,
    125: 5294,
    131: 5810,
}

LEBEDEV_RULES_FILE = (
    Path(__file__).parents[2]
    / "pt"
    / "model"
    / "descriptor"
    / "sezm_nn"
    / "lebedev_rules.npz"
)


def load_lebedev_rule(
    rule_precision: int,
    *,
    float_precision: str = DEFAULT_PRECISION,
) -> tuple[np.ndarray, np.ndarray]:
    rule_key = f"{int(rule_precision):03d}"
    if not LEBEDEV_RULES_FILE.exists():
        raise FileNotFoundError(
            f"Lebedev quadrature data file is missing: {LEBEDEV_RULES_FILE}"
        )
    with np.load(LEBEDEV_RULES_FILE) as rules:
        point_key = f"points_{rule_key}"
        weight_key = f"weights_{rule_key}"
        if point_key not in rules or weight_key not in rules:
            raise ValueError(
                f"Lebedev rule with precision {rule_precision} is not packaged"
            )
        points = rules[point_key]
        weights = rules[weight_key]
    dtype = PRECISION_DICT[float_precision.lower()]
    return np.asarray(points, dtype=dtype), np.asarray(weights, dtype=dtype)
