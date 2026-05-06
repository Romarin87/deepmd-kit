# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.utils.stat import (
    compute_output_stats_global,
)


class TestDPModelStat(unittest.TestCase):
    def test_global_stat_uses_real_natoms_vec_for_mixed_type(self) -> None:
        true_bias = np.array([[1.0], [2.0], [3.0]])
        real_counts = np.array(
            [
                [2, 1, 0],
                [0, 2, 1],
                [1, 0, 2],
            ],
            dtype=np.int32,
        )
        sampled = [
            {
                "energy": real_counts @ true_bias,
                "find_energy": 1.0,
                # Padded mixed-type batches may carry placeholder natoms/type.raw.
                "natoms": np.tile(
                    np.array([[6, 6, 6, 0, 0]], dtype=np.int32),
                    (real_counts.shape[0], 1),
                ),
                "real_natoms_vec": np.concatenate(
                    [
                        np.tile(
                            np.array([[6, 6]], dtype=np.int32),
                            (real_counts.shape[0], 1),
                        ),
                        real_counts,
                    ],
                    axis=1,
                ),
            }
        ]

        bias, std = compute_output_stats_global(
            sampled,
            ntypes=3,
            keys=["energy"],
            global_sampled_idx={"energy": [0]},
        )

        np.testing.assert_allclose(bias["energy"], true_bias)
        np.testing.assert_allclose(std["energy"], np.zeros(1))


if __name__ == "__main__":
    unittest.main()
