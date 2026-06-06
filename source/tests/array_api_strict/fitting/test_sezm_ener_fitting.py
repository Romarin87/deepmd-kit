# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)

from ..common import (
    to_array_api_strict_array,
)
from .fitting import (
    SeZMEnergyFittingNet,
)


class TestSeZMEnergyFittingNet(unittest.TestCase):
    def test_auto_neuron_forward_and_serialize(self) -> None:
        fitting = SeZMEnergyFittingNet(
            ntypes=2,
            dim_descrpt=4,
            neuron=[0],
            activation_function="silu",
            precision="float32",
            mixed_types=True,
            seed=2026,
        )
        self.assertEqual(fitting.neuron, [32])
        self.assertEqual(
            fitting.nets.serialize()["network_type"],
            "sezm_fitting_network",
        )

        descriptor = to_array_api_strict_array(
            np.linspace(-0.2, 0.3, 24, dtype=np.float32).reshape(2, 3, 4)
        )
        atype = to_array_api_strict_array(
            np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
        )
        result = fitting.call(descriptor, atype)
        self.assertEqual(result["energy"].shape, (2, 3, 1))
        self.assertTrue(np.all(np.isfinite(to_numpy_array(result["energy"]))))

        serialized = fitting.serialize()
        self.assertEqual(serialized["type"], "sezm_ener")
        restored = SeZMEnergyFittingNet.deserialize(serialized)
        restored_result = restored.call(descriptor, atype)
        np.testing.assert_allclose(
            to_numpy_array(result["energy"]),
            to_numpy_array(restored_result["energy"]),
        )

    def test_case_film_is_explicitly_unsupported(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "case_film_embd"):
            SeZMEnergyFittingNet(
                ntypes=1,
                dim_descrpt=4,
                neuron=[0],
                dim_case_embd=2,
                case_film_embd=True,
            )


if __name__ == "__main__":
    unittest.main()
