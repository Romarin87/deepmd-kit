# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.loggers.training import (
    format_training_message_per_task,
)


class TestTrainingLogger(unittest.TestCase):
    def test_metric_order_is_preserved(self):
        msg = format_training_message_per_task(
            batch=1,
            task_name="trn",
            rmse={
                "mae_e": 1.0,
                "mae_f": 2.0,
                "mae_h": 3.0,
                "rmse": 4.0,
            },
            learning_rate=1e-3,
        )

        self.assertLess(msg.index("mae_e"), msg.index("mae_f"))
        self.assertLess(msg.index("mae_f"), msg.index("mae_h"))
        self.assertLess(msg.index("mae_h"), msg.index("rmse"))


if __name__ == "__main__":
    unittest.main()
