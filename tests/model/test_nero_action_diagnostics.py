"""Unit handling and padding exclusion for saved-action diagnostics."""
import unittest

import numpy as np

from scripts.model.diagnose_nero_actions import diagnose


class ActionDiagnosticsTests(unittest.TestCase):
    def sample(self):
        anchors = [{"index": i, "episode_index": int(i == 2), "source_episode": f"ep{int(i == 2)}",
                    "local_index": i, "remaining_frames": remaining}
                   for i, remaining in enumerate((50, 2, 1))]
        arrays = {"predictions": np.zeros((3, 50, 16)), "references": np.zeros((3, 50, 16)),
                  "states": np.zeros((3, 16)), "dataset_indices": np.arange(3),
                  "valid_mask": np.arange(50)[None, :] < np.array([50, 2, 1])[:, None]}
        return arrays, anchors

    def test_signed_gripper_bias_and_width_units(self):
        arrays, anchors = self.sample()
        arrays["predictions"][:, :, 15] = 0.002
        result = diagnose(arrays, anchors)
        first = result["horizons"][0]
        self.assertAlmostEqual(first["right_gripper_bias_mm"], 2)
        self.assertAlmostEqual(first["right_gripper_mae_mm"], 2)
        self.assertAlmostEqual(result["right_gripper_first_actions"][0]["predicted_width_mm"], 2)

    def test_invalid_future_padding_is_not_counted(self):
        arrays, anchors = self.sample()
        arrays["predictions"][~arrays["valid_mask"]] = 1000
        result = diagnose(arrays, anchors)
        last = next(row for row in result["horizons"] if row["episode"] == "all" and row["horizon_index"] == 49)
        self.assertEqual(last["count"], 1)
        self.assertEqual(last["joint_mae_deg"], 0)
        self.assertEqual(last["right_gripper_mae_mm"], 0)

    def test_right_seventh_joint_is_kept_and_labeled(self):
        arrays, anchors = self.sample()
        arrays["predictions"][:, :, 14] = np.pi / 180
        result = diagnose(arrays, anchors)
        joint = next(row for row in result["first_action_joints"] if row["joint"] == "right_J7")
        self.assertAlmostEqual(joint["mae_deg"], 1)
        self.assertEqual(len(result["first_action_joints"]), 14)

    def test_mismatched_source_or_nonfinite_data_rejected(self):
        arrays, anchors = self.sample()
        arrays["dataset_indices"] = np.array([1, 0, 2])
        with self.assertRaises(ValueError):
            diagnose(arrays, anchors)
        arrays, anchors = self.sample()
        arrays["states"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            diagnose(arrays, anchors)


if __name__ == "__main__":
    unittest.main()
