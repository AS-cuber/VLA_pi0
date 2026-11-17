"""NERO camera identity, unit preservation and output-channel regression checks."""
import unittest

import numpy as np

from scripts.model.nero_pi0_adapter import CAMERAS, JOINT_MASK, NeroInputs, NeroOutputs, rgb_image


class NeroAdapterTests(unittest.TestCase):
    def sample(self):
        return {
            "state": np.arange(16, dtype=np.float32) / 100,
            "actions": np.arange(48, dtype=np.float32).reshape(3, 16) / 100,
            "images": {
                "base": np.full((3, 8, 10), 10, dtype=np.uint8),
                "left_wrist": np.full((3, 8, 10), 20, dtype=np.uint8),
                "right_wrist": np.full((3, 8, 10), 30, dtype=np.uint8),
            },
            "prompt": "用右臂将物体放入盒子，然后双臂合盖。",
        }

    def test_camera_identity_layout_and_prompt(self):
        source = self.sample()
        result = NeroInputs()(source)
        for name, expected in zip(CAMERAS.values(), (10, 20, 30)):
            self.assertEqual(result["image"][name].shape, (8, 10, 3))
            np.testing.assert_array_equal(result["image"][name], expected)
            self.assertTrue(result["image_mask"][name])
        self.assertEqual(result["prompt"], source["prompt"])

    def test_preserves_all_channels_signs_and_width_units(self):
        source = self.sample()
        source["state"][6] = -0.73
        source["state"][7] = 0.00008
        source["state"][14] = -1.25
        source["state"][15] = 0.04
        result = NeroInputs()(source)
        np.testing.assert_array_equal(result["state"], source["state"])
        np.testing.assert_array_equal(result["actions"], source["actions"])
        self.assertEqual(np.flatnonzero(~JOINT_MASK).tolist(), [7, 15])
        result["state"][:] = 0
        result["actions"][:] = 0
        self.assertEqual(source["state"][14], np.float32(-1.25))
        self.assertNotEqual(source["actions"][0, 1], 0)

    def test_float_image_keeps_rgb_order(self):
        image = np.zeros((3, 8, 10), dtype=np.float32)
        image[0] = 1
        image[1] = 0.5
        output = rgb_image(image)
        np.testing.assert_array_equal(output[0, 0], [255, 128, 0])

    def test_missing_camera_rejected(self):
        source = self.sample()
        del source["images"]["right_wrist"]
        with self.assertRaises(ValueError):
            NeroInputs()(source)

    def test_invalid_or_short_robot_vectors_rejected(self):
        for key, value in (("state", np.zeros(14)),
                           ("state", np.full(16, np.nan)),
                           ("actions", np.zeros((3, 14))),
                           ("actions", np.full((3, 16), np.inf))):
            with self.subTest(key=key, shape=value.shape):
                source = self.sample()
                source[key] = value
                with self.assertRaises(ValueError):
                    NeroInputs()(source)

    def test_invalid_float_pixel_range_rejected(self):
        with self.assertRaises(ValueError):
            rgb_image(np.full((3, 8, 10), 255.0, dtype=np.float32))

    def test_inference_input_without_training_actions(self):
        source = self.sample()
        del source["actions"]
        result = NeroInputs()(source)
        self.assertNotIn("actions", result)
        self.assertEqual(result["state"].shape, (16,))

    def test_output_keeps_right_j7_and_gripper_discards_only_padding(self):
        source = np.arange(96, dtype=np.float32).reshape(3, 32)
        result = NeroOutputs()({"actions": source})["actions"]
        np.testing.assert_array_equal(result, source[:, :16])
        self.assertEqual(result.shape, (3, 16))
        result[:] = 0
        self.assertEqual(source[0, 15], 15)

    def test_short_output_rejected(self):
        with self.assertRaises(ValueError):
            NeroOutputs()({"actions": np.zeros((3, 14))})


if __name__ == "__main__":
    unittest.main()
