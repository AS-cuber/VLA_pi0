"""离线验证数据迁移的路径边界，以及错误划分/输入的拒绝行为。"""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.data.export_nero_lerobot import FEATURE_NAMES, preflight, resolve_image


class ExportInputTests(unittest.TestCase):
    def test_moved_image_uses_new_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = Path(tmp) / "episode_1"
            image = episode / "images/base/frame.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"test")
            self.assertEqual(resolve_image("/old/data/episode_1/images/base/frame.jpg",
                                           "/old/data/episode_1", episode, "base"), image)

    def test_missing_new_root_does_not_fall_back_to_old_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "old"
            image = old / "images/base/frame.jpg"
            image.parent.mkdir(parents=True)
            image.touch()
            with self.assertRaisesRegex(ValueError, "缺少图像"):
                resolve_image(str(image), str(old), Path(tmp) / "new", "base")

    def test_wrong_camera_or_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for stored in ("/old/images/right_wrist/frame.jpg", "/old/images/base/../../../escape.jpg"):
                with self.subTest(stored=stored), self.assertRaises(ValueError):
                    resolve_image(stored, "/old", Path(tmp), "base")

    def test_overlapping_split_is_rejected_before_opening_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            aligned = Path(tmp)
            manifest = {"status": "alignment_prepared_not_lerobot", "fps": 15,
                        "feature_order": FEATURE_NAMES,
                        "camera_order": ["base", "left_wrist", "right_wrist"],
                        "action_semantics": "absolute commanded joint positions (rad) and commanded gripper width (m)",
                        "episodes": [{"episode": "episode_1"}], "episode_count": 1}
            (aligned / "dataset_manifest.json").write_text(json.dumps(manifest))
            (aligned / "split.json").write_text(json.dumps(
                {"unit": "whole episode", "train": ["episode_1"], "validation": ["episode_1"]}))
            with self.assertRaisesRegex(ValueError, "划分存在"):
                preflight(aligned, aligned / "missing_raw")


if __name__ == "__main__":
    unittest.main()
