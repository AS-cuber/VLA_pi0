"""Train/validation isolation, normalization floors and local-path collision checks."""
import copy
import dataclasses
import io
import logging
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import json

import numpy as np

from scripts.model.prepare_nero_pi0 import EXPECTED_FEATURES, calculate_stats, check_report, connect_train
from scripts.model.run_nero_pi0 import EXPECTED_COMMIT, PRESENTATION_SOURCE_VERSIONS, capture_training_output, digest, read_training_metrics, readiness, source_version_matches, training_config


class PreparationTests(unittest.TestCase):
    def report(self):
        return {
            "status": "exported_and_reload_checked_not_yet_openpi_configured",
            "fps": 15,
            "feature_order": EXPECTED_FEATURES,
            "action_semantics": "absolute commanded joint positions (rad) and commanded gripper width (m)",
            "splits": {
                "train": {"repo_id": "local/nero_box_v1_train", "frames": 2710,
                          "episodes": [{"source_episode": f"episode_{i}", "frames": n}
                                       for i, n in enumerate([340] * 7 + [330])]},
                "validation": {"frames": 719, "episodes": [
                    {"source_episode": "episode_8", "frames": 344},
                    {"source_episode": "episode_9", "frames": 375}]},
            },
        }

    def test_validation_overlap_is_rejected(self):
        report = self.report()
        self.assertEqual(len(check_report(report)), 8)
        report["splits"]["validation"]["episodes"][0]["source_episode"] = "episode_0"
        with self.assertRaises(ValueError):
            check_report(report)

    def test_wrong_units_or_channel_order_rejected(self):
        for key, value in (("action_semantics", "relative actions"),
                           ("feature_order", list(reversed(EXPECTED_FEATURES)))):
            with self.subTest(key=key):
                report = copy.deepcopy(self.report())
                report[key] = value
                with self.assertRaises(ValueError):
                    check_report(report)

    def test_constant_grippers_keep_means_and_receive_floors(self):
        data = np.tile(np.linspace(-1, 1, 16), (20, 1))
        data[:, 7], data[:, 15] = 0.00008, 0.04
        before = data.copy()
        stats, audit = calculate_stats(data)
        np.testing.assert_allclose(stats["mean"][[7, 15]], [0.00008, 0.04], atol=1e-12)
        np.testing.assert_array_equal(stats["std"], np.full(16, 0.001))
        self.assertEqual(audit["floored_channels"], list(range(16)))
        normalized = (data - stats["mean"]) / (stats["std"] + 1e-6)
        np.testing.assert_allclose(normalized, 0, atol=1e-10)
        np.testing.assert_array_equal(data, before)

    def test_varying_channels_use_actual_standard_deviation(self):
        data = np.repeat(np.array([[-2.0], [0.0], [2.0]]), 16, axis=1)
        stats, audit = calculate_stats(data)
        np.testing.assert_allclose(stats["std"], np.sqrt(8 / 3))
        self.assertEqual(audit["floored_channels"], [])
        self.assertTrue((stats["q01"] <= stats["mean"]).all())
        self.assertTrue((stats["q99"] >= stats["mean"]).all())

    def test_invalid_statistics_input_rejected(self):
        for data in (np.zeros((1, 16)), np.zeros((2, 14)), np.full((3, 16), np.nan)):
            with self.subTest(shape=data.shape):
                with self.assertRaises(ValueError):
                    calculate_stats(data)

    def test_project_link_is_idempotent_and_does_not_link_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "original/train"
            (train / "meta").mkdir(parents=True)
            (train / "meta/info.json").write_text("{}")
            link = connect_train(root / "cache", train)
            self.assertEqual(link.resolve(), train.resolve())
            self.assertEqual(connect_train(root / "cache", train), link)
            self.assertFalse((root / "cache/local/nero_box_v1_validation").exists())

    def test_existing_foreign_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            (train / "meta").mkdir(parents=True)
            (train / "meta/info.json").write_text("{}")
            occupied = root / "cache/local/nero_box_v1_train"
            occupied.mkdir(parents=True)
            marker = occupied / "keep.txt"
            marker.write_text("keep")
            with self.assertRaises(ValueError):
                connect_train(root / "cache", train)
            self.assertEqual(marker.read_text(), "keep")


class ReadinessTests(unittest.TestCase):
    def test_output_revisions_keep_existing_statistics_usable(self):
        model = Path(__file__).resolve().parents[2] / 'scripts/model'
        for name, (previous, current) in PRESENTATION_SOURCE_VERSIONS.items():
            with self.subTest(name=name):
                self.assertEqual(digest(model / name), current)
                self.assertTrue(source_version_matches(name, previous, current))
                self.assertFalse(source_version_matches(name, previous, 'unreviewed change'))

    def fixture(self, root):
        scripts = root / "scripts"
        scripts.mkdir()
        hashes = {}
        for name in ("nero_pi0_adapter.py", "nero_pi0_config.py", "prepare_nero_pi0.py"):
            (scripts / name).write_text("# fixture")
            hashes[name] = digest(scripts / name)
        assets = root / "assets/pi0_nero_box_v1/local/nero_box_v1_train"
        assets.mkdir(parents=True)
        (assets / "nero_norm_report.json").write_text(json.dumps({
            "status": "train_only_norm_stats_ready_not_trained", "validation_values_used": False,
            "state_vectors": 2710, "action_horizon": 50, "source_sha256": hashes,
        }))
        norm = {key: {"mean": [0.0] * 16, "std": [0.001] * 16} for key in ("state", "actions")}
        (assets / "norm_stats.json").write_text(json.dumps({"norm_stats": norm}))
        params = root / "checkpoints/openpi-assets/checkpoints/pi0_base/params"
        params.mkdir(parents=True)
        (params / "_METADATA").write_text("{}")
        train = root / "data/nero_lerobot_v1/train"
        train.mkdir(parents=True)
        links = root / "data/nero_openpi_cache_v1/local"
        links.mkdir(parents=True)
        (links / "nero_box_v1_train").symlink_to(train, target_is_directory=True)
        return scripts, assets

    @mock.patch("scripts.model.run_nero_pi0.subprocess.check_output", return_value=EXPECTED_COMMIT)
    def test_matching_sources_and_statistics_are_accepted(self, command):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts, _ = self.fixture(root)
            self.assertEqual(readiness(root, scripts)["openpi_commit"], EXPECTED_COMMIT)

    def test_stale_adapter_statistics_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts, _ = self.fixture(root)
            (scripts / "nero_pi0_adapter.py").write_text("# changed")
            with self.assertRaisesRegex(ValueError, "版本不一致"):
                readiness(root, scripts)

    def test_invalid_saved_statistics_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts, assets = self.fixture(root)
            norm = json.loads((assets / "norm_stats.json").read_text())
            norm["norm_stats"]["state"]["std"][7] = 0
            (assets / "norm_stats.json").write_text(json.dumps(norm))
            with self.assertRaisesRegex(ValueError, "尺度异常"):
                readiness(root, scripts)


class TrainingRecordTests(unittest.TestCase):
    def test_longer_experiment_keeps_data_and_lr_unchanged(self):
        @dataclasses.dataclass(frozen=True)
        class Config:
            num_train_steps: int = 100
            save_interval: int = 100
            keep_period: int = 5000
            data: tuple = ("same-nero-data",)
            lr_schedule: tuple = (10, 2.5e-5, 1000, 2.5e-6)
        original = Config()
        extended = training_config(original, steps=1000, save_interval=250)
        self.assertEqual(extended.num_train_steps, 1000)
        self.assertEqual(extended.save_interval, 250)
        self.assertIs(extended.data, original.data)
        self.assertIs(extended.lr_schedule, original.lr_schedule)
        self.assertEqual(original.num_train_steps, 100)
        with self.assertRaises(ValueError):
            training_config(original, steps=0)

    def test_captures_stdout_loss_and_logging_and_restores_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "console.log"
            stream = io.StringIO()
            handler = logging.StreamHandler(stream)
            root = logging.getLogger()
            previous_handlers, previous_level = root.handlers[:], root.level
            root.handlers = [handler]
            root.setLevel(logging.INFO)
            try:
                with mock.patch("sys.stdout", new=io.StringIO()), mock.patch("sys.stderr", new=io.StringIO()):
                    with capture_training_output(target):
                        print("Step 0: loss=1.2000, grad_norm=0.3000, param_norm=10.0")
                        logging.info("checkpoint saved")
                self.assertIs(handler.stream, stream)
                contents = target.read_text()
                self.assertIn("checkpoint saved", contents)
                self.assertEqual(read_training_metrics(contents)[0]["loss"], 1.2)
            finally:
                root.handlers, root.level = previous_handlers, previous_level

    def test_parses_scientific_metrics_and_deduplicates_progress_reprints(self):
        values = read_training_metrics("\rStep 0: loss=1.0, grad_norm=2.0\n"
                                       "Step 10: loss=2.0e-2, grad_norm=1.0\r"
                                       "Step 10: loss=2.0e-2, grad_norm=1.0\n")
        self.assertEqual([v["step"] for v in values], [0, 10])
        self.assertEqual(values[-1]["loss"], 0.02)

    def test_nonfinite_loss_or_missing_metrics_are_rejected(self):
        for contents in ("Step 10: loss=nan, grad_norm=1", "Step 10: loss=1, grad_norm=inf", "no metrics"):
            with self.subTest(contents=contents):
                with self.assertRaises(ValueError):
                    read_training_metrics(contents)


if __name__ == "__main__":
    unittest.main()
