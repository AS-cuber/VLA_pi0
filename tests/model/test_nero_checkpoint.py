"""Held-out selection and physical-unit metrics for checkpoint reload checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.model.check_nero_checkpoint import JOINT_INDICES, checkpoint_assets, compare_reports, error_metrics, validation_anchors


class CheckpointEvaluationTests(unittest.TestCase):
    def report(self):
        return {"splits": {
            "train": {"episodes": [{"source_episode": f"train_{i}"} for i in range(8)]},
            "validation": {"repo_id": "local/nero_box_v1_validation", "frames": 719,
                           "episodes": [{"source_episode": "val_0", "frames": 344},
                                        {"source_episode": "val_1", "frames": 375}]},
        }}

    def test_held_out_anchor_boundaries(self):
        repo, anchors = validation_anchors(self.report())
        self.assertEqual(repo, "local/nero_box_v1_validation")
        self.assertEqual([a["index"] for a in anchors], [0, 172, 343, 344, 531, 718])
        self.assertEqual(sum(min(50, a["remaining_frames"]) for a in anchors), 202)

    def test_training_overlap_rejected(self):
        report = self.report()
        report["splits"]["validation"]["episodes"][0]["source_episode"] = "train_0"
        with self.assertRaises(ValueError):
            validation_anchors(report)

    def test_full_validation_covers_each_frame_and_excludes_terminal_padding(self):
        _, anchors = validation_anchors(self.report(), all_frames=True)
        self.assertEqual([a["index"] for a in anchors], list(range(719)))
        self.assertEqual(sum(min(50, a["remaining_frames"]) for a in anchors), 33500)
        self.assertEqual(sum(a["episode_index"] == 0 for a in anchors), 344)

    def test_joint_and_gripper_units_are_separate(self):
        prediction = np.zeros((2, 3, 16))
        reference = prediction.copy()
        prediction[..., JOINT_INDICES] = np.pi / 180
        prediction[..., 7] = 0.002
        prediction[..., 15] = 0.003
        metrics = error_metrics(prediction, reference, np.ones((2, 3), dtype=bool))
        self.assertAlmostEqual(metrics["first_action"]["joint_mae_deg"], 1)
        self.assertAlmostEqual(metrics["first_action"]["left_gripper_mae_mm"], 2)
        self.assertAlmostEqual(metrics["first_action"]["right_gripper_mae_mm"], 3)
        self.assertEqual(len(metrics["first_action"]["joint_mae_per_channel_deg"]), 14)

    def test_terminal_padding_is_excluded_from_future_metrics(self):
        prediction = np.zeros((1, 3, 16))
        prediction[:, 1:] = 1000
        metrics = error_metrics(prediction, np.zeros_like(prediction), np.array([[True, False, False]]))
        self.assertEqual(metrics["valid_future_vectors"], 1)
        self.assertEqual(metrics["all_valid_future_actions"]["joint_mae_deg"], 0)

    def test_nonfinite_output_rejected(self):
        with self.assertRaises(ValueError):
            error_metrics(np.full((1, 3, 16), np.nan), np.zeros((1, 3, 16)), np.ones((1, 3), bool))

    def test_checkpoint_statistics_must_match_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint/99"
            (checkpoint / "params").mkdir(parents=True)
            (checkpoint / "params/_METADATA").write_text("{}")
            stored = checkpoint / "assets/local/nero_box_v1_train"
            stored.mkdir(parents=True)
            original = root / "training_assets"
            original.mkdir()
            value = {"norm_stats": {key: {"mean": [0.] * 16, "std": [1.] * 16}
                                    for key in ("state", "actions")}}
            for path in (stored, original):
                (path / "norm_stats.json").write_text(json.dumps(value))
            self.assertEqual(checkpoint_assets(checkpoint, original), stored / "norm_stats.json")
            value["norm_stats"]["actions"]["std"][7] = 2
            (stored / "norm_stats.json").write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                checkpoint_assets(checkpoint, original)

    def comparison_report(self, kind, error):
        metrics = {scope: {"joint_mae_deg": error, "left_gripper_mae_mm": error,
                           "right_gripper_mae_mm": error}
                   for scope in ("first_action", "all_valid_future_actions")}
        metrics.update(anchor_count=719, valid_future_vectors=33500)
        hold_metrics = {scope: {"joint_mae_deg": 0.25, "left_gripper_mae_mm": 0.25,
                               "right_gripper_mae_mm": 0.25}
                        for scope in ("first_action", "all_valid_future_actions")}
        return {"evaluated_model": kind, "evaluation_scope": "all_validation_frames",
                "validation_repo": "val", "trained_checkpoint": "checkpoint/99",
                "checkpoint_stats_sha256": "same-normalization", "dataset_export_sha256": "same-data",
                "sampling": {"noise_seed": 42}, "evaluation_script_sha256": "same-script",
                "anchors": list(range(719)), "metrics": metrics, "hold_state_metrics": hold_metrics}

    def test_comparison_reports_worsening_without_relabeling_it(self):
        result = compare_reports(self.comparison_report("trained", 2), self.comparison_report("base", 1))
        joint = result["comparison"]["first_action"]["joint_mae_deg"]
        self.assertEqual(joint["trained_minus_base"], 1)
        self.assertEqual(joint["reduction_percent"], -100)

    def test_comparison_rejects_different_sampling_conditions(self):
        trained = self.comparison_report("trained", 0.5)
        base = self.comparison_report("base", 1)
        base["sampling"] = {"noise_seed": 43}
        with self.assertRaisesRegex(ValueError, "条件不一致"):
            compare_reports(trained, base)

    def test_zero_baseline_error_does_not_produce_infinite_percentages(self):
        result = compare_reports(self.comparison_report("trained", 0), self.comparison_report("base", 0))
        self.assertIsNone(result["comparison"]["first_action"]["joint_mae_deg"]["reduction_percent"])


if __name__ == "__main__":
    unittest.main()
