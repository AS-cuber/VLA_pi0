#!/usr/bin/env python3
"""CPU-only horizon and right-gripper diagnostics from an existing validation NPZ."""
import argparse
import csv
import json
from pathlib import Path
import tempfile

import numpy as np


JOINTS = [*range(7), *range(8, 15)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def diagnose(arrays, anchors):
    predictions, references = arrays["predictions"], arrays["references"]
    states, mask = arrays["states"], arrays["valid_mask"]
    indices = arrays["dataset_indices"]
    n = len(anchors)
    require(predictions.shape == references.shape == (n, 50, 16) and states.shape == (n, 16),
            "预测/目标/状态形状不符")
    require(mask.shape == (n, 50) and mask.dtype == np.bool_, "有效目标掩码形状错误")
    require(np.array_equal(indices, [a["index"] for a in anchors]), "NPZ 与报告的样本顺序不符")
    require(np.array_equal(indices, np.arange(n)) and mask[:, 0].all(), "完整验证样本不连续或首步无效")
    expected = np.arange(50)[None, :] < np.asarray([a["remaining_frames"] for a in anchors])[:, None]
    require(np.array_equal(mask, expected), "末端填充掩码与报告不一致")
    require(all(np.isfinite(v).all() for v in (predictions, references, states)), "动作或状态含非有限值")
    predictions, references, states = [v.astype(np.float64) for v in (predictions, references, states)]
    episodes = np.asarray([a["episode_index"] for a in anchors])
    horizon_rows = []
    for episode in [None, *sorted(set(episodes.tolist()))]:
        group = np.ones(n, bool) if episode is None else episodes == episode
        for h in (0, 1, 4, 9, 24, 49):
            selected = group & mask[:, h]
            if not selected.any():
                continue
            error = predictions[selected, h] - references[selected, h]
            hold_error = states[selected] - references[selected, h]
            horizon_rows.append({
                "episode": "all" if episode is None else str(episode), "horizon_index": h,
                "label_offset_s": h / 15, "count": int(selected.sum()),
                "joint_mae_deg": float(np.rad2deg(np.abs(error[:, JOINTS])).mean()),
                "hold_joint_mae_deg": float(np.rad2deg(np.abs(hold_error[:, JOINTS])).mean()),
                "right_gripper_mae_mm": float(np.abs(error[:, 15]).mean() * 1000),
                "hold_right_gripper_mae_mm": float(np.abs(hold_error[:, 15]).mean() * 1000),
                "right_gripper_bias_mm": float(error[:, 15].mean() * 1000),
                "right_gripper_abs_error_p95_mm": float(np.quantile(np.abs(error[:, 15]), 0.95) * 1000),
            })
    right_rows = []
    for i, anchor in enumerate(anchors):
        error_mm = float((predictions[i, 0, 15] - references[i, 0, 15]) * 1000)
        right_rows.append({
            "dataset_index": int(indices[i]), "source_episode": anchor["source_episode"],
            "local_frame": anchor["local_index"], "aligned_episode_time_s": anchor["local_index"] / 15,
            "state_width_mm": float(states[i, 15] * 1000),
            "predicted_width_mm": float(predictions[i, 0, 15] * 1000),
            "target_width_mm": float(references[i, 0, 15] * 1000),
            "signed_error_mm": error_mm, "absolute_error_mm": abs(error_mm),
        })
    per_joint = []
    for i in JOINTS:
        name = f"left_J{i + 1}" if i < 7 else f"right_J{i - 7}"
        error = predictions[:, 0, i] - references[:, 0, i]
        per_joint.append({"joint": name, "mae_deg": float(np.rad2deg(np.abs(error)).mean()),
                          "signed_bias_deg": float(np.rad2deg(error).mean()),
                          "hold_mae_deg": float(np.rad2deg(np.abs(states[:, i] - references[:, 0, i])).mean())})
    return {"horizons": horizon_rows, "first_action_joints": per_joint,
            "worst_right_gripper_first_actions": sorted(right_rows, key=lambda row: row["absolute_error_mm"], reverse=True)[:10],
            "right_gripper_first_actions": right_rows}


def latest_evaluation(project):
    candidates = []
    for path in (project / "results/pi0_nero_box_v1").glob("comparison_*/trained/report.json"):
        report = json.loads(path.read_text())
        if (report.get("status") == "validation_evaluation_passed"
                and report.get("evaluation_scope") == "all_validation_frames"
                and Path(report["trained_checkpoint"]).name == "999"):
            candidates.append((path.stat().st_mtime, path, report))
    require(candidates, "未找到 999 检查点的完整验证报告，可使用 --evaluation 指定 trained/report.json")
    _, path, report = max(candidates, key=lambda item: item[0])
    return path, report


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/home/clwu/vla-wu"))
    parser.add_argument("--evaluation", type=Path, help="可选：明确指定 trained/report.json")
    args = parser.parse_args()
    project = args.project.resolve()
    if args.evaluation:
        path = args.evaluation.resolve()
        report = json.loads(path.read_text())
    else:
        path, report = latest_evaluation(project)
    require(report["status"] == "validation_evaluation_passed"
            and report["evaluation_scope"] == "all_validation_frames"
            and report["evaluated_model"] == "trained", "请使用训练模型的完整验证报告")
    require(len(report["anchors"]) == 719, "期望 719 个验证时刻")
    with np.load(path.parent / "validation_actions.npz", allow_pickle=False) as archive:
        result = diagnose(archive, report["anchors"])
    parent = project / "results/pi0_nero_box_v1"
    output = Path(tempfile.mkdtemp(prefix="action_diagnostics_", dir=parent))
    result.update(status="offline_action_diagnostics_completed", evaluation_report=str(path),
                  trained_checkpoint=report["trained_checkpoint"],
                  note="Bias = prediction minus target; label_offset_s = horizon_index / 15.")
    (output / "diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_csv(output / "horizon_errors.csv", result["horizons"])
    save_csv(output / "right_gripper_first_actions.csv", result["right_gripper_first_actions"])
    print("读取评估：", path)
    print("标签偏移(s) | 关节MAE/保持状态(度) | 右夹爪MAE/保持状态(mm) | 右夹爪偏差(mm)")
    for row in result["horizons"]:
        if row["episode"] == "all":
            print(f"{row['label_offset_s']:6.3f} | {row['joint_mae_deg']:.3f}/{row['hold_joint_mae_deg']:.3f}"
                  f" | {row['right_gripper_mae_mm']:.3f}/{row['hold_right_gripper_mae_mm']:.3f}"
                  f" | {row['right_gripper_bias_mm']:+.3f}")
    print("右夹爪首步误差最大的位置：")
    for row in result["worst_right_gripper_first_actions"][:5]:
        print(f"{row['source_episode']}，帧 {row['local_frame']}，"
              f"目标 {row['target_width_mm']:.2f} mm，预测 {row['predicted_width_mm']:.2f} mm，"
              f"偏差 {row['signed_error_mm']:+.2f} mm")
    print("分析结果目录：", output)


if __name__ == "__main__":
    main()
