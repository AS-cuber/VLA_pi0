#!/usr/bin/env python3
"""Evaluate a NERO checkpoint on held-out anchors, optionally against pi0_base."""
import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


JOINT_INDICES = [*range(7), *range(8, 15)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checkpoint_assets(checkpoint, training_assets):
    require(checkpoint.name.isdecimal(), "请指定完整保存的数字编号检查点目录，例如 99")
    require((checkpoint / "params/_METADATA").is_file(), "训练参数元数据不存在")
    stored = checkpoint / "assets/local/nero_box_v1_train/norm_stats.json"
    saved = json.loads(stored.read_text())["norm_stats"]
    original = json.loads((training_assets / "norm_stats.json").read_text())["norm_stats"]
    for feature in ("state", "actions"):
        for key in ("mean", "std"):
            a, b = np.asarray(saved[feature][key]), np.asarray(original[feature][key])
            require(a.shape == b.shape == (16,) and np.isfinite(a).all()
                    and np.array_equal(a, b), "检查点归一化与训练准备记录不一致")
    return stored


def validation_anchors(report, all_frames=False):
    train = report["splits"]["train"]
    validation = report["splits"]["validation"]
    train_names = {v["source_episode"] for v in train["episodes"]}
    val_names = [v["source_episode"] for v in validation["episodes"]]
    require(len(val_names) == len(set(val_names)) == 2 and not train_names.intersection(val_names),
            "验证示教划分重复或与训练集交叉")
    require(validation["frames"] == sum(v["frames"] for v in validation["episodes"]) == 719,
            "验证集帧数与已确认记录不符")
    anchors = []
    offset = 0
    for episode_index, episode in enumerate(validation["episodes"]):
        n = episode["frames"]
        require(n >= 3, "验证示教过短")
        for local_index in (range(n) if all_frames else (0, n // 2, n - 1)):
            anchors.append({"index": offset + local_index, "episode_index": episode_index,
                            "source_episode": episode["source_episode"],
                            "local_index": local_index, "remaining_frames": n - local_index})
        offset += n
    return validation["repo_id"], anchors


def error_metrics(predictions, references, valid):
    predictions, references = np.asarray(predictions), np.asarray(references)
    valid = np.asarray(valid)
    require(predictions.shape == references.shape and predictions.ndim == 3
            and predictions.shape[-1] == 16, "预测/参考动作形状错误")
    require(valid.shape == predictions.shape[:2] and valid.dtype == np.bool_
            and valid.any() and valid[:, 0].all(), "有效动作掩码错误")
    require(np.isfinite(predictions).all() and np.isfinite(references).all(), "动作中存在 NaN/Inf")
    errors = np.abs(predictions.astype(np.float64) - references.astype(np.float64))

    def summarize(value):
        return {
            "joint_mae_deg": float(np.rad2deg(value[:, JOINT_INDICES]).mean()),
            "joint_mae_per_channel_deg": np.rad2deg(value[:, JOINT_INDICES].mean(axis=0)).tolist(),
            "left_gripper_mae_mm": float(value[:, 7].mean() * 1000),
            "right_gripper_mae_mm": float(value[:, 15].mean() * 1000),
        }

    return {"anchor_count": len(predictions), "valid_future_vectors": int(valid.sum()),
            "first_action": summarize(errors[:, 0, :]),
            "all_valid_future_actions": summarize(errors[valid]),
            "note": "MAE uses valid_mask; joint errors are in deg, gripper errors in mm."}


def compare_reports(trained, base):
    require(trained["evaluated_model"] == "trained" and base["evaluated_model"] == "base",
            "对照模型标记错误")
    for key in ("evaluation_scope", "validation_repo", "trained_checkpoint", "checkpoint_stats_sha256",
                "dataset_export_sha256", "sampling", "evaluation_script_sha256", "anchors", "hold_state_metrics"):
        require(trained[key] == base[key], f"对照条件不一致：{key}")
    for key in ("anchor_count", "valid_future_vectors"):
        require(trained["metrics"][key] == base["metrics"][key], "评估动作计数不一致")
    comparison = {}
    for scope in ("first_action", "all_valid_future_actions"):
        comparison[scope] = {}
        for key in ("joint_mae_deg", "left_gripper_mae_mm", "right_gripper_mae_mm"):
            a, b = float(base["metrics"][scope][key]), float(trained["metrics"][scope][key])
            hold = float(trained["hold_state_metrics"][scope][key])
            require(np.isfinite([a, b, hold]).all() and min(a, b, hold) >= 0, "误差指标无效")
            comparison[scope][key] = {
                "base": a, "trained": b, "hold_current_state": hold, "trained_minus_base": b - a,
                "reduction_percent": (a - b) / a * 100 if a > 1e-12 else None,
                "reduction_vs_hold_percent": (hold - b) / hold * 100 if hold > 1e-12 else None,
            }
    return {"status": "paired_evaluation_passed", "evaluation_scope": trained["evaluation_scope"],
            "anchor_count": trained["metrics"]["anchor_count"],
            "valid_future_vectors": trained["metrics"]["valid_future_vectors"],
            "comparison": comparison, "note": "reduction_percent = (base - trained) / base * 100."}


def evaluate(project, checkpoint, output, all_frames=False, model_kind="trained"):
    import jax
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from openpi.policies.policy_config import create_trained_policy
    from openpi.shared import normalize
    from nero_pi0_adapter import CAMERAS
    from nero_pi0_config import HORIZON, get_train_config
    from run_nero_pi0 import readiness, digest

    require(jax.default_backend() == "gpu" and jax.device_count() == 1, "请指定一张空闲 GPU")
    provenance = readiness(project, Path(__file__).resolve().parent)
    assets = project / "assets/pi0_nero_box_v1/local/nero_box_v1_train"
    stored_stats = checkpoint_assets(checkpoint, assets)
    export_path = project / "data/nero_lerobot_v1/export_report.json"
    report = json.loads(export_path.read_text())
    repo, anchors = validation_anchors(report, all_frames)
    validation_root = project / "data/nero_lerobot_v1/validation"
    metadata = LeRobotDatasetMetadata(repo, root=validation_root)
    require(metadata.fps == 15, "验证集采样率改变")
    dataset = LeRobotDataset(repo, root=validation_root,
                             delta_timestamps={"action": [i / metadata.fps for i in range(HORIZON)]})
    require(len(dataset) == 719, "验证集读取帧数不符")
    cfg = get_train_config(project, checkpoint.parent.name)
    selected_checkpoint = checkpoint
    extra_kwargs = {}
    if model_kind == "base":
        # Remove the adapters for the original pretrained reference, keeping the same NERO transforms.
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(
            cfg.model, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"))
        selected_checkpoint = project / "checkpoints/openpi-assets/checkpoints/pi0_base"
        extra_kwargs["norm_stats"] = normalize.load(stored_stats.parent)
    logging.info("模型=%s；加载检查点：%s", model_kind, selected_checkpoint)
    start = time.monotonic()
    # This API restores checkpoint/params and checkpoint/assets; it does not use cfg.weight_loader.
    policy = create_trained_policy(cfg, selected_checkpoint, sample_kwargs={"num_steps": 10}, **extra_kwargs)
    load_s = time.monotonic() - start
    logging.info("模型已加载；耗时 %.2fs。开始 %d 个验证集时刻评估。", load_s, len(anchors))
    predictions, references, masks, elapsed_times, states = [], [], [], [], []
    for ordinal, anchor in enumerate(anchors):
        row = dataset[anchor["index"]]
        require(int(row["episode_index"]) == anchor["episode_index"], "验证集 episode 顺序改变")
        observation = {"state": np.asarray(row["observation.state"]),
                       "images": {role: np.asarray(row[f"observation.images.{role}"]) for role in CAMERAS},
                       "prompt": row["task"]}
        require(isinstance(observation["prompt"], str) and observation["prompt"].strip(), "任务文本为空")
        reference = np.asarray(row["action"])
        require(reference.shape == (HORIZON, 16), "参考动作块形状错误")
        noise = np.random.default_rng(42 + anchor["index"]).standard_normal((HORIZON, 32)).astype(np.float32)
        start = time.monotonic()
        inferred = policy.infer(observation, noise=noise)
        elapsed_times.append(time.monotonic() - start)
        predicted = np.asarray(inferred["actions"])
        require(predicted.shape == (HORIZON, 16) and np.isfinite(predicted).all(), "预测动作无效")
        predictions.append(predicted.copy())
        references.append(reference.copy())
        states.append(observation["state"].copy())
        masks.append(np.arange(HORIZON) < anchor["remaining_frames"])
        if len(anchors) <= 6 or ordinal == 0 or (ordinal + 1) % 50 == 0 or ordinal + 1 == len(anchors):
            logging.info("模型=%s，进度 %d/%d，样本 %d：shape=%s，耗时 %.3fs",
                         model_kind, ordinal + 1, len(anchors), anchor["index"], predicted.shape, elapsed_times[-1])
    predictions, references, masks = np.stack(predictions), np.stack(references), np.stack(masks)
    metrics = error_metrics(predictions, references, masks)
    held = np.broadcast_to(np.stack(states)[:, None, :], predictions.shape)
    hold_metrics = error_metrics(held, references, masks)
    np.savez_compressed(output / "validation_actions.npz", predictions=predictions,
                        references=references, valid_mask=masks, states=np.stack(states),
                        dataset_indices=np.asarray([a["index"] for a in anchors]))
    result = {
        **provenance,
        "status": "validation_evaluation_passed",
        "evaluated_model": model_kind,
        "evaluation_scope": "all_validation_frames" if all_frames else "six_anchor_smoke",
        "evaluation_script_sha256": digest(Path(__file__).resolve()),
        "dataset_export_sha256": digest(export_path),
        "initial_checkpoint_params": provenance["checkpoint_params"],
        "checkpoint_params": str(selected_checkpoint / "params"),
        "trained_checkpoint": str(checkpoint),
        "checkpoint_stats_sha256": digest(stored_stats),
        "checkpoint_metadata_sha256": digest(selected_checkpoint / "params/_METADATA"),
        "validation_repo": repo, "anchors": anchors, "metrics": metrics,
        "hold_state_metrics": hold_metrics,
        "load_seconds": load_s, "inference_seconds": elapsed_times,
        "sampling": {"num_steps": 10, "noise_seed": "42 + validation dataset index"},
        "normalization": "Loaded from the trained checkpoint; no validation statistics fitted.",
    }
    logging.info("首步参考误差：关节 %.3f 度；左夹爪 %.3f mm；右夹爪 %.3f mm",
                 metrics["first_action"]["joint_mae_deg"], metrics["first_action"]["left_gripper_mae_mm"],
                 metrics["first_action"]["right_gripper_mae_mm"])
    for episode_index in range(2):
        selection = np.asarray([a["episode_index"] == episode_index for a in anchors])
        result.setdefault("per_episode_metrics", {})[str(episode_index)] = error_metrics(
            predictions[selection], references[selection], masks[selection])
    logging.info("全部有效未来动作误差：关节 %.3f 度；左夹爪 %.3f mm；右夹爪 %.3f mm",
                 metrics["all_valid_future_actions"]["joint_mae_deg"],
                 metrics["all_valid_future_actions"]["left_gripper_mae_mm"],
                 metrics["all_valid_future_actions"]["right_gripper_mae_mm"])
    return result


def paired_evaluation(project, checkpoint, output, all_frames):
    # Separate sequential processes ensure the first model releases its GPU memory before the next loads.
    reports = {}
    for kind in ("trained", "base"):
        child_output = output / kind
        command = [sys.executable, str(Path(__file__).resolve()), "--project", str(project),
                   "--checkpoint", str(checkpoint), "--model", kind, "--output", str(child_output)]
        if all_frames:
            command.append("--all-frames")
        logging.info("启动独立评估进程：%s", kind)
        subprocess.run(command, check=True)
        reports[kind] = json.loads((child_output / "report.json").read_text())
        require(reports[kind]["status"] == "validation_evaluation_passed", "子评估未通过")
    comparison = compare_reports(reports["trained"], reports["base"])
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, values in comparison["comparison"]["first_action"].items():
        logging.info("首步 %s：base=%.6f，trained=%.6f，保持当前状态=%.6f，较 base 误差降低百分比=%s",
                     key, values["base"], values["trained"], values["hold_current_state"], values["reduction_percent"])
    return {**comparison, "trained_report": str(output / "trained/report.json"),
            "base_report": str(output / "base/report.json"), "comparison_file": str(output / "comparison.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/home/clwu/vla-wu"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--all-frames", action="store_true", help="评估两条验证示教的全部 719 帧")
    parser.add_argument("--model", choices=("trained", "base"), default="trained")
    parser.add_argument("--compare-base", action="store_true", help="顺序评估训练模型和基础模型并保存对照")
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    project = args.project.resolve()
    checkpoint = (args.checkpoint or project / "checkpoints/pi0_nero_box_v1/nero_smoke_v1/99").resolve()
    os.environ["HF_LEROBOT_HOME"] = str(project / "data/nero_openpi_cache_v1")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    parent = project / "results/pi0_nero_box_v1"
    parent.mkdir(parents=True, exist_ok=True)
    if args.output is not None:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="comparison_" if args.compare_base else "checkpoint_check_", dir=parent))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")], force=True)
    report = {"status": "running", "trained_checkpoint": str(checkpoint), "result_dir": str(output)}
    logging.info("本次结果目录：%s", output)
    try:
        if args.compare_base:
            report.update(paired_evaluation(project, checkpoint, output, args.all_frames))
            logging.info("基础模型与微调模型对照完成：%d 个验证时刻。", report["anchor_count"])
        else:
            report.update(evaluate(project, checkpoint, output, args.all_frames, args.model))
            logging.info("%s 模型评估完成：%d 个验证时刻。", args.model, report["metrics"]["anchor_count"])
    except BaseException as error:
        report.update(status="failed_or_interrupted", error_type=type(error).__name__, error=str(error))
        logging.exception("本轮失败或中断，结果已保留。")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("本次报告：%s", output / "report.json")


if __name__ == "__main__":
    main()
