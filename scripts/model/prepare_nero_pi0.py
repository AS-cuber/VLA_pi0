#!/usr/bin/env python3
"""Connect the existing NERO train split and save train-only OpenPI norm statistics."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


EXPECTED_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
EXPECTED_FEATURES = ([f"left_joint_{i}_rad" for i in range(1, 8)] + ["left_gripper_m"]
                     + [f"right_joint_{i}_rad" for i in range(1, 8)] + ["right_gripper_m"])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_report(report):
    require(report["status"] == "exported_and_reload_checked_not_yet_openpi_configured",
            "LeRobot 导出尚未完成")
    require(report["fps"] == 15 and report["feature_order"] == EXPECTED_FEATURES,
            "采样率或 16 维字段顺序改变")
    require(report["action_semantics"] ==
            "absolute commanded joint positions (rad) and commanded gripper width (m)",
            "输入动作必须是绝对关节目标和夹爪宽度")
    train = report["splits"]["train"]
    validation = report["splits"]["validation"]
    require((len(train["episodes"]), train["frames"]) == (8, 2710), "训练集记录不符")
    require((len(validation["episodes"]), validation["frames"]) == (2, 719), "验证集记录不符")
    train_names = [v["source_episode"] for v in train["episodes"]]
    val_names = [v["source_episode"] for v in validation["episodes"]]
    require(len(set(train_names + val_names)) == 10, "训练与验证划分重复或交叉")
    require(sum(v["frames"] for v in train["episodes"]) == 2710, "训练分段帧数不符")
    require(train["repo_id"] == "local/nero_box_v1_train", "训练 repo_id 改变")
    return train_names


def connect_train(cache_root, train_root):
    """Use a project-specific cache; refuse to replace an existing different target."""
    train_root = train_root.resolve(strict=True)
    require((train_root / "meta/info.json").is_file(), "训练目录缺少 meta/info.json")
    destination = cache_root / "local/nero_box_v1_train"
    if destination.exists() or destination.is_symlink():
        require(destination.is_symlink() and destination.resolve() == train_root,
                f"目标已存在且不是本训练集链接：{destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(train_root, target_is_directory=True)
    return destination


def calculate_stats(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == 16 and len(values) > 1,
            "统计输入必须为 (N,16)，且 N>1")
    require(np.isfinite(values).all(), "统计输入出现 NaN/Inf")
    raw_std = values.std(axis=0)
    # Initial engineering floors: 1 mrad for joints, 1 mm for gripper widths.
    floor = np.full(16, 1e-3, dtype=np.float64)
    return {
        "mean": values.mean(axis=0),
        "std": np.maximum(raw_std, floor),
        "q01": np.quantile(values, 0.01, axis=0),
        "q99": np.quantile(values, 0.99, axis=0),
    }, {"raw_std": raw_std.tolist(), "scale_floor": floor.tolist(),
        "floored_channels": np.flatnonzero(raw_std < floor).tolist()}


def prepare(project):
    project = project.resolve()
    data_root = project / "data/nero_lerobot_v1"
    report_path = data_root / "export_report.json"
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    train_names = check_report(report)
    commit = subprocess.check_output(
        ["git", "-C", str(project / "vendor/openpi"), "rev-parse", "HEAD"], text=True).strip()
    require(commit == EXPECTED_COMMIT, f"OpenPI 版本不同：{commit}；保留输出后核对")
    output = project / "assets/pi0_nero_box_v1/local/nero_box_v1_train"
    require(not output.exists(), f"统计输出已存在，拒绝覆盖：{output}")
    cache_root = project / "data/nero_openpi_cache_v1"
    link = connect_train(cache_root, data_root / "train")
    os.environ["HF_LEROBOT_HOME"] = str(cache_root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    from lerobot.common.constants import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared import normalize
    from openpi.training import data_loader
    from nero_pi0_config import HORIZON, TRAIN_REPO, statistics_data_config
    from nero_pi0_adapter import JOINT_MASK, NeroOutputs

    require(Path(HF_LEROBOT_HOME).resolve() == cache_root.resolve(),
            "LeRobot 未采用本项目数据根目录；请使用新 Python 进程执行")
    metadata = LeRobotDatasetMetadata(TRAIN_REPO)
    require(Path(metadata.root).resolve() == (data_root / "train").resolve(), "训练路径未接入")
    require(metadata.fps == 15, "LeRobot 采样率改变")
    spec = statistics_data_config()
    dataset = data_loader.create_torch_dataset(spec, HORIZON, Pi0Config())
    require(len(dataset) == 2710, "OpenPI 读取训练帧数错误")
    transform = transforms.compose([*spec.repack_transforms.inputs, *spec.data_transforms.inputs])
    states, actions = [], []
    first = None
    print(f"OpenPI 已按 repo_id 接入训练目录：{link}，{len(dataset)} 帧", flush=True)
    offset = 0
    expected_episodes = np.repeat(np.arange(8), [v["frames"] for v in report["splits"]["train"]["episodes"]])
    for index in range(len(dataset)):
        row = dataset[index]
        require(int(row["episode_index"]) == int(expected_episodes[index]), "训练 episode 顺序改变")
        raw_action = np.asarray(row["action"]).copy()
        transformed = transform(row)
        require(transformed["state"].shape == (16,) and
                transformed["actions"].shape == (HORIZON, 16), "统计前状态/动作形状错误")
        if first is None:
            first = (copy.deepcopy(transformed), raw_action)
        states.append(transformed["state"].copy())
        actions.append(transformed["actions"].copy())
        offset += 1
        if offset % 250 == 0 or offset == len(dataset):
            print(f"统计读取进度：{offset}/{len(dataset)}", flush=True)

    norm_stats, audit = {}, {}
    for key, values in (("state", np.stack(states)), ("actions", np.concatenate(actions))):
        result, audit[key] = calculate_stats(values)
        norm_stats[key] = normalize.NormStats(**result)
        print(f"{key} 启用尺度下限的通道：{audit[key]['floored_channels']}", flush=True)
        print(f"{key} 左/右夹爪原始标准差：{np.asarray(audit[key]['raw_std'])[[7, 15]].tolist()}")

    relative, raw_action = first
    value = transforms.Normalize(norm_stats)(copy.deepcopy(relative))
    value = transforms.PadStatesAndActions(32)(value)
    require(np.isfinite(value["state"]).all() and np.isfinite(value["actions"]).all(),
            "归一化结果出现 NaN/Inf")
    value = transforms.Unnormalize(norm_stats)(value)
    value = transforms.AbsoluteActions(JOINT_MASK)(value)
    restored = NeroOutputs()(value)["actions"]
    require(np.allclose(restored, raw_action, atol=1e-5, rtol=0), "归一化全链路还原失败")

    audit.update(
        status="train_only_norm_stats_ready_not_trained",
        openpi_commit=commit,
        input_export_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        train_repo=TRAIN_REPO,
        train_source_episodes=train_names,
        state_vectors=2710,
        action_vectors=2710 * HORIZON,
        action_horizon=HORIZON,
        feature_order=EXPECTED_FEATURES,
        normalisation="z-score; floors are engineering choices: joints 0.001 rad, grippers 0.001 m",
        action_sampling="all 50 steps per training anchor, including LeRobot terminal padding",
        validation_values_used=False,
        transformation="14 joints relative to anchor state; both grippers absolute width",
        full_normalize_pad_inverse_check=True,
        source_sha256={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                       for name in ("nero_pi0_adapter.py", "nero_pi0_config.py", "prepare_nero_pi0.py")},
    )
    output.mkdir(parents=True, exist_ok=False)
    normalize.save(output, norm_stats)
    (output / "nero_norm_report.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded = normalize.load(output)
    for key in norm_stats:
        require(np.array_equal(loaded[key].mean, norm_stats[key].mean) and
                np.array_equal(loaded[key].std, norm_stats[key].std), "统计文件重载不符")
    print(f"归一化统计：{output / 'norm_stats.json'}")
    print("统计输入：train，8条示教，2710帧；动作块50步。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/home/clwu/vla-wu"))
    args = parser.parse_args()
    prepare(args.project)


if __name__ == "__main__":
    main()
