#!/usr/bin/env python3
"""将 NERO 15 Hz 中间数据导出为服务器已安装的 LeRobot 格式；不联网、不控制硬件。"""
import argparse
import hashlib
import importlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


ROLES = ("base", "left_wrist", "right_wrist")
FEATURE_NAMES = ([f"left_joint_{i}_rad" for i in range(1, 8)] + ["left_gripper_m"]
                 + [f"right_joint_{i}_rad" for i in range(1, 8)] + ["right_gripper_m"])
SERVER_OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image(stored_path, original_episode, raw_episode, role):
    """只按原始 episode 内的相对路径重映射，绝不回退读取旧机器绝对路径。"""
    relative = Path(str(stored_path)).relative_to(Path(original_episode))
    require(relative.parts[:2] == ("images", role), f"图像角色/路径错误：{stored_path}")
    candidate = (raw_episode / relative).resolve()
    require(candidate.is_relative_to(raw_episode.resolve()), "图像路径超出原始 episode")
    require(candidate.is_file(), f"缺少图像：{candidate}")
    return candidate


def read_episode(aligned, report):
    with np.load(aligned / report["episode"] / "aligned.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in
                ("state", "action", "timestamp_s", "host_grid_ns", "image_paths")}


def preflight(aligned, raw_root):
    manifest = read_json(aligned / "dataset_manifest.json")
    split = read_json(aligned / "split.json")
    require(manifest["status"] == "alignment_prepared_not_lerobot", "中间数据未完整生成")
    require(manifest["fps"] == 15, "本轮只接受已核实的 15 Hz 数据")
    require(manifest["feature_order"] == FEATURE_NAMES, "16 维字段顺序改变")
    require(tuple(manifest["camera_order"]) == ROLES, "三路相机顺序改变")
    require(manifest["action_semantics"] ==
            "absolute commanded joint positions (rad) and commanded gripper width (m)",
            "动作语义不符：必须是绝对目标 rad/m")
    reports = {report["episode"]: report for report in manifest["episodes"]}
    require(len(reports) == manifest["episode_count"] == len(manifest["episodes"]),
            "episode 数量不符或重复")
    require(split.get("unit") == "whole episode", "划分单位必须是完整 episode")
    require(split["train"] and split["validation"], "训练/验证集不能为空")
    names = split["train"] + split["validation"]
    require(len(names) == len(set(names)) and set(names) == set(reports),
            "划分存在重复、交叉或遗漏")
    image_count = 0
    total = 0
    for name in names:
        require(Path(name).name == name and name.startswith("episode_"), "非法 episode 名称")
        report = reports[name]
        raw_episode = raw_root / name
        for filename, expected in report["source_hashes"].items():
            require(Path(filename).name == filename, "非法原始来源文件名")
            require(sha256(raw_episode / filename) == expected, f"原始文件校验失败：{name}/{filename}")
        require(read_json(aligned / name / "alignment_report.json") == report,
                f"对齐报告与总清单不符：{name}")
        data = read_episode(aligned, report)
        n = report["frame_count"]
        require(data["state"].shape == data["action"].shape == (n, 16), f"维度错误：{name}")
        require(np.isfinite(data["state"]).all() and np.isfinite(data["action"]).all(),
                f"非有限数值：{name}")
        require(data["image_paths"].shape == (n, 3), f"图像索引维度错误：{name}")
        grid = data["host_grid_ns"]
        require(grid.shape == (n,) and n > 1 and np.all(np.abs(np.diff(grid) - 1e9 / 15) <= 1),
                f"时间轴不是连续 15 Hz：{name}")
        require(data["timestamp_s"].shape == (n,) and
                np.allclose(data["timestamp_s"], np.arange(n) / 15, rtol=0, atol=1e-8),
                f"相对时间不符：{name}")
        require(isinstance(report["task"], str) and bool(report["task"].strip()), "任务指令为空")
        checked = set()
        for paths in data["image_paths"]:
            for role, stored in zip(ROLES, paths):
                path = resolve_image(stored, report["raw_episode"], raw_episode, role)
                if path not in checked:
                    with Image.open(path) as image:
                        require(image.size == (640, 480) and image.mode == "RGB", f"图像规格错误：{path}")
                        image.load()
                    checked.add(path)
        image_count += len(checked)
        total += n
        print(f"{name}: {n} 帧，路径、图像、来源校验通过", flush=True)
    require(total == manifest["total_frames"], "总帧数不符")
    print(f"预检查通过：{len(reports)} 条、{total} 帧、{image_count} 张被引用图像；"
          f"训练 {len(split['train'])} 条，验证 {len(split['validation'])} 条。", flush=True)
    return manifest, split, reports


def load_lerobot():
    os.environ["HF_HUB_OFFLINE"] = "1"
    module = importlib.import_module("lerobot.common.datasets.lerobot_dataset")
    cls = module.LeRobotDataset
    require(metadata.version("lerobot") == "0.1.0", "本脚本针对已确认的 LeRobot 0.1.0；不要自动升级依赖")
    parameters = inspect.signature(cls.create).parameters
    require({"repo_id", "fps", "root", "features", "use_videos"} <= set(parameters),
            "LeRobot create() 接口与截图不符")
    # 已知签名的 save_episode() 没有 task 参数；要求 add_frame() 支持逐帧 task。
    require("task" in inspect.getsource(cls.add_frame),
            "当前 add_frame() 未能确认 task 支持；请提供该函数源码，不要省略语言标签")
    return cls


def features():
    result = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": list(FEATURE_NAMES)},
        "action": {"dtype": "float32", "shape": (16,), "names": list(FEATURE_NAMES)},
    }
    for role in ROLES:
        result[f"observation.images.{role}"] = {
            "dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
    return result


def numpy_value(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def export_split(cls, aligned, raw_root, output, repo_id, names, reports):
    dataset = cls.create(repo_id=repo_id, root=output, fps=15, robot_type="nero_bimanual",
                         features=features(), use_videos=False,
                         image_writer_processes=0, image_writer_threads=0)
    mapping = []
    for episode_index, name in enumerate(names):
        report = reports[name]
        data = read_episode(aligned, report)
        for index in range(report["frame_count"]):
            frame = {"observation.state": data["state"][index].astype(np.float32).copy(),
                     "action": data["action"][index].astype(np.float32).copy(),
                     "task": report["task"]}
            for role, stored in zip(ROLES, data["image_paths"][index]):
                path = resolve_image(stored, report["raw_episode"], raw_root / name, role)
                with Image.open(path) as image:
                    frame[f"observation.images.{role}"] = np.array(image.convert("RGB"), dtype=np.uint8)
            dataset.add_frame(frame)
        dataset.save_episode()
        mapping.append({"episode_index": episode_index, "source_episode": name,
                        "frames": report["frame_count"], "task": report["task"]})
        print(f"已导出 {repo_id}: {name}", flush=True)
    # 部分旧版需 consolidate() 才能重载；新版没有此方法。
    consolidate = getattr(dataset, "consolidate", None)
    if callable(consolidate):
        options = {"run_compute_stats": True} if "run_compute_stats" in inspect.signature(consolidate).parameters else {}
        consolidate(**options)
    del dataset
    loaded = cls(repo_id, root=output)
    expected_count = sum(item["frames"] for item in mapping)
    require(len(loaded) == expected_count, f"重载帧数不符：{repo_id}")
    offset = 0
    for item in mapping:
        data = read_episode(aligned, reports[item["source_episode"]])
        for index in sorted({0, item["frames"] // 2, item["frames"] - 1}):
            row = loaded[offset + index]
            for feature, source in (("observation.state", "state"), ("action", "action")):
                require(np.allclose(numpy_value(row[feature]), data[source][index], rtol=0, atol=1e-7),
                        f"导出后数值改变：{repo_id}/{feature}")
            require(row["task"] == item["task"], "导出后语言指令改变")
            require(int(row["episode_index"]) == item["episode_index"], "episode 边界错误")
            require(abs(float(row["timestamp"]) - index / 15) < 1e-4, "导出后时间轴改变")
            for role in ROLES:
                value = numpy_value(row[f"observation.images.{role}"])
                require(value.shape == (3, 480, 640) and np.isfinite(value).all(),
                        f"导出后图像不可读取：{role}")
        offset += item["frames"]
    write_json(output / "nero_episode_map.json", mapping)
    print(f"重载检查通过：{repo_id}，{expected_count} 帧", flush=True)
    return {"repo_id": repo_id, "frames": expected_count, "episodes": mapping}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, default=root / "data/nero_aligned_v1")
    parser.add_argument("--raw-root", type=Path, default=root / "data/nero_demos")
    parser.add_argument("--output", type=Path, default=root / "data/nero_lerobot_v1")
    parser.add_argument("--repo-prefix", default="local/nero_box_v1")
    parser.add_argument("--dry-run", action="store_true", help="仅检查输入，不需要 LeRobot，不写输出")
    args = parser.parse_args()
    require(args.repo_prefix.count("/") == 1 and all(args.repo_prefix.split("/")), "repo-prefix 应为 local/名称")
    if not args.dry_run:
        require(not args.output.exists(), "输出目录已存在，拒绝覆盖；请使用新的 --output")
    manifest, split, reports = preflight(args.aligned, args.raw_root)
    if args.dry_run:
        print("仅预检查：未生成 LeRobot 数据。")
        return
    cls = load_lerobot()
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "export_report.json"
    result = {"status": "building_do_not_train", "fps": 15,
              "server_openpi_commit_from_user": SERVER_OPENPI_COMMIT,
              "lerobot_version": metadata.version("lerobot"),
              "lerobot_source": inspect.getfile(cls),
              "input_manifest_sha256": sha256(args.aligned / "dataset_manifest.json"),
              "input_split_sha256": sha256(args.aligned / "split.json"),
              "action_semantics": manifest["action_semantics"], "feature_order": FEATURE_NAMES,
              "normalisation": "not applied; use train split only for later OpenPI statistics",
              "image_storage": "image, not video; RGB; no depth",
              "splits": {}}
    write_json(report_path, result)
    try:
        for key in ("train", "validation"):
            result["splits"][key] = export_split(cls, args.aligned, args.raw_root, args.output / key,
                                                f"{args.repo_prefix}_{key}", split[key], reports)
            write_json(report_path, result)
        result["status"] = "exported_and_reload_checked_not_yet_openpi_configured"
        write_json(report_path, result)
    except BaseException as error:
        result.update(status="incomplete_do_not_train", error=str(error))
        write_json(report_path, result)
        raise
    print(f"导出完成：{args.output}\n训练/验证分别保存；未启动训练、未上传数据、未操作硬件。")


if __name__ == "__main__":
    main()
