#!/usr/bin/env python3
"""Verify pi0_base, or run a separate NERO LoRA experiment with saved loss logs."""
import argparse
import contextlib
import csv
import dataclasses
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import numpy as np


EXPECTED_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Accepted output-only revisions; other source differences still reject stale statistics.
PRESENTATION_SOURCE_VERSIONS = {'prepare_nero_pi0.py': ('3962f750db630b91c6d5ad7854c9dc47ee6405eb663d3e15406d96a9df3ec0e0', 'a1d0e768e36c9aa10086f23186291e3674cc3496e9d719ff224b2c692c3dd8d4'), 'nero_batch.py': ('a7962a2f9380c369309f58be21becbee44d08544a811356fb72e6a4f113435ab', '5c5ddb11524d5a10c32ac9566a8d3dc69c28354846b7726a031b2b58a26ed559')}


def source_version_matches(name, recorded, current):
    return recorded == current or PRESENTATION_SOURCE_VERSIONS.get(name) == (recorded, current)


def training_config(cfg, steps=None, save_interval=None):
    steps = cfg.num_train_steps if steps is None else steps
    interval = cfg.save_interval if save_interval is None else save_interval
    require(steps > 0 and 0 < interval <= steps, "训练步数和保存间隔必须为正，且间隔不超过步数")
    # Keep model/data/normalization and the original LR schedule unchanged for a controlled extension.
    return dataclasses.replace(cfg, num_train_steps=steps, save_interval=interval)


class TeeStream:
    def __init__(self, stream, copy_to):
        self.stream, self.copy_to = stream, copy_to

    def write(self, text):
        result = self.stream.write(text)
        self.copy_to.write(text)
        self.copy_to.flush()
        return result

    def flush(self):
        self.stream.flush()
        self.copy_to.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextlib.contextmanager
def capture_training_output(path):
    # tqdm.write() does not go through logging.FileHandler; record stdout/stderr too.
    with path.open("w", encoding="utf-8", buffering=1) as console:
        with contextlib.redirect_stdout(TeeStream(sys.stdout, console)), contextlib.redirect_stderr(TeeStream(sys.stderr, console)):
            handlers = [(handler, handler.stream) for handler in logging.getLogger().handlers
                        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)]
            for handler, _ in handlers:
                handler.setStream(sys.stderr)
            try:
                yield
            finally:
                for handler, stream in handlers:
                    handler.setStream(stream)


def read_training_metrics(console_text):
    records = {}
    for match in re.finditer(r"Step\s+(\d+):\s*([^\r\n]+)", console_text):
        values = {key: float(value) for key, value in re.findall(
            r"([A-Za-z_][A-Za-z_0-9]*)=([^,\s]+)", match.group(2))}
        if "loss" in values:
            require(np.isfinite(list(values.values())).all(), "训练日志出现非有限 loss 或梯度指标")
            step = int(match.group(1))
            records[step] = {"step": step, **values}
    require(records, "未捕获到训练 loss，请保留 console.log 检查")
    return [records[step] for step in sorted(records)]


def readiness(project, script_root):
    asset_root = project / "assets/pi0_nero_box_v1/local/nero_box_v1_train"
    norm_path = asset_root / "norm_stats.json"
    audit = json.loads((asset_root / "nero_norm_report.json").read_text())
    require(audit["status"] == "train_only_norm_stats_ready_not_trained", "归一化报告未完成")
    require(audit["validation_values_used"] is False and audit["state_vectors"] == 2710
            and audit["action_horizon"] == 50, "归一化来源或动作长度不符")
    for name in ("nero_pi0_adapter.py", "nero_pi0_config.py", "prepare_nero_pi0.py"):
        require(source_version_matches(name, audit["source_sha256"][name], digest(script_root / name)),
                f"{name} 与统计时版本不一致，请保留输出后核对")
    norm = json.loads(norm_path.read_text())["norm_stats"]
    for key in ("state", "actions"):
        mean, std = np.asarray(norm[key]["mean"]), np.asarray(norm[key]["std"])
        require(mean.shape == std.shape == (16,) and np.isfinite(mean).all()
                and np.isfinite(std).all() and (std >= 0.001).all(), "归一化值/尺度异常")
    params = project / "checkpoints/openpi-assets/checkpoints/pi0_base/params"
    require((params / "_METADATA").is_file(), "pi0_base 参数元数据不存在")
    link = project / "data/nero_openpi_cache_v1/local/nero_box_v1_train"
    require(link.is_symlink() and link.resolve() == (project / "data/nero_lerobot_v1/train").resolve(),
            "项目训练集链接改变或缺失")
    commit = subprocess.check_output(
        ["git", "-C", str(project / "vendor/openpi"), "rev-parse", "HEAD"], text=True).strip()
    require(commit == EXPECTED_COMMIT, f"OpenPI 版本变化：{commit}")
    return {"openpi_commit": commit, "norm_stats_sha256": digest(norm_path),
            "checkpoint_metadata_sha256": digest(params / "_METADATA"),
            "checkpoint_params": str(params), "source_sha256": audit["source_sha256"]}


def load_official_train(project):
    path = project / "vendor/openpi/scripts/train.py"
    spec = importlib.util.spec_from_file_location("_nero_official_train", path)
    require(spec is not None and spec.loader is not None, "找不到官方训练入口")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify(project, cfg, official, output):
    import jax
    from flax import nnx
    from openpi import transforms
    from openpi.policies.policy import Policy
    from openpi.training import data_loader, sharding

    logging.info("阶段 1：加载 NERO 归一化与模型输入变换；首次可能获取分词器。")
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    require(data.norm_stats is not None, "未加载 NERO 训练集统计")
    dataset = data_loader.create_torch_dataset(data, cfg.model.action_horizon, cfg.model)
    row = dataset[0]
    packed = transforms.compose(data.repack_transforms.inputs)(row)
    model_input = transforms.compose([
        *data.data_transforms.inputs,
        transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
        *data.model_transforms.inputs,
    ])(packed)
    require(model_input["state"].shape == (32,) and model_input["actions"].shape == (50, 32),
            "模型输入形状异常")
    tokens = np.asarray(model_input["tokenized_prompt_mask"])
    require(tokens.shape == (128,) and int(tokens.sum()) < 128,
            "任务文本达到 token 上限，需要先核对是否被截断")
    logging.info("模型输入：state=(32,), actions=(50,32), 有效文本 token=%d", tokens.sum())

    logging.info("阶段 2：加载 pi0_base，初始化 LoRA 和优化器。")
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    start = time.monotonic()
    state, _ = official.init_train_state(cfg, jax.random.key(cfg.seed), mesh, resume=False)
    jax.block_until_ready(state)
    require(int(state.step) == 0, "验证过程出现非零训练步数")
    params = state.params.to_pure_dict()
    leaves = jax.tree.leaves(params)
    require(all(not isinstance(v, jax.ShapeDtypeStruct) for v in leaves), "仍存在未实例化的参数")
    count = sum(int(v.size) for v in leaves)
    load_s = time.monotonic() - start
    logging.info("权重已加载；参数元素数=%d，耗时=%.2fs", count, load_s)

    logging.info("阶段 3：对第一条训练观测执行离线动作生成；首次编译需要等待。")
    model = nnx.merge(state.model_def, state.params)
    if hasattr(model, "eval"):
        model.eval()
    policy = Policy(
        model,
        transforms=[*data.data_transforms.inputs,
                    transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
                    *data.model_transforms.inputs],
        output_transforms=[*data.model_transforms.outputs,
                           transforms.Unnormalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
                           *data.data_transforms.outputs],
        sample_kwargs={"num_steps": 10},
    )
    # Repack again: the input pipeline may mutate the first packed dictionary.
    observation = transforms.compose(data.repack_transforms.inputs)(dataset[0])
    reference = np.asarray(observation.pop("actions")).copy()
    start = time.monotonic()
    inferred = policy.infer(observation)
    actions = np.asarray(inferred["actions"])
    elapsed = time.monotonic() - start
    require(actions.shape == (50, 16) and np.isfinite(actions).all(), "离线动作无效")
    np.save(output / "base_actions.npy", actions, allow_pickle=False)
    np.save(output / "reference_actions.npy", reference, allow_pickle=False)
    sample = {"state": np.asarray(observation["state"]), "task": np.asarray(observation["prompt"])}
    sample.update({f"image_{k}": np.asarray(v) for k, v in observation["images"].items()})
    np.savez_compressed(output / "input.npz", **sample)
    logging.info("离线动作形状=%s；首次推理耗时=%.2fs", actions.shape, elapsed)
    return {"status": "pi0_base_loaded_and_inference_passed_not_trained", "training_step": int(state.step),
            "parameter_elements": count, "load_seconds": load_s, "first_inference_seconds": elapsed,
            "actions_shape": list(actions.shape), "all_finite": True,
            "valid_text_tokens": int(tokens.sum()),
            "gripper_output_ranges_m": {str(i): [float(actions[:, i].min()), float(actions[:, i].max())]
                                         for i in (7, 15)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "train"))
    parser.add_argument("--project", type=Path, default=Path("/home/clwu/vla-wu"))
    parser.add_argument("--exp-name", default="nero_smoke_v1")
    parser.add_argument("--steps", type=int, help="独立实验的训练步数；默认保持配置的 100 步")
    parser.add_argument("--save-interval", type=int, help="检查点保存间隔；默认保持配置值")
    args = parser.parse_args()
    project = args.project.resolve()
    provenance = readiness(project, Path(__file__).resolve().parent)
    os.environ["HF_LEROBOT_HOME"] = str(project / "data/nero_openpi_cache_v1")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    parent = project / "results/pi0_nero_box_v1"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{args.mode}_", dir=parent))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")], force=True)
    logging.info("本次结果目录：%s", output)
    report = {**provenance, "mode": args.mode, "status": "running", "result_dir": str(output)}
    path = output / "report.json"
    try:
        import jax
        from nero_pi0_config import get_train_config

        require(jax.default_backend() == "gpu" and jax.device_count() == 1,
                "请用 CUDA_VISIBLE_DEVICES 选择一张空闲 GPU；本轮要求恰好一张可见 GPU")
        logging.info("使用设备：%s", jax.devices())
        cfg = get_train_config(project, args.exp_name)
        official = load_official_train(project)
        if args.mode == "verify":
            require(args.steps is None and args.save_interval is None, "训练参数仅适用于 train 模式")
            report.update(verify(project, cfg, official, output))
            logging.info("模型加载和动作生成完成：%s", output)
        else:
            cfg = training_config(cfg, args.steps, args.save_interval)
            require(not cfg.checkpoint_dir.exists(), f"检查点目录已存在，保留原结果：{cfg.checkpoint_dir}")
            report.update(num_train_steps=cfg.num_train_steps, checkpoint_dir=str(cfg.checkpoint_dir),
                          exp_name=args.exp_name, runner_sha256=digest(Path(__file__).resolve()),
                          training_settings={"batch_size": cfg.batch_size, "save_interval": cfg.save_interval,
                                             "keep_period": cfg.keep_period,
                                             "lr_schedule": dataclasses.asdict(cfg.lr_schedule)})
            logging.info("开始 %d 步 NERO LoRA 微调；初始权重：%s", cfg.num_train_steps, cfg.weight_loader.params_path)
            with capture_training_output(output / "console.log"):
                official.main(cfg)
            metrics = read_training_metrics((output / "console.log").read_text(encoding="utf-8"))
            (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            with (output / "training_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
                fields = ["step"] + sorted({k for item in metrics for k in item if k != "step"})
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(metrics)
            report.update(status="training_completed_with_finite_logged_metrics",
                          logged_loss={"count": len(metrics), "first": metrics[0], "last": metrics[-1]},
                          console_log=str(output / "console.log"), training_metrics=str(output / "training_metrics.csv"))
            logging.info("训练结束：%d 步；检查点目录：%s", cfg.num_train_steps, cfg.checkpoint_dir)
    except BaseException as error:
        report.update(status="failed_or_interrupted", error_type=type(error).__name__, error=str(error))
        logging.exception("本轮失败或中断，已保留结果目录。")
        raise
    finally:
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("本次报告：%s", path)


if __name__ == "__main__":
    main()
