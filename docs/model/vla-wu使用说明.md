# vla-wu 使用说明

GPU 服务器：`clwu@218.192.172.137`，项目 `/home/clwu/vla-wu`。整理后，模型脚本统一位于 `scripts/model/`。

## 环境与目录

| 位置 | 内容 |
|---|---|
| `vendor/openpi/` | OpenPI 源码 |
| `vendor/openpi/.venv/` | 模型 Python 环境 |
| `scripts/model/` | NERO 适配、归一化、训练与评估入口 |
| `checkpoints/openpi-assets/checkpoints/pi0_base/` | 官方基础权重 |
| `checkpoints/pi0_nero_box_v1/实验名/编号/` | 微调权重及配套统计 |
| `data/nero_lerobot_v1/` | train、validation 和导出报告 |
| `data/nero_input_v1/` | 原始迁移资料及导出来源 |
| `data/nero_openpi_cache_v1/` | OpenPI 查找本地训练集的链接 |
| `assets/pi0_nero_box_v1/local/nero_box_v1_train/` | 训练集归一化统计 |
| `results/pi0_nero_box_v1/` | 训练日志和离线评估输出 |

先查看显卡，选择空闲 GPU：

```bash
nvidia-smi
cd /home/clwu/vla-wu
```

下面使用 GPU 1。Python 均采用 `vendor/openpi/.venv/bin/python`。当前环境需要在进程中取消 `LD_LIBRARY_PATH` 和 `LD_PRELOAD`，避免系统 CUDA 库干扰 JAX。

## 每个模型文件

| 文件 | 用途 / 输入输出 | 启动方式 |
|---|---|---|
| `nero_pi0_adapter.py` | 三相机与 16 维 rad/m 数据转换；输出裁回 16 维 | 公共模块，不单独运行 |
| `nero_pi0_config.py` | NERO 数据变换、LoRA 和训练默认配置 | 配置模块，由训练和评估入口读取 |
| `prepare_nero_pi0.py` | 读取 train，连接数据路径并生成 norm_stats.json | 见归一化命令 |
| `run_nero_pi0.py` | verify 加载基础模型；train 更新参数并保存检查点 | 见训练命令 |
| `check_nero_checkpoint.py` | 重载训练模型，评估验证集，或与基础模型对照 | 见评估命令 |
| `diagnose_nero_actions.py` | 读取已保存 NPZ，分预测距离分析误差 | 可选 CPU 分析，见下文 |
| `__init__.py` | Python 包与相邻模块导入入口 | 不单独运行 |

服务器是否具有某个可选脚本，以 `scripts/model/` 的实际文件为准；整理工具只移动已有代码。

## 训练

新建 1,000 步实验：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  vendor/openpi/.venv/bin/python -u scripts/model/run_nero_pi0.py train \
  --steps 1000 --save-interval 250 \
  --exp-name "nero_1000_$(date +%Y%m%d_%H%M%S)"
```

新实验从官方 pi0_base 开始。输出目录为 `checkpoints/pi0_nero_box_v1/实验名/`，重复实验名会停止。每 250 步保存最新检查点，1,000 步最终编号为 `999`。当前 1,000 步配置仅保留本实验最近的检查点。

训练日志在终端打印的 `results/pi0_nero_box_v1/train_随机后缀/`，包括 `report.json`、`run.log`、`console.log` 和 `training_metrics.csv/json`。

## 检查点评估

将 `实验名` 改成实际名称：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  vendor/openpi/.venv/bin/python -u scripts/model/check_nero_checkpoint.py \
  --checkpoint /home/clwu/vla-wu/checkpoints/pi0_nero_box_v1/实验名/999 \
  --all-frames --compare-base
```

两组评估顺序执行，结果保存在新建的 `comparison_随机后缀/`。不加 `--all-frames` 时仅抽样 6 个时刻；不加 `--compare-base` 时仅评估所选模型。

## 其他入口

基础模型加载验证：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  vendor/openpi/.venv/bin/python -u scripts/model/run_nero_pi0.py verify
```

已有统计位于 `assets/`。首次准备统计时使用 CPU：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD JAX_PLATFORMS=cpu \
  vendor/openpi/.venv/bin/python -u scripts/model/prepare_nero_pi0.py
```

统计输出已存在会拒绝覆盖。可选的离线动作诊断：

```bash
vendor/openpi/.venv/bin/python -u scripts/model/diagnose_nero_actions.py
```

该命令读取现有预测，不使用 GPU，也不启动训练。当前下一阶段见[实机评估](../deployment/实机评估.md)。
