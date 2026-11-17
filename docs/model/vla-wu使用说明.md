# vla-wu 模型工具

服务器 `clwu@218.192.172.137`，项目 `/home/clwu/vla-wu`。新红色批次的传输、导出、统计、训练和评估完整命令见[总流程](../../README.md)第4～8步。

## 两套入口的边界

| 入口 | 适用范围 |
|---|---|
| `nero_batch.py prepare/train/evaluate --dataset red_v1` | 新红色批次；从导出报告读取示教/帧数，独立管理资产 |
| `prepare_nero_pi0.py`、`run_nero_pi0.py`、`check_nero_checkpoint.py` | 黄色历史批次，固定8/2示教、2,710/719帧及旧repo |
| `nero_pi0_config.py`、`nero_pi0_adapter.py` | 两套入口共用的NERO模型和变换基础 |
| `diagnose_nero_actions.py` | 黄色719帧完整预测的可选CPU分析 |

原有适配器、配置和归一化算法保留，日志文字更新兼容黄色已保存的统计。`nero_batch.py` 在运行时替换批次的repo和实验名称。新实验从官方pi0_base开始。

## 环境

| 项目 | 当前约定 |
|---|---|
| Python | `vendor/openpi/.venv/bin/python` |
| OpenPI | 提交 `215abfb217dbac7d5f1273282331b9b1866c0479` |
| LeRobot | 0.1.0 |
| 基础权重 | `checkpoints/openpi-assets/checkpoints/pi0_base/params/` |
| GPU | 单进程使用一张空闲卡；命令示例选择GPU 1 |

调用模型进程时取消 `LD_LIBRARY_PATH`、`LD_PRELOAD`，避免系统CUDA库影响JAX。数据工具和统计用CPU；训练、评估用 `CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false`。不在已有OpenPI环境中安装本地采集依赖或升级LeRobot。

## 红色批次路径

| 路径 | 内容 |
|---|---|
| `data/nero_demos/red/` | 从本地同步的50条原始记录 |
| `data/nero_aligned_red_v1/` | 本地生成并同步的49条成功示教对齐数据 |
| `data/nero_lerobot_red_v1/train/`、`validation/` | LeRobot数据，39/10示教 |
| `data/nero_openpi_cache_red_v1/local/nero_red_v1_train` | 指向本批次训练目录的链接 |
| `assets/pi0_nero_red_v1/local/nero_red_v1_train/` | 训练集统计 `norm_stats.json` 及其来源报告 |
| `checkpoints/pi0_nero_red_v1/实验名/编号/` | 微调检查点，含参数与配套统计 |
| `results/pi0_nero_red_v1/train_*/` | 训练报告、控制台日志、CSV/JSON指标 |
| `results/pi0_nero_red_v1/evaluation_*/` | 验证预测、掩码和误差报告 |

查看新入口参数：

```bash
cd /home/clwu/vla-wu
vendor/openpi/.venv/bin/python scripts/model/nero_batch.py --help
```

`--dataset red_v1` 对应导出参数 `--repo-prefix local/nero_red_v1`。`prepare` 不读取验证集数值；`train` 要求 `--exp-name`，默认批大小2、LoRA、动作块50步；`--steps` 默认1,000、保存间隔250。新入口的学习率衰减长度跟随训练步数。`evaluate` 必须指定本批次的完整数字编号检查点。

## 黄色历史命令

以下命令仅用于复现黄色旧实验，路径与数据不指向red。历史结果见[实验结果](实验结果.md)。服务器原有数据路径：

```text
数据：data/nero_lerobot_v1/
缓存：data/nero_openpi_cache_v1/local/nero_box_v1_train
统计：assets/pi0_nero_box_v1/local/nero_box_v1_train/
模型：checkpoints/pi0_nero_box_v1/
日志：results/pi0_nero_box_v1/
```

原来的归一化已存在，日常不需要重算。首次准备同一旧批次时：

```bash
cd /home/clwu/vla-wu
env -u LD_LIBRARY_PATH -u LD_PRELOAD JAX_PLATFORMS=cpu \
  vendor/openpi/.venv/bin/python scripts/model/prepare_nero_pi0.py
```

独立复现1,000步实验：

```bash
cd /home/clwu/vla-wu
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  vendor/openpi/.venv/bin/python -u scripts/model/run_nero_pi0.py train \
  --steps 1000 --save-interval 250 \
  --exp-name "yellow_1000_$(date +%Y%m%d_%H%M%S)"
```

评估已保存的黄色检查点，交互输入完整路径：

```bash
cd /home/clwu/vla-wu && \
read -r -p '黄色检查点完整路径：' NERO_YELLOW_CHECKPOINT

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  vendor/openpi/.venv/bin/python -u scripts/model/check_nero_checkpoint.py \
  --checkpoint "$NERO_YELLOW_CHECKPOINT" --all-frames --compare-base
```

不加 `--all-frames` 时是两条示教各三个时刻，共6个样本。`diagnose_nero_actions.py` 只读取此前已保存的黄色全量预测：

```bash
cd /home/clwu/vla-wu && \
vendor/openpi/.venv/bin/python scripts/model/diagnose_nero_actions.py
```

这些离线分析是按需工具，不增加固定的重复评估门槛。
