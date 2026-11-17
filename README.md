# W_pro：NERO 双臂示教与 π0 微调

本地项目 `/home/user/W_pro` 负责采集和数据整理；GPU 服务器 `clwu@218.192.172.137:/home/clwu/vla-wu` 负责 LeRobot 导出、归一化、训练和离线评估。模型从官方 **pi0_base 基础预训练权重**开始做 NERO LoRA 微调。

任务是右臂将物体放入透明盒子，随后双臂协作合盖。黄色和红色是两批不同物体的示教，使用各自元数据中的任务文本，当前分别管理。

## 当前数据与进度

| 批次 | 原始记录 | 用于训练/验证的成功示教 | 对齐帧数 | 按整条示教划分 | 当前阶段 |
|---|---:|---:|---:|---|---|
| `yellow` | 10 | 10 | 3,429 | 训练8条/2,710帧，验证2条/719帧 | 已完成历史1,000步微调与离线评估 |
| `red` | 50 | 49 | 15,118 | 训练39条/11,960帧，验证10条/3,158帧 | 本地对齐已生成；尚未在服务器导出或训练 |

红色记录 `episode_20261117_155400_379098` 标记为失败，原始文件保留，默认不进入训练。成功标签由操作员给出，不代表过程完全没有纠正动作。数据采集均为三路 RGB、640×480、15 fps。

## 目录

| 位置 | 内容 |
|---|---|
| `data/nero_demos/yellow/`、`red/` | 原始图像、CAN、相机索引与记录元数据 |
| `data/nero_aligned_v1/` | 黄色历史15 Hz对齐数据，保留原8/2划分 |
| `data/nero_aligned_red_v1/` | 红色49条成功示教的15 Hz对齐数据及39/10划分 |
| `data/archive/` | 黄色10条、红色50条全部原始示教及两批对齐数据的无损备份分卷 |
| `scripts/acquisition/` | 相机查看、总线读取与示教录制 |
| `scripts/data/` | 记录检查、对齐、LeRobot导出、完整备份恢复 |
| `scripts/model/` | NERO变换、按批次准备/训练/评估，以及黄色历史入口 |
| `scripts/maintenance/`、`scripts/_common/` | 旧独立机械臂接线下的维护工具 |
| `scripts/tools/` | 服务器目录整理工具及文档模板 |
| `docs/` | 数据格式、硬件操作、服务器流程、历史结果和文件索引 |
| `packages/nero-training-tools.tar.gz` | 发往服务器的模型工具和导出脚本，不含数据/权重/环境 |
| `pyAgxArm/` | 厂商SDK子模块 |
| `.venv/` | 本地采集与数据处理环境 |
| `tests/` | 数据路径、批次划分、变换及维护逻辑测试 |

每个项目文件的用途、使用场景、运行电脑、完整命令和输出见[逐文件使用说明](docs/maintenance/文件索引.md)。命令框需要连同 `cd` 整段复制；终端在家目录 `~` 时，不能直接执行以 `.venv/` 开头的相对路径。

## 1. 本地环境

现有电脑已有项目和 `.venv`，日常直接使用。新电脑先获取代码（目标目录必须尚未存在）：

```bash
git clone --recurse-submodules https://github.com/AS-cuber/VLA_pi0.git /home/user/W_pro
cd /home/user/W_pro
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

仓库中的完整备份包含黄色10条、红色50条原始示教（包括失败记录）和两批对齐数据。需要恢复时运行：

```bash
cd /home/user/W_pro && \
python3 scripts/data/restore_github_data.py
```

恢复位置为 `data/nero_demos/`（包含 `yellow/` 和 `red/`）、`data/nero_aligned_v1/` 和 `data/nero_aligned_red_v1/`。脚本先校验所有分卷和完整压缩包；任一目标已存在时停止，不覆盖现有数据。备份仅代表本次提交时的快照，新增采集需要重新备份。详见[备份说明](data/archive/README.md)。

## 2. 采集新示教（本地）

已有50条无需重采。继续采集时，先查看 CAN 接口：

```bash
cd /home/user/W_pro
ip -details -statistics link show type can
```

左侧为 `can1`，右侧为 `can0`，速率1 Mbps。接口 DOWN 时启用对应接口；已 UP 且速率正确时跳过：

```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
```

查看反馈和相机：

```bash
cd /home/user/W_pro && \
.venv/bin/python scripts/acquisition/nero_teleop_read.py --seconds 10
.venv/bin/python scripts/acquisition/test_three_cameras.py
```

在 Viewer 中核对基座、左腕、右腕三路 RGB，关闭 Viewer 后再录制。核实反馈来自从臂后，采集红色物体任务：

```bash
cd /home/user/W_pro && \
.venv/bin/python scripts/acquisition/record_nero_demo.py \
  --output data/nero_demos/red --camera-fps 15 \
  --prompt '用右臂将红色物体放入透明盒子，然后双臂协作合上盒盖。' \
  --feedback-source-confirmed \
  --feedback-source-note '两主臂反馈推送关闭，两从臂开启；已对照从臂上位机核实关节反馈'
```

每次运行先在输出目录下的隐藏临时目录录制；窗口中 `S` 开始、`F` 成功保存、`X` 失败丢弃、`Esc` 中止丢弃。只有成功结束且记录器未报错时，才保留为正式的 `episode_日期_时间/`。关闭窗口、Ctrl+C或自动计时结束均丢弃本轮数据。已有历史记录不受影响。前述反馈来源参数必须对应现场实际核对结果。只接收数据的采集器退出后，不会停止机械臂或主从联动。

采集黄色任务时同时把 `--output` 改为 `data/nero_demos/yellow`、`--prompt` 改为黄色物体任务。设备序列号和完整操作见[采集操作](docs/acquisition/采集操作.md)。

## 3. 原始示教到15 Hz对齐数据（本地）

本轮 `data/nero_aligned_red_v1/` 已生成，可以直接进入第4步。首次处理该批次、且输出目录不存在时使用：

```bash
cd /home/user/W_pro
.venv/bin/python scripts/data/prepare_nero_dataset.py \
  --input data/nero_demos/red \
  --output data/nero_aligned_red_v1 \
  --validation-count 10 --seed 42
```

程序保留原始记录，跳过非成功标签，按完整 episode 留出验证集。成功记录存在采集错误、关键流缺失或中间数据过期时会停止，不将这些异常悄悄填平。

新增示教后重新生成，应使用新的输出版本，如 `nero_aligned_red_v2`；后续 LeRobot、repo-prefix 和 `--dataset` 同步使用 `red_v2`，不覆盖旧划分、统计或检查点。

按需定位数据问题：

```bash
cd /home/user/W_pro && \
.venv/bin/python scripts/data/check_nero_demo.py \
  data/nero_demos/red/episode_20261117_154334_742573

.venv/bin/python scripts/data/check_nero_alignment.py \
  --input data/nero_aligned_red_v1 --raw-root data/nero_demos/red
```

黄色旧 NPZ 仍保留移动前的绝对路径，读取时按当前原始目录重映射：

```bash
cd /home/user/W_pro && \
.venv/bin/python scripts/data/check_nero_alignment.py \
  --input data/nero_aligned_v1 --raw-root data/nero_demos/yellow
```

时间、字段、每种数据文件的含义见[数据格式与转换](docs/data/数据格式与转换.md)。

## 4. 将红色数据和工具传到服务器（本地）

本地可先查看导出输入是否可读，命令不生成 LeRobot 数据：

```bash
cd /home/user/W_pro
.venv/bin/python scripts/data/export_nero_lerobot.py \
  --aligned data/nero_aligned_red_v1 --raw-root data/nero_demos/red --dry-run
```

创建服务器目标目录，再传输。使用 `rsync` 保留目录结构，未使用删除远端文件的参数：

```bash
cd /home/user/W_pro && \
ssh clwu@218.192.172.137 'mkdir -p /home/clwu/vla-wu/data/nero_demos'

rsync -a --partial --info=progress2 data/nero_demos/red/ \
  clwu@218.192.172.137:/home/clwu/vla-wu/data/nero_demos/red/

rsync -a --partial --info=progress2 data/nero_aligned_red_v1/ \
  clwu@218.192.172.137:/home/clwu/vla-wu/data/nero_aligned_red_v1/

scp packages/nero-training-tools.tar.gz \
  clwu@218.192.172.137:/home/clwu/vla-wu/
```

这次训练使用红色批次，传输全部50条原始记录以保留来源，导出仅按对齐清单使用49条成功示教。黄色批次与服务器已有历史结果保留原位。

## 5. LeRobot 导出（服务器）

在服务器终端运行：

```bash
cd /home/clwu/vla-wu
tar -xzf nero-training-tools.tar.gz
python3 scripts/tools/organize_vla_wu.py --project /home/clwu/vla-wu --write-index

vendor/openpi/.venv/bin/python scripts/data/export_nero_lerobot.py \
  --aligned data/nero_aligned_red_v1 \
  --raw-root data/nero_demos/red \
  --output data/nero_lerobot_red_v1 \
  --repo-prefix local/nero_red_v1
```

使用服务器已有 OpenPI Python 环境和 LeRobot 0.1.0；不要把本地采集 `requirements.txt` 安装到该环境。输出目录已存在时拒绝覆盖，避免把未完成或不同版本的数据当作本次导出。

输出为 `train/`、`validation/`、`export_report.json`；每个 split 中有 `nero_episode_map.json`。导出携带每条示教自己的任务文本。

## 6. 训练集归一化（服务器CPU）

红色批次使用新入口，数据、缓存、统计和实验路径由 `--dataset red_v1` 统一命名：

```bash
cd /home/clwu/vla-wu
env -u LD_LIBRARY_PATH -u LD_PRELOAD JAX_PLATFORMS=cpu \
  vendor/openpi/.venv/bin/python -u scripts/model/nero_batch.py prepare \
  --project /home/clwu/vla-wu --dataset red_v1
```

只读取训练39条的数值。14个关节目标减去当前观测关节位置，两夹爪保持绝对宽度；随后计算 z-score 统计。统计保存到 `assets/pi0_nero_red_v1/local/nero_red_v1_train/`，不能复用黄色统计。

## 7. 从 pi0_base 微调（服务器GPU）

先查看显卡，选择空闲设备：

```bash
nvidia-smi
```

下面使用 GPU 1，给新实验取唯一名称，首轮以1,000步运行：

```bash
cd /home/clwu/vla-wu
NERO_EXP="red_1000_$(date +%Y%m%d_%H%M%S)"
printf '实验名：%s\n' "$NERO_EXP"

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  vendor/openpi/.venv/bin/python -u scripts/model/nero_batch.py train \
  --project /home/clwu/vla-wu --dataset red_v1 \
  --exp-name "$NERO_EXP" --steps 1000 --save-interval 250
```

训练起点为 `checkpoints/openpi-assets/checkpoints/pi0_base/params`。输出为：

- `checkpoints/pi0_nero_red_v1/实验名/`：模型参数及配套统计，1,000步最终编号为 `999`。
- `results/pi0_nero_red_v1/train_*/`：训练报告、完整控制台日志、loss等指标的CSV/JSON。

1,000步是首轮运行配置，不代表已经收敛。按结果决定后续实验步数；新入口会让学习率衰减长度随 `--steps` 调整。既有黄色1,000步权重不是本轮红色实验的初始化权重。

## 8. 按需离线评估（服务器GPU）

沿用上一步终端中的实验名；换终端时将 `NERO_CHECKPOINT` 填为真实检查点路径：

```bash
cd /home/clwu/vla-wu
NERO_CHECKPOINT="/home/clwu/vla-wu/checkpoints/pi0_nero_red_v1/$NERO_EXP/999"

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  vendor/openpi/.venv/bin/python -u scripts/model/nero_batch.py evaluate \
  --project /home/clwu/vla-wu --dataset red_v1 \
  --checkpoint "$NERO_CHECKPOINT" --all-frames
```

省略 `--all-frames` 时每条验证示教取开始、中间、末尾各一个时刻。全量使用红色验证10条的3,158帧，排除跨越各条示教末尾的未来目标。输出 `evaluation_*/report.json` 和 `validation_actions.npz`；关节误差单位为度，夹爪为mm。离线误差不等于真实任务成功率。

## 补充说明

黄色原实验的环境、旧命令与路径见[vla-wu使用说明](docs/model/vla-wu使用说明.md)，已有结果见[黄色历史实验结果](docs/model/实验结果.md)。旧入口保留原批次约束，不拿它直接训练红色新数据。

维护工具使用旧独立USB-CAN映射，不能直接用于当前主从共线接法，适用范围见[硬件维护](docs/maintenance/硬件维护.md)。

目录与清理规则见[目录维护](docs/maintenance/目录维护.md)，服务器工具包更新方式见[packages说明](packages/README.md)。当前项目不包含仿真和实机评估执行器。
