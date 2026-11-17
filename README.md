# W_pro：NERO 示教采集与 π0 适配

## 从 GitHub 获取完整项目

```bash
git clone --recurse-submodules https://github.com/AS-cuber/VLA_pi0.git
cd VLA_pi0
python3 scripts/data/restore_github_data.py
```

代码与文档直接保存在仓库；厂商 SDK 和仿真模型通过固定版本的 Git 子模块获取。完整原始示教及对齐数据保存在 `data/archive/` 的无损压缩分卷中，还原脚本会先校验所有分卷及压缩包的 SHA-256，再恢复原来的数据目录。虚拟环境和缓存不上传。详见 [GitHub 数据备份](data/archive/README.md)。

任务：右臂将黄色长方体放入透明盒子，随后双臂协作合盖。

本地采集项目为 `/home/user/W_pro`；GPU 服务器项目为 `/home/clwu/vla-wu`。已有基于官方 pi0_base 微调 1,000 步的候选模型；当前进入部署阶段，先用 MuJoCo 预览动作，再接入实机闭环评估。实时推理服务和自主执行接口尚待开发。

## 目录

| 目录 | 内容 |
|---|---|
| `scripts/acquisition/` | 三相机查看、CAN 状态读取、示教录制 |
| `scripts/data/` | 示教检查、15 Hz 对齐、LeRobot 导出 |
| `scripts/model/` | 服务器使用的 NERO 适配、归一化、训练与评估工具 |
| `scripts/maintenance/` | 机械臂使能、失能、急停恢复 |
| `scripts/_common/` | 硬件维护公共模块 |
| `scripts/tools/` | 项目整理工具和服务器文档模板 |
| `data/nero_demos/` | 原始示教、三路 RGB、CAN 报文和元数据 |
| `data/nero_aligned_v1/` | 15 Hz 对齐数据及训练/验证划分 |
| `docs/` | 采集、数据、模型、部署和维护说明 |
| `tests/` | 按数据、硬件和模型分类的测试 |
| `pyAgxArm/` | 厂商 SDK 与接口文档 |
| `.venv/` | 本地采集环境 |
| `packages/` | 发往服务器的整理包 |

## 常用操作

以下命令在本地采集电脑运行：

```bash
cd /home/user/W_pro
```

| 操作 | 命令 |
|---|---|
| 打开相机 Viewer | `.venv/bin/python scripts/acquisition/test_three_cameras.py` |
| 查看左右 CAN 状态 | `.venv/bin/python scripts/acquisition/nero_teleop_read.py --seconds 10` |
| 15 fps 静止试录 | `.venv/bin/python scripts/acquisition/record_nero_demo.py --headless --seconds 10 --camera-fps 15` |
| 查看采集参数 | `.venv/bin/python scripts/acquisition/record_nero_demo.py --help` |
| 检查已有对齐数据 | `.venv/bin/python scripts/data/check_nero_alignment.py` |
| LeRobot 导出预检查 | `.venv/bin/python scripts/data/export_nero_lerobot.py --dry-run` |

完整示教、CAN 启动和录制按键见[采集操作](docs/acquisition/采集操作.md)。模型命令在 GPU 服务器运行，见[vla-wu 使用说明](docs/model/vla-wu使用说明.md)。每个项目文件的说明见[文件索引](docs/maintenance/文件索引.md)。

## 文档索引

- [采集操作](docs/acquisition/采集操作.md)：设备映射、CAN、相机和示教录制。
- [数据格式与转换](docs/data/数据格式与转换.md)：文件含义、单位、对齐、8/2 划分及导出。
- [vla-wu 使用说明](docs/model/vla-wu使用说明.md)：模型文件、环境和训练评估命令。
- [实验结果](docs/model/实验结果.md)：100 步、1,000 步与参考方法的离线误差。
- [实机评估](docs/deployment/实机评估.md)：下一阶段的部署接口和任务记录。
- [硬件维护](docs/maintenance/硬件维护.md)：维护入口及适用接线。
- [目录维护](docs/maintenance/目录维护.md)：本地与服务器的文件归类和迁移方式。

## 环境与维护

本地依赖见 `requirements.txt`，编辑器使用 `.venv/bin/python`。模型工具使用服务器 `vendor/openpi/.venv/bin/python`。

运行项目测试：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests
```
