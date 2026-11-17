# vla-wu：NERO π0 微调

本项目在GPU服务器上使用官方pi0_base基础权重微调NERO数据。红色新批次与黄色历史批次分别管理。

| 批次 | 原始示教 | 使用的成功示教 | 训练/验证帧数 | 入口 |
|---|---:|---:|---|---|
| red_v1 | 50 | 49 | 11,960 / 3,158 | `scripts/model/nero_batch.py` |
| 黄色历史box_v1 | 10 | 10 | 2,710 / 719 | 原prepare/run/check入口 |

红色数据是否已到本机，以 `data/nero_aligned_red_v1/`、`data/nero_lerobot_red_v1/` 的实际文件为准。数据从本地同步，服务器整理工具不生成新示教。

- [运行说明](docs/运行说明.md)：红色导出、统计、训练和评估命令，以及各模型脚本用途。
- [目录说明](docs/目录说明.md)：数据、统计、检查点和日志的位置。
- [逐文件说明](docs/文件索引.md)：每个脚本的用途、使用场景、完整命令和输出。
- [黄色历史结果](docs/实验结果.md)：旧10条黄色示教的实验结果，不是red的训练结果。

使用 `vendor/openpi/.venv/bin/python`。现有环境不需要重新安装本地相机和CAN依赖。完整本地采集、对齐与传输流程在W_pro根目录的README中。

安装或更新工具包后生成逐文件说明：

```bash
cd /home/clwu/vla-wu && \
python3 scripts/tools/organize_vla_wu.py --project /home/clwu/vla-wu --write-index
```
