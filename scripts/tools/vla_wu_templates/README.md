# vla-wu：NERO π0 训练与评估

任务：使用官方 pi0_base 和 NERO 示教数据微调，完成右臂入盒、双臂合盖。

项目目录 `{{PROJECT}}`。模型代码位于 `scripts/model/`，Python 环境为 `vendor/openpi/.venv/bin/python`。

## 文档

- [文件索引](docs/文件索引.md)：每个项目脚本的用途和调用方式。
- [运行说明](docs/运行说明.md)：归一化、训练、检查点评估及结果读取。
- [目录说明](docs/目录说明.md)：数据、权重、统计和日志文件。
- [实验结果](docs/实验结果.md)：当前离线误差和下一阶段。

## 常用命令

```bash
cd {{PROJECT}}
nvidia-smi
vendor/openpi/.venv/bin/python scripts/model/run_nero_pi0.py --help
vendor/openpi/.venv/bin/python scripts/model/check_nero_checkpoint.py --help
```

新实验示例（GPU 1 空闲时）：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  vendor/openpi/.venv/bin/python -u scripts/model/run_nero_pi0.py train \
  --steps 1000 --save-interval 250 \
  --exp-name "nero_1000_$(date +%Y%m%d_%H%M%S)"
```

当前已有 1,000 步模型与全量离线对照结果，开发暂停；下一阶段是实时部署和实机评估。
