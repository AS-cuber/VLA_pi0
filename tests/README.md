# 测试目录

从 W_pro 根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests
```

| 文件 | 覆盖内容 | 单独运行 |
|---|---|---|
| `data/test_nero_export.py` | 图像路径重映射、来源和数据划分 | `.venv/bin/python -m unittest tests.data.test_nero_export` |
| `data/test_vla_wu_layout.py` | 服务器目录移动、冲突、清理和重复运行 | `.venv/bin/python -m unittest tests.data.test_vla_wu_layout` |
| `hardware/test_nero_controls.py` | 维护入口的使能、失能和恢复逻辑 | `.venv/bin/python -m unittest tests.hardware.test_nero_controls` |
| `model/test_nero_pi0_adapter.py` | 相机映射、通道、单位和输出形状 | `.venv/bin/python -m unittest tests.model.test_nero_pi0_adapter` |
| `model/test_nero_pi0_preparation.py` | 统计、划分、训练配置和日志解析 | `.venv/bin/python -m unittest tests.model.test_nero_pi0_preparation` |
| `model/test_nero_checkpoint.py` | 验证帧选择、误差单位和模型对照条件 | `.venv/bin/python -m unittest tests.model.test_nero_checkpoint` |
| `model/test_nero_action_diagnostics.py` | 预测距离、填充排除和夹爪偏差 | `.venv/bin/python -m unittest tests.model.test_nero_action_diagnostics` |

这些测试在本地运行，不连接机械臂或加载 GPU 模型。各级 `__init__.py` 用于 Python 包发现，不单独执行。
