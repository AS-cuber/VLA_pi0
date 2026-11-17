# 测试文件：用途、使用场景与命令

修改对应代码后选择下列测试。它们在本地运行，不连接硬件或加载GPU模型，不是每次采集/训练前的固定步骤。每段命令包含工作目录，需要整段复制。

运行整个项目测试：

```bash
cd /home/user/W_pro && \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests
```

## `acquisition/test_recording_storage.py`

**用途：** 确认只有成功会生成正式目录，旧数据不会被清理。
**什么时候用：** 修改录制保存、失败清理、异常退出或结束按键逻辑后。

```bash
cd /home/user/W_pro && \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.acquisition.test_recording_storage
```

## `data/test_nero_batches.py`

**用途：** 检查历史失败记录保留、批次隔离及恢复不覆盖red。
**什么时候用：** 修改成功示教筛选、训练/验证划分、批次命名或备份恢复位置后。

```bash
cd /home/user/W_pro && \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.data.test_nero_batches
```

## `data/test_nero_export.py`

**用途：** 检查路径重映射、越界与划分交叉的拒绝行为。
**什么时候用：** 修改图像路径迁移或LeRobot导出输入检查后。

```bash
cd /home/user/W_pro && \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.data.test_nero_export
```

## `data/test_vla_wu_layout.py`

**用途：** 检查源文件保留、冲突拒绝和重复运行。
**什么时候用：** 修改服务器目录整理、清理范围或模板更新逻辑后。

```bash
cd /home/user/W_pro && \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.data.test_vla_wu_layout
```

## `hardware/test_nero_controls.py`

**用途：** 用模拟设备检查控制分支，不接触真实机械臂。
**什么时候用：** 修改旧独立接线维护工具的状态检查或使能/复位逻辑后。

```bash
cd /home/user/W_pro && \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.hardware.test_nero_controls
```

## `model/test_nero_pi0_adapter.py`

**用途：** 检查16维数据与右臂第七关节、夹爪的保留。
**什么时候用：** 修改三相机映射、图像布局、状态/动作通道或单位后。

```bash
cd /home/user/W_pro && \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.model.test_nero_pi0_adapter
```

## `model/test_nero_pi0_preparation.py`

**用途：** 检查统计数值和训练/验证隔离。
**什么时候用：** 修改黄色统计、尺度下限、来源检查、训练配置或日志解析后。

```bash
cd /home/user/W_pro && \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.model.test_nero_pi0_preparation
```

## `model/test_nero_checkpoint.py`

**用途：** 检查episode边界和比较条件。
**什么时候用：** 修改黄色验证采样、未来掩码、物理单位指标或基础模型对照后。

```bash
cd /home/user/W_pro && \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.model.test_nero_checkpoint
```

## `model/test_nero_action_diagnostics.py`

**用途：** 检查偏差符号、通道标签及无效未来排除。
**什么时候用：** 修改分预测距离或右夹爪误差分析后。

```bash
cd /home/user/W_pro && \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.model.test_nero_action_diagnostics
```

测试结果打印到终端，临时文件使用临时目录，现有示教不被修改。各目录 `__init__.py` 供Python发现测试模块，没有独立运行命令。
