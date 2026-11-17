# 服务器训练工具包

`nero-training-tools.tar.gz` 包含：模型脚本、LeRobot导出入口、服务器目录维护工具及服务器说明。没有原始图像、CAN数据、模型权重、SDK或虚拟环境。旧 `vla-wu-layout.tar.gz` 已由本包替代。

传输与运行按[总流程](../README.md)第4～8步执行。更新代码/文档后，在项目根目录重新打包：

```bash
cd /home/user/W_pro
tar --hard-dereference --exclude='__pycache__' -czf packages/nero-training-tools.tar.gz \
  scripts/__init__.py scripts/model \
  scripts/data/__init__.py scripts/data/export_nero_lerobot.py \
  scripts/tools/__init__.py scripts/tools/organize_vla_wu.py \
  scripts/tools/vla_wu_templates docs/model \
  -C scripts/tools/vla_wu_templates README.md docs
```

包内根目录README和 `docs/运行说明.md` 等由现有服务器模板提供，解压后可直接阅读；模板自身同时保留供目录工具使用。`docs/model/` 保存黄色历史命令和结果。解压会更新这些说明及项目工具，不操作服务器的数据、统计、权重或OpenPI环境。

服务器安装并生成逐文件说明：

```bash
cd /home/clwu/vla-wu && \
tar -xzf nero-training-tools.tar.gz && \
python3 scripts/tools/organize_vla_wu.py --project /home/clwu/vla-wu --write-index
```
