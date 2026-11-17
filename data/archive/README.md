# GitHub 数据备份

这里完整保存了首次上传时的 `data/nero_demos/` 和 `data/nero_aligned_v1/`，包含全部 RGB JPEG、CAN 报文、相机索引、元数据、对齐 NPZ 和划分文件。

由于采集数据约 900 MB、包含一万多个文件，GitHub 中使用 gzip 无损压缩，并分成每卷 16 MiB 的文件。分卷不是抽样或降精度版本；解压后恢复原来的目录和文件内容。

在项目根目录运行：

```bash
python3 scripts/data/restore_github_data.py
```

脚本使用 Python 标准库，先验证 `manifest.json` 中每卷和整个压缩包的 SHA-256，然后恢复数据。已有目标目录时会停止，避免覆盖本地采集结果。只检查备份完整性可运行：

```bash
python3 scripts/data/restore_github_data.py --verify-only
```

请预留约 1.6 GB 临时可用空间。还原后的数据目录由 `.gitignore` 忽略，分卷本身保存在版本库中。新采集的数据需要另行更新备份；当前分卷是本次上传时的快照。
