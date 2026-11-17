# 完整数据备份

本目录保存本次提交时的完整无损快照：黄色10条、红色50条原始示教（包括失败记录），以及 `nero_aligned_v1/`、`nero_aligned_red_v1/` 的对齐数据。JPEG、CAN报文、索引、元数据和NPZ均按原文件内容保存，不抽样、不降精度。

`manifest.json` 记录分卷顺序、大小、SHA-256、原始文件数和字节数。各分卷属于同一个gzip压缩包，不能单独解压。在项目根目录运行：

```bash
cd /home/user/W_pro && \
python3 scripts/data/restore_github_data.py
```

脚本先校验所有分卷及整个压缩包，再恢复 `data/nero_demos/`（含 `yellow/`、`red/`）、`data/nero_aligned_v1/` 和 `data/nero_aligned_red_v1/`。任一目标已存在时停止，避免覆盖现有数据。旧黄色NPZ中的历史图像路径仍由数据工具的 `--raw-root` 参数重映射。

只核对备份分卷而不恢复：

```bash
cd /home/user/W_pro && \
python3 scripts/data/restore_github_data.py --verify-only
```

校验需要与 `archive_size` 相当的临时空间；恢复还需加上 `original_bytes`，两者均见manifest。脚本只使用Python标准库，兼容旧版黄色快照。恢复后的原始目录由 `.gitignore` 忽略；新增或修改采集数据后，需要另行更新分卷备份。
