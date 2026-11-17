#!/usr/bin/env python3
"""启动厂商 RealSense Viewer，在界面中手动开启 RGB 和 Depth 测试相机。"""
import argparse
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='仅检查 Viewer 是否已安装，不打开窗口')
    args = parser.parse_args()
    viewer = shutil.which('realsense-viewer')
    if viewer is None:
        print('未找到 realsense-viewer，请先安装厂商 RealSense Viewer。', file=sys.stderr)
        return 1
    if args.check:
        print(f'RealSense Viewer 已安装：{viewer}')
        return 0
    print('打开 RealSense Viewer；在界面中开启 RGB Camera 和 Depth，切换到 2D 查看。', flush=True)
    try:
        return subprocess.run([viewer], check=False).returncode
    except OSError as error:
        print(f'Viewer 启动失败：{error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
