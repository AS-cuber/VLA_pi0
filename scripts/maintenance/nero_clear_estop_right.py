#!/usr/bin/env python3
"""右臂软件急停诊断与恢复入口。"""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._common.nero_control_common import recovery_main

if __name__ == "__main__":
    sys.exit(recovery_main("right"))
