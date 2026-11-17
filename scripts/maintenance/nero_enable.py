#!/usr/bin/env python3
"""使能入口，默认只预览。"""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._common.nero_control_common import power_main

if __name__ == "__main__":
    sys.exit(power_main(True))
