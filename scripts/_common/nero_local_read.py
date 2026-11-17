#!/usr/bin/env python3
"""旧独立机械臂的 USB-CAN 身份解析。当前主从总线使用 scripts/acquisition/nero_teleop_read.py。"""
from pathlib import Path

# 保留原维护工具的独立机械臂身份，不自动改为主从共用总线。
ARMS = {'left': '002B001D4648570C20303731', 'right': '0020002C4648570A20303731'}


def usb_serial(interface):
    device = (interface / 'device').resolve()
    for parent in (device, *device.parents):
        if (parent / 'idVendor').is_file() and (parent / 'serial').is_file():
            return (parent / 'serial').read_text().strip()
    return None


def resolve_interfaces(root=Path('/sys/class/net')):
    found = {}
    for interface in sorted(root.glob('can*')):
        serial = usb_serial(interface)
        if serial in ARMS.values():
            if serial in found:
                raise RuntimeError(f'USB 序列号重复：{serial}')
            found[serial] = interface.name
    missing = [f'{side}: {serial}' for side, serial in ARMS.items() if serial not in found]
    if missing:
        raise RuntimeError('未找到旧维护配置中的 USB-CAN：' + '; '.join(missing)
                           + '。请先核实独立维护接线与身份，不要直接改成主从总线执行控制。')
    return {side: found[serial] for side, serial in ARMS.items()}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(resolve_interfaces())
