#!/usr/bin/env python3
"""双侧主从总线只读诊断。不导入 SDK，不发送 CAN 帧或改变接口配置。"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import selectors
import socket
import struct
import time

CHANNELS = {"left": "can1", "right": "can0"}
SERIALS = {"left": "004B003A4648571020363032", "right": "004800254148571320343133"}
TARGETS = {0x155: (0, 1), 0x156: (2, 3), 0x157: (4, 5), 0x170: (6,)}
FEEDBACK = {0x2A5: (0, 1), 0x2A6: (2, 3), 0x2A7: (4, 5), 0x2A9: (6,)}
IDS = set(TARGETS) | set(FEEDBACK) | {0x159, 0x2A8, 0x2A1}
FRAME = struct.Struct("=IB3x8s")


def verify_usb(channel, expected):
    device = (Path('/sys/class/net') / channel / 'device').resolve()
    for parent in (device, *device.parents):
        if (parent / 'idVendor').is_file() and (parent / 'serial').is_file():
            actual = (parent / 'serial').read_text().strip()
            if actual != expected:
                raise RuntimeError(f'{channel} USB 序列号改变：{actual}，请先核对左右映射')
            return
    raise RuntimeError(f'{channel} 未找到预期 USB-CAN；请在连接设备的本地 Ubuntu 运行')


def decode(can_id, data):
    if can_id in TARGETS or can_id in FEEDBACK:
        kind = 'target' if can_id in TARGETS else 'feedback'
        indices = (TARGETS if kind == 'target' else FEEDBACK)[can_id]
        values = [v * 0.001 for v in struct.unpack('>ii', data)[:len(indices)]]
        return kind, {'indices': indices, 'degrees': values}
    if can_id in (0x159, 0x2A8):
        raw, force = struct.unpack('>ih', data[:6])
        mode = ((data[6] >> 2) & 1) if can_id == 0x159 and data[6] <= 7 else (data[7] if can_id == 0x2A8 else -1)
        return ('grip_target' if can_id == 0x159 else 'grip_feedback'), dict(
            raw=raw, value=raw * (0.001 if mode in (0, 1) else 1),
            unit='mm' if mode == 0 else 'deg' if mode == 1 else 'raw',
            force_N=force * 0.001, status=data[6], last_byte=data[7], raw_hex=data.hex(' '))
    if can_id == 0x2A1:
        return 'status', dict(ctrl_mode=data[0], arm_status=data[1],
                              mode_feedback=data[2], err_code=int.from_bytes(data[6:8], 'big'))
    return None


def new_cache():
    return dict(target=[None]*7, feedback=[None]*7, times={}, counts=Counter())


def show(item, now):
    for kind in ('target', 'feedback'):
        required = TARGETS if kind == 'target' else FEEDBACK
        fresh = all(i in item['times'] and now-item['times'][i] < 0.5 for i in required)
        values = ', '.join('--' if v is None else f'{v:.3f}' for v in item[kind])
        print(f'  {kind} deg [{values}] | {"FRESH" if fresh else "MISSING/STALE"}')
    for kind, frame_id in (('grip_target', 0x159), ('grip_feedback', 0x2A8), ('status', 0x2A1)):
        if kind in item:
            print(f'  {kind}: {item[kind]} | age={now-item["times"][frame_id]:.3f}s')
        else:
            print(f'  {kind}: MISSING')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=20)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds 必须是有限正数')
    caches = {side: new_cache() for side in CHANNELS}
    selector = selectors.DefaultSelector()
    buses = []
    print('只读监听：left=can1，right=can0。', flush=True)
    try:
        for side, channel in CHANNELS.items():
            verify_usb(channel, SERIALS[side])
            bus = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            buses.append(bus)
            bus.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
            # CAN_ERR_FLAG in a filter mask selects the kernel error-frame list.
            # Standard data filters must use EFF|RTR|SFF_MASK without ERR.
            filters = b''.join(struct.pack('=II', i, 0xC00007FF) for i in sorted(IDS))
            bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
            bus.bind((channel,))
            bus.setblocking(False)
            selector.register(bus, selectors.EVENT_READ, side)
        end = time.monotonic() + args.seconds
        next_print = time.monotonic() + 1
        while time.monotonic() < end:
            for key, _ in selector.select(timeout=0.05):
                for _ in range(256):
                    try:
                        packet = key.fileobj.recv(FRAME.size)
                    except BlockingIOError:
                        break
                    if len(packet) != FRAME.size:
                        continue
                    can_id, size, data = FRAME.unpack(packet)
                    if can_id & 0xE0000000 or size != 8:
                        continue
                    decoded = decode(can_id, data)
                    if decoded is None:
                        continue
                    kind, value = decoded
                    cache = caches[key.data]
                    cache['times'][can_id] = time.monotonic()
                    cache['counts'][can_id] += 1
                    if kind in ('target', 'feedback'):
                        for i, angle in zip(value['indices'], value['degrees']):
                            cache[kind][i] = angle
                    else:
                        cache[kind] = value
            now = time.monotonic()
            if now >= next_print:
                for side, channel in CHANNELS.items():
                    print(f'\n{side} / {channel}')
                    show(caches[side], now)
                print(flush=True)
                next_print = now + 1
        for side, cache in caches.items():
            print(side, '接收计数:', json.dumps({hex(i): n for i, n in sorted(cache['counts'].items())}))
    finally:
        selector.close()
        for bus in buses:
            bus.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('已停止只读监听。')
