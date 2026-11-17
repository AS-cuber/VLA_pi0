#!/usr/bin/env python3
"""NERO 双臂三相机只读示教记录；默认窗口控制开始、成功或失败结束。"""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
import select
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.acquisition.nero_teleop_read import CHANNELS, SERIALS, TARGETS, FEEDBACK, FRAME, decode, verify_usb

CAMERAS = {'base': '263022078897', 'left_wrist': '263022072396', 'right_wrist': '263022078453'}
PROMPT = '用右臂将黄色长方体物体放入透明盒子，然后双臂协作合上盒盖。'
REQUIRED = set(TARGETS) | set(FEEDBACK) | {0x159, 0x2A8}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def line(stream, data):
    stream.write(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')


def preview_size(source_width, source_height, available_width, available_height):
    """Fit the display image inside its panel without changing its aspect ratio."""
    scale = min(max(1, available_width) / source_width,
                max(1, available_height) / source_height)
    return max(1, int(source_width * scale)), max(1, int(source_height * scale))


def terminal_dimensions():
    """Read the controlling terminal, including pixels when the emulator supplies them."""
    import fcntl
    import termios
    try:
        descriptor = os.open('/dev/tty', os.O_RDONLY | os.O_NOCTTY)
        try:
            rows, columns, pixel_width, pixel_height = struct.unpack(
                '=HHHH', fcntl.ioctl(descriptor, termios.TIOCGWINSZ, b'\0' * 8))
        finally:
            os.close(descriptor)
    except OSError:
        return None
    return (columns, rows, pixel_width, pixel_height) if columns and rows else None


class TerminalSizeFollower:
    """Resize the single preview container when its launching terminal is resized."""
    def __init__(self, gui, initial_size, reader=terminal_dimensions):
        self.gui = gui
        self.reader = reader
        self.initial = reader()
        self.previous = self.initial
        self.initial_size = initial_size

    def update(self):
        current = self.reader()
        if current is None:
            return
        if self.initial is None:
            self.initial = self.previous = current
            return
        if current == self.previous:
            return
        self.previous = current
        columns, rows, pixel_width, pixel_height = current
        width = pixel_width or round(self.initial_size[0] * columns / self.initial[0])
        height = pixel_height or round(self.initial_size[1] * rows / self.initial[1])
        width = max(780, min(width, max(780, self.gui.winfo_screenwidth() - 40)))
        height = max(360, min(height, max(360, self.gui.winfo_screenheight() - 80)))
        self.gui.geometry(f'{width}x{height}')


def interface_stats():
    result = {}
    for side, channel in CHANNELS.items():
        result[side] = {}
        for name in ('rx_packets', 'rx_dropped', 'rx_errors', 'tx_packets'):
            p = Path('/sys/class/net') / channel / 'statistics' / name
            try:
                result[side][name] = int(p.read_text())
            except OSError:
                result[side][name] = None
    return result


class Recorder:
    def __init__(self, output, args):
        self.output, self.args = output, args
        self.lock = threading.Lock()
        self.camera_start_lock = threading.Lock()
        self.stop = threading.Event()
        self.threads = []
        self.errors = []
        self.active = False
        self.start_ns = None
        self.end_ns = None
        self.finish_ns = None
        self.outcome = 'not_recorded'
        self.cache = {side: {} for side in CHANNELS}
        self.images = {}
        self.camera_latest = {}
        self.can_counts = {side: Counter() for side in CHANNELS}
        self.image_counts = Counter()
        self.socket_drops = {side: 0 for side in CHANNELS}
        self.socket_drops_at_start = None
        self.socket_drops_at_end = None
        self.snapshot_warmup_skipped = 0
        self.camera_gaps = Counter()
        self.snapshot_count = 0
        self.interface_before = None

    def launch(self, function, *args):
        def wrapper():
            try:
                function(*args)
            except Exception:
                error = f'{function.__name__} {args}:\n' + traceback.format_exc()
                with self.lock:
                    self.errors.append(error)
                print(error, flush=True)
                self.stop.set()
        worker = threading.Thread(target=wrapper, daemon=False)
        self.threads.append(worker)
        worker.start()

    def receiving(self, now):
        return self.active and self.start_ns is not None and now >= self.start_ns

    def can_worker(self, side, channel):
        with socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as bus:
            bus.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            # Linux SO_RXQ_OVFL reports cumulative socket receive-queue drops.
            bus.setsockopt(socket.SOL_SOCKET, 40, 1)
            bus.bind((channel,))
            bus.setblocking(False)
            with (self.output / f'can_{side}.jsonl').open('w', buffering=262144) as log:
                while not self.stop.is_set():
                    if not select.select([bus], [], [], 0.1)[0]:
                        continue
                    packet, ancillary, flags, _ = bus.recvmsg(FRAME.size, 128)
                    now = time.monotonic_ns()
                    for level, kind, value in ancillary:
                        if level == socket.SOL_SOCKET and kind == 40 and len(value) >= 4:
                            with self.lock:
                                self.socket_drops[side] = struct.unpack('=I', value[:4])[0]
                    if len(packet) != FRAME.size:
                        continue
                    can_id, dlc, data = FRAME.unpack(packet)
                    decoded = decode(can_id, data) if not can_id & 0xE0000000 and dlc == 8 else None
                    with self.lock:
                        if decoded:
                            self.cache[side][can_id] = (now, decoded)
                        recording = self.receiving(now)
                        drop_count = self.socket_drops[side]
                    if recording:
                        line(log, dict(host_receive_monotonic_ns=now, can_id_flags=can_id,
                                       dlc=dlc, data_hex=data[:min(dlc, 8)].hex(), recv_flags=flags,
                                       socket_drop_counter=drop_count))
                        self.can_counts[side][can_id] += 1

    def camera_worker(self, role, serial):
        import numpy as np
        import pyrealsense2 as rs
        from PIL import Image
        pipe = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, self.args.camera_fps)
        started = False
        previous = None
        try:
            with self.camera_start_lock:
                if self.stop.is_set():
                    return
                print(f'启动相机 {role} / {serial}……', flush=True)
                profile = pipe.start(config)
                started = True
            device = profile.get_device()
            details = {'serial': serial, 'name': device.get_info(rs.camera_info.name)}
            if device.supports(rs.camera_info.usb_type_descriptor):
                details['usb_type'] = device.get_info(rs.camera_info.usb_type_descriptor)
            write_json(self.output / f'camera_{role}_device.json', details)
            print(f'相机 {role} / {serial}: USB {details.get("usb_type", "unknown")}', flush=True)
            last_frame_time = time.monotonic()
            received_first = False
            folder = self.output / 'images' / role
            folder.mkdir(parents=True)
            with (self.output / f'camera_{role}.jsonl').open('w') as log:
                while not self.stop.is_set():
                    arrived, frames = pipe.try_wait_for_frames(timeout_ms=250)
                    if self.stop.is_set():
                        break
                    if not arrived:
                        limit = 3 if received_first else 10
                        if time.monotonic() - last_frame_time > limit:
                            raise RuntimeError(f'相机 {role} / {serial} 超过 {limit} 秒未收到图像；'
                                               f'USB={details.get("usb_type", "unknown")}，检查占用、供电和带宽')
                        continue
                    frame = frames.get_color_frame()
                    if not frame:
                        if time.monotonic() - last_frame_time > 10:
                            raise RuntimeError(f'相机 {role} / {serial} 持续缺少 RGB 帧')
                        continue
                    last_frame_time = time.monotonic()
                    if not received_first:
                        print(f'首帧到达 {role} / {serial}', flush=True)
                        received_first = True
                    now = time.monotonic_ns()
                    image = np.asanyarray(frame.get_data()).copy()
                    number = frame.get_frame_number()
                    metadata = dict(host_receive_monotonic_ns=now, frame_number=number,
                                    device_timestamp_ms=frame.get_timestamp(),
                                    timestamp_domain=str(frame.get_frame_timestamp_domain()))
                    with self.lock:
                        self.images[role] = (now, image)
                        recording = self.receiving(now)
                    if recording:
                        if previous is not None:
                            self.camera_gaps[role] += max(0, number - previous - 1)
                        previous = number
                        index = self.image_counts[role]
                        relative = f'images/{role}/{index:07d}.jpg'
                        Image.fromarray(image).save(self.output / relative, quality=95)
                        metadata['path'] = relative
                        line(log, metadata)
                        self.image_counts[role] += 1
                        with self.lock:
                            self.camera_latest[role] = metadata
        finally:
            if started:
                pipe.stop()

    def ready(self):
        now = time.monotonic_ns()
        with self.lock:
            return (not self.errors and
                    all(i in self.cache[side] and now-self.cache[side][i][0] < 200_000_000
                        for side in CHANNELS for i in REQUIRED) and
                    all(role in self.images and now-self.images[role][0] < 500_000_000
                        for role in CAMERAS))

    def start_recording(self):
        if self.start_ns is not None:
            return
        if not self.ready():
            print('尚未就绪：需要三路新鲜图像及两侧完整控制目标/反馈。', flush=True)
            return
        self.interface_before = interface_stats()
        with self.lock:
            self.start_ns = time.monotonic_ns()
            self.socket_drops_at_start = dict(self.socket_drops)
            self.active = True
        self.outcome = 'unreviewed'
        print('RECORDING：开始操作主臂进行示教。', flush=True)

    def finish(self, outcome):
        with self.lock:
            self.end_ns = time.monotonic_ns()
            self.socket_drops_at_end = dict(self.socket_drops)
            self.active = False
        self.outcome = outcome if self.start_ns is not None else 'not_recorded'
        self.stop.set()

    def snapshot(self):
        with self.lock:
            now = time.monotonic_ns()
            row = dict(host_sample_monotonic_ns=now, arms={}, cameras={})
            for side in CHANNELS:
                packets = {}
                for can_id, (stamp, decoded) in self.cache[side].items():
                    packets[hex(can_id)] = dict(host_receive_monotonic_ns=stamp,
                                              age_ms=(now-stamp)/1e6,
                                              kind=decoded[0], value=decoded[1])
                row['arms'][side] = packets
            row['cameras'] = {role: {**entry, 'age_ms': (now-entry['host_receive_monotonic_ns'])/1e6}
                              for role, entry in self.camera_latest.items()}
        return row

    def close(self):
        self.stop.set()
        for worker in self.threads:
            worker.join()
        self.finish_ns = time.monotonic_ns()
        if self.errors:
            self.outcome = 'incomplete_error'
        summary = dict(outcome=self.outcome, errors=self.errors,
                       start_monotonic_ns=self.start_ns, end_monotonic_ns=self.end_ns,
                       snapshot_count=self.snapshot_count,
                       image_counts=dict(self.image_counts),
                       camera_frame_number_gaps=dict(self.camera_gaps),
                       socket_receive_queue_drops=self.socket_drops,
                       socket_drops_at_recording_start=self.socket_drops_at_start,
                       socket_drops_at_recording_end=self.socket_drops_at_end,
                       socket_drops_during_recording=(
                           {side: (self.socket_drops_at_end[side]-self.socket_drops_at_start[side]) % (1 << 32)
                            for side in CHANNELS}
                           if self.socket_drops_at_start is not None and self.socket_drops_at_end is not None else None),
                       snapshot_initial_wait_count=self.snapshot_warmup_skipped,
                       can_counts={s: {hex(i): n for i, n in c.items()} for s, c in self.can_counts.items()},
                       interface_before=self.interface_before, interface_after=interface_stats(),
                       feedback_source_confirmed=self.args.feedback_source_confirmed,
                       training_ready=False)
        write_json(self.output / 'summary.json', summary)
        print('已保存:', self.output, '\n结果标记:', self.outcome, flush=True)


def run(args):
    # Preflight only reads interface settings and USB identity.
    for side, channel in CHANNELS.items():
        verify_usb(channel, SERIALS[side])
        report = subprocess.run(['ip', '-j', '-details', 'link', 'show', 'dev', channel],
                                capture_output=True, text=True, check=True)
        interface = json.loads(report.stdout)[0]
        data = interface.get('linkinfo', {}).get('info_data', {})
        if 'UP' not in interface.get('flags', []) or data.get('bittiming', {}).get('bitrate') != 1000000:
            raise RuntimeError(f'{channel} 需要已启用的 1 Mbps 接口；程序不会更改设置')
        if data.get('state') in ('BUS-OFF', 'STOPPED'):
            raise RuntimeError(f'{channel} 状态异常：{data.get("state")}')
    import numpy  # noqa: F401
    import pyrealsense2  # noqa: F401
    from PIL import Image  # noqa: F401
    gui = None
    terminal_follower = None
    preview_panels = {}
    if not args.headless:
        import tkinter as tk
        gui = tk.Tk()
        gui.title('NERO 示教采集：只读，不控制机械臂')
        gui.resizable(True, True)
        gui.minsize(780, 360)
        width = max(780, min(1400, gui.winfo_screenwidth() - 80))
        height = max(360, min(720, gui.winfo_screenheight() - 100))
        gui.geometry(f'{width}x{height}')
        if not args.no_follow_terminal:
            terminal_follower = TerminalSizeFollower(gui, (width, height))
            if terminal_follower.initial is not None:
                print('已启用终端尺寸跟随：拖动启动终端边缘，三路预览会统一缩放。', flush=True)
            else:
                print('未检测到控制终端，改为手动缩放采集窗口。', flush=True)
        gui.grid_rowconfigure(1, weight=1)
        for column in range(3):
            gui.grid_columnconfigure(column, weight=1, uniform='camera_columns')
    output = args.output / datetime.now().strftime('episode_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    rec = Recorder(output, args)
    clock = dict(monotonic_ns=time.monotonic_ns(), unix_ns=time.time_ns())
    write_json(output / 'metadata.json', dict(schema_version=2, task=args.prompt,
               roles={'right': '抓取、放入、参与协作盖盖', 'left': '协作盖盖'},
               cameras=CAMERAS, channels=CHANNELS, usb_serials=SERIALS,
               camera_mode=f'RGB 640x480 {args.camera_fps}fps requested; JPEG quality 95',
               requested_camera_fps=args.camera_fps,
               target_snapshot_hz=30, clock_anchor=clock,
               time_alignment='asynchronous; host userspace receive times; no hardware sync',
               feedback_source_confirmed=args.feedback_source_confirmed,
               feedback_source_note=args.feedback_source_note,
               success_definition='物体在盒内，双臂离开后盒盖保持关闭；不默认要求扣紧卡扣',
               training_ready=False))
    try:
        status = None
        if gui is not None:
            status = tk.Label(gui, text='等待三相机和两侧 CAN 就绪……')
            status.grid(row=0, column=0, columnspan=3, sticky='ew', pady=8)
            for column, role in enumerate(CAMERAS):
                panel = tk.Frame(gui)
                panel.grid(row=1, column=column, sticky='nsew', padx=4)
                panel.grid_rowconfigure(1, weight=1)
                panel.grid_columnconfigure(0, weight=1)
                tk.Label(panel, text=f'{role} / {CAMERAS[role]}').grid(row=0, column=0, pady=4)
                canvas = tk.Canvas(panel, width=1, height=1, background='#161616',
                                   highlightthickness=0)
                canvas.grid(row=1, column=0, sticky='nsew')
                preview_panels[role] = dict(canvas=canvas, item=None, photo=None, last_key=None)
            tk.Button(gui, text='开始记录 (S)', command=rec.start_recording).grid(row=2, column=0, sticky='ew', padx=4, pady=10, ipady=8)
            tk.Button(gui, text='成功结束 (F)', command=lambda: rec.finish('success_user_marked')).grid(row=2, column=1, sticky='ew', padx=4, pady=10, ipady=8)
            tk.Button(gui, text='失败结束 (X)', command=lambda: rec.finish('failed_user_marked')).grid(row=2, column=2, sticky='ew', padx=4, pady=10, ipady=8)
            gui.bind('<KeyPress-s>', lambda e: rec.start_recording())
            gui.bind('<KeyPress-f>', lambda e: rec.finish('success_user_marked'))
            gui.bind('<KeyPress-x>', lambda e: rec.finish('failed_user_marked'))
            gui.bind('<Escape>', lambda e: rec.finish('aborted'))
            gui.protocol('WM_DELETE_WINDOW', lambda: rec.finish('aborted'))
        for side, channel in CHANNELS.items():
            rec.launch(rec.can_worker, side, channel)
        for role, serial in CAMERAS.items():
            rec.launch(rec.camera_worker, role, serial)
        print('正在打开三路图像和两侧 CAN。不会发送控制指令。', flush=True)
        if args.headless:
            print('无窗口试录模式：所有数据就绪后自动开始，到时标记 diagnostic。', flush=True)
        boot = time.monotonic()
        next_sample = boot
        next_display = boot
        next_log = boot
        next_terminal_check = boot
        with (output / 'samples.jsonl').open('w') as samples:
            while not rec.stop.is_set():
                now = time.monotonic()
                ready = rec.ready()
                if rec.start_ns is None and not ready and now-boot > 30:
                    raise RuntimeError('30 秒内未开始记录：设备未就绪或未点击开始；请检查输出后重新运行')
                if args.headless and rec.start_ns is None and ready and now-boot > 3:
                    rec.start_recording()
                if rec.active and now >= next_sample:
                    row = rec.snapshot()
                    # Raw logs start immediately. Reference snapshots start once
                    # every camera has a saved frame; later stale rows stay visible.
                    if rec.snapshot_count == 0 and len(row['cameras']) != len(CAMERAS):
                        rec.snapshot_warmup_skipped += 1
                    else:
                        line(samples, row)
                        rec.snapshot_count += 1
                    next_sample = now + 1/30
                if rec.active and args.seconds and (time.monotonic_ns()-rec.start_ns)/1e9 >= args.seconds:
                    rec.finish('diagnostic' if args.headless else 'unreviewed_timeout')
                if gui is not None:
                    if terminal_follower is not None and now >= next_terminal_check:
                        terminal_follower.update()
                        next_terminal_check = now + 0.25
                    if now >= next_display:
                        with rec.lock:
                            previews = dict(rec.images)
                        for role, (stamp, pixels) in previews.items():
                            panel = preview_panels[role]
                            canvas = panel['canvas']
                            panel_width, panel_height = canvas.winfo_width(), canvas.winfo_height()
                            if panel_width <= 1 or panel_height <= 1:
                                continue
                            size = preview_size(pixels.shape[1], pixels.shape[0], panel_width, panel_height)
                            key = (stamp, panel_width, panel_height)
                            if panel['last_key'] == key:
                                continue
                            small = Image.fromarray(pixels).resize(size, Image.Resampling.BILINEAR)
                            ppm = f'P6\n{size[0]} {size[1]}\n255\n'.encode('ascii') + small.tobytes()
                            photo = tk.PhotoImage(data=ppm, format='PPM', master=gui)
                            if panel['item'] is None:
                                panel['item'] = canvas.create_image(panel_width/2, panel_height/2,
                                                                    image=photo, anchor='center')
                            else:
                                canvas.itemconfigure(panel['item'], image=photo)
                                canvas.coords(panel['item'], panel_width/2, panel_height/2)
                            panel['photo'] = photo
                            panel['last_key'] = key
                        state = 'RECORDING' if rec.active else 'READY：按 S 开始' if ready else 'WAITING：缺少或过期数据'
                        status.configure(text=state + ' | 左 can1 / 右 can0 | Esc 中止')
                        next_display = now + 0.1
                    gui.update()
                if now >= next_log:
                    elapsed = 0 if rec.start_ns is None else (time.monotonic_ns()-rec.start_ns)/1e9
                    print(f'ready={ready} recording={rec.active} t={elapsed:.1f}s images={dict(rec.image_counts)}', flush=True)
                    next_log = now + 2
                time.sleep(0.002)
    except KeyboardInterrupt:
        rec.finish('aborted')
    except Exception:
        rec.errors.append(traceback.format_exc())
        rec.finish('incomplete_error')
        raise
    finally:
        if rec.end_ns is None:
            rec.finish('incomplete_error' if rec.errors else 'aborted')
        rec.close()
        if gui is not None:
            gui.destroy()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='无窗口静止试录，必须指定 --seconds')
    parser.add_argument('--no-follow-terminal', action='store_true', help='关闭默认的终端尺寸跟随，改为手动缩放预览窗口')
    parser.add_argument('--seconds', type=float, default=0, help='自动结束时长；窗口模式 0 表示手动结束')
    parser.add_argument('--camera-fps', type=int, choices=(15, 30), default=30, help='相机请求帧率；15 用于低带宽诊断')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2]/'data'/'nero_demos')
    parser.add_argument('--prompt', default=PROMPT)
    parser.add_argument('--feedback-source-confirmed', action='store_true', help='仅在已核实反馈来自从臂时填写')
    parser.add_argument('--feedback-source-note', default='未核实同总线反馈的物理来源')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0 or (args.headless and args.seconds <= 0):
        parser.error('时长必须有限且非负；--headless 需要 --seconds > 0')
    return args


if __name__ == '__main__':
    run(parse_args())
