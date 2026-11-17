#!/usr/bin/env python3
"""离线检查一条 NERO 记录的文件完整性和应用接收新鲜度，不连接硬件。"""
import argparse
from collections import Counter
import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.acquisition.record_nero_demo import CAMERAS, CHANNELS, REQUIRED


def rows(path):
    with path.open() as stream:
        for number, text in enumerate(stream, 1):
            try:
                yield json.loads(text)
            except Exception as error:
                raise ValueError(f'{path.name} 第 {number} 行损坏') from error


def check(folder):
    issues = []
    summary = json.loads((folder/'summary.json').read_text())
    metadata = json.loads((folder/'metadata.json').read_text())
    start, end = summary['start_monotonic_ns'], summary['end_monotonic_ns']
    if start is None or end is None or end <= start:
        raise ValueError('没有有效的记录起止时间')
    duration = (end-start)/1e9
    if summary['errors']:
        issues.append('记录器报错')
    for role in CAMERAS:
        count, previous, max_gap = 0, start, 0
        for entry in rows(folder/f'camera_{role}.jsonl'):
            stamp = entry['host_receive_monotonic_ns']
            if stamp < previous or not start <= stamp <= end:
                issues.append(f'{role} 图像接收时间倒退或越界')
            max_gap = max(max_gap, stamp-previous)
            previous = stamp
            image = folder/entry['path']
            if not image.is_file() or image.stat().st_size == 0:
                issues.append(f'{role} 快照文件缺失：{entry["path"]}')
            count += 1
        max_gap = max(max_gap, end-previous)
        rate = count/duration
        print(f'{role}: {count} 张，平均 {rate:.2f} fps，最长接收间隔 {max_gap/1e6:.1f} ms')
        if count != summary['image_counts'].get(role, 0) or count == 0:
            issues.append(f'{role} 图像计数不一致或为空')
        if rate < 0.8 * metadata.get('requested_camera_fps', 30) or max_gap > 500_000_000:
            issues.append(f'{role} 接收偏慢或长时间未更新')
    for side in CHANNELS:
        counts = Counter()
        previous = start
        for entry in rows(folder/f'can_{side}.jsonl'):
            stamp = entry['host_receive_monotonic_ns']
            if stamp < previous or not start <= stamp <= end:
                issues.append(f'{side} CAN 接收时间倒退或越界')
            previous = stamp
            counts[entry['can_id_flags']] += 1
        missing = REQUIRED-set(counts)
        if missing:
            issues.append(f'{side} 缺少关键 CAN ID：{[hex(i) for i in sorted(missing)]}')
        recording_drops = summary.get('socket_drops_during_recording')
        if recording_drops is not None:
            increment = recording_drops.get(side, 0)
            print(f'{side} socket 溢出：启动至退出累计 '
                  f'{summary["socket_receive_queue_drops"].get(side, 0)}；录制区间增量 {increment}')
            if increment:
                issues.append(f'{side} 录制区间 socket 接收队列溢出')
        elif summary['socket_receive_queue_drops'].get(side, 0):
            issues.append(f'{side} socket 累计溢出，旧记录未保存起始基线，无法区分启动期与录制期')
        print(f'{side}: {sum(counts.values())} CAN 帧；关键 ID 缺失 {len(missing)} 个')
    sample_count, stale, previous, max_gap = 0, 0, start, 0
    for row in rows(folder/'samples.jsonl'):
        stamp = row['host_sample_monotonic_ns']
        max_gap = max(max_gap, stamp-previous)
        previous = stamp
        sample_count += 1
        valid = all(hex(i) in row['arms'][side] and 0 <= row['arms'][side][hex(i)]['age_ms'] < 100
                    for side in CHANNELS for i in REQUIRED)
        valid = valid and all(role in row['cameras'] and 0 <= row['cameras'][role]['age_ms'] < 200
                              for role in CAMERAS)
        if not valid:
            stale += 1
    print(f'参考快照: {sample_count}，缺失/过期 {stale}，最长间隔 {max_gap/1e6:.1f} ms')
    if summary.get('snapshot_initial_wait_count'):
        print('参考快照开始前等待三路首张保存图像的次数:', summary['snapshot_initial_wait_count'],
              '；该时段原始图像和 CAN 仍保留')
    if not sample_count or sample_count != summary['snapshot_count']:
        issues.append('参考快照为空或计数不一致')
    if stale:
        issues.append('存在缺失/过期参考快照；需检查时间范围和原始记录')
    if max_gap > 200_000_000:
        issues.append('参考快照出现超过 200ms 的间隔')
    print('任务:', metadata['task'])
    print('人工结果标记:', summary['outcome'])
    if not metadata['feedback_source_confirmed']:
        print('metadata.json: feedback_source_confirmed=False')
    print('发现的问题:', list(dict.fromkeys(issues)) if issues else '无')
    return 1 if issues else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    args = parser.parse_args()
    raise SystemExit(check(args.episode))
