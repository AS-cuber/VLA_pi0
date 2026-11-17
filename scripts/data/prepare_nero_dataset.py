#!/usr/bin/env python3
"""按主机接收时间对齐原始示教，生成 15Hz 中间数据及按 episode 的训练/验证划分。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.acquisition.nero_teleop_read import TARGETS, FEEDBACK, decode

ROLES = ('base', 'left_wrist', 'right_wrist')
SIDES = ('left', 'right')
FEATURES = ([f'left_joint_{i}_rad' for i in range(1, 8)] + ['left_gripper_m']
            + [f'right_joint_{i}_rad' for i in range(1, 8)] + ['right_gripper_m'])


def load_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as file:
        for index, line in enumerate(file, 1):
            try:
                yield json.loads(line)
            except Exception as error:
                raise ValueError(f'{path}: 第 {index} 行无效') from error


def causal_indices(times, queries, max_age_ns):
    times = np.asarray(times, dtype=np.int64)
    if len(times) == 0 or np.any(np.diff(times) < 0):
        raise ValueError('原始数据为空或时间戳倒退')
    indices = np.searchsorted(times, queries, side='right')-1
    if np.any(indices < 0):
        raise ValueError('查询时刻之前没有数据')
    ages = queries-times[indices]
    if np.any(ages > max_age_ns):
        raise ValueError(f'对齐数据过期，最大年龄 {ages.max()/1e6:.2f} ms')
    return indices, ages


def load_can(path):
    required = set(TARGETS) | set(FEEDBACK) | {0x159, 0x2A8}
    streams = {i: {'times': [], 'values': []} for i in required}
    for row in rows(path):
        can_id = row['can_id_flags']
        if can_id not in streams:
            continue
        if row['dlc'] != 8:
            raise ValueError(f'{path}: 关键报文长度不是 8')
        data = bytes.fromhex(row['data_hex'])
        if len(data) != 8:
            raise ValueError(f'{path}: 关键报文载荷损坏')
        kind, value = decode(can_id, data)
        if kind in ('target', 'feedback'):
            values = [math.radians(v) for v in value['degrees']]
        else:
            if value['unit'] != 'mm':
                raise ValueError(f'{path}: 夹爪不是宽度模式，不能转换为米')
            values = [value['value']/1000]
        streams[can_id]['times'].append(row['host_receive_monotonic_ns'])
        streams[can_id]['values'].append(values)
    if any(not v['times'] for v in streams.values()):
        raise ValueError(f'{path}: 关键控制目标或反馈缺失')
    return streams


def prepare_episode(source, destination, fps, max_image_ms, max_can_ms):
    metadata, summary = load_json(source/'metadata.json'), load_json(source/'summary.json')
    if summary['outcome'] != 'success_user_marked' or summary['errors']:
        raise ValueError(f'{source.name}: 不是无错误的成功标记记录')
    if not metadata['feedback_source_confirmed']:
        raise ValueError(f'{source.name}: 未声明核实反馈来源')
    if summary.get('socket_drops_during_recording') is None or any(summary['socket_drops_during_recording'].values()):
        raise ValueError(f'{source.name}: 缺少 socket 基线或录制期发生溢出')
    images = {role: list(rows(source/f'camera_{role}.jsonl')) for role in ROLES}
    streams = {side: load_can(source/f'can_{side}.jsonl') for side in SIDES}
    bounds = [([r['host_receive_monotonic_ns'] for r in v]) for v in images.values()]
    bounds += [v['times'] for side in streams.values() for v in side.values()]
    if any(not seq for seq in bounds):
        raise ValueError('存在空数据流')
    start = summary['start_monotonic_ns']
    common_start = max(seq[0] for seq in bounds)
    common_end = min(seq[-1] for seq in bounds)
    # Integer nanoseconds on a rational 15Hz grid; never index samples.jsonl by row.
    first = math.ceil((common_start-start)*fps/1e9)
    last = math.floor((common_end-start)*fps/1e9)
    frame_numbers = np.arange(first, last+1, dtype=np.int64)
    grid = start + np.rint(frame_numbers.astype(np.float64)*1e9/fps).astype(np.int64)
    if len(grid) < fps*2:
        raise ValueError('公共有效区间少于两秒')
    states, actions = [], []
    ages, selected_times = [], []
    provenance = {}
    for side in SIDES:
        vectors = {}
        for kind, joint_ids, grip_id in (('state', FEEDBACK, 0x2A8), ('action', TARGETS, 0x159)):
            vector = np.empty((len(grid), 8), dtype=np.float64)
            stamps = []
            for can_id in list(joint_ids)+[grip_id]:
                stream = streams[side][can_id]
                selected, age = causal_indices(stream['times'], grid, max_can_ms*1e6)
                values = np.asarray(stream['values'], dtype=np.float64)[selected]
                indices = joint_ids[can_id] if can_id in joint_ids else (7,)
                vector[:, indices] = values
                times = np.asarray(stream['times'], dtype=np.int64)[selected]
                stamps.append(times)
                ages.append(age)
                provenance[f'{side}_{can_id:03x}_receive_ns'] = times
            span = np.ptp(np.stack(stamps, axis=1), axis=1)
            if span.max() > 30_000_000:
                raise ValueError(f'{side}/{kind}: CAN 子包接收跨度超过 30ms')
            vectors[kind] = vector
        states.append(vectors['state'])
        actions.append(vectors['action'])
    image_paths, image_indices, image_timestamps = [], [], []
    for role in ROLES:
        entries = images[role]
        times = np.array([v['host_receive_monotonic_ns'] for v in entries], dtype=np.int64)
        indices, age = causal_indices(times, grid, max_image_ms*1e6)
        chosen = [entries[i] for i in indices]
        paths = [str((source/row['path']).resolve()) for row in chosen]
        if any(not Path(p).is_file() for p in paths):
            raise ValueError(f'{role}: 图像缺失')
        image_paths.append(paths)
        image_indices.append(indices)
        image_timestamps.append(times[indices])
        selected_times.append(times[indices])
    state, action = np.concatenate(states, axis=1), np.concatenate(actions, axis=1)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError('状态或动作存在非有限值')
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination/'aligned.npz', state=state.astype(np.float32),
                        action=action.astype(np.float32), host_grid_ns=grid,
                        timestamp_s=(grid-grid[0]).astype(np.float64)/1e9,
                        source_frame_numbers=frame_numbers,
                        image_paths=np.asarray(image_paths, dtype=str).T,
                        image_source_indices=np.stack(image_indices, axis=1),
                        image_receive_ns=np.stack(image_timestamps, axis=1),
                        **provenance)
    report = dict(episode=source.name, frame_count=len(grid), fps=fps,
                  raw_duration_s=(summary['end_monotonic_ns']-start)/1e9,
                  aligned_span_s=(int(grid[-1])-int(grid[0]))/1e9,
                  head_trim_ms=(int(grid[0])-start)/1e6,
                  tail_trim_ms=(summary['end_monotonic_ns']-int(grid[-1]))/1e6,
                  max_can_age_ms=float(np.max(np.stack(ages))/1e6),
                  max_image_age_ms=float(np.max(grid[:, None]-np.stack(image_timestamps, axis=1))/1e6),
                  max_intercamera_receive_skew_ms=float(np.max(np.ptp(np.stack(image_timestamps, axis=1), axis=1))/1e6),
                  left_gripper_state_range_m=[float(state[:, 7].min()),float(state[:, 7].max())],
                  left_gripper_action_range_m=[float(action[:, 7].min()),float(action[:, 7].max())],
                  task=metadata['task'], raw_episode=str(source.resolve()),
                  source_hashes={name:digest(source/name) for name in ('metadata.json','summary.json','can_left.jsonl','can_right.jsonl','camera_base.jsonl','camera_left_wrist.jsonl','camera_right_wrist.jsonl')},
                  hardware_exposure_alignment_verified=False)
    write_json(destination/'alignment_report.json', report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[2]
    parser.add_argument('--input',type=Path,default=root/'data/nero_demos')
    parser.add_argument('--output',type=Path,default=root/'data/nero_aligned_v1')
    args=parser.parse_args()
    if args.output.exists():
        parser.error('输出目录已存在，为避免覆盖请指定新的 --output')
    episodes=sorted(p for p in args.input.glob('episode_*') if p.is_dir())
    if len(episodes)!=10:
        parser.error(f'本轮预期十条示教，实际 {len(episodes)} 条；请检查输入目录')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'dataset_manifest.json',{'status':'building'})
    try:
        reports=[]
        for source in episodes:
            report=prepare_episode(source,args.output/source.name,15,100,50)
            reports.append(report)
            print(f'{source.name}: {report["frame_count"]} 帧；最大图像年龄 {report["max_image_age_ms"]:.1f}ms；最大 CAN 年龄 {report["max_can_age_ms"]:.1f}ms',flush=True)
        names=[p.name for p in episodes]
        shuffled=np.random.default_rng(42).permutation(names).tolist()
        validation=sorted(shuffled[:2]);train=sorted(shuffled[2:])
        write_json(args.output/'split.json',dict(seed=42,train=train,validation=validation,unit='whole episode'))
        write_json(args.output/'dataset_manifest.json',dict(status='alignment_prepared_not_lerobot',fps=15,
                   feature_order=FEATURES,camera_order=ROLES,episode_count=10,
                   total_frames=sum(r['frame_count'] for r in reports),episodes=reports,
                   sampling='causal latest-at-or-before host receive timestamp; no interpolation; no action time shift',
                   action_semantics='absolute commanded joint positions (rad) and commanded gripper width (m)',
                   state_semantics='measured follower joint positions (rad) and gripper width (m)',
                   language='task is stored in each alignment report',
                   normalisation='not applied; compute training-only statistics after final OpenPI transforms; floor or bypass near-constant gripper scales',
                   limitation='host receipt alignment is not calibrated exposure/control latency alignment; fixed-scene pipeline test only',
                   split_file='split.json',portable_images=False))
        print('完成:',args.output,'；训练 8 条，验证 2 条；原始数据未改动。',flush=True)
    except BaseException:
        write_json(args.output/'dataset_manifest.json',{'status':'incomplete_do_not_train'})
        raise


if __name__=='__main__':
    main()
