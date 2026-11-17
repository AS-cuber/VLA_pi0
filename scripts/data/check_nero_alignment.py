#!/usr/bin/env python3
"""离线核对 15Hz 对齐中间数据：时间、维度、原始来源、单位和整条示教划分。"""
import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def check(output):
    manifest=json.loads((output/'dataset_manifest.json').read_text())
    split=json.loads((output/'split.json').read_text())
    assert manifest['status']=='alignment_prepared_not_lerobot', '数据构建未完成'
    assert len(split['train'])==8 and len(split['validation'])==2
    assert not set(split['train']) & set(split['validation']), '训练/验证轨迹泄漏'
    assert set(split['train']+split['validation'])=={r['episode'] for r in manifest['episodes']}
    total=0
    for report in manifest['episodes']:
        source=Path(report['raw_episode'])
        for name,digest in report['source_hashes'].items():
            assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest, f'原始来源已改变：{source.name}/{name}'
        with np.load(output/report['episode']/'aligned.npz',allow_pickle=False) as data:
            grid=data['host_grid_ns'];n=len(grid)
            assert data['state'].shape==data['action'].shape==(n,16)
            assert data['image_paths'].shape==(n,3)
            assert np.isfinite(data['state']).all() and np.isfinite(data['action']).all()
            assert np.all(np.abs(np.diff(grid)-1e9/15)<=1), '不是统一 15Hz 时间轴'
            assert np.all(data['image_receive_ns']<=grid[:,None]), '使用了未来图像'
            assert np.max(grid[:,None]-data['image_receive_ns'])<=100_000_000
            for path in data['image_paths'].ravel():
                assert Path(path).is_file(),f'图像缺失：{path}'
            # Independently decode raw integers for start/middle/end samples.
            joint_targets={0x155:(0,1),0x156:(2,3),0x157:(4,5),0x170:(6,)}
            joint_feedback={0x2a5:(0,1),0x2a6:(2,3),0x2a7:(4,5),0x2a9:(6,)}
            for side_index,side in enumerate(('left','right')):
                raw={i:[] for i in set(joint_targets)|set(joint_feedback)|{0x159,0x2a8}}
                for text in (source/f'can_{side}.jsonl').open():
                    row=json.loads(text)
                    if row['can_id_flags'] in raw:
                        raw[row['can_id_flags']].append((row['host_receive_monotonic_ns'],bytes.fromhex(row['data_hex'])))
                for can_id,entries in raw.items():
                    times=[row[0] for row in entries]
                    recorded=data[f'{side}_{can_id:03x}_receive_ns']
                    assert np.all(recorded<=grid)
                    assert np.max(grid-recorded)<=50_000_000
                    for index in (0,n//2,n-1):
                        raw_index=bisect_right(times,int(grid[index]))-1
                        assert raw_index>=0 and times[raw_index]==int(recorded[index])
                        payload=entries[raw_index][1]
                        if can_id in joint_targets or can_id in joint_feedback:
                            indices=(joint_targets if can_id in joint_targets else joint_feedback)[can_id]
                            array=data['action'] if can_id in joint_targets else data['state']
                            for value_index,joint in enumerate(indices):
                                raw_value=int.from_bytes(payload[value_index*4:value_index*4+4],'big',signed=True)
                                expected=math.radians(raw_value*.001)
                                assert np.isclose(array[index,side_index*8+joint],expected,rtol=1e-6,atol=1e-7)
                        else:
                            array=data['action'] if can_id==0x159 else data['state']
                            expected=int.from_bytes(payload[:4],'big',signed=True)*1e-6
                            assert np.isclose(array[index,side_index*8+7],expected,rtol=1e-6,atol=1e-9)
            total+=n
        print(report['episode'],n,'帧：维度、时间与原始数值核对通过')
    assert total==manifest['total_frames']
    print(f'全部通过：10 条，{total} 帧，训练 8 条、验证 2 条。未归一化，未生成 LeRobot 格式。')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[2]/'data/nero_aligned_v1')
    args=parser.parse_args()
    check(args.input)
