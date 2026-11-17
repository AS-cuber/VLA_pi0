#!/usr/bin/env python3
"""Preview dual NERO states, prediction chunks, or recorded first-action targets."""
import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.simulation.dual_nero import DEFAULT_LAYOUT, DEFAULT_MODEL, DualNero


def load_trajectory(path, mode='states', episode=0, anchor=0, steps=50):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    selection = np.arange(len(data.get('states', data.get('state', []))))
    report_path = path.with_name('report.json')
    if 'predictions' in data:
        if not report_path.is_file():
            raise ValueError('评估预测旁需要 report.json，以区分两条示教')
        anchors = json.loads(report_path.read_text())['anchors']
        if len(anchors) != len(data['predictions']):
            raise ValueError('报告和 NPZ 帧数不同')
        if not np.array_equal(data['dataset_indices'], [a['index'] for a in anchors]):
            raise ValueError('报告和 NPZ 帧顺序不同')
        selection = np.array([i for i, a in enumerate(anchors) if a['episode_index'] == episode], dtype=int)
        if not len(selection):
            raise ValueError('所选 episode 不存在')
    if mode == 'states':
        values = data.get('states', data.get('state'))
        if values is None or values.ndim != 2:
            raise ValueError('states 模式需要 (N,16) state/states')
        trajectory = values[selection]
    elif mode == 'first-actions':
        trajectory = data['predictions'][selection, 0]
    elif mode == 'references':
        trajectory = data['references'][selection, 0] if 'references' in data else data['action']
    else:
        if 'predictions' in data:
            if not 0 <= anchor < len(selection):
                raise ValueError('--anchor 是所选示教内的行号')
            index = selection[anchor]
            actions = data['predictions'][index]
            mask = data['valid_mask'][index]
            actions = actions[mask]
            initial = data['states'][index]
        else:
            actions = data['actions']
            initial = data['state']
        trajectory = np.concatenate([initial[None], actions[:steps]])
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != 16 or not len(trajectory) or not np.isfinite(trajectory).all():
        raise ValueError('轨迹必须为非空、有限的 (N,16)')
    return trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='aligned.npz / validation_actions.npz / 客户端动作记录')
    parser.add_argument('--mode', choices=('states', 'first-actions', 'chunk', 'references'), default='states')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--anchor', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--speed', type=float, default=1)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--layout', type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument('--allow-out-of-range', action='store_true', help='仅仿真：显示越界目标而不截断')
    parser.add_argument('--headless', action='store_true', help='加载全部轨迹，不打开窗口')
    parser.add_argument('--render', type=Path, help='离屏渲染最后一帧 PNG；需 Pillow')
    parser.add_argument('--loop', action='store_true')
    args = parser.parse_args()
    if not np.isfinite([args.fps, args.speed]).all() or min(args.fps, args.speed) <= 0 or not 1 <= args.steps <= 50:
        parser.error('fps/speed 必须为正数；steps 范围 1～50')
    trajectory = load_trajectory(args.input, args.mode, args.episode, args.anchor, args.steps) if args.input else np.zeros((1, 16))
    robot = DualNero(args.model, args.layout)
    print('运动学预览；未进行物理抓取仿真。安装位姿已标定：', robot.layout.get('calibrated', False))
    if args.headless:
        violations = 0
        for frame in trajectory:
            violations += bool(robot.set_state(frame, args.allow_out_of_range))
        print(f'已读取 {len(trajectory)} 帧；越界帧 {violations}。')
    else:
        with robot.viewer() as viewer:
            while viewer.is_running():
                for frame in trajectory:
                    if not viewer.is_running():
                        break
                    start = time.monotonic()
                    with viewer.lock():
                        bad = robot.set_state(frame, args.allow_out_of_range)
                    if bad:
                        print('越界目标：', ', '.join(bad))
                    viewer.sync()
                    time.sleep(max(0, 1 / args.fps / args.speed - (time.monotonic() - start)))
                if not args.loop:
                    break
    if args.render:
        from PIL import Image
        import mujoco
        robot.set_state(trajectory[-1], args.allow_out_of_range)
        with mujoco.Renderer(robot.model, width=640, height=480) as renderer:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [.2, 0, 1]
            camera.distance, camera.azimuth, camera.elevation = 2.4, 135, -25
            renderer.update_scene(robot.data, camera=camera)
            args.render.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(renderer.render()).save(args.render)
        print('已保存：', args.render)


if __name__ == '__main__':
    main()
