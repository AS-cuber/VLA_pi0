#!/usr/bin/env python3
"""Serve a trained NERO pi0 checkpoint on loopback HTTP, or a labelled mock."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('/home/clwu/vla-wu'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--mock', action='store_true', help='仅测试传输；实机执行器拒绝此服务')
    parser.add_argument('--skip-warmup', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('端口必须为 1～65535')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    from scripts.deployment.service import MockPolicy, PolicyServer
    from scripts.deployment.protocol import PROMPT
    metadata = {'mock': args.mock, 'horizon': 50, 'action_dim': 16, 'action_hz': 15,
                'units': 'rad/m', 'checkpoint': None, 'norm_stats_sha256': None}
    if args.mock:
        policy = MockPolicy()
    else:
        if args.checkpoint is None:
            parser.error('真实服务必须指定 --checkpoint')
        project, checkpoint = args.project.resolve(), args.checkpoint.resolve()
        if not checkpoint.name.isdecimal() or not (checkpoint / 'params/_METADATA').is_file():
            parser.error('请指定包含 params/_METADATA 的数字编号检查点目录')
        stats = checkpoint / 'assets/local/nero_box_v1_train/norm_stats.json'
        saved = json.loads(stats.read_text())['norm_stats']
        for feature in ('state', 'actions'):
            if any(len(saved[feature][key]) != 16 for key in ('mean', 'std')):
                raise ValueError('检查点归一化参数不是 NERO 16 维')
        # Support both server layouts without changing the original training sources.
        config_dir = project / 'scripts/model'
        if not (config_dir / 'nero_pi0_config.py').is_file():
            config_dir = project
        sys.path.insert(0, str(config_dir))
        import jax
        import numpy as np
        from openpi.policies.policy_config import create_trained_policy
        from nero_pi0_config import get_train_config
        if jax.default_backend() != 'gpu' or jax.device_count() != 1:
            raise ValueError('真实服务需要通过 CUDA_VISIBLE_DEVICES 指定一张 GPU')
        cfg = get_train_config(project, checkpoint.parent.name)
        logging.info('加载候选检查点：%s', checkpoint)
        policy = create_trained_policy(cfg, checkpoint, sample_kwargs={'num_steps': 10})
        if not args.skip_warmup:
            logging.info('首次编译预热；完成后才开放服务端口')
            policy.infer({'state': np.zeros(16, np.float32),
                          'images': {r: np.zeros((480, 640, 3), np.uint8)
                                     for r in ('base', 'left_wrist', 'right_wrist')}, 'prompt': PROMPT})
        metadata.update(checkpoint=str(checkpoint),
                        norm_stats_sha256=hashlib.sha256(stats.read_bytes()).hexdigest())
    with PolicyServer(('127.0.0.1', args.port), policy, metadata) as server:
        logging.info('服务就绪：http://127.0.0.1:%d；mock=%s', args.port, args.mock)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            logging.info('服务停止')


if __name__ == '__main__':
    main()
