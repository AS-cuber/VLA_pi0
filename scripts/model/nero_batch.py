#!/usr/bin/env python3
"""Prepare, train and evaluate a named NERO batch without modifying yellow v1 assets."""
import argparse
import csv
import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.model.prepare_nero_pi0 import EXPECTED_COMMIT, EXPECTED_FEATURES, calculate_stats, require
from scripts.model.run_nero_pi0 import capture_training_output, load_official_train, read_training_metrics, source_version_matches

SOURCES = ('nero_pi0_adapter.py', 'nero_pi0_config.py', 'prepare_nero_pi0.py', 'nero_batch.py')


def paths(project, dataset):
    require(bool(re.fullmatch(r'[a-z][a-z0-9_]*', dataset)) and dataset != 'box_v1',
            'dataset 使用小写字母、数字和下划线，例如 red_v1；box_v1 留给历史实验')
    project = Path(project).resolve()
    name, repo = f'pi0_nero_{dataset}', f'local/nero_{dataset}_train'
    return dict(project=project, name=name, repo=repo,
                data=project / f'data/nero_lerobot_{dataset}',
                cache=project / f'data/nero_openpi_cache_{dataset}',
                assets=project / 'assets' / name / repo,
                results=project / 'results' / name)


def check_export(report, repo):
    require(report['status'] == 'exported_and_reload_checked_not_yet_openpi_configured', 'LeRobot 导出未完成')
    require(report['fps'] == 15 and report['feature_order'] == EXPECTED_FEATURES, '采样率或字段顺序不符')
    require(report['action_semantics'] == 'absolute commanded joint positions (rad) and commanded gripper width (m)',
            '动作必须为绝对目标 rad/m')
    names = []
    for key in ('train', 'validation'):
        part = report['splits'][key]
        require(part['episodes'] and all(v['frames'] >= 3 for v in part['episodes']), '示教为空或过短')
        require(sum(v['frames'] for v in part['episodes']) == part['frames'], '分段帧数不符')
        expected_repo = repo if key == 'train' else repo.removesuffix('_train') + '_validation'
        require(part['repo_id'] == expected_repo, 'repo_id 与 --dataset 不一致')
        names.extend(v['source_episode'] for v in part['episodes'])
    require(len(names) == len(set(names)), '训练和验证示教重复或交叉')
    return report['splits']['train']


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(project, dataset):
    p = paths(project, dataset)
    report = json.loads((p['data'] / 'export_report.json').read_text())
    check_export(report, p['repo'])
    commit = subprocess.check_output(['git', '-C', str(p['project'] / 'vendor/openpi'), 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == EXPECTED_COMMIT, f'OpenPI 版本不符：{commit}')
    link = p['cache'] / p['repo']
    train = (p['data'] / 'train').resolve(strict=True)
    require((train / 'meta/info.json').is_file(), '训练目录缺少 meta/info.json')
    if link.exists() or link.is_symlink():
        require(link.is_symlink() and link.resolve() == train, f'缓存位置已被其他数据占用：{link}')
    else:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(train, target_is_directory=True)
    os.environ.update(HF_LEROBOT_HOME=str(p['cache']), HF_HUB_OFFLINE='1',
                      HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled')
    return p, report


def make_config(p, exp_name):
    from nero_pi0_config import get_train_config
    cfg = get_train_config(p['project'], exp_name)
    return dataclasses.replace(cfg, name=p['name'], data=dataclasses.replace(cfg.data, repo_id=p['repo']))


def prepare(p, report):
    require(not p['assets'].exists(), f'统计已存在，拒绝覆盖：{p["assets"]}')
    from lerobot.common.constants import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared import normalize
    from openpi.training import data_loader
    from nero_pi0_config import statistics_data_config
    require(Path(HF_LEROBOT_HOME).resolve() == p['cache'], '请在新的 Python 进程中准备本批次统计')
    metadata = LeRobotDatasetMetadata(p['repo'])
    require(Path(metadata.root).resolve() == (p['data'] / 'train').resolve() and metadata.fps == 15, '训练集路径或采样率不符')
    spec = dataclasses.replace(statistics_data_config(), repo_id=p['repo'], asset_id=p['repo'])
    dataset = data_loader.create_torch_dataset(spec, 50, Pi0Config())
    train = report['splits']['train']
    require(len(dataset) == train['frames'], '训练帧数与导出报告不同')
    transform = transforms.compose([*spec.repack_transforms.inputs, *spec.data_transforms.inputs])
    episodes = np.repeat(np.arange(len(train['episodes'])), [v['frames'] for v in train['episodes']])
    states, actions = [], []
    for index in range(len(dataset)):
        row = dataset[index]
        require(int(row['episode_index']) == int(episodes[index]), '训练示教顺序不同')
        value = transform(row)
        require(value['state'].shape == (16,) and value['actions'].shape == (50, 16), '统计输入维度不符')
        states.append(value['state'].copy())
        actions.append(value['actions'].copy())
        if (index + 1) % 500 == 0 or index + 1 == len(dataset):
            print(f'训练统计：{index + 1}/{len(dataset)}', flush=True)
    stats, details = {}, {}
    for key, values in (('state', np.stack(states)), ('actions', np.concatenate(actions))):
        numbers, details[key] = calculate_stats(values)
        stats[key] = normalize.NormStats(**numbers)
    p['assets'].mkdir(parents=True, exist_ok=False)
    normalize.save(p['assets'], stats)
    audit = dict(dataset=p['name'], train_repo=p['repo'], validation_values_used=False,
                 state_vectors=len(dataset), action_vectors=len(dataset) * 50, action_horizon=50,
                 source_sha256=source_hashes(), export_report_sha256=file_hash(p['data'] / 'export_report.json'),
                 train_source_episodes=[v['source_episode'] for v in train['episodes']],
                 action_sampling='50 steps per anchor, including terminal padding',
                 scale_floor='0.001 rad for joints; 0.001 m for grippers', details=details)
    (p['assets'] / 'nero_batch_report.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print('训练集统计已保存：', p['assets'] / 'norm_stats.json')


def readiness(p, report):
    audit = json.loads((p['assets'] / 'nero_batch_report.json').read_text())
    require(audit['train_repo'] == p['repo'] and audit['validation_values_used'] is False
            and audit['state_vectors'] == report['splits']['train']['frames'], '统计来源与本批次不同')
    current_sources = source_hashes()
    require(set(audit['source_sha256']) == set(current_sources) and all(
        source_version_matches(name, audit['source_sha256'][name], value)
        for name, value in current_sources.items()), '适配/统计代码已改变；请使用新的数据版本准备统计')
    require(audit['export_report_sha256'] == file_hash(p['data'] / 'export_report.json'), '导出报告已改变')
    norm = json.loads((p['assets'] / 'norm_stats.json').read_text())['norm_stats']
    for key in ('state', 'actions'):
        mean, std = np.asarray(norm[key]['mean']), np.asarray(norm[key]['std'])
        require(mean.shape == std.shape == (16,) and np.isfinite(mean).all()
                and np.isfinite(std).all() and (std >= .001).all(), '统计值无效')
    require((p['project'] / 'checkpoints/openpi-assets/checkpoints/pi0_base/params/_METADATA').is_file(),
            '缺少官方 pi0_base 基础权重')


def train(p, report, args):
    import jax
    require(jax.default_backend() == 'gpu' and jax.device_count() == 1, '请指定一张 GPU')
    require(args.exp_name is not None and bool(re.fullmatch(r'[A-Za-z0-9_-]+', args.exp_name)), '训练需要合法的 --exp-name')
    require(args.steps >= 2 and 1 <= args.save_interval <= args.steps, '训练步数/保存间隔不合法')
    cfg = make_config(p, args.exp_name)
    cfg = dataclasses.replace(cfg, num_train_steps=args.steps, save_interval=args.save_interval,
                              lr_schedule=dataclasses.replace(cfg.lr_schedule, decay_steps=args.steps,
                                                              warmup_steps=min(10, args.steps - 1)))
    require(not cfg.checkpoint_dir.exists(), f'实验已存在：{cfg.checkpoint_dir}')
    p['results'].mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='train_', dir=p['results']))
    result = dict(status='running', dataset=p['name'], exp_name=args.exp_name,
                  num_train_steps=args.steps, checkpoint_dir=str(cfg.checkpoint_dir),
                  export_report_sha256=file_hash(p['data'] / 'export_report.json'),
                  norm_stats_sha256=file_hash(p['assets'] / 'norm_stats.json'))
    print('本次结果目录：', output, flush=True)
    try:
        with capture_training_output(output / 'console.log'):
            load_official_train(p['project']).main(cfg)
        metrics = read_training_metrics((output / 'console.log').read_text())
        (output / 'training_metrics.json').write_text(json.dumps(metrics, indent=2))
        with (output / 'training_metrics.csv').open('w', newline='') as stream:
            fields = ['step'] + sorted({k for row in metrics for k in row if k != 'step'})
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(metrics)
        result['status'] = 'training_completed'
    except BaseException as error:
        result.update(status='interrupted', error=str(error))
        raise
    finally:
        (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))


def validation_anchors(report, all_frames):
    anchors, offset = [], 0
    for episode_index, episode in enumerate(report['splits']['validation']['episodes']):
        n = episode['frames']
        for local in (range(n) if all_frames else sorted({0, n // 2, n - 1})):
            anchors.append(dict(index=offset + local, episode_index=episode_index,
                                source_episode=episode['source_episode'], local_index=local,
                                remaining_frames=n - local))
        offset += n
    return anchors


def evaluate(p, report, args):
    import jax
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.policies.policy_config import create_trained_policy
    from scripts.model.check_nero_checkpoint import error_metrics
    require(jax.default_backend() == 'gpu' and jax.device_count() == 1, '请指定一张 GPU')
    require(args.checkpoint is not None, 'evaluate 需要 --checkpoint')
    checkpoint = args.checkpoint.resolve()
    require(checkpoint.parent.parent == (p['project'] / 'checkpoints' / p['name']).resolve()
            and checkpoint.name.isdecimal() and (checkpoint / 'params/_METADATA').is_file(), '检查点与批次不符')
    stored = checkpoint / 'assets' / p['repo'] / 'norm_stats.json'
    saved_norm = json.loads(stored.read_text())['norm_stats']
    train_norm = json.loads((p['assets'] / 'norm_stats.json').read_text())['norm_stats']
    require(all(np.array_equal(saved_norm[feature][key], train_norm[feature][key])
                for feature in ('state', 'actions') for key in ('mean', 'std')),
            '检查点与训练统计不同')
    cfg = make_config(p, checkpoint.parent.name)
    policy = create_trained_policy(cfg, checkpoint, sample_kwargs={'num_steps': 10})
    validation = report['splits']['validation']
    dataset = LeRobotDataset(validation['repo_id'], root=p['data'] / 'validation',
                             delta_timestamps={'action': [i / 15 for i in range(50)]})
    require(len(dataset) == validation['frames'], '验证集帧数不同')
    anchors = validation_anchors(report, args.all_frames)
    predictions, references, states, masks = [], [], [], []
    for number, anchor in enumerate(anchors):
        row = dataset[anchor['index']]
        require(int(row['episode_index']) == anchor['episode_index'], '验证示教顺序不同')
        obs = {'state': np.asarray(row['observation.state']), 'prompt': row['task'],
               'images': {r: np.asarray(row[f'observation.images.{r}']) for r in ('base', 'left_wrist', 'right_wrist')}}
        noise = np.random.default_rng(42 + anchor['index']).standard_normal((50, 32)).astype(np.float32)
        prediction = np.asarray(policy.infer(obs, noise=noise)['actions'])
        require(prediction.shape == (50, 16) and np.isfinite(prediction).all(), '预测动作无效')
        predictions.append(prediction)
        references.append(np.asarray(row['action']))
        states.append(obs['state'])
        masks.append(np.arange(50) < anchor['remaining_frames'])
        if number == 0 or (number + 1) % 50 == 0:
            print(f'验证推理：{number + 1}/{len(anchors)}', flush=True)
    predictions, references, masks = np.stack(predictions), np.stack(references), np.stack(masks)
    metrics = error_metrics(predictions, references, masks)
    p['results'].mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='evaluation_', dir=p['results']))
    np.savez_compressed(output / 'validation_actions.npz', predictions=predictions, references=references,
                        states=np.stack(states), valid_mask=masks,
                        dataset_indices=np.array([a['index'] for a in anchors]))
    result = dict(status='validation_evaluation_passed', dataset=p['name'], evaluated_model='trained',
                  trained_checkpoint=str(checkpoint), anchors=anchors, metrics=metrics,
                  evaluation_scope='all_validation_frames' if args.all_frames else 'episode_start_middle_end')
    (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('离线评估结果：', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'train', 'evaluate'))
    parser.add_argument('--project', type=Path, default=Path('/home/clwu/vla-wu'))
    parser.add_argument('--dataset', default='red_v1', help='与导出的 repo-prefix local/nero_批次 对应')
    parser.add_argument('--exp-name')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--save-interval', type=int, default=250)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--all-frames', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    p, report = setup(args.project, args.dataset)
    if args.mode == 'prepare':
        prepare(p, report)
    else:
        readiness(p, report)
        (train if args.mode == 'train' else evaluate)(p, report, args)


if __name__ == '__main__':
    main()
