"""Batch selection keeps failures, splits complete episodes, and isolates model assets."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.data.prepare_nero_dataset import select_episodes, split_episodes
from scripts.model.nero_batch import check_export, paths, validation_anchors
from scripts.model.prepare_nero_pi0 import EXPECTED_FEATURES


class BatchTests(unittest.TestCase):
    def test_failed_raw_episode_is_kept_and_not_selected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i, outcome in enumerate(('success_user_marked', 'failed_user_marked', 'success_user_marked')):
                path = root / f'episode_{i}'
                path.mkdir()
                (path / 'summary.json').write_text(json.dumps({'outcome': outcome}))
            selected, excluded = select_episodes(root)
            self.assertEqual([p.name for p in selected], ['episode_0', 'episode_2'])
            self.assertEqual(excluded, [{'episode': 'episode_1', 'outcome': 'failed_user_marked'}])
            self.assertTrue((root / 'episode_1/summary.json').is_file())

    def test_49_successes_split_39_10_without_overlap(self):
        names = [f'episode_{i}' for i in range(49)]
        split = split_episodes(names, 10)
        self.assertEqual((len(split['train']), len(split['validation'])), (39, 10))
        self.assertFalse(set(split['train']) & set(split['validation']))
        self.assertEqual(set(split['train'] + split['validation']), set(names))
        self.assertEqual(split, split_episodes(names, 10))
        self.assertEqual(len(split_episodes(names)['validation']), 10)

    def test_yellow_legacy_split_remains_8_2(self):
        names = [f'episode_{i}' for i in range(10)]
        self.assertEqual(len(split_episodes(names)['validation']), 2)
        for count in (0, 10, -1):
            with self.assertRaises(ValueError):
                split_episodes(names, count)

    def test_dataset_names_cannot_escape_or_overwrite_old_assets(self):
        red, yellow = paths('/tmp/project', 'red_v1'), paths('/tmp/project', 'yellow_v2')
        self.assertNotEqual(red['assets'], yellow['assets'])
        self.assertEqual(red['repo'], 'local/nero_red_v1_train')
        for name in ('../red', '/red', 'Red', 'box_v1'):
            with self.assertRaises(ValueError):
                paths('/tmp/project', name)

    def report(self):
        return dict(status='exported_and_reload_checked_not_yet_openpi_configured', fps=15,
                    feature_order=EXPECTED_FEATURES,
                    action_semantics='absolute commanded joint positions (rad) and commanded gripper width (m)',
                    splits={'train': {'repo_id': 'local/nero_red_v1_train', 'frames': 6,
                                      'episodes': [{'source_episode': 'train', 'frames': 6}]},
                            'validation': {'repo_id': 'local/nero_red_v1_validation', 'frames': 8,
                                           'episodes': [{'source_episode': 'val0', 'frames': 3},
                                                        {'source_episode': 'val1', 'frames': 5}]}})

    def test_dynamic_export_report_rejects_leakage_and_repo_mismatch(self):
        report = self.report()
        self.assertEqual(check_export(report, 'local/nero_red_v1_train')['frames'], 6)
        for mutate in ('overlap', 'repo'):
            value = copy.deepcopy(report)
            if mutate == 'overlap':
                value['splits']['validation']['episodes'][0]['source_episode'] = 'train'
            else:
                value['splits']['train']['repo_id'] = 'local/nero_box_v1_train'
            with self.assertRaises(ValueError):
                check_export(value, 'local/nero_red_v1_train')

    def test_validation_anchors_use_batch_boundaries(self):
        anchors = validation_anchors(self.report(), all_frames=True)
        self.assertEqual([a['index'] for a in anchors], list(range(8)))
        self.assertEqual([a['remaining_frames'] for a in anchors], [3, 2, 1, 5, 4, 3, 2, 1])
        self.assertEqual(len(validation_anchors(self.report(), all_frames=False)), 6)

class SnapshotRestoreTests(unittest.TestCase):
    def test_full_snapshot_restores_both_batches_and_rejects_overwrite(self):
        import hashlib
        import io
        import tarfile
        from unittest import mock
        from scripts.data import restore_github_data as restore
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            backup = root / 'data/archive'
            backup.mkdir(parents=True)
            files = {
                'data/nero_demos/yellow/episode_y/metadata.json': b'{"task":"yellow"}',
                'data/nero_demos/red/episode_r/summary.json': b'{"outcome":"failed_user_marked"}',
                'data/nero_aligned_v1/split.json': b'{"train":["yellow"]}',
                'data/nero_aligned_red_v1/split.json': b'{"train":["red"]}',
            }
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode='w:gz') as archive:
                for name, content in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
            content = payload.getvalue()
            digest = hashlib.sha256(content).hexdigest()
            (backup / 'snapshot.part000').write_bytes(content)
            (backup / 'manifest.json').write_text(json.dumps(dict(
                format_version=2,
                directories=['nero_demos', 'nero_aligned_v1', 'nero_aligned_red_v1'],
                archive_size=len(content), archive_sha256=digest,
                original_file_count=len(files), original_bytes=sum(map(len, files.values())),
                parts=[dict(name='snapshot.part000', size=len(content), sha256=digest)])))
            with mock.patch.object(restore, '__file__', str(root / 'scripts/data/restore_github_data.py')):
                with mock.patch('sys.argv', ['restore_github_data.py']):
                    restore.main()
                    with self.assertRaises(SystemExit):
                        restore.main()
            for name, content in files.items():
                self.assertEqual((root / name).read_bytes(), content)

    def test_old_snapshot_restores_into_yellow_and_preserves_existing_red(self):
        import hashlib
        import io
        import tarfile
        from unittest import mock
        from scripts.data import restore_github_data as restore
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            backup = root / 'data/archive'
            backup.mkdir(parents=True)
            red = root / 'data/nero_demos/red/keep.txt'
            red.parent.mkdir(parents=True)
            red.write_text('new red collection')
            payload = io.BytesIO()
            files = {'data/nero_demos/episode_old/metadata.json': b'{"task":"yellow"}',
                     'data/nero_aligned_v1/split.json': b'{}'}
            with tarfile.open(fileobj=payload, mode='w:gz') as archive:
                for name, data in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            data = payload.getvalue()
            digest = hashlib.sha256(data).hexdigest()
            (backup / 'snapshot.part000').write_bytes(data)
            (backup / 'manifest.json').write_text(json.dumps(dict(
                directories=['nero_demos', 'nero_aligned_v1'], archive_size=len(data), archive_sha256=digest,
                original_file_count=2, original_bytes=sum(map(len, files.values())),
                parts=[dict(name='snapshot.part000', size=len(data), sha256=digest)])))
            with mock.patch.object(restore, '__file__', str(root / 'scripts/data/restore_github_data.py')):
                with mock.patch('sys.argv', ['restore_github_data.py']):
                    restore.main()
                    with self.assertRaises(SystemExit):
                        restore.main()
            self.assertEqual(red.read_text(), 'new red collection')
            self.assertEqual((root / 'data/nero_demos/yellow/episode_old/metadata.json').read_bytes(), files['data/nero_demos/episode_old/metadata.json'])
            self.assertTrue((root / 'data/nero_aligned_v1/split.json').is_file())
            self.assertFalse((root / 'data/nero_demos/episode_old').exists())


if __name__ == '__main__':
    unittest.main()
