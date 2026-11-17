"""Directory migration preserves model/data paths and removes only identified leftovers."""
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

from scripts.tools.organize_vla_wu import apply_plan, archive_matches, build_plan, digest


TEMPLATES = Path(__file__).resolve().parents[2] / 'scripts/tools/vla_wu_templates'


class LayoutTests(unittest.TestCase):
    def project(self, root):
        (root / 'vendor/openpi').mkdir(parents=True)
        for name in ('nero_pi0_adapter.py', 'nero_pi0_config.py', 'prepare_nero_pi0.py', 'run_nero_pi0.py'):
            (root / name).write_text('"""fixture module"""\nVALUE = 1\n')
        (root / 'data').mkdir()
        (root / 'data/keep.bin').write_bytes(b'original data')
        checkpoint = root / 'checkpoints/pi0_nero_box_v1/keep/99/params'
        checkpoint.mkdir(parents=True)
        (checkpoint / '_METADATA').write_text('{}')
        return checkpoint

    @mock.patch('scripts.tools.organize_vla_wu.active_model_processes', return_value=[])
    def test_apply_preserves_source_bytes_and_valid_checkpoint_and_is_idempotent(self, active):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.project(root)
            protected = {name: digest(root / name) for name in
                         ('nero_pi0_adapter.py', 'nero_pi0_config.py', 'prepare_nero_pi0.py')}
            full = root / 'results/pi0_nero_box_v1/comparison_ok'
            full.mkdir(parents=True)
            (full / 'report.json').write_text(json.dumps({'status': 'paired_evaluation_passed',
                                                        'evaluation_scope': 'all_validation_frames'}))
            failed = full.parent / 'train_failed'
            failed.mkdir()
            (failed / 'report.json').write_text(json.dumps({'status': 'failed_or_interrupted'}))
            (root / 'check_nero_openpi.py').write_text('pass\n')
            (root / '__pycache__').mkdir()
            (root / '__pycache__/a.pyc').write_bytes(b'cache')
            plan = build_plan(root)
            apply_plan(root, plan, TEMPLATES)
            for name, sha in protected.items():
                self.assertEqual(digest(root / 'scripts/model' / name), sha)
                self.assertFalse((root / name).exists())
            self.assertEqual((root / 'data/keep.bin').read_bytes(), b'original data')
            self.assertTrue((checkpoint / '_METADATA').is_file())
            self.assertFalse(failed.exists())
            self.assertFalse((root / '__pycache__').exists())
            self.assertFalse((root / 'check_nero_openpi.py').exists())
            self.assertTrue((root / 'docs/文件索引.md').is_file())
            before = sorted(str(path.relative_to(root)) for path in root.rglob('*'))
            apply_plan(root, build_plan(root), TEMPLATES)
            self.assertEqual(before, sorted(str(path.relative_to(root)) for path in root.rglob('*')))

    def test_plan_does_not_move_files_and_rejects_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.project(root)
            before = sorted(str(path.relative_to(root)) for path in root.rglob('*'))
            self.assertTrue(build_plan(root)['moves'])
            self.assertEqual(before, sorted(str(path.relative_to(root)) for path in root.rglob('*')))
            target = root / 'scripts/model/nero_pi0_config.py'
            target.parent.mkdir(parents=True)
            target.write_text('different\n')
            with self.assertRaises(RuntimeError):
                build_plan(root)
            self.assertTrue((root / 'nero_pi0_adapter.py').is_file())

    def test_archive_deleted_only_after_all_members_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'unpacked'
            (source / 'data').mkdir(parents=True)
            (source / 'data/a.txt').write_text('original')
            archive = root / 'input.tar.gz'
            with tarfile.open(archive, 'w:gz') as stream:
                stream.add(source / 'data/a.txt', arcname='data/a.txt')
            self.assertTrue(archive_matches(archive, source))
            (source / 'data/a.txt').write_text('modified')
            self.assertFalse(archive_matches(archive, source))

    @mock.patch('scripts.tools.organize_vla_wu.active_model_processes', return_value=[{'pid': 123}])
    def test_active_model_blocks_mutation(self, active):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.project(root)
            with self.assertRaises(RuntimeError):
                apply_plan(root, build_plan(root), TEMPLATES)
            self.assertTrue((root / 'run_nero_pi0.py').is_file())
            self.assertFalse((root / 'logs').exists())

    def test_interrupted_run_with_usable_checkpoint_keeps_its_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.project(root)
            partial = root / 'results/pi0_nero_box_v1/train_partial'
            partial.mkdir(parents=True)
            (partial / 'report.json').write_text(json.dumps({'status': 'failed_or_interrupted',
                'checkpoint_dir': str(checkpoint.parent.parent)}))
            removed = {item['path'] for item in build_plan(root)['deletions']}
            self.assertNotIn(str(partial.relative_to(root)), removed)
            self.assertNotIn(str(checkpoint.parent.parent.relative_to(root)), removed)


if __name__ == '__main__':
    unittest.main()
