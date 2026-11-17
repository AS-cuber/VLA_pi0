"""Only completed, error-free successes become visible demonstration folders."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import time

import numpy  # Import before patching sys.modules, avoiding cache restoration reloads.
from PIL import Image  # noqa: F401
import threading
import unittest
from unittest import mock

from scripts.acquisition import record_nero_demo as recording


class RecordingStorageTests(unittest.TestCase):
    def args(self, output):
        return Namespace(output=output, headless=True, no_follow_terminal=True,
                         camera_fps=15, prompt='红色任务', feedback_source_confirmed=True,
                         feedback_source_note='fixture', seconds=1)

    def run_without_devices(self, output, capture):
        interface = json.dumps([{'flags': ['UP'], 'linkinfo': {'info_data': {
            'bittiming': {'bitrate': 1000000}, 'state': 'ERROR-ACTIVE'}}}])
        with mock.patch.object(recording, 'verify_usb'), \
                mock.patch.object(recording.subprocess, 'run', return_value=Namespace(stdout=interface)), \
                mock.patch.object(recording, 'interface_stats', return_value={}), \
                mock.patch.object(recording, 'capture_session', side_effect=capture), \
                mock.patch.dict('sys.modules', {'pyrealsense2': mock.Mock()}):
            recording.run(self.args(output))

    def capture(self, outcome, started=True, errors=None):
        def implementation(args, rec, gui, follower, panels):
            self.assertTrue(rec.output.name.startswith('.recording_'))
            # Nothing has been published while data is still being collected.
            self.assertEqual(list(args.output.glob('episode_*')), [])
            (rec.output / 'metadata.json').write_text('{"task":"red"}')
            (rec.output / 'images').mkdir()
            (rec.output / 'images/frame.jpg').write_bytes(b'fixture image')
            if started:
                rec.start_ns = 1
            rec.finish(outcome)
            rec.errors.extend(errors or [])
            rec.close()
        return implementation

    def test_success_publishes_complete_files_after_workers_close(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            def capture(args, rec, gui, follower, panels):
                rec.start_ns = 1
                def worker():
                    rec.stop.wait()
                    (rec.output / 'last_frame.jpg').write_bytes(b'last frame')
                thread = threading.Thread(target=worker)
                rec.threads.append(thread)
                thread.start()
                rec.finish('success_user_marked')
                rec.close()
            self.run_without_devices(output, capture)
            episodes = list(output.glob('episode_*'))
            self.assertEqual(len(episodes), 1)
            self.assertEqual((episodes[0] / 'last_frame.jpg').read_bytes(), b'last frame')
            self.assertEqual(json.loads((episodes[0] / 'summary.json').read_text())['outcome'], 'success_user_marked')
            self.assertFalse(list(output.glob('.recording_*')))

    def test_failure_abort_and_timeout_leave_no_recording(self):
        for outcome in ('failed_user_marked', 'aborted', 'diagnostic', 'unreviewed_timeout', 'incomplete_error'):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                self.run_without_devices(output, self.capture(outcome))
                self.assertEqual(list(output.iterdir()), [])

    def test_success_before_start_and_late_camera_error_are_discarded(self):
        for started, errors in ((False, []), (True, ['camera failed during shutdown'])):
            with self.subTest(started=started, errors=errors), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                self.run_without_devices(output, self.capture('success_user_marked', started, errors))
                self.assertEqual(list(output.iterdir()), [])

    def test_exception_and_keyboard_interrupt_clean_temporary_data(self):
        for error in (RuntimeError('capture failed'), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                existing = output / 'existing_record'
                existing.mkdir()
                (existing / 'keep.txt').write_text('previous data')
                def capture(args, rec, gui, follower, panels):
                    (rec.output / 'partial.jpg').write_bytes(b'partial')
                    raise error
                with self.assertRaises(type(error)):
                    self.run_without_devices(output, capture)
                self.assertEqual(list(output.iterdir()), [existing])
                self.assertEqual((existing / 'keep.txt').read_text(), 'previous data')

    def test_summary_write_failure_cannot_publish_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with mock.patch.object(recording, 'write_json', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    self.run_without_devices(output, self.capture('success_user_marked'))
            self.assertEqual(list(output.iterdir()), [])

    def test_existing_episode_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            existing = output / 'episode_existing'
            existing.mkdir()
            (existing / 'keep.txt').write_text('previous data')
            def capture(args, rec, gui, follower, panels):
                rec.start_ns = 1
                rec.finish('success_user_marked')
                rec.close()
            with mock.patch.object(recording, 'datetime') as clock:
                clock.now.return_value.strftime.return_value = 'episode_existing'
                with self.assertRaises(FileExistsError):
                    self.run_without_devices(output, capture)
            self.assertEqual(list(output.iterdir()), [existing])
            self.assertEqual((existing / 'keep.txt').read_text(), 'previous data')

    def test_first_end_decision_is_final(self):
        with tempfile.TemporaryDirectory() as directory:
            rec = recording.Recorder(Path(directory), self.args(Path(directory)))
            rec.start_ns = 1
            rec.finish('failed_user_marked')
            first_end = rec.end_ns
            rec.finish('success_user_marked')
            self.assertEqual(rec.outcome, 'failed_user_marked')
            self.assertEqual(rec.end_ns, first_end)
            rec.start_recording()
            self.assertFalse(rec.active)

    def test_gui_capture_branch_constructs_buttons_without_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = self.args(output)
            args.headless, args.seconds = False, 0
            rec = recording.Recorder(output, args)
            rec.start_ns = time.monotonic_ns()
            gui, toolkit = mock.Mock(), mock.Mock()
            gui.update.side_effect = lambda: rec.finish('success_user_marked')
            with mock.patch.dict('sys.modules', {'tkinter': toolkit}), \
                    mock.patch.object(rec, 'launch'), mock.patch.object(rec, 'ready', return_value=True), \
                    mock.patch.object(recording, 'interface_stats', return_value={}):
                recording.capture_session(args, rec, gui, None, {})
            self.assertEqual(rec.outcome, 'success_user_marked')
            self.assertTrue((output / 'metadata.json').is_file())
            self.assertTrue((output / 'summary.json').is_file())


if __name__ == '__main__':
    unittest.main()
