import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import video_scene_describer4 as v4
from test_video_scene_describer4 import entry, event


def identified_entry(events, second=18., frame_id='front:180/rear:180'):
    result = entry([])
    result.description = v4.validate_frame_evidence(entry(events, second).description, [second],
        [dict(frame_id=frame_id, sec=second, front_sec=second, rear_sec=second)])
    result.captured_frame_id = frame_id
    result.captured_frame_sec = second
    return result


def current_event(**changes):
    return event(**dict(dict(first_seen=18., last_seen=18., evidence_frames=[18.]), **changes))


class FrameIdentityTests(unittest.TestCase):
    def test_decoder_identity_overrides_model_metadata_and_survives_validation(self):
        e = current_event(evidence_frame_ids=['invented'])
        result = identified_entry([e])
        data = v4.validate_observations(json.loads(result.description))
        self.assertEqual(data['latest_frame_id'], 'front:180/rear:180')
        self.assertEqual(data['observations'][0]['evidence_frame_ids'], ['front:180/rear:180'])
        report = v4.build_final_report([result])
        self.assertEqual(report['latest_frame_analysis_status'], 'completed')
        self.assertEqual(len(report['notifications']), 1)

    def test_identity_matches_despite_float_rounding(self):
        result = identified_entry([current_event()])
        result.captured_frame_sec += 1e-12
        self.assertEqual(len(v4.build_final_report([result])['notifications']), 1)

    def test_later_failed_frame_suppresses_old_current_but_keeps_contact(self):
        past = identified_entry([current_event(), current_event(event_id='touch', notification_reason='contact')])
        failed = entry([])
        failed.description = '推論失敗: timeout'
        failed.captured_frame_id = 'front:200/rear:200'
        failed.captured_frame_sec = 20.
        report = v4.build_final_report([past, failed])
        self.assertEqual(report['latest_frame_analysis_status'], 'unavailable')
        self.assertEqual(report['latest_analyzed_frame_sec'], 18.)
        self.assertEqual(report['latest_frame_sec'], 20.)
        self.assertEqual([n['event_id'] for n in report['notifications']], ['scene_0_touch'])
        self.assertIn('離席中', report['notifications'][0]['message'])

    def test_partial_chunk_merge_does_not_invent_analysis_of_last_frame(self):
        result = identified_entry([current_event()])
        merged = v4.merge_chunk_descriptions_locally([
            v4.ChunkAnalysisResult(1, 'one', result.description),
            v4.ChunkAnalysisResult(2, 'two', '', error='timeout')])
        result.description = merged
        result.captured_frame_id = 'front:190/rear:190'
        result.captured_frame_sec = 19.
        report = v4.build_final_report([result])
        self.assertEqual(report['latest_frame_analysis_status'], 'unavailable')
        self.assertEqual(report['notifications'], [])

    def test_newer_empty_frame_and_different_ids_are_not_current(self):
        previous = identified_entry([current_event()])
        latest = identified_entry([], 19., 'front:190/rear:190')
        self.assertEqual(v4.build_final_report([previous, latest])['notifications'], [])
        previous.captured_frame_id = 'front:180/rear:181'
        self.assertEqual(v4.build_final_report([previous])['notifications'], [])

    def test_chunk_analysis_failure_keeps_actual_successful_boundary(self):
        timeline = [dict(frame_id=f'front:{i}/rear:{i}', sec=i / 10) for i in range(10)]
        partial = v4.validate_frame_evidence(entry([event(first_seen=.5, last_seen=.5,
            evidence_frames=[.5])], .5).description, [f['sec'] for f in timeline[:6]], timeline[:6])

        def chunk(**kwargs):
            if kwargs['chunk_index'] == 2:
                return v4.ChunkAnalysisResult(2, 'two', '', error='timeout')
            return v4.ChunkAnalysisResult(1, 'one', partial)

        with patch.object(v4, 'analyze_dual_video_chunk', side_effect=chunk), \
             patch.object(v4, 'log'), v4.requests.Session() as session:
            description = v4.describe_dual_video_scene('http://test', ['img'] * 10, session,
                2048, 0., (1., 1.), 'caution', '', True, 50, sample_fps=10.,
                frame_times=[f['sec'] for f in timeline], frame_timeline=timeline)
        self.assertEqual(json.loads(description)['latest_frame_sec'], .5)
        self.assertEqual(json.loads(description)['latest_frame_id'], 'front:5/rear:5')
        result = entry([])
        result.description = description
        result.captured_frame_id = timeline[-1]['frame_id']
        result.captured_frame_sec = .9
        self.assertEqual(v4.build_final_report([result])['notifications'], [])

    def test_equal_front_times_do_not_assign_evidence_to_a_different_rear_frame(self):
        # Sampling timestamps remain distinct even if a low-FPS front image is repeated.
        timeline = [dict(frame_id='front:0/rear:0', sec=0., front_sec=0., rear_sec=0.),
                    dict(frame_id='front:0/rear:1', sec=.05, front_sec=0., rear_sec=.05)]
        e = event(first_seen=0., last_seen=0., evidence_frames=[0.])
        description = v4.validate_frame_evidence(entry([e], .05).description, [0., .05], timeline)
        self.assertEqual(json.loads(description)['observations'][0]['evidence_frame_ids'], ['front:0/rear:0'])


class NotificationDeduplicationTests(unittest.TestCase):
    def test_overlapping_chunks_keep_observations_and_group_notification(self):
        a = identified_entry([current_event(notification_reason='contact')])
        merged = v4.merge_chunk_descriptions_locally([
            v4.ChunkAnalysisResult(1, 'one', a.description),
            v4.ChunkAnalysisResult(2, 'two', a.description)])
        a.description = merged
        report = v4.build_final_report([a])
        self.assertEqual(len(report['observations']), 2)
        self.assertEqual(len(report['notifications']), 1)
        self.assertEqual(len(report['notifications'][0]['related_event_ids']), 2)

    def test_cross_scene_overlap_and_styles_use_same_group(self):
        a = identified_entry([current_event()])
        reports = [v4.build_final_report([a, a], notification_style=style)
                   for style in ('object_only', 'situation', 'action_oriented')]
        self.assertTrue(all(len(r['notifications']) == 1 for r in reports))
        self.assertTrue(all(r['notifications'][0]['related_event_ids'] ==
                            reports[0]['notifications'][0]['related_event_ids'] for r in reports))

    def test_same_source_and_ambiguous_objects_are_kept_separate(self):
        a = identified_entry([current_event(event_id='a'), current_event(event_id='b')])
        b = identified_entry([current_event(event_id='c')])
        report = v4.build_final_report([a, b])
        self.assertEqual(len(report['notifications']), 3)
        self.assertTrue(all(len(n['related_event_ids']) == 1 for n in report['notifications']))

    def test_different_camera_position_or_time_not_grouped(self):
        a = identified_entry([current_event(notification_reason='contact')])
        alternatives = [
            identified_entry([current_event(direction='前方', notification_reason='contact')]),
            identified_entry([current_event(position='右側', notification_reason='contact')]),
            identified_entry([event(first_seen=19., last_seen=19., evidence_frames=[19.],
                                   notification_reason='contact')], 19., 'front:190/rear:190')]
        for b in alternatives:
            with self.subTest(description=b.description):
                self.assertEqual(len(v4.build_final_report([a, b])['notifications']), 2)

    def test_duplicate_does_not_consume_notification_slot(self):
        a = identified_entry([current_event()])
        b = identified_entry([current_event()])
        c = identified_entry([current_event(direction='前方', notification_reason='blocked_path',
                                          temporal_status='stationary', action='stationary')])
        report = v4.build_final_report([a, b, c])
        self.assertEqual(len(report['notifications']), 2)
        self.assertEqual(report['notifications'][1]['direction'], '前方')

    def test_environment_requires_shared_frame_name_and_position(self):
        a = identified_entry([current_event(object_type='other', object_label='コーン', position='中央',
            action='stationary', temporal_status='stationary', notification_reason='blocked_path')])
        environment = dict(status='completed', latest_sec=18., latest_frame_id=a.captured_frame_id,
            obstacles=[dict(direction='後方', object='コーン', position='中央', initial_state='unknown',
                            latest_visible=True, evidence='進路を塞ぐコーン')])
        report = v4.build_final_report([a], environment=environment)
        self.assertEqual(len(report['observations']), 2)
        self.assertEqual(len(report['notifications']), 1)
        self.assertEqual(len(report['notifications'][0]['related_event_ids']), 2)
        environment['obstacles'][0]['position'] = ''
        self.assertEqual(len(v4.build_final_report([a], environment=environment)['notifications']), 2)


class VideoPipelineTests(unittest.TestCase):
    def test_actual_video_decode_chunking_reports_and_resume(self):
        requests = []

        def inference(base_url, messages, *args, **kwargs):
            requests.append(kwargs.get('request_context'))
            if kwargs.get('request_context') == 'environment_comparison':
                return v4.ApiJsonResult(content='{"obstacles":[]}')
            texts = [part['text'] for part in messages[1]['content']
                     if part['type'] == 'text' and part['text'].startswith('動画開始から ')]
            times = [float(text.split()[1]) for text in texts]
            e = event(first_seen=min(times), last_seen=max(times), evidence_frames=times)
            return v4.ApiJsonResult(content=entry([e], max(times)).description)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            video = directory / 'test.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10., (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(20):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            output = directory / 'output'
            argv = ['v4', '--front-video', str(video), '--back-video', str(video),
                    '--output-dir', str(output), '--log-level', 'quiet']
            with patch('sys.argv', argv), patch.object(v4, 'check_server'), \
                 patch.object(v4, 'request_dual_video_json', side_effect=inference), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(v4.main(), 0)
                report = json.loads((output / 'final_report.json').read_text())
                self.assertEqual(report['latest_frame_id'], 'front:19/rear:19')
                self.assertEqual(report['environment']['latest_frame_id'], report['latest_frame_id'])
                self.assertEqual(report['latest_frame_analysis_status'], 'completed')
                self.assertEqual(report['latest_frame_sec'], 1.9)
                self.assertEqual(len(report['notifications']), 1)
                self.assertEqual(len(requests), 5)  # Four chunks plus environment comparison.
                self.assertEqual(v4.main(), 0)
                resumed = json.loads((output / 'final_report.json').read_text())
                self.assertEqual(resumed, report)
                self.assertEqual(len(requests), 6)  # Only comparison is repeated on resume.
            cap = v4.open_video(video)
            try:
                metadata = []
                v4.read_temporal_frames(cap, 0, 0., 10., 20., 3, 1., metadata)
                self.assertEqual([f['frame_index'] for f in metadata], [0, 0, 1])
                self.assertEqual([f['sec'] for f in metadata], [0., 0., .1])
            finally:
                cap.release()


if __name__ == '__main__':
    unittest.main()
