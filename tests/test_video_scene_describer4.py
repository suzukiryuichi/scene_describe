import json
import tempfile
import unittest
from pathlib import Path
from video_scene_describer4 import (
    ChunkAnalysisResult, DualVideoSceneEntry, build_final_report, write_final_report,
    merge_chunk_descriptions_locally, format_dual_video_description, validate_frame_evidence,
    build_timed_image_content,
)


def event(**changes):
    result = dict(event_id='e1', object_type='bicycle', direction='後方', action='approaching',
                  temporal_status='approaching', first_seen=12., last_seen=18.,
                  evidence_frames=[12., 18.], latest_visible=True, position='',
                  evidence='後方の車両直近に自転車の接近を確認', notification_reason='near_movement',
                  confirmation_action='発進前に後方をご確認ください。')
    result.update(changes)
    return result


def entry(events, latest=18.):
    return DualVideoSceneEntry(timestamp='video 00:18', elapsed_sec=0, image_file='',
        description=json.dumps(dict(observations=events, notifications=[], latest_frame_sec=latest)))


class FeedbackTests(unittest.TestCase):
    def test_approaching(self):
        report = build_final_report([entry([event()])])
        self.assertEqual(len(report['notifications']), 1)
        self.assertIn('接近しています', report['notifications'][0]['message'])
        self.assertEqual(report['risks'], report['notifications'])

    def test_departed_and_stale_are_not_current(self):
        for e in (event(temporal_status='departed'), event(latest_visible=False),
                  event(last_seen=12., evidence_frames=[12.])):
            self.assertEqual(build_final_report([entry([e])])['notifications'], [])
        self.assertEqual(build_final_report([entry([event()]), entry([], latest=30.)])['notifications'], [])

    def test_stationary_and_unknown(self):
        e = event(temporal_status='stationary', action='stationary', notification_reason='blocked_path')
        self.assertIn('滞留', build_final_report([entry([e])])['notifications'][0]['message'])
        e = event(temporal_status='unknown', latest_visible=False, notification_reason='none')
        self.assertEqual(build_final_report([entry([e])])['notifications'], [])

    def test_multiple_events_priority_and_unlimited_observations(self):
        reasons = ['change', 'contact', 'boarding_or_stop', 'blocked_path', 'near_movement', 'none']
        events = [event(event_id=f'e{i}', notification_reason=r) for i, r in enumerate(reasons)]
        report = build_final_report([entry(events)])
        self.assertEqual(len(report['observations']), 6)
        self.assertEqual([n['event_id'] for n in report['notifications']], ['scene_0_e4', 'scene_0_e3', 'scene_0_e2'])

    def test_historical_contact(self):
        e = event(notification_reason='contact', temporal_status='departed', latest_visible=False)
        n = build_final_report([entry([e], latest=30.)])['notifications'][0]
        self.assertIn('離席中', n['message'])
        self.assertNotIn('接近しています', n['message'])

    def test_styles_fix_events_and_selection(self):
        reports = [build_final_report([entry([event()])], notification_style=style)
                   for style in ('object_only', 'situation', 'action_oriented')]
        self.assertTrue(all(r['observations'] == reports[0]['observations'] for r in reports))
        self.assertEqual(len({r['notifications'][0]['event_id'] for r in reports}), 1)
        self.assertEqual(len({r['notifications'][0]['message'] for r in reports}), 3)

    def test_chunk_merge_keeps_events_without_identity_assumptions(self):
        chunks = [ChunkAnalysisResult(i, str(i), entry([event()]).description) for i in range(4)]
        merged = json.loads(merge_chunk_descriptions_locally(chunks))
        self.assertEqual(len(merged['observations']), 4)
        self.assertEqual(len({e['event_id'] for e in merged['observations']}), 4)

    def test_validation_and_exact_frame_times(self):
        description = format_dual_video_description(entry([event()]).description)
        self.assertEqual(json.loads(validate_frame_evidence(description, [12., 18.]))['latest_frame_sec'], 18.)
        with self.assertRaises(ValueError):
            validate_frame_evidence(description, [12., 15.])
        for changes in (dict(first_seen=20.), dict(direction='左後方'),
                        dict(temporal_status='invented'), dict(evidence_frames=[float('nan')])):
            with self.assertRaises(ValueError):
                format_dual_video_description(entry([event(**changes)]).description)
        content = build_timed_image_content(['abc', 'def'], 'jpeg', [12., 18.])
        self.assertIn('18.0', content[2]['text'])

    def test_empty_legacy_and_report_files(self):
        legacy = entry([])
        legacy.description = '【前方】同一人物が車両方向を継続して見ているため、注意'
        self.assertEqual(build_final_report([legacy])['notifications'], [])
        with tempfile.TemporaryDirectory() as directory:
            text = write_final_report(Path(directory), [], interrupted=True)
            self.assertIn('障害物比較は未完了', text)
            self.assertIn('通知すべき情報はありません', text)
            for name in ('final_report.json', 'final_report.txt', 'final_report.html', 'observations.json'):
                self.assertTrue((Path(directory) / name).exists())

    def test_environment_newer_than_traffic(self):
        environment = dict(status='completed', latest_sec=30., obstacles=[dict(direction='前方',
            object='荷物', initial_state='unknown', latest_visible=True, evidence='経路を塞ぐ荷物')])
        report = build_final_report([entry([event()])], environment=environment)
        self.assertEqual(len(report['notifications']), 2)
        self.assertEqual(report['notifications'][0]['event_id'], 'scene_0_e1')
        self.assertEqual(report['notifications'][1]['event_id'], 'environment_0')
        self.assertEqual(report['latest_frame_sec'], 18.)
        self.assertEqual(report['environment_latest_frame_sec'], 30.)


if __name__ == '__main__':
    unittest.main()
