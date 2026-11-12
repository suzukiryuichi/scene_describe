"""Offline regression tests: no VLM server or real driving footage is required."""
import json
import io
from contextlib import redirect_stdout
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import video_scene_describer5 as v5
import numpy as np


def observation(**changes):
    value = dict(event_id='o1', object_type='bicycle', direction='後方', action='stationary',
                 temporal_status='stationary', first_seen=9., last_seen=10.,
                 evidence_frames=[9., 10.], latest_visible=True, position='中央',
                 evidence='後方中央で自転車が停止', notification_reason='none',
                 confirmation_action='', category='traffic', observed_behavior='none',
                 change_type='none', blocks_path=False, observation_certainty='observed')
    value.update(changes)
    return value


def scene(observations, latest=10., **changes):
    return v5.DualVideoSceneEntry(timestamp='video 00:10', elapsed_sec=0, image_file='',
        description=json.dumps(dict(observations=observations, notifications=[], latest_frame_sec=latest),
                               ensure_ascii=False), **changes)


class NotificationTests(unittest.TestCase):
    def test_01_past_bicycle_is_not_current(self):
        for status in ('passing', 'departed', 'unknown'):
            report = v5.build_final_report([scene([observation(
                temporal_status=status, latest_visible=False)], latest=40.)])
            self.assertEqual(report['risks'], [])
            self.assertEqual(len(report['events']), 1)
            self.assertFalse(report['events'][0]['current'])
            self.assertTrue(report['notification_candidates'][0]['exclusion_reason'])

    def test_02_pedestrian_remains_behind(self):
        report = v5.build_final_report([scene([observation(object_type='pedestrian')])],
                                      notification_style='object_only')
        self.assertEqual(report['risks'][0]['message'], '後方に歩行者がいます')
        self.assertEqual(report['events'][0]['current_presence'], 'confirmed_current')
        self.assertEqual(report['events'][0]['first_seen'], 9.)
        self.assertTrue(report['events'][0]['state_history'])

    def test_03_new_parked_vehicle(self):
        report = v5.build_final_report([scene([observation(object_type='vehicle',
            object_label='停車車両', change_type='appeared')])], notification_style='object_only')
        self.assertEqual(report['risks'][0]['message'], '後方に停車車両があります')
        self.assertEqual(report['notification_candidates'][0]['features']['important_change'], 1.)
        self.assertEqual(report['events'][0]['state_changes'][0]['change_type'], 'appeared')

    def test_04_front_blockage_outranks_rear_presence(self):
        events = [observation(event_id='rear', object_type='pedestrian'),
                  observation(event_id='front', direction='前方', object_type='vehicle',
                              blocks_path=True, notification_reason='blocked_path')]
        report = v5.build_final_report([scene(events)], notification_style='situation')
        self.assertEqual(report['risks'][0]['event_id'], 'scene_0_front')
        self.assertEqual(report['risks'][0]['message'], '前方の道路を車両がふさいでいます')

    def test_05_limits_and_decision_trail(self):
        events = [observation(event_id=str(i), object_type=kind, direction=direction,
                              temporal_status='approaching', action='approaching',
                              notification_reason='near_movement')
                  for i, (kind, direction) in enumerate([
                      ('pedestrian', '前方'), ('bicycle', '後方'), ('motorcycle', '前方'),
                      ('vehicle', '後方'), ('bicycle', '前方')])]
        report = v5.build_final_report([scene(events)])
        self.assertEqual(len(report['risks']), 3)
        self.assertEqual(len(report['notification_candidates']), 5)
        self.assertEqual(sum(c['selected'] for c in report['notification_candidates']), 3)
        self.assertEqual(sum(c['exclusion_reason'] == 'notification_limit'
                             for c in report['notification_candidates']), 2)
        # Ordinary presence alone stays within the preferred count.
        for event in events:
            event.update(temporal_status='unknown', notification_reason='none')
        self.assertEqual(len(v5.build_final_report([scene(events)])['risks']), 2)

    def test_06_no_objects(self):
        report = v5.build_final_report([scene([])])
        for key in ('observations', 'events', 'risks', 'notification_candidates'):
            self.assertEqual(report[key], [])

    def test_07_overlap_dedup_and_camera_identity(self):
        event = observation()
        report = v5.build_final_report([scene([event]), scene([event])])
        self.assertEqual(len(report['observations']), 2)
        self.assertEqual(len(report['events']), 1)
        self.assertEqual(len(report['risks']), 1)
        self.assertEqual(len(report['risks'][0]['related_event_ids']), 2)
        # Same local ID/name across cameras is not evidence of a single object.
        report = v5.build_final_report([scene([event, observation(event_id='front', direction='前方')])])
        self.assertEqual(len(report['events']), 2)
        self.assertEqual({r['direction'] for r in report['risks']}, {'前方', '後方'})
        # Indistinguishable subjects retain separate internal events but share a presence message.
        report = v5.build_final_report([scene([event, observation(event_id='other')])])
        self.assertEqual(len(report['events']), 2)
        self.assertEqual(len(report['risks']), 1)
        self.assertEqual(report['notification_candidates'][1]['exclusion_reason'], 'duplicate_message')

    def test_08_sparse_motion_is_unknown(self):
        report = v5.build_final_report([scene([observation(first_seen=0., evidence_frames=[0., 10.],
            temporal_status='approaching', action='approaching', notification_reason='near_movement')])],
            notification_style='situation')
        self.assertEqual(report['events'][0]['temporal_status'], 'unknown')
        self.assertIn('確認されています', report['risks'][0]['message'])
        self.assertEqual(report['observations'][0]['temporal_status'], 'approaching')
        self.assertTrue(report['events'][0]['state_changes'])

    def test_09_approach_is_objective_security_event(self):
        report = v5.build_final_report([scene([observation(object_type='pedestrian',
            observed_behavior='approaching_vehicle', evidence='人物が車両に近づいた')])])
        self.assertEqual(len(report['security_events']), 1)
        self.assertEqual(report['risks'], [])
        self.assertEqual(report['notification_candidates'][0]['exclusion_reason'], 'security_separate_category')
        for term in ('不審者', '盗難目的'):
            self.assertNotIn(term, json.dumps(report, ensure_ascii=False))

    def test_10_styles_hold_observations_and_selection_fixed(self):
        reports = [v5.build_final_report([scene([observation()])], notification_style=style)
                   for style in v5.NOTIFICATION_STYLES]
        for report in reports[1:]:
            for field in ('observations', 'events', 'notification_candidates'):
                self.assertEqual(report[field], reports[0][field])
            self.assertEqual(report['risks'][0]['event_id'], reports[0]['risks'][0]['event_id'])
        self.assertEqual(len({r['risks'][0]['message'] for r in reports}), 3)
        self.assertTrue(all(len(r['risks'][0]['message']) < 50 for r in reports))

    def test_historical_change_is_configurable_and_past_in_every_style(self):
        event = observation(latest_visible=False, temporal_status='passing', change_type='state_changed')
        config = v5.load_notification_config()
        config['notify_historical_changes'] = True
        for style in v5.NOTIFICATION_STYLES:
            report = v5.build_final_report([scene([event], latest=40.)], config=config, notification_style=style)
            self.assertIn('離席中、後方を自転車が通過しました', report['risks'][0]['message'])
            self.assertNotIn('自転車がいます', report['risks'][0]['message'])
        self.assertEqual(v5.build_final_report([scene([event], latest=80.)], config=config)['risks'], [])

    def test_old_collision_survives_departure_but_not_as_current_presence(self):
        event = observation(temporal_status='departed', latest_visible=False, observed_behavior='collision')
        for style in v5.NOTIFICATION_STYLES:
            report = v5.build_final_report([scene([event], latest=100.)], notification_style=style)
            self.assertIn('離席中、後方で接触', report['risks'][0]['message'])
            self.assertFalse(report['notification_candidates'][0]['current'])

    def test_uncertain_observation_cannot_be_promoted(self):
        for changes in (dict(observation_certainty='uncertain'), dict(evidence_frames=[], first_seen=None,
            last_seen=None), dict(evidence='')):
            self.assertEqual(v5.build_final_report([scene([observation(**changes)])])['risks'], [])

    def test_failure_of_latest_frame_never_promotes_older_image(self):
        failure = scene([], captured_frame_sec=12., captured_frame_id='front:120/rear:120')
        failure.description = '推論失敗: offline'
        report = v5.build_final_report([scene([observation()]), failure])
        self.assertEqual(report['risks'], [])
        self.assertEqual(report['latest_frame_analysis_status'], 'unavailable')

    def test_environment_preserves_appearance_without_inventing_blockage(self):
        environment = dict(status='completed', latest_sec=10., obstacles=[dict(direction='後方',
            object='停車車両', initial_state='absent', latest_visible=True, evidence='車両が出現', position='中央')])
        report = v5.build_final_report([], environment=environment, notification_style='situation')
        self.assertEqual(report['events'][0]['change_type'], 'appeared')
        self.assertNotIn('ふさいで', report['risks'][0]['message'])
        environment['latest_sec'] = 40.
        report = v5.build_final_report([scene([observation()])], environment=environment)
        self.assertTrue(all(r['event_id'].startswith('environment') for r in report['risks']))

    def test_environment_named_objects_keep_types_in_all_styles(self):
        environment = dict(status='completed', latest_sec=10., obstacles=[
            dict(direction='前方', object='黒色SUV', initial_state='absent', latest_visible=True,
                 blocks_path=True, evidence='道路上に車両が出現', position='中央'),
            dict(direction='前方', object='歩行者', initial_state='absent', latest_visible=True,
                 blocks_path=False, evidence='歩行者が出現', position='左側')])
        report = v5.build_final_report([], environment=environment, notification_style='object_only')
        self.assertEqual([n['message'] for n in report['notifications']],
                         ['前方に車両があります', '前方に歩行者がいます'])
        for style in v5.NOTIFICATION_STYLES:
            messages = [n['variants'][style] for n in report['notifications']]
            self.assertEqual(len(messages), len(set(messages)))
        self.assertEqual(v5.environment_object_type(dict(object='原動機付自転車')), 'motorcycle')
        self.assertEqual(v5.environment_object_type(dict(object='歩行者用信号')), 'other')
        with self.assertRaises(ValueError):
            v5.environment_object_type(dict(object='人物', object_type='suspicious'))

    def test_dedup_considers_all_styles_without_changing_selection(self):
        events = [observation(event_id='moving', temporal_status='passing',
                              notification_reason='near_movement'),
                  observation(event_id='stopped')]
        reports = [v5.build_final_report([scene(events)], notification_style=style)
                   for style in v5.NOTIFICATION_STYLES]
        for report in reports:
            self.assertEqual(len(report['notifications']), 1)
            self.assertEqual(report['notifications'][0]['event_id'], 'scene_0_moving')
            self.assertEqual(report['notification_candidates'][1]['exclusion_reason'], 'duplicate_message')

    def test_security_does_not_displace_traffic(self):
        events = [observation(event_id=str(i), object_type='pedestrian', observed_behavior='touching_vehicle')
                  for i in range(5)] + [observation(event_id='traffic')]
        report = v5.build_final_report([scene(events)])
        self.assertEqual(len(report['security_events']), 5)
        self.assertEqual(report['risks'][0]['event_id'], 'scene_0_traffic')

    def test_config_changes_weights_limits_and_motion_gap(self):
        config = v5.load_notification_config()
        events = [observation(), observation(event_id='front', direction='前方', object_type='vehicle', blocks_path=True)]
        config['weights']['rear_or_overlooked'] = 100.
        config.update(max_notifications=1, preferred_notifications=1)
        self.assertEqual(v5.build_final_report([scene(events)], config=config)['risks'][0]['direction'], '後方')
        config.update(max_notifications=0, preferred_notifications=0)
        self.assertEqual(v5.build_final_report([scene(events)], config=config)['risks'], [])
        config = v5.load_notification_config()
        config['max_motion_gap_sec'] = 20.
        event = observation(first_seen=0., evidence_frames=[0., 10.], temporal_status='approaching')
        self.assertEqual(v5.build_final_report([scene([event])], config=config)['events'][0]['temporal_status'], 'approaching')

    def test_invalid_config_and_schema(self):
        for key, value in [('max_notifications', 4), ('minimum_score', float('nan')),
                           ('preferred_notifications', True), ('weights', {}), ('unknown', 1)]:
            config = v5.load_notification_config()
            config[key] = value
            with self.assertRaises(ValueError):
                v5.build_final_report([], config=config)
        for changes in (dict(direction='左後方'), dict(observation_certainty='certain'),
                        dict(blocks_path='yes'), dict(object_type='alien'), dict(category='suspicious'),
                        dict(first_seen=11.), dict(evidence_frames=[float('nan')])):
            with self.assertRaises(ValueError):
                v5.format_dual_video_description(scene([observation(**changes)]).description)
        with self.assertRaises(ValueError):
            v5.format_dual_video_description('{broken')

    def test_motorcycle_is_distinct(self):
        report = v5.build_final_report([scene([observation(object_type='motorbike')])], notification_style='object_only')
        self.assertEqual(report['observations'][0]['object_type'], 'motorcycle')
        self.assertEqual(report['risks'][0]['message'], '後方にバイクがいます')

    def test_frame_validation_and_repeated_camera_image(self):
        timeline = [dict(sec=9., frame_id='front:90/rear:90', front_sec=9., rear_sec=9.),
                    dict(sec=10., frame_id='front:90/rear:100', front_sec=9., rear_sec=10.)]
        event = observation(direction='前方', temporal_status='approaching',
                            camera_evidence_times=[10.], evidence_frame_ids=['invented'])
        text = v5.validate_frame_evidence(v5.format_dual_video_description(scene([event]).description),
                                         [9., 10.], timeline)
        entry = scene([])
        entry.description = text
        report = v5.build_final_report([entry])
        self.assertEqual(report['events'][0]['temporal_status'], 'unknown')
        self.assertEqual(report['events'][0]['age_sec'], 1.)
        config = v5.load_notification_config()
        config['current_max_age_sec']['bicycle'] = .5
        self.assertEqual(v5.build_final_report([entry], config=config)['risks'], [])
        with self.assertRaises(ValueError):
            v5.validate_frame_evidence(text, [8., 10.])

    def test_report_outputs_and_legacy_records(self):
        legacy = scene([])
        legacy.description = '【前方】対象の確認\n【後方】対象の確認'
        self.assertEqual(v5.build_final_report([legacy])['risks'], [])
        with tempfile.TemporaryDirectory() as directory:
            text = v5.write_final_report(Path(directory), [scene([observation()])], interrupted=True)
            self.assertIn('直接確認', text)
            for filename in ('final_report.json', 'final_report.txt', 'final_report.html', 'observations.json'):
                self.assertTrue((Path(directory) / filename).exists())
            data = json.loads((Path(directory) / 'final_report.json').read_text())
            self.assertEqual(data['risks'], data['notifications'])
            self.assertEqual(data['metadata']['max_notifications'], 3)
            self.assertEqual(data['metadata']['config'], v5.load_notification_config())

    def test_external_context_does_not_invent_objects(self):
        class Provider(v5.ContextProvider):
            def facts_at(self, video_sec):
                return [v5.ContextFact('location', '学校周辺', 'test-map', video_sec)]
        report = v5.build_final_report([scene([])], context_provider=Provider())
        self.assertEqual(report['risks'], [])
        self.assertEqual(report['events'], [])
        self.assertEqual(report['metadata']['external_context'][0]['source'], 'test-map')

    def test_schema_failure_is_retried_with_correction(self):
        valid = scene([observation(object_type='motorcycle')]).description
        responses = [v5.ApiJsonResult('{"observations":[{}]}'), v5.ApiJsonResult(valid)]
        with patch.object(v5, 'request_dual_video_json', side_effect=responses) as request:
            _, description = v5.request_observation_json('unused', [], None, 2048, 0., (1., 1.),
                                                         True, frame_times=[9., 10.])
        self.assertEqual(request.call_count, 2)
        self.assertEqual(json.loads(description)['observations'][0]['object_type'], 'motorcycle')
        self.assertIn('検証エラー', request.call_args.args[1][-1]['content'])

    def test_cli_video_to_reports_and_resume_with_mock_vlm(self):
        def response(base_url, messages, *args, **kwargs):
            if kwargs.get('request_context') == 'environment_comparison':
                return v5.ApiJsonResult('{"obstacles":[]}')
            times = [float(match.group(1)) for message in messages
                     if isinstance(message.get('content'), list)
                     for part in message['content'] if part.get('type') == 'text'
                     for match in [re.search(r'動画開始から ([0-9.]+) 秒', part['text'])] if match]
            event = observation(first_seen=min(times), last_seen=max(times), evidence_frames=times)
            return v5.ApiJsonResult(json.dumps(dict(observations=[event], notifications=[],
                                                    latest_frame_sec=max(times)), ensure_ascii=False))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'camera.avi'
            writer = v5.cv2.VideoWriter(str(video), v5.cv2.VideoWriter_fourcc(*'MJPG'), 10., (96, 64))
            self.assertTrue(writer.isOpened())
            try:
                for index in range(20):
                    writer.write(np.full((64, 96, 3), index * 8, dtype=np.uint8))
            finally:
                writer.release()
            argv = ['video_scene_describer5.py', '--front-video', str(video), '--back-video', str(video),
                    '--output-dir', str(root / 'reports'), '--max-samples', '1', '--sample-fps', '2',
                    '--temporal-frames', '3', '--chunk-frames', '2', '--chunk-overlap', '1',
                    '--notification-style', 'object_only', '--location-info', '学校周辺', '--log-level', 'quiet']
            with patch('sys.argv', argv), patch.object(v5, 'check_server'), patch.object(
                    v5, 'request_dual_video_json', side_effect=response) as request:
                self.assertEqual(v5.main(), 0)
                first_calls = request.call_count
                self.assertGreater(first_calls, 1)
                report = json.loads((root / 'reports/final_report.json').read_text())
                self.assertEqual(report['risks'][0]['message'], '後方に自転車がいます')
                self.assertEqual(report['latest_frame_analysis_status'], 'completed')
                self.assertEqual(report['environment']['status'], 'completed')
                self.assertEqual(report['metadata']['external_context'][0]['value'], '学校周辺')
                self.assertEqual(v5.main(), 0)
                # Only the environment comparison runs again; scene/chunk results resume.
                self.assertEqual(request.call_count, first_calls + 1)
                resumed = json.loads((root / 'reports/final_report.json').read_text())
                for field in ('observations', 'events', 'notification_candidates', 'risks'):
                    self.assertEqual(report[field], resumed[field])
                for filename in ('final_report.txt', 'final_report.html', 'observations.json', 'index.html'):
                    self.assertTrue((root / 'reports' / filename).exists())


class CleanupCompatibilityTests(unittest.TestCase):
    def test_prompt_example_and_retry_agree_on_observation_output(self):
        prompt = v5.DUAL_VIDEO_SYSTEM_PROMPT_CONTEXT
        example = json.loads(v5.extract_json_object(prompt))
        self.assertEqual(v5.validate_observations(example)['notifications'], [])
        self.assertNotIn('confirmation_action', example['observations'][0])
        self.assertIn('notifications は空配列', prompt)
        self.assertIn('notifications は空配列', v5.OBSERVATION_REPAIR_INSTRUCTIONS)

    def test_chunk_request_records_observations_without_notification_selection(self):
        valid = scene([observation()]).description
        with patch.object(v5, 'request_dual_video_json', return_value=v5.ApiJsonResult(valid)) as request:
            result = v5.analyze_dual_video_chunk(base_url='unused', chunk_image_b64=['a','b'],
                image_format='jpeg', system_prompt=v5.DUAL_VIDEO_SYSTEM_PROMPT_CONTEXT,
                user_prompt=v5.DUAL_VIDEO_USER_PROMPT_CONTEXT, chunk_index=1, chunk_count=2,
                start=0, end=2, max_tokens=2048, temperature=0., timeout=(1.,1.),
                retry_on_length=True, api_max_retries=1, scene_id='s', video_label='test',
                request_context='test', fallback_name_value='normal', intermediate_path=None,
                running_state={'count':0,'lock':v5.threading.Lock()}, scene_position_sec=9.,
                sample_fps=1., frame_times=[9.,10.])
        self.assertEqual(result.error, '')
        text = request.call_args.args[1][1]['content'][-1]['text']
        self.assertIn('件数制限なく観測JSONに記録', text)
        self.assertNotIn('確認できる注意点だけ', text)
        self.assertEqual(json.loads(result.description)['observations'][0]['object_type'], 'bicycle')

    def test_legacy_vlm_merge_option_uses_local_merge_even_for_text_logs(self):
        for descriptions in ([scene([observation()]).description]*2,
                             ['【前方】荷物があります','【後方】コーンがあります']):
            with self.subTest(structured=descriptions[0].startswith('{')):
                completed = {('s','01-01_01'):{'description':descriptions[0]},
                             ('s','02-02_02'):{'description':descriptions[1]}}
                with patch.object(v5, 'request_dual_video_json', side_effect=AssertionError('unexpected VLM call')):
                    result = v5.describe_dual_video_scene('unused', ['a','b'], None, 2048, 0., (1.,1.),
                        'caution', '', True, 50, scene_id='s', chunk_frames=1, chunk_overlap=0,
                        chunk_merge_mode='vlm', completed_chunks=completed, frame_times=[9.,10.])
                if descriptions[0].startswith('{'):
                    self.assertEqual(len(json.loads(result)['observations']), 2)
                else:
                    self.assertIn('荷物',result)
                    self.assertIn('コーン',result)

    def test_old_observation_fields_and_notification_refs_remain_readable(self):
        event = observation(confirmation_action='旧形式の確認行動')
        data = dict(observations=[event], notifications=[{'event_id':event['event_id']}], latest_frame_sec=10.)
        parsed = json.loads(v5.format_dual_video_description(json.dumps(data)))
        self.assertEqual(parsed['observations'][0]['confirmation_action'], '旧形式の確認行動')
        entry = scene([])
        entry.description = json.dumps(parsed)
        report = v5.build_final_report([entry], notification_style='object_only')
        self.assertEqual(report['notifications'][0]['message'], '後方に自転車がいます')
        self.assertNotIn('旧形式',report['notifications'][0]['message'])

    def test_resume_ignores_partial_and_non_object_json_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'intermediate.jsonl'
            records = ['null','[]','42','"text"','{"partial":',
                       json.dumps({'type':'scene_complete','scene_id':'s','description':'saved'}),
                       json.dumps({'type':'chunk_complete','scene_id':'s','chunk_id':'c','description':'saved'}),
                       json.dumps({'type':'scene_complete','scene_id':'bad','description':123})]
            path.write_text('\n'.join(records))
            self.assertEqual(set(v5.load_completed_scene_results(path)), {'s'})
            self.assertEqual(set(v5.load_completed_chunk_results(path)), {('s','c')})
            self.assertEqual(v5.load_completed_scene_results(Path(directory)/'missing'), {})
        for data in (None, [], 1, 'text'):
            with self.assertRaises(ValueError):
                v5.validate_observations(data)

    def test_render_cli_preserves_observations_selection_and_source_without_server(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'saved.json'
            report = v5.build_final_report([scene([observation()])],notification_style='object_only')
            source.write_text(json.dumps(report,ensure_ascii=False))
            original = source.read_bytes()
            for style in v5.NOTIFICATION_STYLES:
                output = root/style
                argv = ['video_scene_describer5.py','--render-report',str(source),
                        '--notification-style',style,'--output-dir',str(output)]
                with patch('sys.argv',argv), redirect_stdout(io.StringIO()), patch.object(
                        v5,'check_server',side_effect=AssertionError('unexpected server access')), patch.object(
                        v5,'build_final_report',side_effect=AssertionError('unexpected re-selection')):
                    self.assertEqual(v5.main(),0)
                rendered = json.loads((output/'final_report.json').read_text())
                for key in ('observations','events','notification_candidates','security_events','environment'):
                    self.assertEqual(rendered[key],report[key])
                self.assertEqual(rendered['notifications'][0]['event_id'],report['notifications'][0]['event_id'])
                self.assertEqual(rendered['notifications'][0]['message'],report['notifications'][0]['variants'][style])
                self.assertEqual(rendered['risks'],rendered['notifications'])
                self.assertEqual(rendered['metadata']['notification_style'],style)
                for name in ('final_report.txt','final_report.html','observations.json'):
                    self.assertTrue((output/name).exists())
            self.assertEqual(source.read_bytes(),original)

    def test_render_rejects_missing_variants_and_keeps_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'final_report.json'
            report = v5.build_final_report([scene([observation()])])
            source.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                v5.render_saved_report(source,root,'situation')
            for bad in ('missing_variants','missing_style','bad_limit','empty_message'):
                broken = json.loads(json.dumps(report))
                if bad == 'missing_variants':
                    broken['notifications'][0].pop('variants')
                elif bad == 'missing_style':
                    broken['notifications'][0]['variants'].pop('situation')
                elif bad == 'bad_limit':
                    broken['metadata']['max_notifications'] = 0
                else:
                    broken['notifications'][0]['variants']['situation'] = ''
                source.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    v5.render_saved_report(source,root/bad,'situation')
                self.assertFalse((root/bad/'final_report.json').exists())

    def test_render_empty_notifications_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.json'
            source.write_text(json.dumps(v5.build_final_report([])))
            text = v5.render_saved_report(source,Path(directory)/'output','situation')
            self.assertIn('通知すべき情報はありません',text)

    def test_retry_exhaustion_reports_validation_error(self):
        with patch.object(v5,'request_dual_video_json',return_value=v5.ApiJsonResult('{"observations":[{}]}')) as request:
            with self.assertRaisesRegex(ValueError,'3回失敗'):
                v5.request_observation_json('unused',[],None,2048,0.,(1.,1.),True,frame_times=[9.,10.])
        self.assertEqual(request.call_count,3)

    def test_prompt_profile_changes_for_chunk_or_repair_edits_only(self):
        with patch('sys.argv',['test','--front-video','front.mp4','--back-video','rear.mp4']):
            args = v5.parse_args()
        original = v5.analysis_profile(args,768,72)
        args.notification_style='situation'
        self.assertEqual(v5.analysis_profile(args,768,72),original)
        for key in ('CHUNK_OBSERVATION_PROMPT','OBSERVATION_REPAIR_INSTRUCTIONS'):
            with patch.object(v5,key,getattr(v5,key)+' changed'):
                self.assertNotEqual(v5.analysis_profile(args,768,72),original)


if __name__ == '__main__':
    unittest.main()
