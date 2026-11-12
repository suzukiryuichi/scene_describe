import json
import unittest
from unittest.mock import Mock, patch

import video_scene_describer4 as v4
from test_video_scene_describer4 import entry, event


def response(content, finish='stop'):
    result = Mock(status_code=200)
    result.json.return_value = dict(choices=[dict(message=dict(content=content), finish_reason=finish)],
                                   usage=dict(completion_tokens=100))
    return result


class ApiRecoveryTests(unittest.TestCase):
    def test_aliases_preserve_event_and_unknown_types_fail(self):
        for alias, expected in [('car', 'vehicle'), ('motorcycle', 'vehicle'),
                                ('person', 'pedestrian'), ('cyclist', 'bicycle')]:
            data = json.loads(v4.format_dual_video_description(entry([event(object_type=alias)]).description))
            self.assertEqual(data['observations'][0]['object_type'], expected)
            self.assertEqual(data['observations'][0]['evidence_frames'], [12., 18.])
        with self.assertRaises(ValueError):
            v4.format_dual_video_description(entry([event(object_type='unrecognized')]).description)

    def test_truncation_grows_budget_before_parsing(self):
        session = Mock()
        complete = entry([event(object_type='car')]).description
        session.post.side_effect = [response('{"observations":[', 'length'), response(complete)]
        result, description = v4.request_observation_json('http://test', [], session, 240, 0, (1, 1), True,
                                                         frame_times=[12., 18.])
        self.assertEqual([c.kwargs['json']['max_tokens'] for c in session.post.call_args_list], [240, 480])
        self.assertEqual(result.retry_count, 1)
        self.assertEqual(json.loads(description)['observations'][0]['object_type'], 'vehicle')

    def test_invalid_syntax_without_length_also_retries(self):
        session = Mock()
        session.post.side_effect = [response('{"observations":['), response(entry([]).description)]
        v4.request_dual_video_json('http://test', [], session, 2048, 0, (1, 1), True)
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(session.post.call_args.kwargs['json']['max_tokens'], 4096)

    def test_semantic_retry_contains_specific_error_and_previous_answer(self):
        bad = entry([event(direction='左後方')]).description
        session = Mock()
        session.post.side_effect = [response(bad), response(entry([event()]).description)]
        _, description = v4.request_observation_json('http://test', [], session, 2048, 0, (1, 1), True,
                                                     frame_times=[12., 18.])
        payload = session.post.call_args.kwargs['json']
        self.assertEqual(payload['max_tokens'], 4096)
        self.assertEqual(payload['messages'][-2]['content'], bad)
        self.assertIn('観測の方向が不正', payload['messages'][-1]['content'])
        self.assertEqual(json.loads(description)['observations'][0]['direction'], '後方')

    def test_exhausted_truncation_is_explicit_and_bounded(self):
        session = Mock()
        session.post.return_value = response('{', 'length')
        with self.assertRaisesRegex(ValueError, 'JSON出力が不完全'):
            v4.request_dual_video_json('http://test', [], session, 2048, 0, (1, 1), True)
        self.assertEqual(session.post.call_count, 3)

    def test_defaults_and_opt_out(self):
        with patch('sys.argv', ['v4', '--front-video', 'front', '--back-video', 'back']):
            args = v4.parse_args()
        self.assertEqual(args.max_tokens, 2048)
        self.assertTrue(args.retry_on_length)
        with patch('sys.argv', ['v4', '--front-video', 'front', '--back-video', 'back', '--no-retry-on-length']):
            self.assertFalse(v4.parse_args().retry_on_length)


if __name__ == '__main__':
    unittest.main()
