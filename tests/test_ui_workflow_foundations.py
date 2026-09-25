import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app import app
from services import offline_tagger, webp_converter


class UiWorkflowFoundationTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_home_is_small_and_heavy_tools_load_on_direct_link(self):
        home = self.client.get('/')
        self.assertEqual(home.status_code, 200)
        html = home.get_data(as_text=True)
        self.assertIn('data-lazy-tool="tags"', html)
        self.assertIn('data-lazy-tool="pipeline"', html)
        self.assertNotIn('cdn.jsdelivr.net', html)
        self.assertIn('data-bs-target="#tab-tags"', html)
        direct = self.client.get('/?tab=tags').get_data(as_text=True)
        self.assertIn('id="tags-form"', direct)
        self.assertNotIn('data-lazy-tool="tags"', direct)
        self.assertEqual(self.client.get('/tool-pane/settings').status_code, 200)

    def test_path_roles_do_not_share_a_root_attribute(self):
        html = self.client.get('/').get_data(as_text=True)
        self.assertNotIn('data-root-sync', html)
        self.assertIn('id="folder_tags" name="folder" type="text" class="form-control" data-path-role="project"', self.client.get('/?tab=tags').get_data(as_text=True))
        self.assertIn('id="src_png" name="src_png" type="text" class="form-control" placeholder="D:\\path\\to\\images" data-path-role="source"', html)
        self.assertIn('id="dst_png" name="dst_png" type="text" class="form-control" data-path-role="output"', html)

    def test_auto_tag_form_forces_proposals_without_touching_caption(self):
        image = self.root / 'image.png'
        caption = self.root / 'image.txt'
        image.write_bytes(b'image')
        caption.write_text('existing tag\n', encoding='utf-8')
        original = (caption.read_bytes(), caption.stat().st_mtime_ns)
        seen = {}

        def fake_run(opts, **kwargs):
            seen['preview_only'] = opts.preview_only
            return True, ['Preview mode made no filesystem changes.']

        with patch('services.offline_tagger.run_tagger', side_effect=fake_run):
            result = offline_tagger.handle({'folder':str(self.root), 'preview_only':'0', 'replace_existing_captions':'1'}, {})
        self.assertTrue(seen['preview_only'])
        self.assertTrue(result[2]['ok'])
        self.assertEqual((caption.read_bytes(), caption.stat().st_mtime_ns), original)

    def test_packaged_tagger_worker_returns_review_proposals(self):
        received = []
        proposal = {'image': 'image.png', 'caption': 'image.txt', 'text': 'tag'}
        def fake_worker(command, **kwargs):
            result_path = Path(command[command.index('--result') + 1])
            result_path.write_text(json.dumps({'ok': True, 'lines': ['Preview ready'], 'proposals': [proposal]}), encoding='utf-8')
            return SimpleNamespace(stdout='', stderr='', returncode=0)
        with patch.object(offline_tagger.sys, 'frozen', True, create=True), \
             patch('services.offline_tagger._candidate_source_roots', return_value=[self.root]), \
             patch('services.offline_tagger._external_worker_python', return_value=self.root / 'python.exe'), \
             patch('services.offline_tagger.subprocess.run', side_effect=fake_worker):
            result = offline_tagger._run_external_worker(SimpleNamespace(dataset_path=self.root), [], received.append)
        self.assertTrue(result[0])
        self.assertEqual(received, [proposal])

    def test_converter_result_success_partial_and_invalid_folder(self):
        source = self.root / 'source'; source.mkdir()
        output = self.root / 'output'
        Image.new('RGB', (2, 2), 'red').save(source / 'good.jpg')
        invalid = webp_converter.handle({'src_png':str(self.root / 'missing')}, {})[2]
        self.assertEqual(invalid['state'], 'failed')
        success = webp_converter.handle({'src_png':str(source), 'dst_png':str(output)}, {})[2]
        self.assertEqual(success['state'], 'completed')
        self.assertEqual(success['counts']['changed'], 1)
        original_output = (output / 'good.png').read_bytes()
        repeated = webp_converter.handle({'src_png':str(source), 'dst_png':str(output)}, {})[2]
        self.assertEqual(repeated['counts']['conflicts'], 1)
        self.assertEqual((output / 'good.png').read_bytes(), original_output)
        self.assertTrue((output / 'good_jpg.png').exists())
        (source / 'bad.png').write_bytes(b'not an image')
        partial = webp_converter.handle({'src_png':str(source), 'dst_png':str(output)}, {})[2]
        self.assertEqual(partial['state'], 'partial')
        self.assertEqual(partial['counts']['errors'], 1)
        self.assertTrue(partial['issues'])

    def test_dataset_health_reuses_pipeline_audit(self):
        dataset = self.root / 'dataset'; dataset.mkdir()
        (dataset / 'missing.png').write_bytes(b'fixture')
        response = self.client.post('/api/dataset-health/scan', json={'dataset_path':str(dataset)})
        self.assertEqual(response.status_code, 200)
        report = response.get_json()['report']
        self.assertEqual(report['summary']['missing_txt'], 1)
        self.assertIn('missing_captions', [item['code'] for item in report['errors']])

    def test_dataset_health_caption_edit_is_guarded(self):
        dataset = self.root / 'dataset'; dataset.mkdir()
        (dataset / 'space name.png').write_bytes(b'fixture')
        route = '/api/dataset-health/caption'
        body = {'dataset_path': str(dataset), 'relative_path': 'space name.png'}
        opened = self.client.post(route, json=body).get_json()
        self.assertFalse(opened['exists'])
        save = self.client.post(route + '/save', json={**body, 'content': 'long hair, blue eyes\n', 'fingerprint': None})
        self.assertEqual(save.status_code, 200)
        self.assertEqual((dataset / 'space name.txt').read_text(encoding='utf-8'), 'long hair, blue eyes\n')
        stale = self.client.post(route + '/save', json={**body, 'content': 'wrong', 'fingerprint': None})
        self.assertEqual(stale.status_code, 409)
        self.assertEqual((dataset / 'space name.txt').read_text(encoding='utf-8'), 'long hair, blue eyes\n')
        outside = self.client.post(route, json={'dataset_path': str(dataset), 'relative_path': '../escape.txt'})
        self.assertEqual(outside.status_code, 400)
        image = self.client.get('/api/dataset-health/image', query_string={'dataset_path': str(dataset), 'relative_path': 'space name.txt'})
        self.assertEqual(image.status_code, 200)
        rejected = self.client.get('/api/dataset-health/image', query_string={'dataset_path': str(dataset), 'relative_path': '../escape.png'})
        self.assertEqual(rejected.status_code, 400)

    def test_clip_scan_returns_caption_files_for_review(self):
        dataset = self.root / 'caption set'; dataset.mkdir()
        (dataset / 'long.png').write_bytes(b'image')
        (dataset / 'long.txt').write_text('long hair, blue eyes, white shirt, outdoor background', encoding='utf-8')
        response = self.client.post('/api/clip-tokens/scan', json={
            'folder_clip_tokens': str(dataset), 'exts_clip_tokens': '.png',
            'limit_clip_tokens': 4, 'mode_clip_tokens': 'estimate',
        })
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result['over_limit'], 1)
        self.assertEqual(result['offenders'][0]['file'], 'long.png')


if __name__ == '__main__':
    unittest.main()
