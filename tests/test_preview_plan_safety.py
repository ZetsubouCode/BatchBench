import json
import tempfile
import unittest
from pathlib import Path

from services import normalizer, offline_tagger


class PreviewPlanSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()
        (self.dataset / 'image.png').write_bytes(b'png fixture')
        (self.dataset / 'image.txt').write_text('old_tag\n', encoding='utf-8')
        self.presets = self.root / 'presets' / 'anime'
        self.presets.mkdir(parents=True)
        (self.presets / 'rules.json').write_text(json.dumps({'rules': {'replace_map': {'old_tag': 'new_tag'}}}), encoding='utf-8')
        self.opts = normalizer.NormalizeOptions(dataset_path=self.dataset, preset_type='anime', preset_file='rules.json', image_exts=['.png'], backup_enabled=False)

    def test_apply_requires_a_preview_and_is_one_time(self):
        self.assertFalse(normalizer.apply_preview_plan(self.opts, self.presets.parent, '')['ok'])
        preview = normalizer.preview_plan(self.opts, self.presets.parent)
        self.assertTrue(preview['ok'])
        self.assertEqual(preview['changed_files'], 1)
        token = preview['plan_token']
        self.assertTrue(normalizer.apply_preview_plan(self.opts, self.presets.parent, token)['ok'])
        self.assertFalse(normalizer.apply_preview_plan(self.opts, self.presets.parent, token)['ok'])

    def test_option_and_caption_changes_reject_stale_plan(self):
        token = normalizer.preview_plan(self.opts, self.presets.parent)['plan_token']
        changed_opts = normalizer.NormalizeOptions(**{**self.opts.__dict__, 'normalize_order': False})
        self.assertFalse(normalizer.apply_preview_plan(changed_opts, self.presets.parent, token)['ok'])
        self.assertEqual((self.dataset / 'image.txt').read_text(encoding='utf-8'), 'old_tag\n')
        token = normalizer.preview_plan(self.opts, self.presets.parent)['plan_token']
        (self.dataset / 'image.txt').write_text('external edit\n', encoding='utf-8')
        self.assertFalse(normalizer.apply_preview_plan(self.opts, self.presets.parent, token)['ok'])
        self.assertEqual((self.dataset / 'image.txt').read_text(encoding='utf-8'), 'external edit\n')

    def test_reviewed_tag_proposal_checks_caption_state_and_keeps_backup(self):
        caption = self.dataset / 'image.txt'
        image = self.dataset / 'image.png'
        stat = image.stat()
        item = {
            'dataset': str(self.dataset), 'caption': str(caption), 'image': str(image),
            'text': 'new tag\n', 'changed': True, 'backup_existing': True,
            'old_hash': offline_tagger.hashlib.sha256(caption.read_bytes()).hexdigest(),
            'image_state': (stat.st_size, stat.st_mtime_ns),
        }
        token = 'test-proposal-stale'
        offline_tagger._PROPOSAL_PLANS[token] = (offline_tagger.time.monotonic(), [item])
        caption.write_text('external edit', encoding='utf-8')
        self.assertFalse(offline_tagger.apply_reviewed_proposals(token, [0])['ok'])
        self.assertEqual(caption.read_text(encoding='utf-8'), 'external edit')
        item['old_hash'] = offline_tagger.hashlib.sha256(caption.read_bytes()).hexdigest()
        token = 'test-proposal-apply'
        offline_tagger._PROPOSAL_PLANS[token] = (offline_tagger.time.monotonic(), [item])
        result = offline_tagger.apply_reviewed_proposals(token, [0])
        self.assertTrue(result['ok'])
        self.assertEqual(caption.read_text(encoding='utf-8'), 'new tag\n')
        self.assertTrue(list((self.dataset / '.batchbench_backup').rglob('image.txt')))


if __name__ == '__main__':
    unittest.main()
