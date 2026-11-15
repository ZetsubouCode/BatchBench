import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import tag_editor


class TaggingCaptionSaveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'dataset' / '_temp').mkdir(parents=True)
        (self.root / 'database').mkdir()
        (self.root / 'prompt.txt').write_text('trigger\n', encoding='utf-8')
        (self.root / 'dataset' / 'sample.png').write_bytes(b'image')
        self.caption = self.root / 'dataset' / 'sample.txt'
        self.caption.write_text('original tag', encoding='utf-8')
        self.session = tag_editor.start_tagging_session(
            self.root, ['.png'],
            settings={'segments': [{'id': 'appearance', 'label': 'Appearance', 'order': 1}]},
            mapping_rows=[{'left_sections': [], 'right_segment': 'appearance'}],
            recommendations={'appearance': []}, recommendation_sources={},
        )['session']

    def save(self):
        return tag_editor.save_tagging_quiz_image(
            self.root, 'dataset/sample.png',
            {'appearance': {'selected': ['long_hair', 'long hair'], 'manual': ['blue_eyes']}},
            session_payload=self.session, backup=True,
        )

    def test_caption_write_reports_both_persistence_results(self):
        result = self.save()
        self.assertTrue(result['ok'])
        self.assertTrue(result['session_saved'])
        self.assertEqual(self.caption.read_text(encoding='utf-8'), 'long hair, blue eyes')
        self.assertEqual(self.caption.with_suffix('.txt.bak').read_text(encoding='utf-8'), 'original tag')

    def test_caption_success_with_failed_draft_is_explicit(self):
        for outcome in [OSError('disk full'), {'ok': False, 'error': 'session refused'}]:
            with self.subTest(outcome=outcome):
                kwargs = {'side_effect': outcome} if isinstance(outcome, Exception) else {'return_value': outcome}
                with patch('services.tag_editor.save_tagging_session', **kwargs):
                    result = self.save()
                self.assertTrue(result['ok'], 'caption was actually written')
                self.assertFalse(result['session_saved'])
                self.assertTrue(result['session_error'])
                self.assertIn('[WARN]', '\n'.join(result['logs']))
                self.assertEqual(self.caption.read_text(encoding='utf-8'), 'long hair, blue eyes')

    def test_failed_caption_backup_does_not_claim_success(self):
        with patch.object(Path, 'write_text', side_effect=OSError('read only')):
            result = self.save()
        self.assertFalse(result['ok'])
        self.assertIn('back up', result['error'])
        self.assertEqual(self.caption.read_text(encoding='utf-8'), 'original tag')

    def test_other_project_session_is_rejected_before_caption_or_backup_write(self):
        self.session['project_root'] = str(self.root / 'another-project')
        result = self.save()
        self.assertFalse(result['ok'])
        self.assertIn('different project', result['error'])
        self.assertEqual(self.caption.read_text(encoding='utf-8'), 'original tag')
        self.assertFalse(self.caption.with_suffix('.txt.bak').exists())


if __name__ == '__main__':
    unittest.main()
