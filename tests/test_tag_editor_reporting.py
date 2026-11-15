import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import tag_editor


class TagEditorReportingTests(unittest.TestCase):
    def test_batch_reports_changed_unchanged_missing_and_failed_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temp = root / 'dataset' / '_temp'
            temp.mkdir(parents=True)
            (root / 'database').mkdir()
            (root / 'prompt.txt').write_text('trigger\n', encoding='utf8')
            for name in ['changed', 'unchanged', 'missing', 'failed']:
                (temp / f'{name}.png').write_bytes(b'image')
            (temp / 'changed.txt').write_text('base', encoding='utf8')
            (temp / 'unchanged.txt').write_text('base, marker', encoding='utf8')
            (temp / 'failed.txt').write_text('base', encoding='utf8')
            read_text = tag_editor.read_text_best_effort

            def read(path):
                if path.name == 'failed.txt':
                    raise OSError('unreadable fixture')
                return read_text(path)

            with patch('services.tag_editor.read_text_best_effort', side_effect=read):
                _, log, result = tag_editor.handle({
                    'folder': str(root), 'mode': 'insert', 'tags': 'marker',
                    'exts': '.png', 'backup': '1',
                }, {})
            self.assertFalse(result['ok'])
            self.assertEqual(result['counts'], {
                'scanned': 4, 'changed': 1, 'created': 0, 'skipped': 1,
                'errors': 1, 'unchanged': 1,
            })
            self.assertEqual(result['artifacts']['backups_created'], 1)
            self.assertEqual(result['output_path'], str(temp))
            self.assertIn('unreadable fixture', log)
            self.assertEqual((temp / 'changed.txt.bak').read_text(), 'base')
            self.assertFalse((temp / 'unchanged.txt.bak').exists())


if __name__ == '__main__':
    unittest.main()
