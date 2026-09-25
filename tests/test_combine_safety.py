import tempfile
import unittest
from pathlib import Path

from services import combine_datasets


class CombineDatasetSafetyTests(unittest.TestCase):
    def _make_pair(self, root: Path, stem: str):
        (root / f"{stem}.png").write_bytes(b"img")
        (root / f"{stem}.txt").write_text("tag\n", encoding="utf-8")

    def test_requires_output_folder(self):
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a"
            b = Path(td) / "b"
            a.mkdir(parents=True, exist_ok=True)
            b.mkdir(parents=True, exist_ok=True)
            self._make_pair(a, "one")
            self._make_pair(b, "two")

            _, _, meta = combine_datasets.handle(
                {
                    "source_folders": f"{a}\n{b}",
                    "out_dir": "",
                    "exts_combine": ".png",
                },
                {},
            )

            self.assertFalse(meta.get("ok"), msg=meta)
            self.assertIn("Output folder is required", meta.get("error", ""))

    def test_move_requires_current_preview_and_preserves_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); a = root / 'a'; b = root / 'b'; out = root / 'out'
            a.mkdir(); b.mkdir(); out.mkdir()
            self._make_pair(a, 'same'); self._make_pair(b, 'same')
            form = {'source_folders': f'{a}\n{b}', 'out_dir': str(out),
                    'exts_combine': '.png', 'move_instead': '1'}
            _, _, denied = combine_datasets.handle(form, {})
            self.assertFalse(denied['ok'])
            self.assertTrue((a / 'same.png').exists())
            preview = combine_datasets.preview_move(form)
            self.assertTrue(preview['ok'])
            self.assertEqual(preview['pairs'], 2)
            (a / 'same.txt').write_text('changed externally', encoding='utf-8')
            _, _, stale = combine_datasets.handle({**form, 'move_plan_token': preview['plan_token']}, {})
            self.assertFalse(stale['ok'])
            self.assertTrue((a / 'same.png').exists())
            fresh = combine_datasets.preview_move(form)
            _, _, applied = combine_datasets.handle({**form, 'move_plan_token': fresh['plan_token']}, {})
            self.assertTrue(applied['ok'], applied)
            self.assertFalse((a / 'same.png').exists())
            self.assertFalse((a / 'same.txt').exists())
            self.assertEqual(len(list(out.glob('*.png'))), 2)
            self.assertEqual(len(list(out.glob('*.txt'))), 2)


if __name__ == "__main__":
    unittest.main()
