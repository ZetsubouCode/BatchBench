import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import shutil as _shutil

from services import tag_editor


class TagEditorTransactionTests(unittest.TestCase):
    def _make_pair(self, root: Path, stem: str = "sample"):
        img = root / f"{stem}.png"
        txt = root / f"{stem}.txt"
        img.write_bytes(b"img")
        txt.write_text("cat, dog\n", encoding="utf-8")
        return img, txt

    def test_move_and_undo_use_root_temp_and_preserve_structure(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "dataset"
            base.mkdir(parents=True, exist_ok=True)
            custom_temp = Path(td) / "custom_temp"
            nested = base / "set_a" / "batch_1"
            nested.mkdir(parents=True, exist_ok=True)
            self._make_pair(nested)

            _, _, move_meta = tag_editor.handle(
                {
                    "folder": str(base),
                    "mode": "move",
                    "edit_target": "base",
                    "tags": "cat",
                    "exts": ".png",
                    "temp_dir": str(custom_temp),
                },
                {},
            )

            self.assertTrue(move_meta.get("ok"), msg=move_meta)
            expected_temp = base / "_temp" / "set_a" / "batch_1"
            self.assertTrue((expected_temp / "sample.png").exists())
            self.assertTrue((expected_temp / "sample.txt").exists())
            self.assertFalse((nested / "sample.png").exists())
            self.assertFalse((nested / "sample.txt").exists())
            self.assertFalse((custom_temp / "sample.png").exists())
            self.assertFalse((custom_temp / "sample.txt").exists())

            _, _, undo_meta = tag_editor.handle(
                {
                    "folder": str(base),
                    "mode": "undo",
                    "edit_target": "base",
                    "exts": ".png",
                    "temp_dir": str(custom_temp),
                },
                {},
            )

            self.assertTrue(undo_meta.get("ok"), msg=undo_meta)
            self.assertTrue((nested / "sample.png").exists())
            self.assertTrue((nested / "sample.txt").exists())
            self.assertFalse((expected_temp / "sample.png").exists())
            self.assertFalse((expected_temp / "sample.txt").exists())

    def test_move_rolls_back_if_txt_move_fails(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "dataset"
            base.mkdir(parents=True, exist_ok=True)
            self._make_pair(base)

            real_move = _shutil.move
            calls = {"n": 0}

            def flaky_move(src, dst):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise OSError("forced failure")
                return real_move(src, dst)

            with patch("services.tag_editor.shutil.move", side_effect=flaky_move):
                _, _, meta = tag_editor.handle(
                    {
                        "folder": str(base),
                        "mode": "move",
                        "edit_target": "base",
                        "tags": "cat",
                        "exts": ".png",
                    },
                    {},
                )

            self.assertFalse(meta.get("ok"), msg=meta)
            self.assertTrue((base / "sample.png").exists())
            self.assertTrue((base / "sample.txt").exists())
            self.assertFalse((base / "_temp" / "sample.png").exists())
            self.assertFalse((base / "_temp" / "sample.txt").exists())

    def test_edit_modes_only_touch_temp_folder(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "dataset"
            temp = base / "_temp"
            base.mkdir(parents=True, exist_ok=True)
            temp.mkdir(parents=True, exist_ok=True)
            self._make_pair(base)
            self._make_pair(temp)

            _, _, meta = tag_editor.handle(
                {
                    "folder": str(base),
                    "mode": "insert",
                    "edit_target": "base",
                    "tags": "bird",
                    "exts": ".png",
                },
                {},
            )

            self.assertTrue(meta.get("ok"), msg=meta)
            base_txt = (base / "sample.txt").read_text(encoding="utf-8").strip()
            temp_txt = (temp / "sample.txt").read_text(encoding="utf-8").strip()
            self.assertEqual(base_txt, "cat, dog")
            self.assertEqual(temp_txt, "cat, dog, bird")


if __name__ == "__main__":
    unittest.main()
