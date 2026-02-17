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

    def test_move_and_undo_with_temp_override(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "dataset"
            base.mkdir(parents=True, exist_ok=True)
            temp_dir = Path(td) / "custom_temp"
            self._make_pair(base)

            _, _, move_meta = tag_editor.handle(
                {
                    "folder": str(base),
                    "mode": "move",
                    "edit_target": "base",
                    "tags": "cat",
                    "exts": ".png",
                    "temp_dir": str(temp_dir),
                },
                {},
            )

            self.assertTrue(move_meta.get("ok"), msg=move_meta)
            self.assertTrue((temp_dir / "sample.png").exists())
            self.assertTrue((temp_dir / "sample.txt").exists())
            self.assertFalse((base / "sample.png").exists())
            self.assertFalse((base / "sample.txt").exists())

            _, _, undo_meta = tag_editor.handle(
                {
                    "folder": str(base),
                    "mode": "undo",
                    "edit_target": "base",
                    "exts": ".png",
                    "temp_dir": str(temp_dir),
                },
                {},
            )

            self.assertTrue(undo_meta.get("ok"), msg=undo_meta)
            self.assertTrue((base / "sample.png").exists())
            self.assertTrue((base / "sample.txt").exists())

    def test_move_rolls_back_if_txt_move_fails(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "dataset"
            base.mkdir(parents=True, exist_ok=True)
            temp_dir = Path(td) / "temp"
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
                        "temp_dir": str(temp_dir),
                    },
                    {},
                )

            self.assertFalse(meta.get("ok"), msg=meta)
            self.assertTrue((base / "sample.png").exists())
            self.assertTrue((base / "sample.txt").exists())
            self.assertFalse((temp_dir / "sample.png").exists())
            self.assertFalse((temp_dir / "sample.txt").exists())


if __name__ == "__main__":
    unittest.main()
