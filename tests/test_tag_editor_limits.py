import tempfile
import unittest
from pathlib import Path

from services import tag_editor


class TagEditorLimitTests(unittest.TestCase):
    def test_list_images_with_tags_applies_tag_limit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img = root / "sample.png"
            txt = root / "sample.txt"
            img.write_bytes(b"")
            txt.write_text(", ".join([f"tag_{i}" for i in range(30)]), encoding="utf-8")

            result = tag_editor.list_images_with_tags(
                root,
                exts=[".png"],
                recursive=False,
                limit=10,
                tag_limit=5,
            )

            self.assertTrue(result.get("ok"))
            self.assertEqual(len(result["images"]), 1)
            item = result["images"][0]
            self.assertEqual(len(item["tags"]), 5)
            self.assertEqual(item["tag_count"], 30)
            self.assertTrue(item["tags_truncated"])


if __name__ == "__main__":
    unittest.main()
