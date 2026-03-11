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

    def test_list_images_with_tags_keeps_temp_items_when_scanning_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            temp = root / "_temp"
            temp.mkdir(parents=True, exist_ok=True)

            (root / "root.png").write_bytes(b"")
            (root / "root.txt").write_text("alpha", encoding="utf-8")
            (temp / "temp.png").write_bytes(b"")
            (temp / "temp.txt").write_text("beta", encoding="utf-8")

            result = tag_editor.list_images_with_tags(
                root,
                exts=[".png"],
                recursive=True,
                limit=10,
                tag_limit=10,
            )

            self.assertTrue(result.get("ok"))
            rels = {item["rel"]: item for item in result["images"]}
            self.assertIn("root.png", rels)
            self.assertIn("_temp/temp.png", rels)
            self.assertFalse(rels["root.png"]["in_temp"])
            self.assertTrue(rels["_temp/temp.png"]["in_temp"])
            self.assertEqual(rels["_temp/temp.png"]["parent_rel"], "_temp")
            self.assertIn("created_ts", rels["_temp/temp.png"])

    def test_add_tags_can_create_missing_txt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            txt = root / "sample.txt"

            result = tag_editor.add_tags(
                txt,
                ["cat", "blue sky"],
                backup=False,
                create_missing_txt=True,
            )

            self.assertTrue(result.get("ok"), msg=result)
            self.assertTrue(result.get("created"))
            self.assertEqual(txt.read_text(encoding="utf-8").strip(), "cat, blue_sky")


if __name__ == "__main__":
    unittest.main()
