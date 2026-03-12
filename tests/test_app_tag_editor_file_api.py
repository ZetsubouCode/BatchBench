import tempfile
import unittest
from pathlib import Path

from app import app


class TagEditorFileApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_tag_add_creates_txt_for_existing_image(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            temp = root / "_temp"
            temp.mkdir(parents=True, exist_ok=True)
            image = temp / "sample.png"
            image.write_bytes(b"img")

            resp = self.client.post(
                "/api/tags/tag-add",
                json={
                    "folder": str(root),
                    "rel": "_temp/sample.png",
                    "tags": "cat, blue sky",
                    "backup": False,
                    "create_missing_txt": True,
                },
            )

            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("ok"), msg=data)
            self.assertTrue(data.get("created"))
            self.assertEqual(data.get("added"), ["cat", "blue_sky"])
            self.assertEqual((temp / "sample.txt").read_text(encoding="utf-8").strip(), "cat, blue_sky")

    def test_move_file_moves_image_and_txt_between_root_and_temp(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            temp = root / "_temp"
            temp.mkdir(parents=True, exist_ok=True)
            image = root / "sample.png"
            txt = root / "sample.txt"
            image.write_bytes(b"img")
            txt.write_text("cat", encoding="utf-8")

            to_temp = self.client.post(
                "/api/tags/move-file",
                json={"folder": str(root), "src": "sample.png", "dst": "_temp"},
            )
            self.assertEqual(to_temp.status_code, 200)
            move_data = to_temp.get_json()
            self.assertTrue(move_data.get("ok"), msg=move_data)
            self.assertTrue((temp / "sample.png").exists())
            self.assertTrue((temp / "sample.txt").exists())
            self.assertFalse(image.exists())
            self.assertFalse(txt.exists())

            back_to_root = self.client.post(
                "/api/tags/move-file",
                json={"folder": str(root), "src": "_temp/sample.png", "dst": ""},
            )
            self.assertEqual(back_to_root.status_code, 200)
            restore_data = back_to_root.get_json()
            self.assertTrue(restore_data.get("ok"), msg=restore_data)
            self.assertTrue(image.exists())
            self.assertTrue(txt.exists())
            self.assertFalse((temp / "sample.png").exists())
            self.assertFalse((temp / "sample.txt").exists())

    def test_move_files_moves_multiple_pairs_in_one_request(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            temp = root / "_temp"
            temp.mkdir(parents=True, exist_ok=True)
            for stem in ("a", "b"):
                (root / f"{stem}.png").write_bytes(b"img")
                (root / f"{stem}.txt").write_text(stem, encoding="utf-8")

            resp = self.client.post(
                "/api/tags/move-files",
                json={"folder": str(root), "srcs": ["a.png", "b.png"], "dst": "_temp"},
            )

            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("ok"), msg=data)
            self.assertEqual(len(data.get("moved") or []), 2)
            for stem in ("a", "b"):
                self.assertTrue((temp / f"{stem}.png").exists())
                self.assertTrue((temp / f"{stem}.txt").exists())
                self.assertFalse((root / f"{stem}.png").exists())
                self.assertFalse((root / f"{stem}.txt").exists())

    def test_return_temp_moves_all_nested_pairs_back_to_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            nested = root / "_temp" / "batch_a" / "variant"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "sample.png").write_bytes(b"img")
            (nested / "sample.txt").write_text("tag_a", encoding="utf-8")
            (nested / "sample_2.png").write_bytes(b"img")
            (nested / "sample_2.txt").write_text("tag_b", encoding="utf-8")

            resp = self.client.post(
                "/api/tags/return-temp",
                json={"folder": str(root), "exts": ".png,.jpg,.jpeg,.webp"},
            )

            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("ok"), msg=data)
            self.assertEqual(data.get("total"), 2)
            self.assertEqual(len(data.get("moved") or []), 2)
            self.assertTrue((root / "sample.png").exists())
            self.assertTrue((root / "sample.txt").exists())
            self.assertTrue((root / "sample_2.png").exists())
            self.assertTrue((root / "sample_2.txt").exists())
            self.assertFalse((nested / "sample.png").exists())
            self.assertFalse((nested / "sample.txt").exists())
            self.assertFalse((nested / "sample_2.png").exists())
            self.assertFalse((nested / "sample_2.txt").exists())
            self.assertTrue((root / "_temp").exists())
            self.assertFalse((root / "_temp" / "batch_a").exists())

    def test_return_temp_with_source_all_moves_from_any_subfolder_to_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            temp_nested = root / "_temp" / "batch_a"
            other_nested = root / "placeholderA" / "round_1"
            temp_nested.mkdir(parents=True, exist_ok=True)
            other_nested.mkdir(parents=True, exist_ok=True)

            (temp_nested / "from_temp.png").write_bytes(b"img")
            (temp_nested / "from_temp.txt").write_text("tag_temp", encoding="utf-8")
            (other_nested / "from_other.png").write_bytes(b"img")
            (other_nested / "from_other.txt").write_text("tag_other", encoding="utf-8")
            (root / "already_root.png").write_bytes(b"img")
            (root / "already_root.txt").write_text("tag_root", encoding="utf-8")

            resp = self.client.post(
                "/api/tags/return-temp",
                json={"folder": str(root), "exts": ".png,.jpg,.jpeg,.webp", "source": "all"},
            )

            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("ok"), msg=data)
            self.assertEqual(data.get("source"), "all")
            self.assertEqual(data.get("total"), 2)
            self.assertEqual(len(data.get("moved") or []), 2)
            self.assertTrue((root / "from_temp.png").exists())
            self.assertTrue((root / "from_temp.txt").exists())
            self.assertTrue((root / "from_other.png").exists())
            self.assertTrue((root / "from_other.txt").exists())
            self.assertTrue((root / "already_root.png").exists())
            self.assertTrue((root / "already_root.txt").exists())
            self.assertFalse((temp_nested / "from_temp.png").exists())
            self.assertFalse((temp_nested / "from_temp.txt").exists())
            self.assertFalse((other_nested / "from_other.png").exists())
            self.assertFalse((other_nested / "from_other.txt").exists())

    def test_cheatsheet_reads_root_level_txt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            root.mkdir(parents=True, exist_ok=True)
            cheat = root / "main_tags.txt"
            cheat.write_text(
                "trigger_word\n\nappearance: blue_sky, long_hair\nsmiling, looking_at_viewer\noutfit: jacket, skirt\n",
                encoding="utf-8",
            )

            resp = self.client.post(
                "/api/tags/cheatsheet",
                json={"folder": str(root), "rel": "main_tags.txt"},
            )

            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("ok"), msg=data)
            self.assertEqual(data.get("rel"), "main_tags.txt")
            self.assertEqual(data.get("trigger"), "trigger_word")
            self.assertEqual(data.get("tags"), ["blue_sky", "long_hair", "smiling", "looking_at_viewer", "jacket", "skirt"])
            sections = data.get("sections") or []
            self.assertEqual(len(sections), 2)
            self.assertEqual(sections[0]["category"], "appearance")
            self.assertEqual(sections[0]["tags"], ["blue_sky", "long_hair"])
            self.assertEqual(sections[0]["conditionals"], [["smiling", "looking_at_viewer"]])
            self.assertEqual(sections[1]["category"], "outfit")


if __name__ == "__main__":
    unittest.main()
