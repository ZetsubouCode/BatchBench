import tempfile
import unittest
from pathlib import Path

from services import tag_editor


class TagEditorInitializeTests(unittest.TestCase):
    def test_initialize_copies_root_images_to_database_and_dataset_and_generates_txt(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project_init"
            dataset = project / "dataset"
            dataset.mkdir(parents=True, exist_ok=True)
            (project / "root.png").write_bytes(b"img")
            (dataset / "train_a.png").write_bytes(b"img")
            (dataset / "train_b.png").write_bytes(b"img")
            (dataset / "train_b.txt").write_text("keep_existing", encoding="utf-8")
            (project / "prompt.txt").write_text("my_trigger\n", encoding="utf-8")

            result = tag_editor.initialize_project_layout(project, [".png"], create_prompt=True)

            self.assertTrue(result.get("ok"), msg=result)
            database = project / "database"
            self.assertTrue((database / "root.png").exists())
            self.assertFalse((database / "root.txt").exists())
            self.assertTrue((dataset / "root.png").exists())
            self.assertTrue((dataset / "root.txt").exists())
            self.assertEqual((dataset / "root.txt").read_text(encoding="utf-8").strip(), "my_trigger")
            self.assertTrue((dataset / "train_a.txt").exists())
            self.assertEqual((dataset / "train_a.txt").read_text(encoding="utf-8").strip(), "my_trigger")
            self.assertEqual((dataset / "train_b.txt").read_text(encoding="utf-8").strip(), "keep_existing")
            self.assertIn("train_a.txt", result.get("generated_txt") or [])
            self.assertIn("root.txt", result.get("generated_txt") or [])
            self.assertTrue((result.get("copied_dataset") or []))
            self.assertTrue(any("dataset/train_b.txt: already exists" == item for item in (result.get("skipped") or [])))

    def test_inspect_preview_counts_dataset_txt_generation_excluding_temp(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project_preview"
            dataset = project / "dataset"
            temp = dataset / "_temp"
            project.mkdir(parents=True, exist_ok=True)
            dataset.mkdir(parents=True, exist_ok=True)
            temp.mkdir(parents=True, exist_ok=True)
            (project / "root_a.png").write_bytes(b"img")
            (project / "root_b.png").write_bytes(b"img")
            (dataset / "train_a.png").write_bytes(b"img")
            (dataset / "train_b.png").write_bytes(b"img")
            (dataset / "train_b.txt").write_text("exists", encoding="utf-8")
            (temp / "staged.png").write_bytes(b"img")

            result = tag_editor.inspect_project_layout(project, [".png"])

            self.assertTrue(result.get("ok"), msg=result)
            preview = result.get("init_preview") or {}
            self.assertEqual(preview.get("copy_images"), 2)
            self.assertEqual(preview.get("copy_dataset_images"), 2)
            self.assertEqual(preview.get("dataset_images_found"), 2)
            self.assertEqual(preview.get("generate_txt"), 3)
            self.assertEqual(preview.get("existing_txt"), 1)

    def test_inspect_requires_init_when_dataset_has_missing_txt_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project_ready_check"
            database = project / "database"
            dataset = project / "dataset"
            temp = dataset / "_temp"
            project.mkdir(parents=True, exist_ok=True)
            database.mkdir(parents=True, exist_ok=True)
            dataset.mkdir(parents=True, exist_ok=True)
            temp.mkdir(parents=True, exist_ok=True)

            (project / "prompt.txt").write_text("trigger_word\n\nappearance:\nhair_color\n", encoding="utf-8")
            (project / "sample.png").write_bytes(b"img")
            (database / "sample.png").write_bytes(b"img")
            (dataset / "sample.png").write_bytes(b"img")

            result = tag_editor.inspect_project_layout(project, [".png"])

            self.assertTrue(result.get("ok"), msg=result)
            self.assertFalse(result.get("ready"))
            self.assertTrue(result.get("needs_init"))
            missing = result.get("missing") or []
            self.assertTrue(any(str(item).startswith("dataset(txt pairs:") for item in missing), msg=missing)


if __name__ == "__main__":
    unittest.main()
