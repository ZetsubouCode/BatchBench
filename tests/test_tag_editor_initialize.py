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
            existing_prompt = "my_trigger\n"
            (project / "prompt.txt").write_text(existing_prompt, encoding="utf-8")

            result = tag_editor.initialize_project_layout(project, [".png"], create_prompt=True)

            self.assertTrue(result.get("ok"), msg=result)
            self.assertEqual((project / "prompt.txt").read_text(encoding="utf-8"), existing_prompt)
            database = project / "database"
            self.assertFalse((project / "root.png").exists())
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
            self.assertTrue((result.get("moved_database") or []))
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

    def test_multi_trigger_generates_folder_captions(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project_multi"
            (project / "default outfit").mkdir(parents=True)
            (project / "bikini").mkdir(parents=True)
            (project / "prompt.txt").write_text("my_character\n", encoding="utf-8")
            (project / "default outfit" / "a.png").write_bytes(b"img")
            (project / "bikini" / "b.png").write_bytes(b"img")

            result = tag_editor.initialize_project_layout(project, [".png"], multi_trigger_mode=True)

            self.assertTrue(result.get("ok"), msg=result)
            self.assertEqual((project / "dataset" / "a.txt").read_text(encoding="utf-8").strip(), "my_character, default_outfit")
            self.assertEqual((project / "dataset" / "b.txt").read_text(encoding="utf-8").strip(), "my_character, bikini")

    def test_multi_trigger_ignores_underscore_source_folder(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project_ignore"
            (project / "_reference").mkdir(parents=True)
            (project / "prompt.txt").write_text("my_character\n", encoding="utf-8")
            (project / "_reference" / "skip.png").write_bytes(b"img")

            result = tag_editor.initialize_project_layout(project, [".png"], multi_trigger_mode=True)

            self.assertTrue(result.get("ok"), msg=result)
            self.assertFalse((project / "dataset" / "skip.png").exists())
            self.assertFalse((project / "dataset" / "skip.txt").exists())
            preview = result.get("multi_trigger_preview") or {}
            self.assertIn("_reference", preview.get("ignored_folders") or [])

    def test_multi_trigger_root_level_image_uses_primary_only(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project_root_fallback"
            project.mkdir(parents=True)
            (project / "prompt.txt").write_text("my_character\n", encoding="utf-8")
            (project / "root.png").write_bytes(b"img")

            result = tag_editor.initialize_project_layout(project, [".png"], multi_trigger_mode=True)

            self.assertTrue(result.get("ok"), msg=result)
            self.assertEqual((project / "dataset" / "root.txt").read_text(encoding="utf-8").strip(), "my_character")

    def test_multi_trigger_existing_caption_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project_existing_caption"
            dataset = project / "dataset"
            (project / "default outfit").mkdir(parents=True)
            dataset.mkdir(parents=True)
            (project / "prompt.txt").write_text("my_character\n", encoding="utf-8")
            (project / "default outfit" / "a.png").write_bytes(b"img")
            (dataset / "a.png").write_bytes(b"img")
            (dataset / "a.txt").write_text("custom_existing_caption", encoding="utf-8")

            result = tag_editor.initialize_project_layout(project, [".png"], multi_trigger_mode=True)

            self.assertTrue(result.get("ok"), msg=result)
            self.assertEqual((dataset / "a.txt").read_text(encoding="utf-8").strip(), "custom_existing_caption")

    def test_multi_trigger_nested_source_behavior(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project_nested"
            (project / "bikini" / "chapter_01").mkdir(parents=True)
            (project / "bikini" / "_draft").mkdir(parents=True)
            (project / "_hidden").mkdir(parents=True)
            (project / "prompt.txt").write_text("my_character\n", encoding="utf-8")
            (project / "bikini" / "chapter_01" / "a.png").write_bytes(b"img")
            (project / "_hidden" / "skip_a.png").write_bytes(b"img")
            (project / "bikini" / "_draft" / "skip_b.png").write_bytes(b"img")

            result = tag_editor.initialize_project_layout(project, [".png"], multi_trigger_mode=True)

            self.assertTrue(result.get("ok"), msg=result)
            self.assertEqual((project / "dataset" / "a.txt").read_text(encoding="utf-8").strip(), "my_character, bikini")
            self.assertFalse((project / "dataset" / "skip_a.png").exists())
            self.assertFalse((project / "dataset" / "skip_b.png").exists())
            preview = result.get("multi_trigger_preview") or {}
            self.assertIn("_hidden", preview.get("ignored_folders") or [])
            self.assertIn("bikini/_draft", preview.get("ignored_nested_folders") or [])

    def test_multi_trigger_preview_matches_collected_sources(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project_preview_multi"
            (project / "default outfit").mkdir(parents=True)
            (project / "bikini").mkdir(parents=True)
            (project / "_reference").mkdir(parents=True)
            (project / "prompt.txt").write_text("my_character\n", encoding="utf-8")
            (project / "default outfit" / "a.png").write_bytes(b"img")
            (project / "bikini" / "b.png").write_bytes(b"img")
            (project / "_reference" / "skip.png").write_bytes(b"img")

            result = tag_editor.inspect_project_layout(project, [".png"], multi_trigger_mode=True)

            self.assertTrue(result.get("ok"), msg=result)
            preview = result.get("multi_trigger_preview") or {}
            self.assertTrue(preview.get("enabled"))
            self.assertEqual(preview.get("source_folder_count"), 2)
            self.assertEqual(preview.get("source_image_count"), 2)
            self.assertIn("_reference", preview.get("ignored_folders") or [])
            self.assertTrue(preview.get("examples"))
            self.assertEqual(preview.get("source_image_count"), len(result.get("missing_root_in_dataset") or []))


if __name__ == "__main__":
    unittest.main()
