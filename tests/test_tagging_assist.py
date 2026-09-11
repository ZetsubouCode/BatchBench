import tempfile
import unittest
from pathlib import Path

from services import tagging_assist


class TaggingAssistTests(unittest.TestCase):
    def _project(self, td: str) -> Path:
        root = Path(td)
        (root / "dataset" / "_temp").mkdir(parents=True)
        (root / "database").mkdir()
        return root

    def test_pack_crud_and_apply_deduplicates_tags(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._project(td)
            created = tagging_assist.upsert_pack(
                root,
                {"name": "Outfit base", "segment_id": "outfit", "tags": ["shirt", "blue shirt", "shirt"], "pinned": True},
            )
            pack_id = created["pack"]["id"]

            listed = tagging_assist.list_packs(root, "outfit")
            applied = tagging_assist.apply_pack(root, pack_id, ["solo", "shirt"])
            tagging_assist.delete_pack(root, pack_id)
            after_delete = tagging_assist.list_packs(root, "outfit")

            self.assertEqual(len(listed["packs"]), 1)
            self.assertEqual(applied["tags"], ["solo", "shirt", "blue_shirt"])
            self.assertEqual(applied["added"], ["blue_shirt"])
            self.assertEqual(after_delete["packs"], [])

    def test_sibling_preview_does_not_write_and_append_apply_is_append_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._project(td)
            temp = root / "dataset" / "_temp"
            (temp / "a.jpg").write_bytes(b"")
            (temp / "a.txt").write_text("solo, blue_hair", encoding="utf-8")
            (temp / "b.jpg").write_bytes(b"")
            (temp / "b.txt").write_text("solo, close-up", encoding="utf-8")

            preview = tagging_assist.sibling_preview(root, "temp", "a.jpg", ["b.jpg"], ["blue_hair", "solo"])
            before_apply = (temp / "b.txt").read_text(encoding="utf-8")
            applied = tagging_assist.sibling_apply(root, "temp", "a.jpg", ["b.jpg"], ["blue_hair", "solo"])
            after_apply = (temp / "b.txt").read_text(encoding="utf-8")

            self.assertTrue(preview["changes"][0]["changed"])
            self.assertEqual(before_apply, "solo, close-up")
            self.assertEqual(after_apply, "solo, close-up, blue_hair")
            self.assertEqual(applied["applied"][0]["added"], ["blue_hair"])

    def test_lint_reports_issues_without_changing_caption(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._project(td)
            temp = root / "dataset" / "_temp"
            (temp / "a.jpg").write_bytes(b"")
            caption = "solo, solo, custom_trigger"
            (temp / "a.txt").write_text(caption, encoding="utf-8")
            tagging_assist.save_state(root, {**tagging_assist.default_state(), "required_trigger_tags": ["main_trigger"]})

            report = tagging_assist.lint_captions(root, "temp")

            self.assertEqual((temp / "a.txt").read_text(encoding="utf-8"), caption)
            kinds = {row["issue_type"] for row in report["rows"]}
            self.assertIn("duplicate", kinds)
            self.assertIn("unknown", kinds)
            self.assertIn("missing_required_trigger", kinds)

    def test_machine_suggestions_store_separately_and_ignore(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._project(td)
            temp = root / "dataset" / "_temp"
            (temp / "a.jpg").write_bytes(b"")
            (temp / "a.txt").write_text("solo", encoding="utf-8")

            tagging_assist.store_machine_suggestions(
                root,
                "a.jpg",
                [{"tag": "indoors", "confidence": 0.91}, {"tag": "character_name", "confidence": 0.8}],
                {"model_id": "test"},
            )
            tagging_assist.ignore_machine_suggestion(root, "a.jpg", "character_name")
            listed = tagging_assist.list_machine_suggestions(root, "a.jpg")

            self.assertEqual((temp / "a.txt").read_text(encoding="utf-8"), "solo")
            self.assertEqual([row["tag"] for row in listed["suggestions"]], ["indoors"])
            self.assertEqual(listed["metadata"]["model_id"], "test")


if __name__ == "__main__":
    unittest.main()
