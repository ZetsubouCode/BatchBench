import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import review_quiz


class ReviewQuizTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.config_path = Path(self.temp_dir.name) / "_config" / "review_quiz.json"
        self.config_patch = patch.object(review_quiz, "REVIEW_QUIZ_CONFIG_PATH", self.config_path)
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        review_quiz._CONFIG_CACHE["signature"] = None
        review_quiz._CONFIG_CACHE["payload"] = None

    def _project(self):
        project = Path(self.temp_dir.name) / "project"
        temp = project / "dataset" / "_temp"
        temp.mkdir(parents=True, exist_ok=True)
        return project, temp

    def _pair(self, root: Path, stem: str, tags: str):
        (root / f"{stem}.png").write_bytes(b"img")
        (root / f"{stem}.txt").write_text(tags, encoding="utf-8")

    def _save_config(self, steps):
        payload = review_quiz.default_review_quiz_config()
        payload["quiz_review"]["steps"] = steps
        return review_quiz.save_review_quiz_config(payload)

    def test_default_config_is_created_and_tags_are_normalized(self):
        config = review_quiz.load_review_quiz_config()

        self.assertTrue(self.config_path.exists())
        self.assertTrue(config["quiz_review"]["steps"])

        config["quiz_review"]["steps"] = [
            {
                "id": "Camera Angle",
                "label": "Camera Angle",
                "mode": "single",
                "required": True,
                "tags": [" From Side ", "from side", "FROM FRONT"],
            }
        ]
        saved = review_quiz.save_review_quiz_config(config)
        step = saved["quiz_review"]["steps"][0]
        self.assertEqual(step["id"], "camera_angle")
        self.assertEqual(step["tags"], ["from_side", "from_front"])
        self.assertFalse(step["danbooru_autosuggest"])

    def test_danbooru_autosuggest_step_setting_persists(self):
        config = review_quiz.default_review_quiz_config()
        config["quiz_review"]["steps"] = [
            {
                "id": "camera_angle",
                "label": "Camera Angle",
                "mode": "multi",
                "required": False,
                "danbooru_autosuggest": True,
                "tags": ["from_below"],
            }
        ]

        saved = review_quiz.save_review_quiz_config(config)

        self.assertTrue(saved["quiz_review"]["steps"][0]["danbooru_autosuggest"])

    def test_uncertain_required_step_stays_in_missing_and_uncertain_queue(self):
        with tempfile.TemporaryDirectory() as project_td:
            root = Path(project_td)
            temp_root = root / "dataset" / "_temp"
            temp_root.mkdir(parents=True)
            (temp_root / "sample.jpg").write_bytes(b"")
            (temp_root / "sample.txt").write_text("solo", encoding="utf-8")
            config = review_quiz.default_review_quiz_config()
            config["quiz_review"]["steps"] = [
                {
                    "id": "manual_review",
                    "label": "Manual Review",
                    "mode": "manual",
                    "required": True,
                    "auto_advance": False,
                    "allow_not_applicable": False,
                    "queue_mode": "missing_only",
                    "tags": [],
                }
            ]
            review_quiz.save_review_quiz_config(config)

            saved = review_quiz.save_quiz_item(
                root,
                "temp",
                "sample.jpg",
                "manual_review",
                selected_tags=[],
                manual_tags=["solo", "blue_hair"],
                uncertain=True,
                uncertain_note="Need collar tag",
            )
            missing = review_quiz.list_quiz_items(root, "temp", "manual_review", "missing_only")
            uncertain = review_quiz.list_quiz_items(root, "temp", "manual_review", "uncertain_only")

            self.assertFalse(saved["reviewed"])
            self.assertTrue(saved["uncertain"])
            self.assertEqual(saved["uncertain_note"], "Need collar tag")
            self.assertEqual(missing["items"][0]["rel"], "sample.jpg")
            self.assertEqual(uncertain["items"][0]["uncertain_note"], "Need collar tag")

    def test_default_steps_match_guided_tagging_quiz_segments(self):
        config = review_quiz.default_review_quiz_config()
        expected_ids = [segment["id"] for segment in review_quiz.tag_editor.DEFAULT_QUIZ_SEGMENTS]
        actual_ids = [step["id"] for step in config["quiz_review"]["steps"]]

        self.assertEqual(actual_ids, expected_ids)

    def test_hyphen_is_valid_step_tag(self):
        config = review_quiz.default_review_quiz_config()
        config["quiz_review"]["steps"] = [
            {
                "id": "placeholder",
                "label": "Placeholder",
                "mode": "multi",
                "required": True,
                "tags": ["-", "from-side"],
            }
        ]

        saved = review_quiz.save_review_quiz_config(config)

        self.assertEqual(saved["quiz_review"]["steps"][0]["tags"], ["-", "from-side"])

    def test_legacy_global_manual_setting_migrates_steps_to_manual_mode(self):
        config = review_quiz.default_review_quiz_config()
        config["quiz_review"]["manual_tagging"] = True

        normalized = review_quiz.normalize_review_quiz_config(config)

        self.assertTrue(normalized["quiz_review"]["steps"])
        self.assertTrue(all(step["mode"] == "manual" for step in normalized["quiz_review"]["steps"]))
        self.assertNotIn("manual_tagging", normalized["quiz_review"])

    def test_single_choice_save_removes_conflict_and_preserves_other_tag_order(self):
        project, temp = self._project()
        self._save_config(
            [
                {
                    "id": "body",
                    "label": "Body",
                    "mode": "single",
                    "required": True,
                    "auto_advance": True,
                    "allow_not_applicable": False,
                    "tags": ["full_body", "upper_body", "cowboy_shot"],
                }
            ]
        )
        self._pair(temp, "sample", "trigger, upper_body, from_side, full_body")

        result = review_quiz.save_quiz_item(project, "temp", "sample.png", "body", ["cowboy_shot"], backup=True)

        self.assertTrue(result["ok"])
        self.assertEqual(result["removed"], ["upper_body", "full_body"])
        self.assertEqual(result["added"], ["cowboy_shot"])
        self.assertEqual((temp / "sample.txt").read_text(encoding="utf-8"), "trigger, cowboy_shot, from_side")
        self.assertTrue((temp / "sample.txt.bak").exists())
        metadata = json.loads((temp / ".bb_review.json").read_text(encoding="utf-8"))
        self.assertIn("body", metadata["items"]["sample.png"]["reviewed_steps"])

    def test_multi_choice_save_and_restore_exact_previous_state(self):
        project, temp = self._project()
        self._save_config(
            [
                {
                    "id": "background",
                    "label": "Background",
                    "mode": "multi",
                    "required": True,
                    "allow_not_applicable": True,
                    "tags": ["indoors", "bedroom", "street"],
                }
            ]
        )
        self._pair(temp, "sample", "trigger, street, smile")

        saved = review_quiz.save_quiz_item(project, "temp", "sample.png", "background", ["indoors", "bedroom"])
        self.assertEqual(saved["tags"], ["trigger", "indoors", "bedroom", "smile"])

        restored = review_quiz.restore_quiz_item(
            project,
            "temp",
            "sample.png",
            saved["previous_tags"],
            saved["previous_metadata"],
        )
        self.assertEqual(restored["tags"], ["trigger", "street", "smile"])
        self.assertEqual((temp / "sample.txt").read_text(encoding="utf-8"), "trigger, street, smile")

    def test_not_applicable_leaves_caption_clean_and_removes_item_from_missing_queue(self):
        project, temp = self._project()
        self._save_config(
            [
                {
                    "id": "angle",
                    "label": "Angle",
                    "mode": "single",
                    "required": True,
                    "allow_not_applicable": True,
                    "tags": ["from_front", "from_side"],
                }
            ]
        )
        self._pair(temp, "sample", "trigger")

        before = review_quiz.list_quiz_items(project, "temp", "angle", "missing_only")
        self.assertEqual([item["rel"] for item in before["items"]], ["sample.png"])

        review_quiz.save_quiz_item(project, "temp", "sample.png", "angle", [], not_applicable=True)
        after = review_quiz.list_quiz_items(project, "temp", "angle", "missing_only")

        self.assertEqual(after["items"], [])
        self.assertEqual((temp / "sample.txt").read_text(encoding="utf-8"), "trigger")
        self.assertNotIn("not_applicable", (temp / "sample.txt").read_text(encoding="utf-8"))

    def test_conflict_queue_lists_only_single_step_conflicts(self):
        project, temp = self._project()
        self._save_config(
            [
                {
                    "id": "body",
                    "label": "Body",
                    "mode": "single",
                    "required": True,
                    "tags": ["full_body", "upper_body"],
                }
            ]
        )
        self._pair(temp, "conflict", "trigger, full_body, upper_body")
        self._pair(temp, "clean", "trigger, full_body")

        result = review_quiz.list_quiz_items(project, "temp", "body", "conflict_only")

        self.assertEqual([item["rel"] for item in result["items"]], ["conflict.png"])
        self.assertEqual(result["summary"]["conflict"], 1)

    def test_save_rejects_missing_caption(self):
        project, temp = self._project()
        config = review_quiz.default_review_quiz_config()
        review_quiz.save_review_quiz_config(config)
        (temp / "sample.png").write_bytes(b"img")

        with self.assertRaisesRegex(ValueError, "does not create caption"):
            review_quiz.save_quiz_item(project, "temp", "sample.png", "body_composition", ["full_body"])

    def test_manual_tagging_replaces_full_caption_and_can_create_missing_caption(self):
        project, temp = self._project()
        config = review_quiz.default_review_quiz_config()
        next(step for step in config["quiz_review"]["steps"] if step["id"] == "body_composition")["mode"] = "manual"
        review_quiz.save_review_quiz_config(config)
        (temp / "sample.png").write_bytes(b"img")

        created = review_quiz.save_quiz_item(
            project,
            "temp",
            "sample.png",
            "body_composition",
            [],
            manual_tags=[" Trigger ", "long hair", "smile"],
        )
        self.assertEqual(created["tags"], ["trigger", "long_hair", "smile"])
        self.assertFalse(created["backup_created"])
        self.assertEqual((temp / "sample.txt").read_text(encoding="utf-8"), "trigger, long_hair, smile")
        review_quiz.restore_quiz_item(
            project,
            "temp",
            "sample.png",
            created["previous_tags"],
            created["previous_metadata"],
            had_txt=created["previous_has_txt"],
        )
        self.assertFalse((temp / "sample.txt").exists())
        (temp / "sample.txt").write_text("trigger, long_hair, smile", encoding="utf-8")

        saved = review_quiz.save_quiz_item(
            project,
            "temp",
            "sample.png",
            "body_composition",
            [],
            manual_tags=["trigger", "closed eyes"],
        )
        self.assertEqual(saved["tags"], ["trigger", "closed_eyes"])
        self.assertTrue(saved["backup_created"])

    def test_manual_caption_payload_requires_manual_tagging_mode(self):
        project, temp = self._project()
        review_quiz.save_review_quiz_config(review_quiz.default_review_quiz_config())
        self._pair(temp, "sample", "trigger")

        with self.assertRaisesRegex(ValueError, "manual Quiz Review step"):
            review_quiz.save_quiz_item(
                project,
                "temp",
                "sample.png",
                "body_composition",
                [],
                manual_tags=["trigger", "smile"],
            )

    def test_mixed_flow_keeps_fixed_and_manual_step_behavior_separate(self):
        project, temp = self._project()
        self._save_config(
            [
                {
                    "id": "body",
                    "label": "Body",
                    "mode": "single",
                    "required": True,
                    "tags": ["full_body", "upper_body"],
                },
                {
                    "id": "freeform",
                    "label": "Freeform",
                    "mode": "manual",
                    "required": True,
                    "queue_mode": "missing_only",
                    "tags": [],
                },
            ]
        )
        self._pair(temp, "sample", "trigger, full_body")

        fixed = review_quiz.list_quiz_items(project, "temp", "body", "missing_only")
        manual_before = review_quiz.list_quiz_items(project, "temp", "freeform", "missing_only")
        self.assertEqual(fixed["items"], [])
        self.assertEqual([item["rel"] for item in manual_before["items"]], ["sample.png"])

        review_quiz.save_quiz_item(
            project,
            "temp",
            "sample.png",
            "freeform",
            [],
            manual_tags=["trigger", "full_body", "smile"],
        )
        manual_after = review_quiz.list_quiz_items(project, "temp", "freeform", "missing_only")
        self.assertEqual(manual_after["items"], [])


if __name__ == "__main__":
    unittest.main()
