import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import app
from services import review_quiz


class ReviewQuizApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.config_path = Path(self.temp_dir.name) / "_config" / "review_quiz.json"
        self.config_patch = patch.object(review_quiz, "REVIEW_QUIZ_CONFIG_PATH", self.config_path)
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        review_quiz._CONFIG_CACHE["signature"] = None
        review_quiz._CONFIG_CACHE["payload"] = None
        self.client = app.test_client()

    def _project(self):
        project = Path(self.temp_dir.name) / "project"
        temp = project / "dataset" / "_temp"
        temp.mkdir(parents=True, exist_ok=True)
        return project, temp

    def test_config_get_creates_default_and_post_persists_normalized_step(self):
        get_response = self.client.get("/api/settings/review-quiz")
        self.assertEqual(get_response.status_code, 200)
        self.assertTrue(self.config_path.exists())

        config = get_response.get_json()["config"]
        config["quiz_review"]["steps"] = [
            {
                "id": "Pose Type",
                "label": "Pose Type",
                "mode": "multi",
                "required": False,
                "autosuggest_segment_only": True,
                "tags": ["Standing Pose", "standing_pose", "SITTING"],
            }
        ]
        post_response = self.client.post("/api/settings/review-quiz", json={"config": config})

        self.assertEqual(post_response.status_code, 200)
        step = post_response.get_json()["config"]["quiz_review"]["steps"][0]
        self.assertEqual(step["id"], "pose_type")
        self.assertEqual(step["tags"], ["standing_pose", "sitting"])
        self.assertTrue(step["autosuggest_segment_only"])

    def test_quiz_list_save_and_restore_routes(self):
        project, temp = self._project()
        (temp / "sample.png").write_bytes(b"img")
        (temp / "sample.txt").write_text("trigger, upper_body, from_side", encoding="utf-8")
        review_quiz.save_review_quiz_config(review_quiz.default_review_quiz_config())

        list_response = self.client.post(
            "/api/tags/quiz/list",
            json={"folder": str(project), "area": "temp", "step_id": "body_composition", "queue": "all", "exts": ".png"},
        )
        self.assertEqual(list_response.status_code, 200)
        listed = list_response.get_json()
        self.assertEqual(len(listed["items"]), 1)
        self.assertIn("/api/tags/image?", listed["items"][0]["image_url"])

        save_response = self.client.post(
            "/api/tags/quiz/save",
            json={
                "folder": str(project),
                "area": "temp",
                "rel": "sample.png",
                "step_id": "body_composition",
                "selected_tags": ["cowboy_shot"],
                "backup": True,
            },
        )
        self.assertEqual(save_response.status_code, 200)
        saved = save_response.get_json()
        self.assertEqual(saved["tags"], ["trigger", "cowboy_shot", "from_side"])

        restore_response = self.client.post(
            "/api/tags/quiz/restore",
            json={
                "folder": str(project),
                "area": "temp",
                "rel": "sample.png",
                "tags": saved["previous_tags"],
                "metadata": saved["previous_metadata"],
            },
        )
        self.assertEqual(restore_response.status_code, 200)
        self.assertEqual((temp / "sample.txt").read_text(encoding="utf-8"), "trigger, upper_body, from_side")

    def test_cheatsheet_conversion_returns_editable_steps_without_saving(self):
        project, _ = self._project()
        (project / "prompt.txt").write_text(
            "trigger\n\ncamera angle: from_front, from_side\nfrom_above\nbackground: indoors, outdoors\n",
            encoding="utf-8",
        )
        review_quiz.load_review_quiz_config()
        original = self.config_path.read_text(encoding="utf-8")

        response = self.client.post(
            "/api/settings/review-quiz/from-cheatsheet",
            json={"folder": str(project), "rel": "prompt.txt"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual([step["id"] for step in data["steps"]], ["camera_angle", "background"])
        self.assertEqual(data["steps"][0]["tags"], ["from_front", "from_side", "from_above"])
        self.assertEqual(self.config_path.read_text(encoding="utf-8"), original)

    def test_tagging_quiz_starter_uses_review_quiz_settings_steps(self):
        project, _ = self._project()
        dataset = project / "dataset"
        (dataset / "sample.png").write_bytes(b"img")
        config = review_quiz.default_review_quiz_config()
        config["quiz_review"]["steps"] = [
            {
                "id": "custom_camera",
                "label": "Custom Camera",
                "mode": "single",
                "required": True,
                "auto_advance": True,
                "allow_not_applicable": False,
                "queue_mode": "missing_only",
                "tags": ["from_front", "from_side"],
            }
        ]
        review_quiz.save_review_quiz_config(config)

        settings_response = self.client.get("/api/tagging-quiz/settings")
        self.assertEqual(settings_response.status_code, 200)
        segment = settings_response.get_json()["settings"]["segments"][0]
        self.assertEqual(segment["id"], "custom_camera")
        self.assertEqual(segment["label"], "Custom Camera")

        start_response = self.client.post(
            "/api/tagging-quiz/session/start",
            json={
                "project_root": str(project),
                "exts": ".png",
                "mapping_rows": [{"left_sections": [], "right_segment": "custom_camera"}],
                "recommendations": {"custom_camera": []},
            },
        )
        self.assertEqual(start_response.status_code, 200)
        session = start_response.get_json()["session"]
        self.assertEqual([step["id"] for step in session["quiz_segments"]], ["custom_camera"])

    def test_tagging_quiz_left_only_mapping_row_creates_quiz_segment(self):
        project, _ = self._project()
        dataset = project / "dataset"
        (dataset / "sample.png").write_bytes(b"img")
        (project / "prompt.txt").write_text(
            "trigger\n\naccessory: hair ornament, necklace\n\nbackground: indoors\n",
            encoding="utf-8",
        )
        review_quiz.save_review_quiz_config(review_quiz.default_review_quiz_config())

        response = self.client.post(
            "/api/tagging-quiz/session/start",
            json={
                "project_root": str(project),
                "exts": ".png",
                "mapping_rows": [
                    {"left_sections": ["accessory"], "right_segment": None},
                    {"left_sections": ["background"], "right_segment": "background"},
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        session = response.get_json()["session"]
        self.assertEqual([step["id"] for step in session["quiz_segments"][:2]], ["accessory", "background"])
        segment = next(seg for seg in session["quiz_segments"] if seg["id"] == "accessory")
        self.assertEqual(segment["label"], "Accessory")
        self.assertEqual(session["mapping_rows"][0], {"left_sections": ["accessory"], "right_segment": "accessory"})
        self.assertEqual(session["recommendations"]["accessory"], ["hair_ornament", "necklace"])

    def test_tagging_quiz_sessions_are_restored_by_dataset_image_set(self):
        project, _ = self._project()
        dataset = project / "dataset"
        (dataset / "first.png").write_bytes(b"img")
        review_quiz.save_review_quiz_config(review_quiz.default_review_quiz_config())

        first_response = self.client.post(
            "/api/tagging-quiz/session/start",
            json={"project_root": str(project), "exts": ".png", "mapping_rows": []},
        )
        self.assertEqual(first_response.status_code, 200)
        first_session = first_response.get_json()["session"]
        first_session["images"]["dataset/first.png"]["status"] = "in_progress"
        save_first = self.client.post(
            "/api/tagging-quiz/session/save",
            json={"project_root": str(project), "session": first_session},
        )
        self.assertEqual(save_first.status_code, 200)

        (dataset / "first.png").unlink()
        (dataset / "second.png").write_bytes(b"img")
        second_response = self.client.post(
            "/api/tagging-quiz/session/start",
            json={"project_root": str(project), "exts": ".png", "mapping_rows": []},
        )
        self.assertEqual(second_response.status_code, 200)
        second_session = second_response.get_json()["session"]
        second_session["images"]["dataset/second.png"]["status"] = "in_progress"
        save_second = self.client.post(
            "/api/tagging-quiz/session/save",
            json={"project_root": str(project), "session": second_session},
        )
        self.assertEqual(save_second.status_code, 200)

        (dataset / "second.png").unlink()
        (dataset / "first.png").write_bytes(b"img")
        restored_first = self.client.post("/api/tagging-quiz/session/load", json={"project_root": str(project)})
        self.assertEqual(restored_first.status_code, 200)
        first_loaded = restored_first.get_json()["session"]
        self.assertEqual(list(first_loaded["images"].keys()), ["dataset/first.png"])
        self.assertEqual(first_loaded["images"]["dataset/first.png"]["status"], "in_progress")

        (dataset / "first.png").unlink()
        (dataset / "second.png").write_bytes(b"img")
        restored_second = self.client.post("/api/tagging-quiz/session/load", json={"project_root": str(project)})
        self.assertEqual(restored_second.status_code, 200)
        second_loaded = restored_second.get_json()["session"]
        self.assertEqual(list(second_loaded["images"].keys()), ["dataset/second.png"])
        self.assertEqual(second_loaded["images"]["dataset/second.png"]["status"], "in_progress")

    def test_tagging_quiz_save_rejects_session_from_different_project(self):
        project, _ = self._project()
        other = Path(self.temp_dir.name) / "other_project"
        (project / "dataset" / "sample.png").write_bytes(b"img")
        (other / "dataset" / "_temp").mkdir(parents=True, exist_ok=True)
        (other / "dataset" / "other.png").write_bytes(b"img")
        review_quiz.save_review_quiz_config(review_quiz.default_review_quiz_config())

        start_response = self.client.post(
            "/api/tagging-quiz/session/start",
            json={"project_root": str(project), "exts": ".png", "mapping_rows": []},
        )
        self.assertEqual(start_response.status_code, 200)
        session = start_response.get_json()["session"]

        save_response = self.client.post(
            "/api/tagging-quiz/session/save",
            json={"project_root": str(other), "session": session},
        )
        self.assertEqual(save_response.status_code, 400)
        self.assertIn("different project root", save_response.get_json()["error"])

    def test_tagging_quiz_load_keeps_active_session_when_image_list_changed(self):
        project, _ = self._project()
        dataset = project / "dataset"
        (dataset / "first.png").write_bytes(b"img")
        review_quiz.save_review_quiz_config(review_quiz.default_review_quiz_config())

        start_response = self.client.post(
            "/api/tagging-quiz/session/start",
            json={"project_root": str(project), "exts": ".png", "mapping_rows": []},
        )
        self.assertEqual(start_response.status_code, 200)

        (dataset / "first.png").unlink()
        (dataset / "second.png").write_bytes(b"img")
        load_response = self.client.post("/api/tagging-quiz/session/load", json={"project_root": str(project)})

        self.assertEqual(load_response.status_code, 200)
        data = load_response.get_json()
        self.assertTrue(data["session"])
        self.assertIn("Dataset image list changed", " | ".join(data.get("warnings") or []))
        self.assertTrue(data["session"]["images"]["dataset/first.png"]["missing"])
        self.assertIn("dataset/second.png", data["session"]["images"])

    def test_manual_quiz_save_creates_caption_when_enabled(self):
        project, temp = self._project()
        (temp / "sample.png").write_bytes(b"img")
        config = review_quiz.default_review_quiz_config()
        next(step for step in config["quiz_review"]["steps"] if step["id"] == "body_composition")["mode"] = "manual"
        review_quiz.save_review_quiz_config(config)

        response = self.client.post(
            "/api/tags/quiz/save",
            json={
                "folder": str(project),
                "area": "temp",
                "rel": "sample.png",
                "step_id": "body_composition",
                "selected_tags": [],
                "manual_tags": ["trigger", "long hair"],
            },
        )

        self.assertEqual(response.status_code, 200)
        saved = response.get_json()
        self.assertEqual(saved["tags"], ["trigger", "long_hair"])
        self.assertEqual((temp / "sample.txt").read_text(encoding="utf-8"), "trigger, long_hair")

        restore = self.client.post(
            "/api/tags/quiz/restore",
            json={
                "folder": str(project),
                "area": "temp",
                "rel": "sample.png",
                "tags": saved["previous_tags"],
                "metadata": saved["previous_metadata"],
                "had_txt": saved["previous_has_txt"],
            },
        )
        self.assertEqual(restore.status_code, 200)
        self.assertFalse((temp / "sample.txt").exists())


if __name__ == "__main__":
    unittest.main()
