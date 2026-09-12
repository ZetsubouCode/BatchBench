import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import app
from services import context_suggestions


class ContextSuggestionApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.project = Path(self.temp_dir.name) / "project"
        (self.project / "dataset" / "_temp").mkdir(parents=True)

    def test_model_registry_status_is_shared_by_ui_endpoints(self):
        response = self.client.get("/api/tagger-models")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual({item["key"] for item in data["models"]}, {"caformer_s36_dbv4", "wd_swinv2_v3"})
        self.assertTrue(data["classifier"]["ready"])

    def test_image_suggestion_route_marks_different_model_stale(self):
        context_suggestions._atomic_json(
            context_suggestions.cache_path(self.project),
            {
                "schema_version": context_suggestions.CACHE_SCHEMA_VERSION,
                "entries": {
                    "sample.png": {
                        "config": {"model_profile": "caformer_s36_dbv4", "sensitivity": "normal"},
                        "segments": {"expression": [{"tag": "smile", "score": 0.9}]},
                    }
                },
            },
        )
        ready = self.client.post(
            "/api/tagging-quiz/suggestions/image",
            json={"project_root": str(self.project), "image_rel": "sample.png", "model_profile": "caformer_s36_dbv4", "sensitivity": "normal"},
        )
        self.assertEqual(ready.status_code, 200)
        self.assertTrue(ready.get_json()["ready"])

        stale = self.client.post(
            "/api/tagging-quiz/suggestions/image",
            json={"project_root": str(self.project), "image_rel": "sample.png", "model_profile": "wd_swinv2_v3", "sensitivity": "normal"},
        )
        self.assertEqual(stale.status_code, 200)
        self.assertTrue(stale.get_json()["stale"])
        self.assertIsNone(stale.get_json()["entry"])

    def test_image_suggestion_route_accepts_session_dataset_prefix(self):
        context_suggestions._atomic_json(
            context_suggestions.cache_path(self.project),
            {
                "schema_version": context_suggestions.CACHE_SCHEMA_VERSION,
                "entries": {
                    "nested/sample.png": {
                        "config": {"model_profile": "caformer_s36_dbv4", "sensitivity": "normal"},
                        "segments": {"expression": [{"tag": "smile", "score": 0.9}]},
                    }
                },
            },
        )

        response = self.client.post(
            "/api/tagging-quiz/suggestions/image",
            json={
                "project_root": str(self.project),
                "image_rel": "dataset\\nested\\sample.png",
                "model_profile": "caformer_s36_dbv4",
                "sensitivity": "normal",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["ready"])
        self.assertEqual(data["entry"]["segments"]["expression"][0]["tag"], "smile")

    def test_local_install_route_returns_actionable_service_result(self):
        with patch("app.tagger_model_manager.install_from_local", return_value={"ok": False, "error": "missing model.onnx"}):
            response = self.client.post(
                "/api/tagger-models/install-local",
                json={"model_profile": "caformer_s36_dbv4", "source_folder": str(self.project)},
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("model.onnx", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
