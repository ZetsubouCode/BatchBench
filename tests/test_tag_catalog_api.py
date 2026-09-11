import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import app
from services import tag_catalog


class TagCatalogApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def _patch_paths(self, root: Path):
        return patch.multiple(
            tag_catalog,
            CATALOG_ROOT=root,
            CSV_PATH=root / "danbooru_tags.csv",
            DB_PATH=root / "danbooru_tags.sqlite3",
            STATE_PATH=root / "catalog_state.json",
            STAGING_ROOT=root / "_staging",
            SETTINGS_PATH=root / "settings.json",
        )

    def _install_catalog(self, root: Path):
        source = root / "source.csv"
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tag_catalog.CSV_FIELDS)
            writer.writeheader()
            writer.writerow({"id": 1, "name": "from_below", "category": 0, "category_name": "general", "post_count": 245000, "is_deprecated": "false", "updated_at": ""})
            writer.writerow({"id": 2, "name": "from_behind", "category": 0, "category_name": "general", "post_count": 180000, "is_deprecated": "false", "updated_at": ""})
        result = tag_catalog.import_csv(source)
        self.assertTrue(result["ok"], msg=result)

    def test_suggest_clamps_limit_and_hides_existing_tags(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self._patch_paths(root):
                self._install_catalog(root)
                tag_catalog.save_settings({"enabled": True, "max_suggestions": 30})
                resp = self.client.post(
                    "/api/tag-catalog/suggest",
                    json={"query": "from_b", "section": "manual_tags", "limit": 500, "existing_tags": ["from_behind"]},
                )

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["enabled"])
        self.assertEqual([row["tag"] for row in data["suggestions"]], ["from_below"])

    def test_suggest_disabled_section_returns_no_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self._patch_paths(root):
                self._install_catalog(root)
                tag_catalog.save_settings({"enabled": True, "sections": {"manual_tags": {"enabled": False}}})
                resp = self.client.post("/api/tag-catalog/suggest", json={"query": "from_b", "section": "manual_tags"})

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertFalse(data["enabled"])
        self.assertEqual(data["suggestions"], [])

    def test_guided_step_must_enable_danbooru_autosuggest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = {
                "quiz_review": {
                    "steps": [
                        {
                            "id": "camera_angle",
                            "label": "Camera Angle",
                            "danbooru_autosuggest": False,
                            "tags": ["from_below"],
                        }
                    ]
                }
            }
            with self._patch_paths(root), patch("app.review_quiz.load_review_quiz_config", return_value=config):
                self._install_catalog(root)
                tag_catalog.save_settings({"enabled": True})
                disabled = self.client.post("/api/tag-catalog/suggest", json={"query": "from_b", "section": "camera_angle"})
                config["quiz_review"]["steps"][0]["danbooru_autosuggest"] = True
                enabled = self.client.post("/api/tag-catalog/suggest", json={"query": "from_b", "section": "camera_angle"})

        self.assertFalse(disabled.get_json()["enabled"])
        self.assertTrue(enabled.get_json()["enabled"])
        self.assertEqual(enabled.get_json()["suggestions"][0]["tag"], "from_below")

    def test_guided_step_autosuggest_works_when_global_catalog_switch_is_off(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = {
                "quiz_review": {
                    "steps": [
                        {
                            "id": "camera_angle",
                            "label": "Camera Angle",
                            "danbooru_autosuggest": True,
                            "tags": ["from_below"],
                        }
                    ]
                }
            }
            with self._patch_paths(root), patch("app.review_quiz.load_review_quiz_config", return_value=config):
                self._install_catalog(root)
                tag_catalog.save_settings({"enabled": False})
                resp = self.client.post("/api/tag-catalog/suggest", json={"query": "from_b", "section": "camera_angle"})

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["enabled"])
        self.assertEqual(data["suggestions"][0]["tag"], "from_below")

    def test_suggest_rejects_oversized_query(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self._patch_paths(root):
                resp = self.client.post("/api/tag-catalog/suggest", json={"query": "x" * 121})

        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.get_json()["ok"])

    def test_settings_api_persists(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self._patch_paths(root):
                post = self.client.post("/api/tag-suggestion-settings", json={"enabled": True, "min_post_count": 0})
                get = self.client.get("/api/tag-suggestion-settings")

        self.assertEqual(post.status_code, 200)
        self.assertTrue(post.get_json()["settings"]["enabled"])
        self.assertEqual(get.get_json()["settings"]["min_post_count"], 0)


if __name__ == "__main__":
    unittest.main()
