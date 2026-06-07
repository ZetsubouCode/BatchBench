import tempfile
import unittest
from pathlib import Path

from app import app


class IndexFlashTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def _empty_tag_editor_project(self, base: Path) -> Path:
        project = base / "project"
        (project / "database").mkdir(parents=True, exist_ok=True)
        (project / "dataset" / "_temp").mkdir(parents=True, exist_ok=True)
        (project / "prompt.txt").write_text("tag", encoding="utf-8")
        return project

    def test_ajax_tag_editor_error_does_not_create_global_flash(self):
        with tempfile.TemporaryDirectory() as td:
            project = self._empty_tag_editor_project(Path(td))

            response = self.client.post(
                "/",
                data={
                    "tool": "tags",
                    "folder": str(project),
                    "mode": "insert",
                    "tags": "cat",
                    "exts": ".png",
                },
                headers={"X-Requested-With": "fetch"},
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.get_json()["error"], "No image files found in _temp.")

            page = self.client.get("/?tab=guide")
            self.assertNotIn(b"No image files found in _temp.", page.data)

    def test_redirected_tag_editor_error_flash_is_scoped_to_tags_tab(self):
        with tempfile.TemporaryDirectory() as td:
            project = self._empty_tag_editor_project(Path(td))

            response = self.client.post(
                "/",
                data={
                    "tool": "tags",
                    "folder": str(project),
                    "mode": "insert",
                    "tags": "cat",
                    "exts": ".png",
                },
            )
            self.assertEqual(response.status_code, 302)

            page = self.client.get(response.headers["Location"])
            self.assertIn(b"No image files found in _temp.", page.data)
            self.assertIn(b'data-flash-tab="tags"', page.data)
