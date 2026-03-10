import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import app


class TagEditorGlossaryApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_get_returns_default_when_file_missing(self):
        with tempfile.TemporaryDirectory() as td:
            glossary_path = Path(td) / "tag_editor_glossary.json"
            with patch("app.TAG_EDITOR_GLOSSARY_PATH", glossary_path):
                resp = self.client.get("/api/tags/glossary")

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("ok"))
        self.assertEqual(data["glossary"]["categories"], {"Unsorted": []})

    def test_post_normalizes_and_persists_glossary(self):
        with tempfile.TemporaryDirectory() as td:
            glossary_path = Path(td) / "tag_editor_glossary.json"
            payload = {
                "glossary": {
                    "version": 99,
                    "categories": {
                        "Favorites": ["Long Hair", "long_hair", "cat girl"],
                        "": ["ignored"]
                    },
                    "updated_at": "42"
                }
            }
            with patch("app.TAG_EDITOR_GLOSSARY_PATH", glossary_path):
                resp = self.client.post("/api/tags/glossary", json=payload)

                self.assertEqual(resp.status_code, 200)
                data = resp.get_json()
                self.assertTrue(data.get("ok"))
                self.assertEqual(
                    data["glossary"]["categories"]["Favorites"],
                    ["long_hair", "cat_girl"],
                )
                self.assertIn("Unsorted", data["glossary"]["categories"])

                on_disk = json.loads(glossary_path.read_text(encoding="utf-8"))
                self.assertEqual(on_disk, data["glossary"])


if __name__ == "__main__":
    unittest.main()
