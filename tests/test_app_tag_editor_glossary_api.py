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
                        "Other": ["long__hair", "solo tag"],
                        "": ["ignored"]
                    },
                    "tag_meta": {
                        "long_hair": {
                            "found": True,
                            "post_count": "1234",
                            "category": "0",
                            "category_name": "general",
                            "fetched_at": "100",
                        },
                        "not_in_glossary": {"post_count": 9},
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
                self.assertEqual(data["glossary"]["categories"]["Other"], ["solo_tag"])
                self.assertIn("Unsorted", data["glossary"]["categories"])
                self.assertEqual(data["glossary"]["categories"]["Unsorted"], ["ignored"])
                self.assertEqual(data["glossary"]["version"], 2)
                self.assertEqual(data["glossary"]["tag_meta"]["long_hair"]["post_count"], 1234)
                self.assertNotIn("not_in_glossary", data["glossary"]["tag_meta"])

                on_disk = json.loads(glossary_path.read_text(encoding="utf-8"))
                self.assertEqual(on_disk, data["glossary"])

    def test_post_does_not_overwrite_newer_glossary(self):
        with tempfile.TemporaryDirectory() as td:
            glossary_path = Path(td) / "tag_editor_glossary.json"
            glossary_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "categories": {"Unsorted": ["newer_tag"]},
                        "updated_at": 100,
                    }
                ),
                encoding="utf-8",
            )
            with patch("app.TAG_EDITOR_GLOSSARY_PATH", glossary_path):
                resp = self.client.post(
                    "/api/tags/glossary",
                    json={
                        "glossary": {
                            "version": 1,
                            "categories": {"Unsorted": ["stale_tag"]},
                            "updated_at": 99,
                        }
                    },
                )

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("ok"))
        self.assertEqual(data["glossary"]["categories"]["Unsorted"], ["newer_tag"])


if __name__ == "__main__":
    unittest.main()
