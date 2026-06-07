import unittest
from unittest.mock import patch

from app import app


class DanbooruApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_api_returns_success_payload(self):
        payload = {
            "ok": True,
            "tag": "long_hair",
            "found": True,
            "cached": False,
            "info": {"name": "long_hair", "category": 0, "category_name": "general", "post_count": 10},
            "wiki": {"title": "long_hair", "body": "desc"},
            "related": ["short_hair"],
            "error": "",
        }
        with patch("app.danbooru_client.lookup_tag_info", return_value=payload):
            resp = self.client.post("/api/danbooru/taginfo", json={"tag": "long_hair", "include_related": True})

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("ok"))
        self.assertEqual(data.get("tag"), "long_hair")

    def test_api_maps_invalid_input_to_400_and_hides_error_code(self):
        payload = {
            "ok": False,
            "tag": "",
            "found": False,
            "cached": False,
            "info": None,
            "wiki": None,
            "related": [],
            "error": "tag is required",
            "error_code": "invalid_input",
        }
        with patch("app.danbooru_client.lookup_tag_info", return_value=payload):
            resp = self.client.post("/api/danbooru/taginfo", json={"tag": "   "})

        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data.get("ok"))
        self.assertNotIn("error_code", data)

    def test_api_maps_upstream_failure_to_502(self):
        payload = {
            "ok": False,
            "tag": "long_hair",
            "found": False,
            "cached": False,
            "info": None,
            "wiki": None,
            "related": [],
            "error": "Danbooru request failed",
            "error_code": "fetch_failed",
        }
        with patch("app.danbooru_client.lookup_tag_info", return_value=payload):
            resp = self.client.post("/api/danbooru/taginfo", json={"tag": "long_hair"})

        self.assertEqual(resp.status_code, 502)
        data = resp.get_json()
        self.assertFalse(data.get("ok"))

    def test_api_forwards_preview_options(self):
        payload = {
            "ok": True,
            "tag": "long_hair",
            "found": True,
            "cached": False,
            "info": None,
            "wiki": None,
            "related": [],
            "previews": [],
            "error": "",
        }
        with patch("app.danbooru_client.lookup_tag_info", return_value=payload) as mocked:
            resp = self.client.post(
                "/api/danbooru/taginfo",
                json={"tag": "long_hair", "include_related": False, "include_preview": True, "preview_limit": 12},
            )

        self.assertEqual(resp.status_code, 200)
        mocked.assert_called_once_with(
            "long_hair",
            include_related=False,
            include_preview=True,
            preview_limit=12,
        )

    def test_summary_api_forwards_tags(self):
        payload = {
            "ok": True,
            "summaries": {"long_hair": {"name": "long_hair", "post_count": 12}},
            "errors": {},
        }
        with patch("app.danbooru_client.lookup_tag_summaries", return_value=payload) as mocked:
            resp = self.client.post("/api/danbooru/tag-summaries", json={"tags": ["long_hair"]})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), payload)
        mocked.assert_called_once_with(["long_hair"])

    def test_wiki_api_forwards_preview_limit(self):
        payload = {
            "ok": True,
            "tag": "long_hair",
            "found": True,
            "cached": False,
            "info": None,
            "wiki": None,
            "related": [],
            "previews": [],
            "error": "",
        }
        with patch("app.danbooru_client.lookup_tag_wiki", return_value=payload) as mocked:
            resp = self.client.post(
                "/api/danbooru/tag-wiki",
                json={"tag": "long_hair", "preview_limit": 15},
            )

        self.assertEqual(resp.status_code, 200)
        mocked.assert_called_once_with("long_hair", preview_limit=15)


if __name__ == "__main__":
    unittest.main()
