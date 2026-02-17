import unittest
from unittest.mock import patch

from services import danbooru_client


class DanbooruClientTests(unittest.TestCase):
    def setUp(self):
        danbooru_client.clear_cache()

    def test_lookup_success_then_cache_hit(self):
        def fake_get(path, params, timeout_seconds):  # noqa: ARG001
            if path == "/tags.json":
                return (
                    [
                        {
                            "name": "long_hair",
                            "category": 0,
                            "post_count": 1234,
                        }
                    ],
                    None,
                )
            if path == "/wiki_pages.json":
                return ([{"title": "long_hair", "body": "Wiki body"}], None)
            if path == "/related_tag.json":
                return ({"tags": [["short_hair", 0.8], ["ponytail", 0.6]]}, None)
            if path == "/posts.json":
                return (
                    [
                        {
                            "id": 99,
                            "preview_file_url": "/preview/abc.jpg",
                            "large_file_url": "/sample/abc.jpg",
                            "rating": "s",
                            "image_width": 1024,
                            "image_height": 768,
                            "file_ext": "jpg",
                        }
                    ],
                    None,
                )
            return (None, "unexpected")

        with patch("services.danbooru_client._http_get_json", side_effect=fake_get) as mocked:
            first = danbooru_client.lookup_tag_info(" Long Hair ", include_related=True)
            second = danbooru_client.lookup_tag_info("long_hair", include_related=True)

        self.assertTrue(first.get("ok"))
        self.assertEqual(first.get("tag"), "long_hair")
        self.assertTrue(first.get("found"))
        self.assertFalse(first.get("cached"))
        self.assertEqual(first.get("related"), ["short_hair", "ponytail"])
        self.assertEqual(len(first.get("previews") or []), 1)
        self.assertEqual((first.get("previews") or [])[0]["id"], 99)

        self.assertTrue(second.get("ok"))
        self.assertTrue(second.get("cached"))
        self.assertEqual(mocked.call_count, 4)

    def test_lookup_not_found_when_primary_lookups_are_empty(self):
        def fake_get(path, params, timeout_seconds):  # noqa: ARG001
            if path == "/related_tag.json":
                return ({"tags": []}, None)
            if path == "/posts.json":
                return ([], None)
            return ([], None)

        with patch("services.danbooru_client._http_get_json", side_effect=fake_get):
            out = danbooru_client.lookup_tag_info("unknown_tag", include_related=True)

        self.assertTrue(out.get("ok"))
        self.assertFalse(out.get("found"))
        self.assertEqual(out.get("related"), [])
        self.assertEqual(out.get("previews"), [])

    def test_lookup_returns_error_when_primary_lookup_fails(self):
        def fake_get(path, params, timeout_seconds):  # noqa: ARG001
            if path == "/tags.json":
                return (None, "upstream down")
            if path == "/wiki_pages.json":
                return ([], None)
            if path == "/posts.json":
                return ([], None)
            return ({"tags": []}, None)

        with patch("services.danbooru_client._http_get_json", side_effect=fake_get):
            out = danbooru_client.lookup_tag_info("long_hair", include_related=True)

        self.assertFalse(out.get("ok"))
        self.assertEqual(out.get("error_code"), "fetch_failed")
        self.assertIn("upstream", out.get("error", ""))

    def test_lookup_can_disable_preview_fetch(self):
        calls = []

        def fake_get(path, params, timeout_seconds):  # noqa: ARG001
            calls.append(path)
            if path == "/tags.json":
                return ([{"name": "long_hair", "category": 0, "post_count": 5}], None)
            if path == "/wiki_pages.json":
                return ([{"title": "long_hair", "body": "ok"}], None)
            if path == "/related_tag.json":
                return ({"tags": []}, None)
            if path == "/posts.json":
                return ([{"id": 1, "preview_file_url": "/x.jpg"}], None)
            return ([], None)

        with patch("services.danbooru_client._http_get_json", side_effect=fake_get):
            out = danbooru_client.lookup_tag_info("long_hair", include_related=True, include_preview=False)

        self.assertTrue(out.get("ok"))
        self.assertEqual(out.get("previews"), [])
        self.assertNotIn("/posts.json", calls)


if __name__ == "__main__":
    unittest.main()
