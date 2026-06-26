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

    def test_lookup_tag_summaries_is_lightweight_and_cached(self):
        calls = []

        def fake_get(path, params, timeout_seconds):  # noqa: ARG001
            calls.append((path, params))
            if params.get("search[name]") == "long_hair":
                return ([{"name": "long_hair", "category": 0, "post_count": 123}], None)
            return ([], None)

        with patch("services.danbooru_client._http_get_json", side_effect=fake_get):
            first = danbooru_client.lookup_tag_summaries([" Long Hair ", "missing_tag", "long_hair"])
            second = danbooru_client.lookup_tag_summaries(["long_hair", "missing_tag"])

        self.assertTrue(first.get("ok"))
        self.assertEqual(first["summaries"]["long_hair"]["post_count"], 123)
        self.assertFalse(first["summaries"]["missing_tag"]["found"])
        self.assertEqual(first["summaries"]["missing_tag"]["post_count"], 0)
        self.assertEqual(second["summaries"], first["summaries"])
        self.assertEqual([path for path, _ in calls], ["/tags.json", "/tags.json"])

    def test_fetch_tag_page_uses_danbooru_cursor_pagination(self):
        calls = []

        def fake_get(path, params, timeout_seconds):  # noqa: ARG001
            calls.append((path, dict(params)))
            return (
                [
                    {"id": 99, "name": "from_below", "category": 0, "post_count": 10, "updated_at": "2024-01-01T00:00:00Z"}
                ],
                None,
            )

        with patch("services.danbooru_client._http_get_json", side_effect=fake_get):
            rows, err = danbooru_client.fetch_tag_page(after_id=100, limit=5000)

        self.assertIsNone(err)
        self.assertEqual(rows[0]["name"], "from_below")
        self.assertEqual(calls[0][0], "/tags.json")
        self.assertEqual(calls[0][1]["page"], "b100")
        self.assertEqual(calls[0][1]["limit"], 1000)
        self.assertNotIn("search[id_gt]", calls[0][1])

    def test_wiki_lookup_adds_guidance_groups_and_relationships(self):
        def fake_get(path, params, timeout_seconds):  # noqa: ARG001
            if path == "/tags.json":
                return ([{"name": "miniskirt", "category": 0, "post_count": 100}], None)
            if path == "/wiki_pages.json":
                return (
                    [
                        {
                            "title": "miniskirt",
                            "body": "A short [[skirt]]. Use [[micro skirt]] instead for extremely short skirts.\n\nh4. See also\n\n* [[Tag Group:Attire]]",
                            "other_names": ["mini skirt"],
                        }
                    ],
                    None,
                )
            if path == "/posts.json":
                return ([{"id": 1, "preview_file_url": "/x.jpg"}], None)
            if path == "/related_tag.json":
                return (
                    {
                        "related_tags": [
                            {"tag": {"name": "pleated_skirt", "category": 0, "post_count": 10}, "frequency": 0.5}
                        ],
                        "wiki_page_tags": [{"name": "micro_skirt", "category": 0, "post_count": 5}],
                    },
                    None,
                )
            if path == "/tag_implications.json" and params.get("search[antecedent_name]") == "miniskirt":
                return ([{"antecedent_name": "miniskirt", "consequent_name": "skirt", "status": "active"}], None)
            if path == "/tag_implications.json":
                return ([], None)
            if path == "/tag_aliases.json":
                return ([], None)
            return ([], None)

        with patch("services.danbooru_client._http_get_json", side_effect=fake_get):
            out = danbooru_client.lookup_tag_wiki("miniskirt", preview_limit=4)

        self.assertTrue(out.get("ok"), msg=out)
        self.assertIn("short skirt", out.get("wiki_plain", ""))
        self.assertEqual(out.get("tag_groups"), ["Attire"])
        self.assertEqual((out.get("relationships") or {}).get("implies"), ["skirt"])
        self.assertEqual((out.get("related_details") or [])[0]["name"], "pleated_skirt")
        self.assertEqual((out.get("wiki_linked_tags") or [])[0]["name"], "micro_skirt")
        self.assertTrue((out.get("guidance") or {}).get("avoid_when"))


if __name__ == "__main__":
    unittest.main()
