import tempfile
import unittest
from pathlib import Path

from services import danbooru_client, normalizer, offline_tagger, tag_editor, tag_policy
from utils.tags import (
    normalize_caption_tags,
    tag_compare_key,
    to_caption_tag,
    to_danbooru_tag,
)


class TagRepresentationTests(unittest.TestCase):
    def test_shared_conversion_and_mixed_dedupe(self):
        self.assertEqual(to_caption_tag("long_hair"), "long hair")
        self.assertEqual(to_caption_tag("long   hair"), "long hair")
        self.assertEqual(to_caption_tag("long__hair"), "long hair")
        self.assertEqual(to_danbooru_tag("long hair"), "long_hair")
        self.assertEqual(tag_compare_key("long_hair"), tag_compare_key("long hair"))
        self.assertEqual(normalize_caption_tags(["long_hair", "long hair"]), ["long hair"])

    def test_protected_trigger_remains_literal(self):
        self.assertEqual(
            normalize_caption_tags(
                ["my_trigger_token", "long_hair"],
                protected_literals=["my_trigger_token"],
            ),
            ["my_trigger_token", "long hair"],
        )

    def test_tag_editor_loads_legacy_and_saves_caption_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            caption = Path(tmp) / "sample.txt"
            caption.write_text("long_hair, blue_eyes, long hair", encoding="utf-8")
            self.assertEqual(tag_editor._read_text_tags(caption), ["long hair", "blue eyes"])
            result = tag_editor.add_tags(caption, ["cowboy_shot"], backup=False)
            self.assertTrue(result["ok"])
            self.assertEqual(caption.read_text(encoding="utf-8"), "long hair, blue eyes, cowboy shot")

    def test_tag_editor_save_preserves_configured_trigger_literal(self):
        with tempfile.TemporaryDirectory() as tmp:
            caption = Path(tmp) / "sample.txt"
            caption.write_text("my_trigger_token, long_hair", encoding="utf-8")
            result = tag_editor.add_tags(
                caption,
                ["blue_eyes"],
                backup=False,
                protected_literals=["my_trigger_token"],
            )
            self.assertTrue(result["ok"])
            self.assertEqual(
                caption.read_text(encoding="utf-8"),
                "my_trigger_token, long hair, blue eyes",
            )

    def test_normalizer_matches_underscore_rules_and_migrates_output(self):
        record = normalizer.TagFile(
            path=Path("sample.txt"),
            main=["looking at viewer", "long hair", "long hair"],
            original_text="looking_at_viewer, long_hair, long hair\n",
        )
        preset = {"rules": {"trim": True, "dedup": True, "remove_tags": ["looking_at_viewer"]}}
        after, meta = normalizer.normalize_record(
            record,
            preset,
            normalizer.NormalizeOptions(dataset_path=Path(".")),
            {"total_files": 1, "tag_counts": {}},
        )
        self.assertEqual(after.main, ["long hair"])
        self.assertEqual(meta["after"], "long hair\n")
        self.assertTrue(meta["changed"])

    def test_policy_and_auto_tag_boundary_accept_both_forms(self):
        profile = tag_policy.get_profile(tag_policy.DEFAULT_TAG_POLICY)
        self.assertTrue(tag_policy.decide_tag(profile, "long hair", None)[0])
        merged = offline_tagger.merge_caption_tags(
            ["my_trigger_token"],
            [],
            ["looking_at_viewer", "cowboy_shot"],
        )
        self.assertEqual(merged, ["my_trigger_token", "looking at viewer", "cowboy shot"])

    def test_danbooru_boundary_uses_internal_key(self):
        self.assertEqual(danbooru_client.normalize_tag("long hair"), "long_hair")


if __name__ == "__main__":
    unittest.main()
