import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
import re

from PIL import Image

from services import offline_tagger
from services import normalizer


class OfflineTaggerTests(unittest.TestCase):
    def _opts(self, **overrides):
        base = offline_tagger.TaggerOptions(
            dataset_path=Path("."),
            recursive=False,
            image_exts=[".jpg"],
            model_id="model",
            device="cpu",
            batch_size=1,
            general_threshold=0.35,
            character_threshold=0.75,
            threshold_mode="fixed",
            min_threshold_floor=0.2,
            include_general=True,
            include_character=False,
            include_rating=False,
            include_meta=False,
            include_copyright=False,
            include_artist=False,
            replace_underscore=False,
            write_mode="overwrite",
            preview_only=True,
            preview_limit=0,
            limit=0,
            max_tags=0,
            max_general_tags=30,
            max_character_tags=5,
            max_meta_tags=10,
            character_topk=0,
            skip_empty=False,
            local_only=True,
            exclude_tags=[],
            exclude_regex=[],
            use_normalizer_remove_as_exclude=False,
            backend="transformers",
            use_amp=False,
            trigger_tag="",
            dedupe=True,
            sort_tags=True,
            keep_existing_tags=True,
            newline_end=True,
            strip_whitespace=True,
            force_wd_bgr_fix=True,
            general_category_id=None,
            character_category_id=None,
            rating_category_id=None,
            normalizer_preset_root=None,
            normalizer_preset_type="",
            normalizer_preset_file="",
            enable_color_sanity=True,
            color_ratio_threshold=0.006,
            color_min_saturation=0.2,
            color_min_value=0.15,
            color_keep_if_score_ge=0.92,
            color_downscale=256,
            debug_color_sanity=False,
        )
        return replace(base, **overrides)

    def test_build_tags_filters_and_budgets(self):
        labels = ["g1", "g2", "char_a", "meta_tag", "rating:safe"]
        categories = [0, 0, 3, 4, 9]
        probs = [0.9, 0.8, 0.9, 0.7, 0.95]
        opts = self._opts(
            include_general=True,
            include_character=True,
            include_meta=True,
            include_rating=True,
            max_general_tags=1,
            max_character_tags=1,
            max_meta_tags=1,
        )
        category_ids = offline_tagger.CategoryIds(general=0, character=3, meta=4, rating=9)
        tags = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            opts,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
        )
        self.assertEqual(tags[0], "rating:safe")
        self.assertIn("g1", tags)
        self.assertIn("char_a", tags)
        self.assertIn("meta_tag", tags)
        self.assertEqual(len(tags), 4)

    def test_excludes_apply(self):
        labels = ["g1", "g2"]
        categories = [0, 0]
        probs = [0.9, 0.8]
        opts = self._opts(include_general=True)
        category_ids = offline_tagger.CategoryIds(general=0)
        tags = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            opts,
            category_ids,
            exclude_tags={"g1"},
            exclude_regex=[re.compile(r"^g2$", flags=re.IGNORECASE)],
        )
        self.assertEqual(tags, [])

    def test_color_presence_basic(self):
        opts = self._opts()
        blue = Image.new("RGB", (32, 32), color=(0, 0, 255))
        red = Image.new("RGB", (32, 32), color=(255, 0, 0))
        gray = Image.new("RGB", (32, 32), color=(128, 128, 128))

        blue_presence = offline_tagger._estimate_color_presence(blue, opts)
        self.assertGreater(blue_presence["blue"], 0.8)
        self.assertLess(blue_presence["red"], 0.05)

        red_presence = offline_tagger._estimate_color_presence(red, opts)
        self.assertGreater(red_presence["red"], 0.8)
        self.assertLess(red_presence["blue"], 0.05)

        gray_presence = offline_tagger._estimate_color_presence(gray, opts)
        self.assertGreater(gray_presence["gray"], 0.8)
        self.assertLess(gray_presence["blue"], 0.05)

    def test_color_tag_gating(self):
        labels = ["blue hair"]
        categories = [0]
        opts = self._opts(enable_color_sanity=True, color_ratio_threshold=0.01, color_keep_if_score_ge=0.92)
        category_ids = offline_tagger.CategoryIds(general=0)

        tags = offline_tagger._build_tags(
            [0.7],
            labels,
            categories,
            opts,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
            color_presence={"blue": 0.0},
        )
        self.assertEqual(tags, [])

        tags = offline_tagger._build_tags(
            [0.97],
            labels,
            categories,
            opts,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
            color_presence={"blue": 0.0},
        )
        self.assertEqual(tags, ["blue hair"])

    def test_non_color_tags_untouched(self):
        labels = ["blue sky"]
        categories = [0]
        opts = self._opts(enable_color_sanity=True, color_ratio_threshold=0.01)
        category_ids = offline_tagger.CategoryIds(general=0)
        tags = offline_tagger._build_tags(
            [0.7],
            labels,
            categories,
            opts,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
            color_presence={"blue": 0.0},
        )
        self.assertEqual(tags, ["blue sky"])

    def test_trigger_tag_pinned(self):
        tags = offline_tagger._apply_trigger_tag(["b", "trigger", "a", "trigger"], "trigger")
        self.assertEqual(tags[0], "trigger")
        self.assertEqual(tags.count("trigger"), 1)

        tags = offline_tagger._apply_trigger_tag([], "trigger")
        self.assertEqual(tags, ["trigger"])

    def test_normalizer_keeps_trigger_first(self):
        preset = {"rules": {"trim": True, "dedup": True, "sort": {"enabled": True, "priority_groups": []}}}
        opts = normalizer.NormalizeOptions(
            dataset_path=Path("."),
            preset_type="x",
            preset_file="y",
            pinned_tags=["trigger"],
        )
        record = normalizer.TagFile(path=Path("x.txt"), main=["b", "trigger", "a", "trigger"], optional=[])
        after, _ = normalizer.normalize_record(record, preset, opts, {"total_files": 1, "tag_counts": Counter()})
        self.assertEqual(after.main[0], "trigger")
        self.assertEqual(after.main.count("trigger"), 1)


if __name__ == "__main__":
    unittest.main()
