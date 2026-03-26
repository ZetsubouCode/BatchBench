import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
import re
from unittest.mock import patch

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
            mcut_relax_general=0.08,
            mcut_relax_character=0.02,
            mcut_relax_meta=0.05,
            mcut_min_general_tags=8,
            mcut_min_character_tags=0,
            mcut_min_meta_tags=0,
            output_profile="standard_full",
            selective_keep_background_place=True,
            selective_keep_object_prop=True,
            selective_keep_pose_action=True,
            selective_keep_appearance=False,
            selective_keep_clothing=False,
            selective_keep_character_names=False,
            selective_keep_rating_meta=False,
            selective_keep_unknown_general=False,
            tag_focus_mode="all",
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
            non_character_regex=[],
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
            danbooru_safenet=False,
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

    def test_mcut_tuning_keeps_min_general_tags(self):
        labels = ["g1", "g2", "g3", "g4"]
        categories = [0, 0, 0, 0]
        probs = [0.95, 0.84, 0.83, 0.82]
        category_ids = offline_tagger.CategoryIds(general=0)

        strict = self._opts(
            threshold_mode="mcut",
            mcut_relax_general=0.0,
            mcut_min_general_tags=0,
            include_general=True,
        )
        strict_tags = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            strict,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
        )
        self.assertEqual(strict_tags, ["g1"])

        tuned = self._opts(
            threshold_mode="mcut",
            mcut_relax_general=0.0,
            mcut_min_general_tags=3,
            include_general=True,
        )
        tuned_tags = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            tuned,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
        )
        self.assertEqual(tuned_tags, ["g1", "g2", "g3"])

    def test_tag_focus_mode_character_vs_non_character(self):
        labels = ["1girl", "school uniform", "classroom", "window", "char_a"]
        categories = [0, 0, 0, 0, 3]
        probs = [0.92, 0.9, 0.88, 0.86, 0.95]
        category_ids = offline_tagger.CategoryIds(general=0, character=3)

        char_only = self._opts(
            include_general=True,
            include_character=True,
            tag_focus_mode="character",
            non_character_regex=[r"classroom", r"window"],
        )
        tags_char_only = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            char_only,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
        )
        self.assertIn("char_a", tags_char_only)
        self.assertIn("school uniform", tags_char_only)
        self.assertNotIn("classroom", tags_char_only)
        self.assertNotIn("window", tags_char_only)

        non_char_only = self._opts(
            include_general=True,
            include_character=False,
            tag_focus_mode="non_character",
            non_character_regex=[r"classroom", r"window"],
        )
        tags_non_char = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            non_char_only,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
        )
        self.assertIn("classroom", tags_non_char)
        self.assertIn("window", tags_non_char)
        self.assertNotIn("char_a", tags_non_char)

    def test_trigger_tag_pinned(self):
        tags = offline_tagger._apply_trigger_tag(["b", "trigger", "a", "trigger"], "trigger")
        self.assertEqual(tags[0], "trigger")
        self.assertEqual(tags.count("trigger"), 1)

        tags = offline_tagger._apply_trigger_tag([], "trigger")
        self.assertEqual(tags, ["trigger"])

    def test_effective_skip_empty_disables_for_overwrite(self):
        self.assertTrue(offline_tagger._effective_skip_empty("append", True))
        self.assertTrue(offline_tagger._effective_skip_empty("skip", True))
        self.assertFalse(offline_tagger._effective_skip_empty("overwrite", True))
        self.assertFalse(offline_tagger._effective_skip_empty("overwrite", False))

    def test_is_literal_overwrite(self):
        self.assertTrue(offline_tagger._is_literal_overwrite("overwrite"))
        self.assertTrue(offline_tagger._is_literal_overwrite(" Overwrite "))
        self.assertFalse(offline_tagger._is_literal_overwrite("append"))

    def test_cache_lookup_model_bundle_prefers_exact(self):
        offline_tagger._MODEL_CACHE.clear()
        try:
            bundle_exact = {"backend": "transformers", "device": "cpu", "model": object()}
            bundle_other = {"backend": "transformers", "device": "cuda", "model": object()}
            offline_tagger._MODEL_CACHE[("repo", "cpu", "transformers")] = bundle_exact
            offline_tagger._MODEL_CACHE[("repo", "cuda", "transformers")] = bundle_other
            got = offline_tagger._cache_lookup_model_bundle("repo", "cpu", "transformers")
            self.assertIs(got, bundle_exact)
        finally:
            offline_tagger._MODEL_CACHE.clear()

    def test_cache_lookup_model_bundle_resolves_auto_alias(self):
        offline_tagger._MODEL_CACHE.clear()
        try:
            bundle_cpu = {"backend": "transformers", "device": "cpu", "model": object()}
            offline_tagger._MODEL_CACHE[("repo", "cpu", "transformers")] = bundle_cpu
            got = offline_tagger._cache_lookup_model_bundle("repo", "auto", "transformers")
            self.assertIs(got, bundle_cpu)
        finally:
            offline_tagger._MODEL_CACHE.clear()

    def test_classify_general_tag_priority(self):
        self.assertEqual(
            offline_tagger.classify_general_tag("blue_sky"),
            offline_tagger.BUCKET_BACKGROUND_PLACE,
        )
        self.assertEqual(
            offline_tagger.classify_general_tag("blue_hair"),
            offline_tagger.BUCKET_APPEARANCE_IDENTITY,
        )
        self.assertEqual(
            offline_tagger.classify_general_tag("hair_ribbon"),
            offline_tagger.BUCKET_CLOTHING_OUTFIT,
        )
        self.assertEqual(
            offline_tagger.classify_general_tag("holding_sword"),
            offline_tagger.BUCKET_OBJECT_PROP,
        )

    def test_output_profile_background_pose_only(self):
        labels = [
            "forest",
            "blue_sky",
            "sword",
            "standing",
            "arms_up",
            "blue_hair",
            "dress",
            "char_a",
            "meta_tag",
            "rating:safe",
        ]
        categories = [0, 0, 0, 0, 0, 0, 0, 3, 4, 9]
        probs = [0.95, 0.93, 0.92, 0.91, 0.9, 0.89, 0.88, 0.94, 0.93, 0.99]
        opts = self._opts(
            output_profile="background_pose_only",
            include_general=True,
            include_character=True,
            include_meta=True,
            include_rating=True,
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
        self.assertIn("forest", tags)
        self.assertIn("blue_sky", tags)
        self.assertIn("sword", tags)
        self.assertIn("standing", tags)
        self.assertIn("arms_up", tags)
        self.assertNotIn("blue_hair", tags)
        self.assertNotIn("dress", tags)
        self.assertNotIn("char_a", tags)
        self.assertNotIn("meta_tag", tags)
        self.assertNotIn("rating:safe", tags)

    def test_output_profile_custom_selective_appearance_on_clothing_off(self):
        labels = ["forest", "standing", "blue_hair", "dress"]
        categories = [0, 0, 0, 0]
        probs = [0.95, 0.9, 0.89, 0.88]
        opts = self._opts(
            output_profile="custom_selective",
            selective_keep_background_place=True,
            selective_keep_object_prop=False,
            selective_keep_pose_action=True,
            selective_keep_appearance=True,
            selective_keep_clothing=False,
        )
        category_ids = offline_tagger.CategoryIds(general=0)
        tags = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            opts,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
        )
        self.assertIn("blue_hair", tags)
        self.assertIn("forest", tags)
        self.assertIn("standing", tags)
        self.assertNotIn("dress", tags)

    def test_output_profile_custom_selective_background_only(self):
        labels = ["forest", "sword", "standing", "blue_hair", "dress"]
        categories = [0, 0, 0, 0, 0]
        probs = [0.95, 0.94, 0.93, 0.92, 0.91]
        opts = self._opts(
            output_profile="custom_selective",
            selective_keep_background_place=True,
            selective_keep_object_prop=False,
            selective_keep_pose_action=False,
            selective_keep_appearance=False,
            selective_keep_clothing=False,
        )
        category_ids = offline_tagger.CategoryIds(general=0)
        tags = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            opts,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
        )
        self.assertEqual(tags, ["forest"])

    def test_selective_sparse_unknown_fallback_keeps_top_unknown(self):
        labels = ["unknown_a", "unknown_b", "unknown_c"]
        categories = [0, 0, 0]
        probs = [0.95, 0.85, 0.75]
        opts = self._opts(output_profile="background_pose_only")
        category_ids = offline_tagger.CategoryIds(general=0)
        debug_state = {}
        tags = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            opts,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
            debug_state=debug_state,
        )
        self.assertEqual(tags, ["unknown_a", "unknown_b"])
        self.assertEqual(debug_state.get("unknown_fallback_kept"), 2)

    def test_selective_sparse_unknown_fallback_not_used_when_already_dense(self):
        labels = ["forest", "standing", "unknown_a"]
        categories = [0, 0, 0]
        probs = [0.95, 0.92, 0.9]
        opts = self._opts(output_profile="background_pose_only")
        category_ids = offline_tagger.CategoryIds(general=0)
        tags = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            opts,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
        )
        self.assertEqual(tags, ["forest", "standing"])

    def test_output_profile_safenet_rescues_unknown_background(self):
        labels = ["forest", "standing", "castle_ruins"]
        categories = [0, 0, 0]
        probs = [0.97, 0.96, 0.95]
        category_ids = offline_tagger.CategoryIds(general=0)

        opts_off = self._opts(
            output_profile="background_pose_only",
            danbooru_safenet=False,
        )
        tags_off = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            opts_off,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
        )
        self.assertEqual(tags_off, ["forest", "standing"])

        opts_on = self._opts(
            output_profile="background_pose_only",
            danbooru_safenet=True,
        )
        state = offline_tagger.DanbooruSafeNetState(enabled=True, max_lookups=5)
        with patch(
            "services.offline_tagger._lookup_danbooru_bucket",
            return_value=offline_tagger.BUCKET_BACKGROUND_PLACE,
        ):
            tags_on = offline_tagger._build_tags(
                probs,
                labels,
                categories,
                opts_on,
                category_ids,
                exclude_tags=set(),
                exclude_regex=[],
                danbooru_safenet_state=state,
            )
        self.assertEqual(tags_on, ["forest", "standing", "castle_ruins"])
        self.assertEqual(state.lookups, 1)
        self.assertEqual(state.resolved, 1)

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
