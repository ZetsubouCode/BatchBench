import unittest
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
import re
import tempfile
from unittest.mock import patch

from PIL import Image

from services import offline_tagger
from services import normalizer
from services import tag_policy
from services.pipeline_workflows import compile_workflow


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
            simple_mode=False,
            tag_strength=50,
            min_threshold_floor=0.2,
            mcut_relax_general=0.08,
            mcut_relax_character=0.02,
            mcut_relax_meta=0.05,
            mcut_min_general_tags=8,
            policy_mcut_min_general_tags=0,
            mcut_min_character_tags=0,
            mcut_min_meta_tags=0,
            output_profile="standard_full",
            max_auto_tags=24,
            selective_keep_background_place=True,
            selective_keep_object_prop=True,
            selective_keep_pose_action=True,
            selective_keep_appearance=False,
            selective_keep_clothing=False,
            selective_keep_character_names=False,
            selective_keep_artist_copyright=False,
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
            tag_policy="none",
            policy_keep_tags=[],
            policy_block_tags=[],
            policy_block_regex=[],
            block_permanent_marks=False,
            replace_existing_captions=True,
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
            blocked_tags=[],
            exclude_regex=[],
            non_character_regex=[],
            use_normalizer_remove_as_exclude=False,
            backend="transformers",
            use_amp=False,
            trigger_tag="",
            prefix_tags=[],
            backup_existing=True,
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
        self.assertEqual(tags, ["blue_hair"])

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
        self.assertEqual(tags, ["blue_sky"])

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

    def test_tag_strength_maps_to_fixed_threshold(self):
        self.assertEqual(offline_tagger.tag_strength_to_threshold(0), 0.50)
        self.assertEqual(offline_tagger.tag_strength_to_threshold(50), 0.40)
        self.assertEqual(offline_tagger.tag_strength_to_threshold(100), 0.30)
        self.assertEqual(offline_tagger.tag_strength_to_threshold(-10), 0.50)
        self.assertEqual(offline_tagger.tag_strength_to_threshold(999), 0.30)

    def test_simple_effective_opts_force_fixed_guided_flow(self):
        opts = offline_tagger._effective_opts(
            {
                "folder": ".",
                "ui_mode": "simple",
                "tag_strength": "50",
                "threshold_mode": "mcut",
                "mcut_min_general_tags": "10",
                "prefix_tags": "Character Trigger, greyscale",
                "blocked_tags": "simple background",
            },
            offline_tagger.TAGGER_POLICY,
        )
        self.assertTrue(opts.simple_mode)
        self.assertEqual(opts.threshold_mode, "fixed")
        self.assertEqual(opts.general_threshold, 0.40)
        self.assertEqual(opts.output_profile, offline_tagger.OUTPUT_PROFILE_GUIDED_FLOW_STRICT)
        self.assertFalse(opts.include_character)
        self.assertFalse(opts.include_rating)
        self.assertEqual(opts.prefix_tags, ["character_trigger", "greyscale"])
        self.assertEqual(opts.blocked_tags, ["simple_background"])

    def test_simple_mode_ignores_mcut_and_unknown_fallback(self):
        labels = ["unknown_a", "unknown_b", "unknown_c"]
        categories = [0, 0, 0]
        probs = [0.95, 0.94, 0.93]
        opts = self._opts(
            simple_mode=True,
            output_profile=offline_tagger.OUTPUT_PROFILE_GUIDED_FLOW_STRICT,
            threshold_mode="mcut",
            mcut_min_general_tags=3,
            general_threshold=0.40,
            max_auto_tags=24,
        )
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
        self.assertEqual(tags, [])
        self.assertEqual(debug_state.get("unknown_fallback_kept"), 0)

    def test_guided_flow_strict_semantic_filtering(self):
        labels = [
            "forest",
            "sword",
            "standing",
            "from_below",
            "sunset",
            "blue_hair",
            "red_eyes",
            "school_uniform",
            "boots",
            "character_name",
            "rating:safe",
            "unknown_tag",
        ]
        categories = [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 9, 0]
        probs = [0.95] * len(labels)
        opts = self._opts(
            simple_mode=True,
            output_profile=offline_tagger.OUTPUT_PROFILE_GUIDED_FLOW_STRICT,
            threshold_mode="fixed",
            general_threshold=0.40,
            include_general=True,
            include_character=False,
            include_rating=False,
        )
        category_ids = offline_tagger.CategoryIds(general=0, character=3, rating=9)
        tags = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            opts,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
        )
        self.assertEqual(tags, ["forest", "sword", "standing", "from_below", "sunset"])

    def test_prefix_merge_order_and_dedupe(self):
        merged = offline_tagger.merge_caption_tags(
            ["character_trigger", "greyscale"],
            ["manual_tag", "skirt"],
            ["standing", "forest"],
        )
        self.assertEqual(
            merged,
            ["character_trigger", "greyscale", "manual_tag", "skirt", "standing", "forest"],
        )

        merged = offline_tagger.merge_caption_tags(
            ["forest", "character_trigger"],
            ["manual_tag", "forest"],
            ["standing", "manual_tag", "forest"],
        )
        self.assertEqual(merged, ["forest", "character_trigger", "manual_tag", "standing"])

    def test_blacklist_only_removes_auto_tags(self):
        auto, blocked = offline_tagger._apply_blocked_auto_tags(
            ["standing", "forest", "simple_background"],
            ["forest", "simple background"],
        )
        merged = offline_tagger.merge_caption_tags(["trigger"], ["forest", "manual_tag"], auto)
        self.assertEqual(auto, ["standing"])
        self.assertEqual(blocked, ["forest", "simple_background"])
        self.assertEqual(merged, ["trigger", "forest", "manual_tag", "standing"])

    def test_blacklist_supports_wildcards(self):
        auto, blocked = offline_tagger._apply_blocked_auto_tags(
            ["standing", "blue_hair", "rating_safe", "forest"],
            ["*_hair", "rating:*"],
        )
        self.assertEqual(auto, ["standing", "forest"])
        self.assertEqual(blocked, ["blue_hair", "rating_safe"])

    def test_simple_explicit_cleanup_controls_map_to_selective_options(self):
        opts = offline_tagger._effective_opts(
            {
                "folder": ".",
                "ui_mode": "simple",
                "min_general": "0.42",
                "remove_appearance_identity": "1",
                "remove_outfit_accessory": "0",
                "remove_character_names": "1",
                "remove_artist_copyright": "1",
                "remove_rating_meta": "0",
                "remove_unclassified": "1",
            },
            offline_tagger.TAGGER_POLICY,
        )
        self.assertTrue(opts.simple_mode)
        self.assertEqual(opts.general_threshold, 0.42)
        self.assertEqual(opts.output_profile, offline_tagger.OUTPUT_PROFILE_CUSTOM_SELECTIVE)
        self.assertFalse(opts.selective_keep_appearance)
        self.assertTrue(opts.selective_keep_clothing)
        self.assertFalse(opts.selective_keep_character_names)
        self.assertFalse(opts.selective_keep_artist_copyright)
        self.assertTrue(opts.selective_keep_rating_meta)
        self.assertFalse(opts.selective_keep_unknown_general)
        self.assertFalse(opts.include_character)
        self.assertTrue(opts.include_rating)
        self.assertTrue(opts.include_meta)
        self.assertFalse(opts.include_artist)
        self.assertFalse(opts.include_copyright)

    def test_simple_cleanup_checked_vs_unchecked_changes_auto_tags(self):
        labels = [
            "forest",
            "standing",
            "blue_hair",
            "breasts",
            "dress",
            "glasses",
            "unknown_style",
            "char_a",
            "artist_a",
            "meta_tag",
            "rating:safe",
        ]
        categories = [0, 0, 0, 0, 0, 0, 0, 3, 1, 4, 9]
        probs = [0.95] * len(labels)
        category_ids = offline_tagger.CategoryIds(general=0, character=3, artist=1, meta=4, rating=9)

        unchecked = offline_tagger._effective_opts(
            {
                "folder": ".",
                "ui_mode": "simple",
                "remove_appearance_identity": "0",
                "remove_outfit_accessory": "0",
                "remove_character_names": "0",
                "remove_artist_copyright": "0",
                "remove_rating_meta": "0",
                "remove_unclassified": "0",
            },
            offline_tagger.TAGGER_POLICY,
        )
        unchecked_tags = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            unchecked,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
        )
        self.assertIn("blue_hair", unchecked_tags)
        self.assertIn("breasts", unchecked_tags)
        self.assertIn("dress", unchecked_tags)
        self.assertIn("glasses", unchecked_tags)
        self.assertIn("unknown_style", unchecked_tags)
        self.assertIn("char_a", unchecked_tags)
        self.assertIn("artist_a", unchecked_tags)
        self.assertIn("meta_tag", unchecked_tags)
        self.assertIn("rating_safe", unchecked_tags)

        checked = offline_tagger._effective_opts(
            {
                "folder": ".",
                "ui_mode": "simple",
                "remove_appearance_identity": "1",
                "remove_outfit_accessory": "1",
                "remove_character_names": "1",
                "remove_artist_copyright": "1",
                "remove_rating_meta": "1",
                "remove_unclassified": "1",
            },
            offline_tagger.TAGGER_POLICY,
        )
        checked_debug = {}
        checked_tags = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            checked,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
            debug_state=checked_debug,
        )
        self.assertEqual(checked_tags, ["forest", "standing"])
        self.assertEqual(checked_debug.get("unknown_fallback_kept"), 0)

    def test_no_forced_tags_when_every_prediction_filtered(self):
        final_tags = offline_tagger.merge_caption_tags(
            ["character_trigger"],
            ["manual_tag"],
            [],
        )
        self.assertEqual(final_tags, ["character_trigger", "manual_tag"])

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
        self.assertIn("school_uniform", tags_char_only)
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
            offline_tagger.classify_general_tag("breasts"),
            offline_tagger.BUCKET_APPEARANCE_IDENTITY,
        )
        self.assertEqual(
            offline_tagger.classify_general_tag("glasses"),
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

    def _fake_onnx_bundle(self, labels, categories, probs):
        import numpy as np

        class Processor:
            def __call__(self, images, return_tensors):
                return {"pixel_values": np.zeros((len(images), 1), dtype=np.float32)}

        class Session:
            def run(self, _outputs, _inputs):
                clipped = np.clip(np.array([probs], dtype=np.float32), 0.001, 0.999)
                logits = np.log(clipped / (1.0 - clipped))
                return [logits]

        return {
            "model": object(),
            "backend": "onnx",
            "onnx_session": Session(),
            "onnx_input": "pixel_values",
            "processor": Processor(),
            "labels": labels,
            "categories": categories,
            "device": "cpu",
            "torch": None,
            "warn": [],
            "tag_meta_loaded": True,
            "tag_meta_count": len(labels),
        }

    def test_simple_preview_writes_no_caption_or_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            caption_path = root / "image.txt"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            caption_path.write_text("manual_tag, forest\n", encoding="utf-8")
            opts = self._opts(
                dataset_path=root,
                image_exts=[".png"],
                simple_mode=True,
                output_profile=offline_tagger.OUTPUT_PROFILE_GUIDED_FLOW_STRICT,
                preview_only=True,
                preview_limit=1,
                write_mode="append",
                prefix_tags=["character_trigger"],
                blocked_tags=["forest"],
                max_auto_tags=24,
                sort_tags=False,
            )
            bundle = self._fake_onnx_bundle(
                ["forest", "standing", "blue_hair"],
                [0, 0, 0],
                [0.95, 0.93, 0.92],
            )
            with patch("services.offline_tagger._load_model_bundle", return_value=bundle):
                ok, lines = offline_tagger.run_tagger(opts)

            self.assertTrue(ok)
            self.assertEqual(caption_path.read_text(encoding="utf-8"), "manual_tag, forest\n")
            self.assertFalse((root / ".batchbench_backup").exists())
            self.assertIn("Preview mode made no filesystem changes.", "\n".join(lines))

    def test_simple_preview_summary_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            opts = self._opts(
                dataset_path=root,
                image_exts=[".png"],
                simple_mode=True,
                output_profile=offline_tagger.OUTPUT_PROFILE_CUSTOM_SELECTIVE,
                preview_only=True,
                preview_limit=1,
                write_mode="append",
                prefix_tags=["character_trigger"],
                blocked_tags=["forest"],
                general_threshold=0.40,
                max_auto_tags=24,
                sort_tags=False,
                selective_keep_background_place=True,
                selective_keep_object_prop=True,
                selective_keep_pose_action=True,
                selective_keep_appearance=False,
                selective_keep_clothing=False,
                selective_keep_character_names=False,
                selective_keep_artist_copyright=False,
                selective_keep_rating_meta=False,
                selective_keep_unknown_general=False,
            )
            bundle = self._fake_onnx_bundle(
                ["forest", "standing", "blue_hair", "dress", "char_a", "meta_tag", "artist_a", "rating:safe"],
                [0, 0, 0, 0, 3, 4, 1, 9],
                [0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.99],
            )
            with patch("services.offline_tagger._load_model_bundle", return_value=bundle):
                ok, lines = offline_tagger.run_tagger(opts)

            output = "\n".join(lines)
            self.assertTrue(ok)
            self.assertIn("Active cleanup filters:", output)
            self.assertIn("* outfit and accessory tags", output)
            self.assertIn("Sample results:", output)
            self.assertIn("threshold: 0.40", output)
            self.assertIn("WD candidates above threshold: 8", output)
            self.assertIn("removed by appearance/identity cleanup: 1", output)
            self.assertIn("removed by outfit cleanup: 1", output)
            self.assertIn("removed by character-name cleanup: 1", output)
            self.assertIn("removed by artist/copyright cleanup: 1", output)
            self.assertIn("removed by rating/meta cleanup: 2", output)
            self.assertIn("removed by blacklist: 1", output)
            self.assertIn("final automatic tags: 1", output)

    def test_simple_apply_creates_backup_and_preserves_recursive_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            image_path = nested / "image.png"
            caption_path = nested / "image.txt"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            caption_path.write_text("manual_tag, forest\n", encoding="utf-8")
            opts = self._opts(
                dataset_path=root,
                recursive=True,
                image_exts=[".png"],
                simple_mode=True,
                output_profile=offline_tagger.OUTPUT_PROFILE_GUIDED_FLOW_STRICT,
                preview_only=False,
                preview_limit=0,
                write_mode="append",
                prefix_tags=["character_trigger"],
                blocked_tags=["forest"],
                max_auto_tags=24,
                sort_tags=False,
            )
            bundle = self._fake_onnx_bundle(
                ["forest", "standing", "blue_hair"],
                [0, 0, 0],
                [0.95, 0.93, 0.92],
            )
            with patch("services.offline_tagger._load_model_bundle", return_value=bundle):
                ok, lines = offline_tagger.run_tagger(opts)

            self.assertTrue(ok)
            self.assertEqual(
                caption_path.read_text(encoding="utf-8"),
                "character_trigger, manual_tag, forest, standing\n",
            )
            backups = list((root / ".batchbench_backup").rglob("offline_tagger_*/nested/image.txt"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), "manual_tag, forest\n")
            self.assertIn("- Backups created: 1", "\n".join(lines))

    def test_no_backup_when_caption_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            caption_path = root / "image.txt"
            caption_path.write_text("manual_tag\n", encoding="utf-8")
            changed, backup_created, backup_path = offline_tagger._write_caption_safely(
                root,
                caption_path,
                "manual_tag\n",
                backup_existing=True,
                timestamp="20260101-000000",
            )
            self.assertFalse(changed)
            self.assertFalse(backup_created)
            self.assertIsNone(backup_path)
            self.assertFalse((root / ".batchbench_backup").exists())

    def test_overwrite_mode_replaces_existing_only_when_explicit_and_forces_backup(self):
        form_opts = {
            "folder": ".",
            "ui_mode": "simple",
            "write_mode": "overwrite",
            "backup_existing": "0",
        }
        opts = offline_tagger._effective_opts(form_opts, offline_tagger.TAGGER_POLICY)
        self.assertEqual(opts.write_mode, "overwrite")
        self.assertFalse(opts.keep_existing_tags)
        self.assertTrue(opts.backup_existing)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            caption_path = root / "image.txt"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            caption_path.write_text("manual_tag, skirt\n", encoding="utf-8")
            opts = self._opts(
                dataset_path=root,
                image_exts=[".png"],
                simple_mode=True,
                output_profile=offline_tagger.OUTPUT_PROFILE_GUIDED_FLOW_STRICT,
                preview_only=False,
                preview_limit=0,
                write_mode="overwrite",
                keep_existing_tags=False,
                prefix_tags=["character_trigger"],
                max_auto_tags=24,
                sort_tags=False,
                backup_existing=True,
            )
            bundle = self._fake_onnx_bundle(["forest", "standing"], [0, 0], [0.95, 0.93])
            with patch("services.offline_tagger._load_model_bundle", return_value=bundle):
                ok, _lines = offline_tagger.run_tagger(opts)

            self.assertTrue(ok)
            self.assertEqual(caption_path.read_text(encoding="utf-8"), "character_trigger, forest, standing\n")
            backups = list((root / ".batchbench_backup").rglob("offline_tagger_*/image.txt"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), "manual_tag, skirt\n")

    def test_character_policy_removes_identity_and_keeps_controllable_tags(self):
        labels = [
            "1girl",
            "solo",
            "char_a",
            "blue_hair",
            "long_hair",
            "blue_eyes",
            "large_breasts",
            "closed_eyes",
            "half-closed_eyes",
            "wink",
            "hair_ornament",
            "smile",
            "looking_at_viewer",
            "red_dress",
            "sitting",
            "indoors",
        ]
        categories = [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        probs = [0.95] * len(labels)
        category_ids = offline_tagger.CategoryIds(general=0, character=3)
        opts = self._opts(
            tag_policy="character_identity_omitted",
            include_general=True,
            include_character=True,
            max_general_tags=0,
        )
        compiled = tag_policy.compile_policy(opts.tag_policy, labels, categories, category_ids.character)
        tags = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            opts,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
            compiled_policy=compiled,
        )
        for removed in ["1girl", "solo", "char_a", "blue_hair", "long_hair", "blue_eyes", "large_breasts"]:
            self.assertNotIn(removed, tags)
        for kept in [
            "closed_eyes",
            "half_closed_eyes",
            "wink",
            "hair_ornament",
            "smile",
            "looking_at_viewer",
            "red_dress",
            "sitting",
            "indoors",
        ]:
            self.assertIn(kept, tags)

    def test_policy_applies_before_topk_and_tag_cap(self):
        labels = ["blue_hair", "long_hair", "blue_eyes", "smile", "sitting", "looking_at_viewer", "indoors", "window"]
        categories = [0] * len(labels)
        probs = [0.99, 0.98, 0.96, 0.72, 0.68, 0.65, 0.61, 0.58]
        category_ids = offline_tagger.CategoryIds(general=0)
        opts = self._opts(
            tag_policy="character_identity_omitted",
            include_general=True,
            max_general_tags=5,
            threshold_mode="fixed",
            general_threshold=0.01,
        )
        compiled = tag_policy.compile_policy(opts.tag_policy, labels, categories, category_ids.character)
        tags = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            opts,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
            compiled_policy=compiled,
        )
        self.assertEqual(tags, ["smile", "sitting", "looking_at_viewer", "indoors", "window"])

    def test_policy_applies_before_mcut_threshold(self):
        labels = ["blue_hair", "long_hair", "blue_eyes", "smile", "sitting", "indoors"]
        categories = [0] * len(labels)
        probs = [0.99, 0.98, 0.97, 0.72, 0.68, 0.65]
        category_ids = offline_tagger.CategoryIds(general=0)
        opts = self._opts(
            tag_policy="character_identity_omitted",
            include_general=True,
            threshold_mode="mcut",
            min_threshold_floor=0.0,
            mcut_relax_general=0.0,
            mcut_min_general_tags=0,
            policy_mcut_min_general_tags=0,
        )
        compiled = tag_policy.compile_policy(opts.tag_policy, labels, categories, category_ids.character)
        tags = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            opts,
            category_ids,
            exclude_tags=set(),
            exclude_regex=[],
            compiled_policy=compiled,
        )
        self.assertEqual(tags, ["smile"])

    def test_trigger_survives_policy_exclude_and_sorting(self):
        labels = ["blue_hair", "red_dress"]
        categories = [0, 0]
        probs = [0.99, 0.8]
        category_ids = offline_tagger.CategoryIds(general=0)
        opts = self._opts(
            tag_policy="character_identity_omitted",
            trigger_tag="blue_hair",
            sort_tags=True,
            exclude_tags=["blue_hair"],
        )
        compiled = tag_policy.compile_policy(opts.tag_policy, labels, categories, category_ids.character)
        tags = offline_tagger._build_tags(
            probs,
            labels,
            categories,
            opts,
            category_ids,
            exclude_tags=set(opts.exclude_tags),
            exclude_regex=[],
            compiled_policy=compiled,
        )
        final_tags = offline_tagger._apply_trigger_tag(sorted(tags), opts.trigger_tag)
        self.assertEqual(final_tags, ["blue_hair", "red_dress"])
        self.assertEqual(tag_policy.audit_tags(final_tags, compiled, trigger_tag=opts.trigger_tag), [])

    def test_permanent_mark_toggle(self):
        labels = ["tattoo", "smile"]
        categories = [0, 0]
        category_ids = offline_tagger.CategoryIds(general=0)
        off = tag_policy.compile_policy("character_identity_omitted", labels, categories, category_ids.character)
        on = tag_policy.compile_policy(
            "character_identity_omitted",
            labels,
            categories,
            category_ids.character,
            permanent_marks=True,
        )
        self.assertFalse(off.decision_for_tag("tattoo")[0])
        self.assertTrue(on.decision_for_tag("tattoo")[0])

    def test_policy_custom_keep_and_block_overrides(self):
        labels = ["blue_hair", "smile"]
        categories = [0, 0]
        category_ids = offline_tagger.CategoryIds(general=0)
        keep = tag_policy.compile_policy(
            "character_identity_omitted",
            labels,
            categories,
            category_ids.character,
            custom_keep=["blue_hair"],
        )
        block = tag_policy.compile_policy(
            "character_identity_omitted",
            labels,
            categories,
            category_ids.character,
            custom_block=["red_dress"],
        )
        self.assertFalse(keep.decision_for_tag("blue_hair")[0])
        self.assertTrue(block.decision_for_tag("red_dress")[0])

    def test_replacement_defaults_to_enabled_and_legacy_append_accepted(self):
        opts = offline_tagger._effective_opts({"folder": "."}, offline_tagger.TAGGER_POLICY)
        self.assertTrue(opts.replace_existing_captions)
        self.assertEqual(opts.write_mode, "overwrite")
        legacy = offline_tagger._effective_opts({"folder": ".", "write_mode": "append"}, offline_tagger.TAGGER_POLICY)
        self.assertEqual(legacy.write_mode, "append")

    def test_replacement_disabled_skips_existing_caption(self):
        opts = offline_tagger._effective_opts(
            {"folder": ".", "replace_existing_captions": "0"},
            offline_tagger.TAGGER_POLICY,
        )
        self.assertFalse(opts.replace_existing_captions)
        self.assertEqual(opts.write_mode, "skip")

    def test_replacement_backup_and_write_new_caption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            caption_path = root / "image.txt"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            caption_path.write_text("old_tag\n", encoding="utf-8")
            opts = self._opts(
                dataset_path=root,
                image_exts=[".png"],
                preview_only=False,
                preview_limit=0,
                write_mode="overwrite",
                replace_existing_captions=True,
                trigger_tag="mytrigger",
                sort_tags=False,
                tag_policy="character_identity_omitted",
            )
            bundle = self._fake_onnx_bundle(["blue_hair", "smile", "indoors"], [0, 0, 0], [0.99, 0.8, 0.7])
            with patch("services.offline_tagger._load_model_bundle", return_value=bundle):
                ok, lines = offline_tagger.run_tagger(opts)
            self.assertTrue(ok)
            self.assertEqual(caption_path.read_text(encoding="utf-8"), "mytrigger, smile, indoors\n")
            backups = list((root / ".batchbench_backup").rglob("offline_tagger_*/image.txt"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), "old_tag\n")
            self.assertIn("Final policy leaks: 0", "\n".join(lines))

    def test_replacement_disabled_skips_non_empty_file_in_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            caption_path = root / "image.txt"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            caption_path.write_text("manual_tag\n", encoding="utf-8")
            opts = self._opts(
                dataset_path=root,
                image_exts=[".png"],
                preview_only=False,
                write_mode="skip",
                replace_existing_captions=False,
            )
            bundle = self._fake_onnx_bundle(["smile"], [0], [0.9])
            with patch("services.offline_tagger._load_model_bundle", return_value=bundle):
                ok, lines = offline_tagger.run_tagger(opts)
            self.assertTrue(ok)
            self.assertEqual(caption_path.read_text(encoding="utf-8"), "manual_tag\n")
            self.assertIn("Skipped all files (1).", "\n".join(lines))

    def test_preview_mode_creates_no_caption_or_backup_for_missing_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            caption_path = root / "image.txt"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            opts = self._opts(
                dataset_path=root,
                image_exts=[".png"],
                preview_only=True,
                preview_limit=1,
                write_mode="overwrite",
                replace_existing_captions=True,
            )
            bundle = self._fake_onnx_bundle(["smile"], [0], [0.9])
            with patch("services.offline_tagger._load_model_bundle", return_value=bundle):
                ok, _lines = offline_tagger.run_tagger(opts)
            self.assertTrue(ok)
            self.assertFalse(caption_path.exists())
            self.assertFalse((root / ".batchbench_backup").exists())

    def test_recursive_backup_preserves_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "subfolder"
            nested.mkdir()
            image_path = nested / "image.png"
            caption_path = nested / "image.txt"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            caption_path.write_text("old\n", encoding="utf-8")
            opts = self._opts(
                dataset_path=root,
                recursive=True,
                image_exts=[".png"],
                preview_only=False,
                preview_limit=0,
                write_mode="overwrite",
                replace_existing_captions=True,
                sort_tags=False,
            )
            bundle = self._fake_onnx_bundle(["smile"], [0], [0.9])
            with patch("services.offline_tagger._load_model_bundle", return_value=bundle):
                ok, _lines = offline_tagger.run_tagger(opts)
            self.assertTrue(ok)
            backups = list((root / ".batchbench_backup").rglob("offline_tagger_*/subfolder/image.txt"))
            self.assertEqual(len(backups), 1)

    def test_backup_failure_aborts_before_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            caption_path = root / "image.txt"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            caption_path.write_text("old\n", encoding="utf-8")
            opts = self._opts(
                dataset_path=root,
                image_exts=[".png"],
                preview_only=False,
                write_mode="overwrite",
                replace_existing_captions=True,
            )
            bundle = self._fake_onnx_bundle(["smile"], [0], [0.9])
            with patch("services.offline_tagger._load_model_bundle", return_value=bundle), patch(
                "services.offline_tagger.shutil.copy2",
                side_effect=OSError("copy failed"),
            ):
                ok, lines = offline_tagger.run_tagger(opts)
            self.assertFalse(ok)
            self.assertEqual(caption_path.read_text(encoding="utf-8"), "old\n")
            self.assertIn("Caption backup failed before writing", "\n".join(lines))

    def test_frozen_exe_routes_to_external_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "services").mkdir()
            (root / "services" / "offline_tagger.py").write_text("# worker module marker\n", encoding="utf-8")
            worker_python = root / ".venv" / "Scripts" / "python.exe"
            worker_python.parent.mkdir(parents=True)
            worker_python.write_text("", encoding="utf-8")
            opts = self._opts(dataset_path=root, preview_only=True, preview_limit=1)
            seen = {}

            def fake_run(cmd, cwd, env, text, capture_output):
                seen["cmd"] = list(cmd)
                seen["cwd"] = cwd
                seen["env"] = dict(env)
                payload_path = Path(cmd[4])
                result_path = Path(cmd[6])
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["options"]["dataset_path"], str(root))
                self.assertEqual(payload["deprecated_keys"], ["legacy"])
                result_path.write_text(
                    json.dumps({"ok": True, "lines": ["worker completed"]}),
                    encoding="utf-8",
                )

                class Proc:
                    returncode = 0
                    stdout = "worker stdout\n"
                    stderr = ""

                return Proc()

            with patch.object(offline_tagger.sys, "frozen", True, create=True), patch(
                "services.offline_tagger._candidate_source_roots",
                return_value=[root],
            ), patch("services.offline_tagger.subprocess.run", side_effect=fake_run):
                ok, lines = offline_tagger.run_tagger(opts, deprecated_keys=["legacy"])

            self.assertTrue(ok)
            self.assertEqual(seen["cmd"][:3], [str(worker_python), "-m", "services.offline_tagger"])
            self.assertEqual(seen["cwd"], str(root))
            self.assertEqual(seen["env"][offline_tagger._WORKER_ENV_FLAG], "1")
            self.assertIn(f"Offline Tagger worker: {worker_python}", lines)
            self.assertIn("worker completed", lines)

    def test_pipeline_and_standalone_option_parsing_match(self):
        workflow = compile_workflow(
            "raw_images",
            {
                "auto_tag": True,
                "trigger_tag": "mytrigger",
                "tag_policy": "character_identity_omitted",
                "replace_existing_captions": True,
                "policy_mcut_min_general_tags": 0,
            },
        )
        step_cfg = next(step["config"] for step in workflow["steps"] if step["id"] == "offline_tagger")
        pipeline_opts = offline_tagger._effective_opts({"folder": ".", **step_cfg}, offline_tagger.TAGGER_POLICY)
        standalone_opts = offline_tagger._effective_opts(
            {
                "folder": ".",
                "trigger_tag": "mytrigger",
                "tag_policy": "character_identity_omitted",
                "replace_existing_captions": "1",
                "policy_mcut_min_general_tags": "0",
            },
            offline_tagger.TAGGER_POLICY,
        )
        self.assertEqual(pipeline_opts.tag_policy, standalone_opts.tag_policy)
        self.assertEqual(pipeline_opts.write_mode, standalone_opts.write_mode)
        self.assertEqual(pipeline_opts.trigger_tag, standalone_opts.trigger_tag)
        self.assertEqual(pipeline_opts.policy_mcut_min_general_tags, standalone_opts.policy_mcut_min_general_tags)

    def test_final_audit_zero_for_valid_character_policy_output(self):
        labels = ["blue_hair", "smile"]
        categories = [0, 0]
        compiled = tag_policy.compile_policy("character_identity_omitted", labels, categories, None)
        self.assertEqual(tag_policy.audit_tags(["mytrigger", "smile"], compiled, trigger_tag="mytrigger"), [])

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
