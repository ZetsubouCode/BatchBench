from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from services import context_suggestions, jio7_tags, offline_tagger, tagger_model_manager
from services.tagger_models.animetimm import AnimeTimmDbv4Adapter
from services.tagger_models.base import TagPrediction
from services.tagger_models.registry import (
    CAFORMER_PROFILE,
    LEGACY_WD_PROFILE,
    get_model_profile,
    resolve_model_profile,
)
from services.tagger_models.smilingwolf import SmilingWolfWdAdapter


class TaggerRegistryTests(unittest.TestCase):
    def test_registry_dispatches_explicit_adapters(self):
        self.assertIsInstance(get_model_profile(CAFORMER_PROFILE).adapter(), AnimeTimmDbv4Adapter)
        wd_profile = get_model_profile(LEGACY_WD_PROFILE)
        self.assertIsInstance(wd_profile.adapter(), SmilingWolfWdAdapter)
        self.assertEqual(wd_profile.runtime, "timm")
        self.assertEqual(
            wd_profile.required_files,
            ("model.safetensors", "config.json", "selected_tags.csv"),
        )

    def test_legacy_model_id_resolves_to_wd_profile(self):
        profile = resolve_model_profile(None, "SmilingWolf/wd-swinv2-tagger-v3")
        self.assertEqual(profile.key, LEGACY_WD_PROFILE)
        opts = offline_tagger._effective_opts(
            {"folder": ".", "model_id": "SmilingWolf/wd-swinv2-tagger-v3"},
            offline_tagger.TAGGER_POLICY,
        )
        self.assertEqual(opts.model_profile, LEGACY_WD_PROFILE)

    def test_preprocessing_dispatch_keeps_caformer_rgb_and_swaps_wd(self):
        image = Image.new("RGB", (1, 1), (240, 20, 5))
        wd_pixel = SmilingWolfWdAdapter().preprocess(image).getpixel((0, 0))
        self.assertEqual(wd_pixel, (5, 20, 240))

        caformer = AnimeTimmDbv4Adapter()
        tensor = caformer.preprocess(image)
        self.assertEqual(tensor.shape, (3, 384, 384))
        # Red remains the strongest channel after RGB/ImageNet normalization.
        center = (tensor[:, 192, 192] * caformer.std) + caformer.mean
        self.assertGreater(float(center[0]), float(center[2]))

    def test_caformer_metadata_exposes_best_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected_tags.csv"
            path.write_text("name,category,best_threshold\nsmile,0,0.37\n", encoding="utf-8")
            metadata = AnimeTimmDbv4Adapter().tag_metadata(Path(tmp))
        self.assertEqual(metadata[0].tag_key, "smile")
        self.assertAlmostEqual(metadata[0].recommended_threshold or 0, 0.37)


class Jio7ClassificationTests(unittest.TestCase):
    def test_expected_categories_and_space_lookup(self):
        classifier = jio7_tags.load()
        self.assertIsNotNone(classifier)
        expected = {
            "long_hair": "feature",
            "shirt": "attire",
            "smile": "expression",
            "sitting": "action",
            "bedroom": "setting",
            "cowboy_shot": "other",
        }
        for tag, category in expected.items():
            self.assertEqual(classifier.category_for(tag), category)
        self.assertEqual(classifier.category_for("cowboy shot"), classifier.category_for("cowboy_shot"))

    def test_context_filter_blocks_model_character_and_rating_even_if_other(self):
        classifier = jio7_tags.Jio7Classification(
            Path("."), {"char_x": "other", "rating_x": "other", "cowboy_shot": "other"}, {}, {}, "test", ()
        )
        routed = context_suggestions.filter_and_route_predictions(
            [
                TagPrediction("char_x", 0.99, "character", 0.3),
                TagPrediction("rating_x", 0.99, "rating", 0.3),
                TagPrediction("cowboy_shot", 0.90, "general", 0.3),
            ],
            classifier,
            [{"id": "camera_angle", "label": "Camera Angle"}],
        )
        self.assertEqual([row["tag"] for row in routed["camera_angle"]], ["cowboy shot"])

    def test_auto_tag_default_semantic_filter(self):
        opts = offline_tagger._effective_opts({"folder": ".", "ui_mode": "simple"}, offline_tagger.TAGGER_POLICY)
        labels = ["long_hair", "shirt", "smile", "sitting", "bedroom", "cowboy_shot"]
        scores, dropped = offline_tagger.filter_prediction_scores_with_jio7(
            [0.9] * len(labels), labels, [0] * len(labels), opts, offline_tagger.CategoryIds(general=0), jio7_tags.load()
        )
        kept = [tag for tag, score in zip(labels, scores) if score >= 0]
        self.assertEqual(kept, ["smile", "sitting", "bedroom", "cowboy_shot"])
        self.assertEqual(dropped, 2)

    def test_guided_routing_uses_semantics_and_existing_rules(self):
        segments = [
            {"id": "expression", "label": "Expression"},
            {"id": "body_composition", "label": "Body Composition / Pose"},
            {"id": "background", "label": "Background"},
            {"id": "camera_angle", "label": "Camera Angle"},
        ]
        self.assertEqual(context_suggestions.route_segment("smile", "expression", segments), "expression")
        self.assertEqual(context_suggestions.route_segment("sitting", "action", segments), "body_composition")
        self.assertEqual(context_suggestions.route_segment("bedroom", "setting", segments), "background")
        self.assertEqual(context_suggestions.route_segment("cowboy_shot", "other", segments), "camera_angle")
        self.assertEqual(context_suggestions.route_segment("full_body", "other", segments), "body_composition")


class ContextSuggestionCacheTests(unittest.TestCase):
    def _job(self, root: Path, profile: str = CAFORMER_PROFILE):
        return context_suggestions.InspectionJob(
            "test-job", root, profile, "normal", [{"id": "expression", "label": "Expression"}]
        )

    def test_empty_dataset_fails_with_actionable_image_discovery_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dataset").mkdir()
            job = self._job(root)

            context_suggestions.inspect_dataset_sync(job, predictor=lambda _path: [])

            self.assertEqual(job.status, "failed")
            self.assertEqual(job.phase, "failed")
            self.assertEqual(job.total, 0)
            self.assertIn("No supported images found", job.error)
            self.assertIn(str(root / "dataset"), job.error)

    def test_cache_reuse_invalidation_resume_and_no_caption_write(self):
        classifier = jio7_tags.load()
        self.assertIsNotNone(classifier)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            image = dataset / "sample.png"
            caption = dataset / "sample.txt"
            Image.new("RGB", (8, 8), (255, 0, 0)).save(image)
            caption.write_text("manual tag\n", encoding="utf-8")
            calls = []

            def predict(path):
                calls.append(path)
                return [TagPrediction("smile", 0.9, "general", 0.3)]

            first = self._job(root)
            context_suggestions.inspect_dataset_sync(first, predictor=predict, classifier=classifier)
            self.assertEqual(first.status, "completed")
            self.assertEqual(first.phase, "completed")
            self.assertEqual(first.total, 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(caption.read_text(encoding="utf-8"), "manual tag\n")

            second = self._job(root)
            context_suggestions.inspect_dataset_sync(second, predictor=predict, classifier=classifier)
            self.assertEqual(second.cached, 1)
            self.assertEqual(len(calls), 1)
            stale = context_suggestions.suggestions_for_image(root, "sample.png", expected_profile=LEGACY_WD_PROFILE)
            self.assertTrue(stale["stale"])
            self.assertIsNone(stale["entry"])

            Image.new("RGB", (9, 8), (255, 0, 0)).save(image)
            changed = self._job(root)
            context_suggestions.inspect_dataset_sync(changed, predictor=predict, classifier=classifier)
            self.assertEqual(len(calls), 2)

            different_model = self._job(root, LEGACY_WD_PROFILE)
            context_suggestions.inspect_dataset_sync(different_model, predictor=predict, classifier=classifier)
            self.assertEqual(len(calls), 3)

            changed_classifier = replace(classifier, version="changed")
            changed_data = self._job(root, LEGACY_WD_PROFILE)
            context_suggestions.inspect_dataset_sync(changed_data, predictor=predict, classifier=changed_classifier)
            self.assertEqual(len(calls), 4)

            image2 = dataset / "second.png"
            Image.new("RGB", (8, 8), (0, 0, 0)).save(image2)
            cancelled = self._job(root, LEGACY_WD_PROFILE)
            cancelled.cancel_event.set()
            context_suggestions.inspect_dataset_sync(cancelled, predictor=predict, classifier=changed_classifier)
            self.assertEqual(cancelled.status, "cancelled")
            resumed = self._job(root, LEGACY_WD_PROFILE)
            context_suggestions.inspect_dataset_sync(resumed, predictor=predict, classifier=changed_classifier)
            self.assertEqual(resumed.status, "completed")
            self.assertGreaterEqual(resumed.cached, 1)


class ModelManagerTests(unittest.TestCase):
    def test_gated_download_error_is_actionable_and_does_not_echo_tokens(self):
        message = tagger_model_manager._friendly_download_error(RuntimeError("403 gated repository; token=secret"))
        self.assertIn("Accept the model conditions", message)
        self.assertNotIn("secret", message)

    def test_downloader_uses_curated_required_files_and_managed_storage(self):
        profile = get_model_profile(CAFORMER_PROFILE)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            managed = root / "managed"
            source.mkdir()
            for name in profile.required_files:
                (source / name).write_bytes(b"fixture")

            job = tagger_model_manager.DownloadJob("download-test", profile.key)
            with patch.object(tagger_model_manager, "model_root", return_value=managed), patch(
                "huggingface_hub.hf_hub_download",
                side_effect=lambda repo_id, filename: str(source / filename),
            ):
                tagger_model_manager._run_download(job)

            self.assertEqual(job.status, "completed")
            self.assertEqual(job.completed_files, len(profile.required_files))
            self.assertTrue(all((managed / profile.key / name).is_file() for name in profile.required_files))

    def test_downloader_reports_gated_access_without_traceback(self):
        profile = get_model_profile(CAFORMER_PROFILE)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            tagger_model_manager, "model_root", return_value=Path(tmp)
        ), patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("403 gated token=secret")):
            job = tagger_model_manager.DownloadJob("gated-test", profile.key)
            tagger_model_manager._run_download(job)
        self.assertEqual(job.status, "failed")
        self.assertIn("Accept the model conditions", job.error)
        self.assertNotIn("secret", job.error)


if __name__ == "__main__":
    unittest.main()
