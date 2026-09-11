import sys
import time
import types
import unittest
from unittest import mock

from services.discord_presence import (
    ACTIVITY_PAYLOADS,
    DiscordPresenceService,
    NullDiscordPresence,
)


def _presence_module(presence_cls):
    module = types.ModuleType("pypresence")
    module.Presence = presence_cls
    return module


class FakePresence:
    instances = []
    connect_error = None
    update_error = None

    def __init__(self, application_id):
        self.application_id = application_id
        self.updates = []
        self.cleared = False
        self.closed = False
        FakePresence.instances.append(self)

    def connect(self):
        if FakePresence.connect_error:
            raise FakePresence.connect_error

    def update(self, **payload):
        if FakePresence.update_error:
            raise FakePresence.update_error
        self.updates.append(payload)

    def clear(self):
        self.cleared = True

    def close(self):
        self.closed = True


class DiscordPresenceTests(unittest.TestCase):
    def setUp(self):
        FakePresence.instances = []
        FakePresence.connect_error = None
        FakePresence.update_error = None

    def test_disabled_config_returns_null_presence(self):
        with mock.patch.dict("os.environ", {"DISCORD_RICH_PRESENCE_ENABLED": "false"}, clear=True):
            self.assertIsInstance(DiscordPresenceService.from_env(), NullDiscordPresence)

    def test_missing_application_id_returns_null_presence(self):
        with mock.patch.dict("os.environ", {"DISCORD_RICH_PRESENCE_ENABLED": "true"}, clear=True):
            self.assertIsInstance(DiscordPresenceService.from_env(), NullDiscordPresence)

    def test_invalid_application_id_returns_null_presence(self):
        env = {
            "DISCORD_RICH_PRESENCE_ENABLED": "true",
            "DISCORD_APPLICATION_ID": "not-a-number",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            self.assertIsInstance(DiscordPresenceService.from_env(), NullDiscordPresence)

    def test_unknown_activity_key_becomes_home(self):
        service = DiscordPresenceService("123", asset_key="batchbench")
        payload = service._build_payload("C:/Users/example/private/file.png")
        self.assertEqual(payload["details"], ACTIVITY_PAYLOADS["home"]["details"])
        self.assertEqual(payload["state"], ACTIVITY_PAYLOADS["home"]["state"])

    def test_payload_contains_fixed_presence_fields(self):
        service = DiscordPresenceService("123", asset_key="batchbench-test")
        payload = service._build_payload("tags")
        self.assertEqual(payload["details"], "Editing dataset tags")
        self.assertEqual(payload["state"], "Dataset Preparation")
        self.assertEqual(payload["large_image"], "batchbench-test")
        self.assertEqual(payload["large_text"], "BatchBench")
        self.assertIsInstance(payload["start"], int)

    def test_payload_never_includes_arbitrary_user_input(self):
        service = DiscordPresenceService("123", asset_key="batchbench")
        payload = service._build_payload("D:/datasets/private/image001.png")
        rendered = " ".join(str(value) for value in payload.values())
        self.assertNotIn("D:/datasets", rendered)
        self.assertNotIn("image001.png", rendered)
        self.assertNotIn("private", rendered)

    def test_tagging_context_formats_image_progress(self):
        service = DiscordPresenceService("123", asset_key="batchbench")
        payload = service._build_payload({"tool": "tags", "phase": "default", "current": 1, "total": 63})
        self.assertEqual(payload["details"], "Tagging dataset")
        self.assertEqual(payload["state"], "Image 1 of 63")

    def test_guided_outfit_context_formats_image_progress(self):
        service = DiscordPresenceService("123", asset_key="batchbench")
        payload = service._build_payload({"tool": "tags", "phase": "guided_outfit", "current": 18, "total": 63})
        self.assertEqual(payload["details"], "Reviewing outfit details")
        self.assertEqual(payload["state"], "Image 18 of 63")

    def test_unknown_guided_step_uses_safe_generic_copy(self):
        service = DiscordPresenceService("123", asset_key="batchbench")
        payload = service._build_payload({"tool": "tags", "phase": "client_character_name", "current": 2, "total": 5})
        self.assertEqual(payload["details"], "Reviewing image tags")
        self.assertEqual(payload["state"], "Image 2 of 5")
        self.assertNotIn("client_character_name", " ".join(str(v) for v in payload.values()))

    def test_invalid_progress_is_ignored_without_leaking_raw_values(self):
        service = DiscordPresenceService("123", asset_key="batchbench")
        cases = [
            {"tool": "tags", "phase": "default", "current": -1, "total": 63},
            {"tool": "tags", "phase": "default", "current": 1, "total": 0},
            {"tool": "tags", "phase": "default", "current": "file.png", "total": 63},
            {"tool": "tags", "phase": "default", "current": 1, "total": "D:/private"},
            {"tool": "tags", "phase": "default", "details": "raw detail", "state": "raw state"},
        ]
        for context in cases:
            with self.subTest(context=context):
                payload = service._build_payload(context)
                rendered = " ".join(str(value) for value in payload.values())
                self.assertEqual(payload["details"], "Tagging dataset")
                self.assertEqual(payload["state"], "Dataset Preparation")
                self.assertNotIn("file.png", rendered)
                self.assertNotIn("D:/private", rendered)
                self.assertNotIn("raw detail", rendered)
                self.assertNotIn("raw state", rendered)

    def test_offline_progress_uses_processed_format(self):
        service = DiscordPresenceService("123", asset_key="batchbench")
        payload = service._build_payload({"tool": "offline", "phase": "running", "current": 42, "total": 196})
        self.assertEqual(payload["details"], "Auto-tagging images")
        self.assertEqual(payload["state"], "Processed 42 of 196")

    def test_pipeline_progress_uses_safe_step_label(self):
        service = DiscordPresenceService("123", asset_key="batchbench")
        payload = service._build_payload({
            "tool": "pipeline",
            "phase": "running",
            "current": 3,
            "total": 7,
            "step_key": "normalize",
        })
        self.assertEqual(payload["details"], "Running dataset pipeline")
        self.assertEqual(payload["state"], "Step 3 of 7 - Normalizing captions")

        unsafe = service._build_payload({
            "tool": "pipeline",
            "phase": "running",
            "current": 3,
            "total": 7,
            "step_key": "D:/client/model/path",
        })
        self.assertEqual(unsafe["state"], "Step 3 of 7")
        self.assertNotIn("client", unsafe["state"])

    def test_each_main_menu_has_specific_presence_copy(self):
        service = DiscordPresenceService("123", asset_key="batchbench")
        expected = {
            "guide": "Reviewing workflow guide",
            "webp": "Converting images to PNG",
            "batch": "Tuning image adjustments",
            "blur_brush": "Brushing soft focus",
            "color_brush": "Painting color details",
            "palette_helper": "Planning manga palettes",
            "epub_extractor": "Extracting EPUB artwork",
            "webtoon": "Splitting webtoon panels",
            "merge": "Stitching image groups",
            "rename": "Renaming dataset files",
            "combine": "Combining datasets",
            "normalize": "Normalizing dataset",
            "tags": "Editing dataset tags",
            "offline": "Auto-tagging images",
            "clip_tokens": "Checking CLIP tokens",
            "pipeline": "Preparing pipeline",
            "tag_wiki": "Browsing tag glossary",
            "settings": "Configuring BatchBench",
        }
        for key, details in expected.items():
            with self.subTest(key=key):
                self.assertEqual(service._build_payload(key)["details"], details)

    def test_rapid_activity_calls_coalesce_to_latest(self):
        service = DiscordPresenceService("123", asset_key="batchbench")
        service.set_activity("webp")
        service.set_activity("batch")
        service.report_activity("tags", phase="guided_outfit", current=4, total=8)
        self.assertEqual(service._activity_queue.get_nowait(), {
            "tool": "tags",
            "phase": "guided_outfit",
            "current": 4,
            "total": 8,
        })
        with self.assertRaises(Exception):
            service._activity_queue.get_nowait()

    def test_rate_limiter_does_not_send_more_than_one_update_inside_window(self):
        service = DiscordPresenceService("123", rate_limit_seconds=12, retry_base_seconds=0.01)
        with mock.patch.dict(sys.modules, {"pypresence": _presence_module(FakePresence)}):
            service.start()
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                if FakePresence.instances and FakePresence.instances[0].updates:
                    break
                time.sleep(0.02)
            service.set_activity("tags")
            time.sleep(0.2)
            updates = list(FakePresence.instances[0].updates)
            service.stop()
        self.assertLessEqual(len(updates), 1)

    def test_completion_state_can_bypass_rate_limit_once(self):
        service = DiscordPresenceService("123", rate_limit_seconds=12, retry_base_seconds=0.01)
        with mock.patch.dict(sys.modules, {"pypresence": _presence_module(FakePresence)}):
            service.start()
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                if FakePresence.instances and FakePresence.instances[0].updates:
                    break
                time.sleep(0.02)
            service.report_activity("pipeline", phase="complete", current=7, total=7)
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                if len(FakePresence.instances[0].updates) >= 2:
                    break
                time.sleep(0.02)
            updates = list(FakePresence.instances[0].updates)
            service.stop()
        self.assertGreaterEqual(len(updates), 2)
        self.assertEqual(updates[-1]["details"], "Pipeline complete")

    def test_connection_failure_does_not_raise(self):
        FakePresence.connect_error = RuntimeError("discord closed")
        service = DiscordPresenceService("123", retry_base_seconds=0.01, retry_max_seconds=0.02)
        with mock.patch.dict(sys.modules, {"pypresence": _presence_module(FakePresence)}):
            service.start()
            time.sleep(0.05)
            service.stop()

    def test_reconnect_failure_does_not_raise(self):
        FakePresence.update_error = RuntimeError("ipc disconnected")
        service = DiscordPresenceService("123", rate_limit_seconds=0.01, retry_base_seconds=0.01, retry_max_seconds=0.02)
        with mock.patch.dict(sys.modules, {"pypresence": _presence_module(FakePresence)}):
            service.start()
            time.sleep(0.05)
            service.stop()

    def test_stop_is_safe_before_start(self):
        DiscordPresenceService("123").stop()

    def test_stop_is_safe_after_failed_connection(self):
        FakePresence.connect_error = RuntimeError("discord closed")
        service = DiscordPresenceService("123", retry_base_seconds=0.01, retry_max_seconds=0.02)
        with mock.patch.dict(sys.modules, {"pypresence": _presence_module(FakePresence)}):
            service.start()
            time.sleep(0.03)
            service.stop()

    def test_worker_thread_is_daemonized(self):
        service = DiscordPresenceService("123", retry_base_seconds=0.01)
        with mock.patch.dict(sys.modules, {"pypresence": _presence_module(FakePresence)}):
            service.start()
            self.assertIsNotNone(service.worker_thread)
            self.assertTrue(service.worker_thread.daemon)
            service.stop()


if __name__ == "__main__":
    unittest.main()
