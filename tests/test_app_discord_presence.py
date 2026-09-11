import unittest

from app import app


class FakePresenceService:
    def __init__(self, *, should_raise=False):
        self.calls = []
        self.should_raise = should_raise

    def set_activity(self, activity):
        self.calls.append(activity)
        if self.should_raise:
            raise RuntimeError("application 123456789 private stack trace")

    def report_activity(self, tool, **context):
        self.calls.append((tool, context))
        if self.should_raise:
            raise RuntimeError("application 123456789 private stack trace")


class AppDiscordPresenceTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.previous_presence = app.extensions.pop("discord_presence", None)

    def tearDown(self):
        app.extensions.pop("discord_presence", None)
        if self.previous_presence is not None:
            app.extensions["discord_presence"] = self.previous_presence

    def test_activity_endpoint_returns_ok(self):
        response = self.client.post("/api/discord-presence/activity", json={"activity": "tags"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ok": True})

    def test_valid_activity_calls_attached_service(self):
        service = FakePresenceService()
        app.extensions["discord_presence"] = service
        self.client.post("/api/discord-presence/activity", json={"activity": "tags"})
        self.assertEqual(service.calls, ["tags"])

    def test_epub_extractor_activity_calls_attached_service(self):
        service = FakePresenceService()
        app.extensions["discord_presence"] = service
        self.client.post("/api/discord-presence/activity", json={"activity": "epub_extractor"})
        self.assertEqual(service.calls, ["epub_extractor"])

    def test_structured_context_calls_attached_service(self):
        service = FakePresenceService()
        app.extensions["discord_presence"] = service
        response = self.client.post(
            "/api/discord-presence/activity",
            json={"tool": "tags", "phase": "guided_outfit", "current": 18, "total": 63},
        )
        self.assertEqual(response.get_json(), {"ok": True})
        self.assertEqual(service.calls, [("tags", {
            "phase": "guided_outfit",
            "current": 18,
            "total": 63,
            "step_key": None,
        })])

    def test_unknown_activity_becomes_home(self):
        service = FakePresenceService()
        app.extensions["discord_presence"] = service
        self.client.post("/api/discord-presence/activity", json={"activity": "D:/private/file.png"})
        self.assertEqual(service.calls, ["home"])

    def test_endpoint_works_with_no_presence_service_configured(self):
        response = self.client.post("/api/discord-presence/activity", json={"activity": "pipeline"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ok": True})

    def test_endpoint_response_does_not_leak_application_id_or_errors(self):
        app.extensions["discord_presence"] = FakePresenceService(should_raise=True)
        response = self.client.post(
            "/api/discord-presence/activity",
            json={
                "tool": "tags",
                "phase": "guided_outfit",
                "current": 1,
                "total": 2,
                "details": "D:/private/file.png",
                "state": "secret prompt",
            },
        )
        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ok": True})
        self.assertNotIn("123456789", body)
        self.assertNotIn("private stack trace", body)
        self.assertNotIn("secret prompt", body)
        self.assertNotIn("file.png", body)


if __name__ == "__main__":
    unittest.main()
