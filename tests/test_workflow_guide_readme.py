import unittest

from app import _readme_workflow_guide_items, app


class WorkflowGuideReadmeTests(unittest.TestCase):
    def test_dataset_tag_editor_guide_uses_readme_content(self):
        items = _readme_workflow_guide_items()
        tags_item = next(item for item in items if item["id"] == "tags")

        joined_steps = " ".join(tags_item["steps"])
        joined_details = " ".join(
            row for detail in tags_item["details"] for row in detail["items"]
        )

        self.assertIn("Guided Tagging Flow", joined_steps)
        self.assertIn("Bulk Tag CRUD", joined_details)

    def test_rendered_guide_mentions_readme_source(self):
        client = app.test_client()
        response = client.get("/?tab=guide")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Edit that README section to update this guide", response.data)
        self.assertIn(b"Bulk Tag CRUD", response.data)
