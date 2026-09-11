import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app import WORKFLOW_GUIDE_META, _readme_workflow_guide_items, app


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

    def test_all_workflow_items_have_readme_usage(self):
        items = _readme_workflow_guide_items()

        self.assertEqual(len(items), len(WORKFLOW_GUIDE_META))
        missing = [item["source_title"] for item in items if not item["steps"]]
        fallback = [
            item["source_title"]
            for item in items
            if item["summary"] == "See README for this tool's usage notes."
        ]

        self.assertEqual(missing, [])
        self.assertEqual(fallback, [])

    def test_packaged_nested_readme_layout_is_supported(self):
        with TemporaryDirectory() as td:
            nested = Path(td) / "README.md"
            nested.mkdir()
            (nested / "README.md").write_text(
                "\n".join(
                    [
                        "## 5) Tool Guide",
                        "### Tag Tools",
                        "#### Dataset Tag Editor",
                        "How to use:",
                        "- Use nested packaged README content.",
                    ]
                ),
                encoding="utf-8",
            )

            with patch("app.README_PATH", nested):
                items = _readme_workflow_guide_items()

        tags_item = next(item for item in items if item["id"] == "tags")
        self.assertEqual(tags_item["steps"], ["Use nested packaged README content."])
