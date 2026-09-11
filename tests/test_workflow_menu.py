import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from app import app


WORKFLOW_MENU_TABS = {
    "guide": ("#tab-guide", "Workflow Guide", None),
    "webp": ("#tab-webp", "Image -> PNG Converter", "Image Tools"),
    "batch": ("#tab-batch", "Photo Adjust (preset)", "Image Tools"),
    "blur_brush": ("#tab-blur-brush", "Brush Blur", "Image Tools"),
    "color_brush": ("#tab-color-brush", "Color Brush", "Image Tools"),
    "palette_helper": ("#tab-palette-helper", "Manga Palette Helper", "Image Tools"),
    "epub_extractor": ("#tab-epub-extractor", "EPUB Extractor", "Dataset Assembly"),
    "webtoon": ("#tab-webtoon", "Webtoon Panel Splitter", "Dataset Assembly"),
    "merge": ("#tab-merge", "Stitch Groups", "Dataset Assembly"),
    "rename": ("#tab-rename", "Flatten & Renumber", "Dataset Assembly"),
    "combine": ("#tab-combine", "Combine Dataset", "Dataset Assembly"),
    "tags": ("#tab-tags", "Dataset Tag Editor", "Tag Tools"),
    "normalize": ("#tab-normalize", "Dataset Normalization", "Tag Tools"),
    "offline_tagger": ("#tab-offline-tagger", "Auto Tag Assist", "Tag Tools"),
    "clip_tokens": ("#tab-clip-tokens", "CLIP Token Check", "Tag Tools"),
    "pipeline": ("#tab-pipeline", "Dataset Workflow", None),
    "tag_wiki": ("#tab-tag-wiki", "Tag Glossary Wiki", None),
    "settings": ("#tab-settings", "Settings", None),
}


class ParsedDocument(HTMLParser):
    VOID_TAGS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.elements = []
        self._stack = []

    def handle_starttag(self, tag, attrs):
        element = {
            "tag": tag,
            "attrs": dict(attrs),
            "text": "",
            "parent": self._stack[-1] if self._stack else None,
        }
        self.elements.append(element)
        if tag not in self.VOID_TAGS:
            self._stack.append(len(self.elements) - 1)

    def handle_startendtag(self, tag, attrs):
        element = {
            "tag": tag,
            "attrs": dict(attrs),
            "text": "",
            "parent": self._stack[-1] if self._stack else None,
        }
        self.elements.append(element)

    def handle_data(self, data):
        for idx in self._stack:
            self.elements[idx]["text"] += data

    def handle_endtag(self, tag):
        while self._stack:
            idx = self._stack.pop()
            if self.elements[idx]["tag"] == tag:
                break

    def by_id(self, element_id):
        return [el for el in self.elements if el["attrs"].get("id") == element_id]

    def descendants_of(self, parent):
        parent_idx = self.elements.index(parent)
        descendants = []
        for el in self.elements:
            idx = el["parent"]
            while idx is not None:
                if idx == parent_idx:
                    descendants.append(el)
                    break
                idx = self.elements[idx]["parent"]
        return descendants

    def closest(self, element, predicate):
        idx = element["parent"]
        while idx is not None:
            candidate = self.elements[idx]
            if predicate(candidate):
                return candidate
            idx = candidate["parent"]
        return None


def _classes(element):
    return set((element["attrs"].get("class") or "").split())


def _text(element):
    return " ".join(element["text"].split())


def _parse(response):
    parser = ParsedDocument()
    parser.feed(response.get_data(as_text=True))
    return parser


class WorkflowMenuTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def _menu_controls(self, parsed):
        targets = {target for target, _, _ in WORKFLOW_MENU_TABS.values()}
        return [
            el
            for el in parsed.elements
            if el["tag"] == "button"
            and el["attrs"].get("data-bs-toggle") == "tab"
            and el["attrs"].get("data-bs-target") in targets
        ]

    def _top_level_panes(self, parsed):
        pane_ids = {target[1:] for target, _, _ in WORKFLOW_MENU_TABS.values()}
        return [
            el
            for el in parsed.elements
            if el["attrs"].get("id") in pane_ids and "tab-pane" in _classes(el)
        ]

    def test_workflow_menu_contract_matches_real_rendered_panes(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        parsed = _parse(response)

        controls = self._menu_controls(parsed)
        panes = self._top_level_panes(parsed)

        self.assertEqual(
            {control["attrs"].get("data-bs-target") for control in controls},
            {target for target, _, _ in WORKFLOW_MENU_TABS.values()},
        )
        self.assertEqual(
            {pane["attrs"].get("id") for pane in panes},
            {target[1:] for target, _, _ in WORKFLOW_MENU_TABS.values()},
        )

        for tab, (target, label, _) in WORKFLOW_MENU_TABS.items():
            matching_controls = [
                control
                for control in controls
                if control["attrs"].get("data-bs-target") == target
            ]
            matching_panes = [pane for pane in panes if pane["attrs"].get("id") == target[1:]]

            self.assertEqual(len(matching_controls), 1, f"{tab} should have one menu control")
            self.assertEqual(len(matching_panes), 1, f"{tab} should have one content pane")
            self.assertEqual(matching_controls[0]["attrs"].get("type"), "button")
            self.assertIn(label, _text(matching_controls[0]))

    def test_each_workflow_menu_tab_round_trips_to_one_active_control_and_pane(self):
        for tab, (target, _, dropdown_label) in WORKFLOW_MENU_TABS.items():
            with self.subTest(tab=tab):
                response = self.client.get(f"/?tab={tab}")
                self.assertEqual(response.status_code, 200)
                parsed = _parse(response)
                controls = self._menu_controls(parsed)
                panes = self._top_level_panes(parsed)

                active_controls = [
                    control for control in controls if "active" in _classes(control)
                ]
                active_panes = [
                    pane
                    for pane in panes
                    if {"active", "show"}.issubset(_classes(pane))
                ]

                self.assertEqual(
                    [control["attrs"].get("data-bs-target") for control in active_controls],
                    [target],
                )
                self.assertEqual([pane["attrs"].get("id") for pane in active_panes], [target[1:]])

                active_control = active_controls[0]
                active_dropdown = parsed.closest(
                    active_control, lambda el: "dropdown" in _classes(el)
                )
                if dropdown_label is None:
                    self.assertIsNone(active_dropdown)
                    continue

                self.assertIsNotNone(active_dropdown)
                toggles = [
                    el
                    for el in parsed.descendants_of(active_dropdown)
                    if el["attrs"].get("data-bs-toggle") == "dropdown"
                    and "dropdown-toggle" in _classes(el)
                ]
                self.assertEqual(len(toggles), 1)
                self.assertIn("active", _classes(toggles[0]))
                self.assertIn(dropdown_label, _text(toggles[0]))

    def test_dataset_workflow_menu_opens_complete_guided_workflow_surface(self):
        response = self.client.get("/?tab=pipeline")
        self.assertEqual(response.status_code, 200)
        parsed = _parse(response)
        pane = parsed.by_id("tab-pipeline")[0]
        pane_descendants = parsed.descendants_of(pane)

        required_controls = {
            "workflow-source": "source folder input",
            "workflow-output": "final output input",
            "workflow-workdir": "working folder input",
            "workflow-starting-cards": "workflow picker",
            "workflow-preview": "guided preview button",
            "workflow-run": "guided run button",
            "workflow-plan": "plan preview output",
            "workflow-status-badge": "job status badge",
            "custom-preview": "custom workflow preview button",
        }
        descendant_ids = {el["attrs"].get("id") for el in pane_descendants}
        for element_id, description in required_controls.items():
            self.assertIn(element_id, descendant_ids, f"Missing {description}: #{element_id}")

        workflow_choices = {
            el["attrs"].get("data-workflow-id")
            for el in pane_descendants
            if el["attrs"].get("data-workflow-id")
        }
        self.assertEqual(
            workflow_choices,
            {"raw_images", "captioned_dataset", "image_preparation"},
        )

        self.assertIn("Dataset Workflow", _text(pane))
        self.assertIn("Step 1 - Source and destination", _text(pane))
        self.assertIn("Step 4 - Workflow plan preview", _text(pane))
        self.assertIn("Advanced: Custom Workflow Builder", _text(pane))

    def test_dataset_workflow_preview_endpoint_accepts_payload_from_menu(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            output = root / "output"
            workdir = root / "work"
            source.mkdir()
            output.mkdir()
            workdir.mkdir()
            (source / "sample.png").write_bytes(b"image")

            response = self.client.post(
                "/api/pipeline/plan",
                json={
                    "dataset_path": str(source),
                    "output_dir": str(output),
                    "working_dir": str(workdir),
                    "image_exts": ".png",
                    "recursive": False,
                    "copy_mode": "copy",
                    "workflow_id": "raw_images",
                    "workflow_options": {
                        "auto_tag": False,
                        "manual_review": False,
                        "preset_type": "anime",
                        "preset_file": "normalize_v1.json",
                        "create_zip": True,
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["workflow_id"], "raw_images")
        self.assertEqual(
            [step["id"] for step in payload["steps"]],
            ["normalize", "dataset_audit", "export_final"],
        )
        self.assertEqual(payload["plan"][0]["id"], "prepare_workspace")
        self.assertEqual(payload["plan"][-1]["id"], "export_final")


if __name__ == "__main__":
    unittest.main()
