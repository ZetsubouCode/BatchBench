import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

from app import app


class TagEditorFileApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_mobile_guided_tagging_controls_are_wired(self):
        response = self.client.get("/?tab=tags")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        for section_label in ("Selected / Preselected", "Recommended", "Manual Tags", "Glossarium Tags"):
            with self.subTest(section_label=section_label):
                self.assertIn(section_label, html)
        guided_html = html.split('id="tagging-quiz-modal"', 1)[1].split('id="tagging-quiz-overview-modal"', 1)[0]
        self.assertLess(guided_html.index('id="tagging-quiz-glossary-list"'), guided_html.index('quiz-mobile-footer'))
        self.assertLess(guided_html.index('quiz-mobile-footer'), guided_html.index('quiz-desktop-footer'))
        self.assertIn("function qMobileMappingControls", html)
        self.assertIn("quiz-map-mobile-section-select", html)
        self.assertIn("quiz-map-mobile-segment-select", html)
        self.assertIn("window.visualViewport?.addEventListener('resize'", html)

        project_root = Path(__file__).resolve().parents[1]
        mobile_css = (project_root / "static/css/dataset_tag_editor_mobile.css").read_text(encoding="utf-8")
        mobile_js = (project_root / "static/js/dataset_tag_editor_mobile.js").read_text(encoding="utf-8")
        self.assertIn("#tagging-quiz-setup-modal .quiz-map-mobile-controls", mobile_css)
        self.assertIn("-webkit-overflow-scrolling: touch", mobile_css)
        self.assertIn("touch-action: pan-y", mobile_css)
        self.assertIn('id="tagging-mobile-more-sheet"', guided_html)
        self.assertIn('id="tagging-mobile-more-trigger"', guided_html)
        self.assertIn('id="tagging-quiz-mobile-progress"', guided_html)
        self.assertIn('class="quiz-mobile-image-progress-count"', guided_html)
        self.assertNotIn('class="quiz-mobile-more"', guided_html)
        mobile_footer = guided_html.split('quiz-mobile-footer', 1)[1].split('quiz-desktop-footer', 1)[0]
        desktop_footer = guided_html.split('quiz-desktop-footer', 1)[1]
        self.assertLess(mobile_footer.index('id="tagging-mobile-next-segment"'), mobile_footer.index('id="tagging-mobile-more-sheet"'))
        self.assertGreater(mobile_footer.index('id="tagging-mobile-segment-wrap"'), mobile_footer.index('id="tagging-mobile-more-sheet"'))
        self.assertLess(desktop_footer.index('id="tagging-quiz-next-segment"'), desktop_footer.index('quiz-desktop-more-menu'))
        self.assertIn("#tagging-quiz-modal .quiz-mobile-footer {\n    display: grid;\n    position: static;", mobile_css)
        self.assertIn("#tagging-quiz-modal .quiz-image-stage {\n    min-height: 0;\n    height: min(30dvh, 210px);\n    max-height: 32dvh;\n    position: sticky;", mobile_css)
        self.assertIn("#tagging-quiz-modal .quiz-tag-pane {\n    min-width: 0;\n    overflow: visible;", mobile_css)
        self.assertIn("syncGuidedInputState", mobile_js)
        self.assertIn("syncCaptionPlacement", mobile_js)
        self.assertIn("setMoreOpen", mobile_js)

    def test_tag_add_creates_txt_for_existing_image(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            temp = root / "_temp"
            temp.mkdir(parents=True, exist_ok=True)
            image = temp / "sample.png"
            image.write_bytes(b"img")

            resp = self.client.post(
                "/api/tags/tag-add",
                json={
                    "folder": str(root),
                    "rel": "_temp/sample.png",
                    "tags": "cat, blue sky",
                    "backup": False,
                    "create_missing_txt": True,
                },
            )

            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("ok"), msg=data)
            self.assertTrue(data.get("created"))
            self.assertEqual(data.get("added"), ["cat", "blue sky"])
            self.assertEqual((temp / "sample.txt").read_text(encoding="utf-8").strip(), "cat, blue sky")

    def test_move_file_moves_image_and_txt_between_root_and_temp(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            temp = root / "_temp"
            temp.mkdir(parents=True, exist_ok=True)
            image = root / "sample.png"
            txt = root / "sample.txt"
            image.write_bytes(b"img")
            txt.write_text("cat", encoding="utf-8")

            to_temp = self.client.post(
                "/api/tags/move-file",
                json={"folder": str(root), "src": "sample.png", "dst": "_temp"},
            )
            self.assertEqual(to_temp.status_code, 200)
            move_data = to_temp.get_json()
            self.assertTrue(move_data.get("ok"), msg=move_data)
            self.assertTrue((temp / "sample.png").exists())
            self.assertTrue((temp / "sample.txt").exists())
            self.assertFalse(image.exists())
            self.assertFalse(txt.exists())

            back_to_root = self.client.post(
                "/api/tags/move-file",
                json={"folder": str(root), "src": "_temp/sample.png", "dst": ""},
            )
            self.assertEqual(back_to_root.status_code, 200)
            restore_data = back_to_root.get_json()
            self.assertTrue(restore_data.get("ok"), msg=restore_data)
            self.assertTrue(image.exists())
            self.assertTrue(txt.exists())
            self.assertFalse((temp / "sample.png").exists())
            self.assertFalse((temp / "sample.txt").exists())

    def test_move_files_moves_multiple_pairs_in_one_request(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            temp = root / "_temp"
            temp.mkdir(parents=True, exist_ok=True)
            for stem in ("a", "b"):
                (root / f"{stem}.png").write_bytes(b"img")
                (root / f"{stem}.txt").write_text(stem, encoding="utf-8")

            resp = self.client.post(
                "/api/tags/move-files",
                json={"folder": str(root), "srcs": ["a.png", "b.png"], "dst": "_temp"},
            )

            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("ok"), msg=data)
            self.assertEqual(len(data.get("moved") or []), 2)
            for stem in ("a", "b"):
                self.assertTrue((temp / f"{stem}.png").exists())
                self.assertTrue((temp / f"{stem}.txt").exists())
                self.assertFalse((root / f"{stem}.png").exists())
                self.assertFalse((root / f"{stem}.txt").exists())

    def test_move_preview_scope_and_stale_file_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'dataset'; temp = root / '_temp'; temp.mkdir(parents=True)
            (root / 'a.png').write_bytes(b'image')
            (root / 'a.txt').write_text('first', encoding='utf-8')
            body = {'folder': str(root), 'srcs': ['a.png'], 'dst': '_temp'}
            preview = self.client.post('/api/tags/move-files/preview', json=body)
            self.assertEqual(preview.status_code, 200)
            plan = preview.get_json()
            self.assertEqual(plan['paired_captions'], 1)
            self.assertEqual(plan['missing_captions'], 0)
            (root / 'a.txt').write_text('external change', encoding='utf-8')
            stale = self.client.post('/api/tags/move-files', json={**body, 'preview_token': plan['preview_token']})
            self.assertEqual(stale.status_code, 409)
            self.assertTrue((root / 'a.png').exists())
            self.assertFalse((temp / 'a.png').exists())

    def test_return_temp_preserves_nested_pair_folders(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            nested = root / "_temp" / "batch_a" / "variant"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "sample.png").write_bytes(b"img")
            (nested / "sample.txt").write_text("tag_a", encoding="utf-8")
            (nested / "sample_2.png").write_bytes(b"img")
            (nested / "sample_2.txt").write_text("tag_b", encoding="utf-8")

            resp = self.client.post(
                "/api/tags/return-temp",
                json={"folder": str(root), "exts": ".png,.jpg,.jpeg,.webp"},
            )

            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("ok"), msg=data)
            self.assertEqual(data.get("total"), 2)
            self.assertEqual(len(data.get("moved") or []), 2)
            self.assertTrue((root / "batch_a" / "variant" / "sample.png").exists())
            self.assertTrue((root / "batch_a" / "variant" / "sample.txt").exists())
            self.assertTrue((root / "batch_a" / "variant" / "sample_2.png").exists())
            self.assertTrue((root / "batch_a" / "variant" / "sample_2.txt").exists())
            self.assertFalse((nested / "sample.png").exists())
            self.assertFalse((nested / "sample.txt").exists())
            self.assertFalse((nested / "sample_2.png").exists())
            self.assertFalse((nested / "sample_2.txt").exists())
            self.assertTrue((root / "_temp").exists())
            self.assertFalse((root / "_temp" / "batch_a").exists())

    def test_return_temp_with_source_all_moves_from_any_subfolder_to_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            temp_nested = root / "_temp" / "batch_a"
            other_nested = root / "placeholderA" / "round_1"
            temp_nested.mkdir(parents=True, exist_ok=True)
            other_nested.mkdir(parents=True, exist_ok=True)

            (temp_nested / "from_temp.png").write_bytes(b"img")
            (temp_nested / "from_temp.txt").write_text("tag_temp", encoding="utf-8")
            (other_nested / "from_other.png").write_bytes(b"img")
            (other_nested / "from_other.txt").write_text("tag_other", encoding="utf-8")
            (root / "already_root.png").write_bytes(b"img")
            (root / "already_root.txt").write_text("tag_root", encoding="utf-8")

            resp = self.client.post(
                "/api/tags/return-temp",
                json={"folder": str(root), "exts": ".png,.jpg,.jpeg,.webp", "source": "all"},
            )

            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("ok"), msg=data)
            self.assertEqual(data.get("source"), "all")
            self.assertEqual(data.get("total"), 2)
            self.assertEqual(len(data.get("moved") or []), 2)
            self.assertTrue((root / "from_temp.png").exists())
            self.assertTrue((root / "from_temp.txt").exists())
            self.assertTrue((root / "from_other.png").exists())
            self.assertTrue((root / "from_other.txt").exists())
            self.assertTrue((root / "already_root.png").exists())
            self.assertTrue((root / "already_root.txt").exists())
            self.assertFalse((temp_nested / "from_temp.png").exists())
            self.assertFalse((temp_nested / "from_temp.txt").exists())
            self.assertFalse((other_nested / "from_other.png").exists())
            self.assertFalse((other_nested / "from_other.txt").exists())

    def test_cheatsheet_reads_root_level_txt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            root.mkdir(parents=True, exist_ok=True)
            cheat = root / "main_tags.txt"
            cheat.write_text(
                "trigger_word\n\nappearance: blue_sky, long_hair\nsmiling, looking_at_viewer\noutfit: jacket, skirt\n",
                encoding="utf-8",
            )

            resp = self.client.post(
                "/api/tags/cheatsheet",
                json={"folder": str(root), "rel": "main_tags.txt"},
            )

            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("ok"), msg=data)
            self.assertEqual(data.get("rel"), "main_tags.txt")
            self.assertEqual(data.get("trigger"), "trigger_word")
            self.assertEqual(data.get("tags"), ["blue sky", "long hair", "smiling", "looking at viewer", "jacket", "skirt"])
            sections = data.get("sections") or []
            self.assertEqual(len(sections), 2)
            self.assertEqual(sections[0]["category"], "appearance")
            self.assertEqual(sections[0]["tags"], ["blue sky", "long hair"])
            self.assertEqual(sections[0]["conditionals"], [["smiling", "looking at viewer"]])
            self.assertEqual(sections[1]["category"], "outfit")

    def test_dataset_zip_excludes_temp_and_writes_next_to_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project_a"
            dataset = project / "dataset"
            temp = dataset / "_temp"
            nested = dataset / "set_1"
            nested.mkdir(parents=True, exist_ok=True)
            temp.mkdir(parents=True, exist_ok=True)
            (dataset / "root.png").write_bytes(b"img")
            (nested / "nested.png").write_bytes(b"img")
            (temp / "hidden.png").write_bytes(b"img")

            resp = self.client.post(
                "/api/tags/dataset-zip",
                json={"folder": str(project)},
            )

            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("ok"), msg=data)
            self.assertTrue(data.get("download_url"), msg=data)
            zip_path = Path(data.get("zip_path") or "")
            self.assertTrue(zip_path.exists())
            self.assertEqual(zip_path.parent, dataset.parent)
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = set(zf.namelist())
            self.assertIn("root.png", names)
            self.assertIn("set_1/nested.png", names)
            self.assertNotIn("_temp/hidden.png", names)

            download = self.client.get(data["download_url"])
            try:
                self.assertEqual(download.status_code, 200)
                self.assertEqual(download.mimetype, "application/zip")
                self.assertIn("attachment", download.headers.get("Content-Disposition", "").lower())
                self.assertIn(data["zip_name"], download.headers.get("Content-Disposition", ""))
                downloaded_bytes = bytes(download.data)
            finally:
                download.close()
            with zipfile.ZipFile(BytesIO(downloaded_bytes), "r") as zf:
                downloaded_names = set(zf.namelist())
            self.assertIn("root.png", downloaded_names)
            self.assertIn("set_1/nested.png", downloaded_names)
            self.assertNotIn("_temp/hidden.png", downloaded_names)

    def test_dataset_zip_download_rejects_unsafe_or_arbitrary_paths(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project_a"
            dataset = project / "dataset"
            dataset.mkdir(parents=True, exist_ok=True)
            outside = Path(td) / "private.zip"
            outside.write_bytes(b"not available through dataset download")

            cases = [
                "notes.txt",
                "../dataset__20260919-123456.zip",
                str(outside),
                "nested/dataset__20260919-123456.zip",
            ]
            for name in cases:
                with self.subTest(name=name):
                    response = self.client.get(
                        "/api/tags/dataset-zip/download",
                        query_string={"folder": str(project), "name": name},
                    )
                    self.assertEqual(response.status_code, 400, msg=response.get_data(as_text=True))

            missing = self.client.get(
                "/api/tags/dataset-zip/download",
                query_string={"folder": str(project), "name": "dataset__20260919-123456.zip"},
            )
            self.assertEqual(missing.status_code, 404)

    def test_temp_move_all_moves_content_to_timestamped_folder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            temp = root / "_temp"
            nested = temp / "batch_a" / "variant"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "sample.png").write_bytes(b"img")
            (nested / "sample.txt").write_text("tag_a", encoding="utf-8")
            (temp / "notes.md").write_text("memo", encoding="utf-8")

            resp = self.client.post(
                "/api/tags/temp-move-all",
                json={"folder": str(root), "exts": ".png,.jpg,.jpeg,.webp"},
            )

            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("ok"), msg=data)
            dst_rel = data.get("destination") or ""
            self.assertTrue(dst_rel.startswith("_moved_"), msg=data)
            dst_root = root / dst_rel
            self.assertTrue((dst_root / "batch_a" / "variant" / "sample.png").exists())
            self.assertTrue((dst_root / "batch_a" / "variant" / "sample.txt").exists())
            self.assertTrue((dst_root / "notes.md").exists())
            self.assertFalse((temp / "batch_a" / "variant" / "sample.png").exists())
            self.assertFalse((temp / "batch_a" / "variant" / "sample.txt").exists())
            self.assertFalse((temp / "notes.md").exists())

    def test_cleanup_subdirs_removes_empty_folders_but_keeps_temp(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            (root / "_temp").mkdir(parents=True, exist_ok=True)
            (root / "set_a" / "nested").mkdir(parents=True, exist_ok=True)
            (root / "set_b").mkdir(parents=True, exist_ok=True)

            resp = self.client.post(
                "/api/tags/cleanup-subdirs",
                json={"folder": str(root), "keep_temp": True},
            )

            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("ok"), msg=data)
            self.assertTrue(data.get("cleaned"), msg=data)
            self.assertIn("set_a", data.get("removed") or [])
            self.assertIn("set_b", data.get("removed") or [])
            self.assertFalse((root / "set_a").exists())
            self.assertFalse((root / "set_b").exists())
            self.assertTrue((root / "_temp").exists())

    def test_cleanup_subdirs_warns_and_skips_when_any_folder_has_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            (root / "_temp").mkdir(parents=True, exist_ok=True)
            (root / "set_a").mkdir(parents=True, exist_ok=True)
            (root / "set_a" / "img.png").write_bytes(b"img")
            (root / "set_b").mkdir(parents=True, exist_ok=True)

            resp = self.client.post(
                "/api/tags/cleanup-subdirs",
                json={"folder": str(root), "keep_temp": True},
            )

            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("ok"), msg=data)
            self.assertFalse(data.get("cleaned"), msg=data)
            self.assertIn("set_a", data.get("blocked") or [])
            self.assertEqual(data.get("removed"), [])
            self.assertTrue((root / "set_a").exists())
            self.assertTrue((root / "set_b").exists())
            self.assertTrue((root / "_temp").exists())


if __name__ == "__main__":
    unittest.main()
