import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import tag_catalog


class TagCatalogTests(unittest.TestCase):
    def _patch_paths(self, root: Path):
        return patch.multiple(
            tag_catalog,
            CATALOG_ROOT=root,
            CSV_PATH=root / "danbooru_tags.csv",
            DB_PATH=root / "danbooru_tags.sqlite3",
            STATE_PATH=root / "catalog_state.json",
            STAGING_ROOT=root / "_staging",
            SETTINGS_PATH=root / "settings.json",
        )

    def _write_csv(self, path: Path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tag_catalog.CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_normalization_and_category_mapping(self):
        self.assertEqual(tag_catalog.normalize_tag_name("  Long  Hair  "), "long_hair")
        self.assertEqual(tag_catalog.category_name(4), "character")
        self.assertEqual(tag_catalog.category_name(99), "unknown")

    def test_import_merges_duplicates_and_searches_filters(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.csv"
            self._write_csv(
                source,
                [
                    {"id": 1, "name": "from below", "category": 0, "category_name": "general", "post_count": 10, "is_deprecated": "false", "updated_at": "2024-01-01T00:00:00Z"},
                    {"id": 2, "name": "from_below", "category": 0, "category_name": "general", "post_count": 20, "is_deprecated": "false", "updated_at": "2024-01-02T00:00:00Z"},
                    {"id": 3, "name": "from_behind", "category": 0, "category_name": "general", "post_count": 30, "is_deprecated": "false", "updated_at": ""},
                    {"id": 4, "name": "old_tag", "category": 2, "category_name": "deprecated", "post_count": 99, "is_deprecated": "true", "updated_at": ""},
                    {"id": 5, "name": "", "category": 0, "category_name": "general", "post_count": 1, "is_deprecated": "false", "updated_at": ""},
                    {"id": 6, "name": "from", "category": 0, "category_name": "general", "post_count": 0, "is_deprecated": "false", "updated_at": ""},
                ],
            )
            with self._patch_paths(root):
                result = tag_catalog.import_csv(source)
                self.assertTrue(result["ok"], msg=result)
                status = tag_catalog.get_catalog_status()
                self.assertTrue(status["ready"])
                self.assertEqual(status["tag_count"], 3)

                rows = tag_catalog.search_suggestions(
                    query="from_b",
                    limit=10,
                    min_post_count=15,
                    include_categories=["general"],
                    include_deprecated=False,
                    existing_tags=["from_behind"],
                    project_tags=["from_below"],
                    glossary_tags=[],
                )
                csv_text = tag_catalog.CSV_PATH.read_text(encoding="utf-8")

            self.assertEqual([row["tag"] for row in rows], ["from_below"])
            self.assertEqual(rows[0]["source"], "project")
            self.assertNotIn(",from,", csv_text)

    def test_zero_post_tags_are_not_lookup_or_suggestion_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.csv"
            self._write_csv(
                source,
                [
                    {"id": 1, "name": "arms", "category": 0, "category_name": "general", "post_count": 0, "is_deprecated": "false", "updated_at": ""},
                    {"id": 2, "name": "arm_support", "category": 0, "category_name": "general", "post_count": 12, "is_deprecated": "false", "updated_at": ""},
                ],
            )
            with self._patch_paths(root):
                self.assertTrue(tag_catalog.import_csv(source)["ok"])
                lookup = tag_catalog.lookup_tag("arms")
                rows = tag_catalog.search_suggestions("arm", 10, 0, ["general"], False, [], [], [])

            self.assertIsNone(lookup)
            self.assertEqual([row["tag"] for row in rows], ["arm_support"])

    def test_rebuild_rewrites_csv_without_zero_post_tags(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "danbooru_tags.csv"
            self._write_csv(
                csv_path,
                [
                    {"id": 1, "name": "from", "category": 0, "category_name": "general", "post_count": 0, "is_deprecated": "false", "updated_at": ""},
                    {"id": 2, "name": "from_below", "category": 0, "category_name": "general", "post_count": 50, "is_deprecated": "false", "updated_at": ""},
                ],
            )
            with self._patch_paths(root):
                result = tag_catalog.rebuild_sqlite_from_csv()
                text = tag_catalog.CSV_PATH.read_text(encoding="utf-8")
                rows = tag_catalog.search_suggestions("from", 10, 0, ["general"], False, [], [], [])

            self.assertTrue(result["ok"], msg=result)
            self.assertNotIn(",from,", text)
            self.assertIn("from_below", text)
            self.assertEqual([row["tag"] for row in rows], ["from_below"])

    def test_import_failure_leaves_existing_catalog_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good = root / "good.csv"
            bad = root / "bad.csv"
            self._write_csv(
                good,
                [{"id": 1, "name": "kept_tag", "category": 0, "category_name": "general", "post_count": 5, "is_deprecated": "false", "updated_at": ""}],
            )
            bad.write_text("bad,header\nx,y\n", encoding="utf-8")
            with self._patch_paths(root):
                self.assertTrue(tag_catalog.import_csv(good)["ok"])
                before = (tag_catalog.CSV_PATH.read_text(encoding="utf-8"), tag_catalog.DB_PATH.read_bytes())
                result = tag_catalog.import_csv(bad)
                after = (tag_catalog.CSV_PATH.read_text(encoding="utf-8"), tag_catalog.DB_PATH.read_bytes())

            self.assertFalse(result["ok"])
            self.assertEqual(before, after)

    def test_settings_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self._patch_paths(root):
                saved = tag_catalog.save_settings({"enabled": True, "max_suggestions": 99, "aliases": {"grey hair": "gray_hair"}, "sections": {"manual_tags": {"enabled": False}}})
                loaded = tag_catalog.load_settings()

            self.assertTrue(saved["enabled"])
            self.assertEqual(saved["max_suggestions"], 30)
            self.assertEqual(loaded["aliases"]["grey_hair"], "gray_hair")
            self.assertFalse(loaded["sections"]["manual_tags"]["enabled"])

    def test_state_write_retries_transient_windows_access_denied(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            original_replace = tag_catalog.os.replace
            attempts = {"count": 0}

            def flaky_replace(src, dst):
                attempts["count"] += 1
                if attempts["count"] < 3:
                    raise PermissionError(5, "Access is denied", str(dst))
                return original_replace(src, dst)

            with self._patch_paths(root), patch("services.tag_catalog.os.replace", side_effect=flaky_replace), patch("services.tag_catalog.time.sleep"):
                tag_catalog._save_state({"state": "running", "logs": ["retry test"]})
                loaded = tag_catalog._load_state()

            self.assertEqual(loaded["state"], "running")
            self.assertEqual(attempts["count"], 3)
            self.assertFalse(list(root.glob("catalog_state.json.*.tmp")))

    def test_status_marks_stale_running_sync_as_failed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self._patch_paths(root):
                tag_catalog._save_state({"state": "running", "started_at": "2026-01-01T00:00:00Z", "logs": []})
                status = tag_catalog.get_catalog_status()

            self.assertEqual(status["state"]["state"], "failed")
            self.assertIn("interrupted", status["state"]["last_error"])

    def test_alias_resolution_uses_explicit_local_alias_map(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.csv"
            self._write_csv(
                source,
                [{"id": 1, "name": "gray_hair", "category": 0, "category_name": "general", "post_count": 50, "is_deprecated": "false", "updated_at": ""}],
            )
            with self._patch_paths(root):
                self.assertTrue(tag_catalog.import_csv(source)["ok"])
                tag_catalog.save_settings({"aliases": {"grey_hair": "gray_hair"}})
                resolved = tag_catalog.resolve_alias("grey hair")
                rows = tag_catalog.search_suggestions("grey_hair", 5, 0, ["general"], False, [], [], [])

            self.assertEqual(resolved["canonical"], "gray_hair")
            self.assertEqual(resolved["validation_status"], "Alias that resolves to a canonical tag")
            self.assertEqual(rows[0]["canonical"], "gray_hair")
            self.assertEqual(rows[0]["alias"], "grey_hair")

    def test_cancelled_sync_leaves_old_catalog(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_csv = root / "danbooru_tags.csv"
            old_db = root / "danbooru_tags.sqlite3"
            self._write_csv(
                old_csv,
                [{"id": 1, "name": "old_tag", "category": 0, "category_name": "general", "post_count": 5, "is_deprecated": "false", "updated_at": ""}],
            )
            with self._patch_paths(root):
                rows, _, _ = tag_catalog._read_csv_records(old_csv)
                tag_catalog.build_sqlite(rows, old_db)
                tag_catalog._CANCEL_EVENT.set()
                tag_catalog._run_full_sync(root / "_staging" / "cancel_test")

            self.assertIn("old_tag", old_csv.read_text(encoding="utf-8"))

    def test_successful_sync_swaps_staged_catalog_and_counts_malformed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self._patch_paths(root):
                staging = root / "_staging" / "sync_test"
                staging.mkdir(parents=True)
                pages = [
                    [
                        {"id": 10, "name": "from below", "category": 0, "post_count": 12, "updated_at": ""},
                        {"id": 11, "name": "", "category": 0, "post_count": 1, "updated_at": ""},
                    ],
                    [],
                ]
                cursors = []

                def fake_fetch(after_id=None, limit=1000):  # noqa: ARG001
                    cursors.append(after_id)
                    return pages.pop(0), None

                with patch("services.tag_catalog.danbooru_client.fetch_tag_page", side_effect=fake_fetch):
                    tag_catalog._CANCEL_EVENT.clear()
                    tag_catalog._run_full_sync(staging)
                status = tag_catalog.get_catalog_status()

            self.assertTrue(status["ready"])
            self.assertEqual(status["tag_count"], 1)
            self.assertEqual(status["state"]["malformed_rows"], 1)
            self.assertEqual(cursors, [None, 10])


if __name__ == "__main__":
    unittest.main()
