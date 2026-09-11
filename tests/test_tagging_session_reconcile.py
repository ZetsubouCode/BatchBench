import json
import tempfile
import unittest
from pathlib import Path

from services import tag_editor


def _settings(*segment_ids):
    return {
        "segments": [
            {"id": seg_id, "label": seg_id.replace("_", " ").title(), "order": idx + 1}
            for idx, seg_id in enumerate(segment_ids)
        ],
        "prompt_init": {},
    }


def _mapping(*segment_ids):
    return [{"left_sections": [], "right_segment": seg_id} for seg_id in segment_ids]


def _start(project: Path, *segment_ids):
    return tag_editor.start_tagging_session(
        project,
        [".png"],
        mapping_rows=_mapping(*segment_ids),
        session_defaults={},
        recommendations={seg_id: [] for seg_id in segment_ids},
        recommendation_sources={seg_id: [] for seg_id in segment_ids},
        settings=_settings(*segment_ids),
        replace=True,
    )


def _make_project(root: Path, count: int, prefix: str = "image"):
    dataset = root / "dataset"
    temp = dataset / "_temp"
    temp.mkdir(parents=True, exist_ok=True)
    (root / "database").mkdir(parents=True, exist_ok=True)
    (root / "prompt.txt").write_text("trigger\n", encoding="utf-8")
    for idx in range(count):
        (dataset / f"{prefix}_{idx:03d}.png").write_bytes(f"{prefix}-{idx}".encode("utf-8"))
        (dataset / f"{prefix}_{idx:03d}.txt").write_text("old_tag\n", encoding="utf-8")


def _complete(session, rels=None, seg_id="appearance"):
    wanted = set(rels or session["images"].keys())
    for rel, entry in session["images"].items():
        if rel not in wanted:
            continue
        entry["status"] = "completed"
        entry["final_tags_written"] = True
        entry.setdefault("segments", {}).setdefault(
            seg_id,
            {"selected": [], "manual": [], "removed_defaults": [], "skipped": False},
        )
        entry["segments"][seg_id]["selected"] = [f"done_{Path(rel).stem}"]
        entry["pending_segment_ids"] = []
    session["status"] = "completed"
    session["current"] = {"image_index": 0, "image_rel": "", "segment_index": 0, "segment_id": seg_id}


class TaggingSessionReconcileTests(unittest.TestCase):
    def test_completed_dataset_with_new_images_queues_only_new_images(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project"
            _make_project(project, 10)
            session = _start(project, "appearance")["session"]
            old_rels = list(session["images"])
            _complete(session, old_rels)
            tag_editor.save_tagging_session(project, session)

            dataset = project / "dataset"
            for idx in range(10, 13):
                (dataset / f"image_{idx:03d}.png").write_bytes(f"image-{idx}".encode("utf-8"))
                (dataset / f"image_{idx:03d}.txt").write_text("new_tag\n", encoding="utf-8")

            result = _start(project, "appearance")
            self.assertTrue(result.get("ok"), msg=result)
            session = result["session"]
            for rel in old_rels:
                self.assertEqual(session["images"][rel]["status"], "completed")
            new_rels = [f"dataset/image_{idx:03d}.png" for idx in range(10, 13)]
            self.assertEqual([session["images"][rel]["status"] for rel in new_rels], ["pending", "pending", "pending"])
            self.assertEqual(session["status"], "active")
            self.assertEqual(session["current"]["image_rel"], new_rels[0])

    def test_ongoing_dataset_survives_restart_with_cursor_and_partial_choices(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project"
            _make_project(project, 10)
            session = _start(project, "appearance", "outfit", "camera")["session"]
            rels = list(session["images"])
            _complete(session, [rels[0]], "appearance")
            partial = session["images"][rels[1]]
            partial["status"] = "in_progress"
            partial["segments"]["outfit"]["manual"] = ["red_dress"]
            session["status"] = "active"
            session["current"] = {
                "image_index": 1,
                "image_rel": rels[1],
                "segment_index": 1,
                "segment_id": "outfit",
                "free_tagging": False,
            }
            tag_editor.save_tagging_session(project, session)

            result = tag_editor.load_tagging_session(project)
            self.assertTrue(result.get("ok"), msg=result)
            loaded = result["session"]
            self.assertEqual(loaded["images"][rels[0]]["status"], "completed")
            self.assertEqual(loaded["images"][rels[1]]["segments"]["outfit"]["manual"], ["red_dress"])
            self.assertEqual(loaded["current"]["image_rel"], rels[1])
            self.assertEqual(loaded["current"]["segment_id"], "outfit")

    def test_added_images_do_not_erase_partial_work(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project"
            _make_project(project, 3)
            session = _start(project, "appearance", "outfit")["session"]
            rels = list(session["images"])
            session["images"][rels[0]]["status"] = "in_progress"
            session["images"][rels[0]]["segments"]["outfit"]["selected"] = ["boots"]
            session["current"] = {"image_index": 0, "image_rel": rels[0], "segment_index": 1, "segment_id": "outfit"}
            tag_editor.save_tagging_session(project, session)

            (project / "dataset" / "image_003.png").write_bytes(b"image-3")
            (project / "dataset" / "image_003.txt").write_text("new_tag\n", encoding="utf-8")
            result = _start(project, "appearance", "outfit")
            loaded = result["session"]

            self.assertEqual(loaded["images"][rels[0]]["segments"]["outfit"]["selected"], ["boots"])
            self.assertEqual(loaded["current"]["image_rel"], rels[0])
            self.assertEqual(loaded["images"]["dataset/image_003.png"]["status"], "pending")

    def test_stale_late_cursor_repairs_to_first_unfinished_image(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project"
            _make_project(project, 30)
            session = _start(project, "appearance")["session"]
            rels = list(session["images"])
            session["status"] = "active"
            session["current"] = {
                "image_index": 24,
                "image_rel": rels[24],
                "segment_index": 0,
                "segment_id": "appearance",
                "free_tagging": False,
            }
            tag_editor.save_tagging_session(project, session)

            result = tag_editor.load_tagging_session(project)
            loaded = result["session"]

            self.assertEqual(loaded["current"]["image_rel"], rels[0])
            self.assertEqual(loaded["current"]["image_index"], 0)

    def test_reconcile_keeps_active_images_in_dataset_order(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project"
            _make_project(project, 5)
            session = _start(project, "appearance")["session"]
            rels = list(session["images"])
            session["images"] = {rel: session["images"][rel] for rel in [rels[3], rels[4], rels[0], rels[1], rels[2]]}
            session["current"] = {
                "image_index": 0,
                "image_rel": rels[3],
                "segment_index": 0,
                "segment_id": "appearance",
                "free_tagging": False,
            }
            tag_editor.save_tagging_session(project, session)

            result = tag_editor.load_tagging_session(project)
            loaded = result["session"]

            self.assertEqual(list(loaded["images"].keys()), rels)
            self.assertEqual(loaded["current"]["image_rel"], rels[0])
            self.assertEqual(loaded["current"]["image_index"], 0)

    def test_special_prefixed_numeric_names_start_in_natural_order(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project"
            dataset = project / "dataset"
            (dataset / "_temp").mkdir(parents=True, exist_ok=True)
            (project / "database").mkdir(parents=True, exist_ok=True)
            (project / "prompt.txt").write_text("trigger\n", encoding="utf-8")
            for name in ("[16] subject.png", "[2] subject.png", "[10] subject.png", "[1] subject.png"):
                (dataset / name).write_bytes(name.encode("utf-8"))
                (dataset / Path(name).with_suffix(".txt")).write_text("old_tag\n", encoding="utf-8")

            result = _start(project, "appearance")
            session = result["session"]
            expected = [
                "dataset/[1] subject.png",
                "dataset/[2] subject.png",
                "dataset/[10] subject.png",
                "dataset/[16] subject.png",
            ]

            self.assertEqual(list(session["images"].keys()), expected)
            self.assertEqual(session["current"]["image_rel"], expected[0])
            self.assertEqual(session["current"]["image_index"], 0)

            saved = tag_editor.save_tagging_quiz_image(
                project,
                expected[0],
                session["images"][expected[0]]["segments"],
                session_payload=session,
                backup=False,
            )["session"]

            self.assertEqual(saved["current"]["image_rel"], expected[1])
            self.assertEqual(saved["current"]["image_index"], 1)

    def test_removed_image_is_missing_history_and_does_not_block_completion(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project"
            _make_project(project, 2)
            session = _start(project, "appearance")["session"]
            rels = list(session["images"])
            _complete(session, rels)
            tag_editor.save_tagging_session(project, session)
            (project / "dataset" / "image_001.png").unlink()

            result = tag_editor.load_tagging_session(project)
            loaded = result["session"]
            self.assertTrue(loaded["images"][rels[1]]["missing"])
            self.assertEqual(loaded["images"][rels[0]]["status"], "completed")
            self.assertEqual(loaded["status"], "completed")

    def test_new_segment_requeues_only_missing_segment(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project"
            _make_project(project, 2)
            session = _start(project, "appearance")["session"]
            rels = list(session["images"])
            _complete(session, rels, "appearance")
            tag_editor.save_tagging_session(project, session)

            result = _start(project, "appearance", "outfit")
            loaded = result["session"]
            first = loaded["images"][rels[0]]
            self.assertEqual(first["segments"]["appearance"]["selected"], ["done_image_000"])
            self.assertIn("outfit", first["segments"])
            self.assertEqual(first["pending_segment_ids"], ["outfit"])
            self.assertEqual(first["status"], "in_progress")
            self.assertEqual(loaded["current"]["segment_id"], "outfit")

    def test_renamed_image_with_unique_content_hash_transfers_progress(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project"
            _make_project(project, 1)
            session = _start(project, "appearance")["session"]
            rel = next(iter(session["images"]))
            _complete(session, [rel], "appearance")
            tag_editor.save_tagging_session(project, session)

            old_path = project / "dataset" / "image_000.png"
            new_path = project / "dataset" / "renamed_000.png"
            old_path.rename(new_path)
            (project / "dataset" / "image_000.txt").rename(project / "dataset" / "renamed_000.txt")

            result = tag_editor.load_tagging_session(project)
            loaded = result["session"]
            self.assertIn("dataset/renamed_000.png", loaded["images"])
            self.assertEqual(loaded["images"]["dataset/renamed_000.png"]["segments"]["appearance"]["selected"], ["done_image_000"])
            self.assertTrue(any("renamed images matched safely: 1" in line for line in result.get("logs") or []))

    def test_duplicate_content_rename_ambiguity_leaves_new_files_pending(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project"
            _make_project(project, 1)
            session = _start(project, "appearance")["session"]
            rel = next(iter(session["images"]))
            _complete(session, [rel], "appearance")
            tag_editor.save_tagging_session(project, session)

            old_path = project / "dataset" / "image_000.png"
            data = old_path.read_bytes()
            old_path.unlink()
            (project / "dataset" / "copy_a.png").write_bytes(data)
            (project / "dataset" / "copy_a.txt").write_text("", encoding="utf-8")
            (project / "dataset" / "copy_b.png").write_bytes(data)
            (project / "dataset" / "copy_b.txt").write_text("", encoding="utf-8")

            result = tag_editor.load_tagging_session(project)
            loaded = result["session"]
            self.assertTrue(loaded["images"][rel]["missing"])
            self.assertEqual(loaded["images"]["dataset/copy_a.png"]["status"], "pending")
            self.assertEqual(loaded["images"]["dataset/copy_b.png"]["status"], "pending")
            self.assertTrue(any("ambiguous renamed images left pending: 2" in line for line in result.get("logs") or []))

    def test_legacy_session_migration_preserves_exact_paths_and_marks_unmatched_missing(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project"
            _make_project(project, 1)
            session_path = tag_editor.tagging_session_path(project)
            session_path.parent.mkdir(parents=True, exist_ok=True)
            legacy = {
                "status": "active",
                "current": {"image_rel": "dataset/image_000.png", "segment_id": "appearance"},
                "quiz_segments": [{"id": "appearance", "label": "Appearance", "order": 1}],
                "images": {
                    "dataset/image_000.png": {
                        "status": "completed",
                        "segments": {"appearance": {"selected": ["keep_me"], "manual": ["manual_tag"]}},
                        "final_tags_written": True,
                    },
                    "dataset/gone.png": {
                        "status": "in_progress",
                        "segments": {"appearance": {"selected": ["old"]}},
                    },
                },
            }
            session_path.write_text(json.dumps(legacy), encoding="utf-8")

            result = tag_editor.load_tagging_session(project)
            loaded = result["session"]
            self.assertEqual(loaded["session_version"], 2)
            self.assertEqual(loaded["images"]["dataset/image_000.png"]["status"], "completed")
            self.assertEqual(loaded["images"]["dataset/image_000.png"]["segments"]["appearance"]["selected"], ["keep_me"])
            self.assertTrue(loaded["images"]["dataset/gone.png"]["missing"])
            saved = json.loads(session_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["session_version"], 2)


if __name__ == "__main__":
    unittest.main()
