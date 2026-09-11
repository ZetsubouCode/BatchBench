import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import paths


class RuntimePathTests(unittest.TestCase):
    def setUp(self):
        self.old_cwd = Path.cwd()
        self.addCleanup(os.chdir, self.old_cwd)

    def test_env_override_sets_shared_data_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "shared"
            with patch.dict(os.environ, {"BATCHBENCH_DATA_DIR": str(root)}, clear=False):
                paths._ENV_LOADED = True
                self.addCleanup(setattr, paths, "_ENV_LOADED", False)
                self.assertEqual(paths.user_data_root(), root.resolve())

    def test_frozen_dist_layout_finds_source_repo_data(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "BatchBench"
            exe_dir = repo / "dist" / "BatchBench"
            exe_dir.mkdir(parents=True)
            (exe_dir / "tag_editor_glossary.json").write_text("{}", encoding="utf-8")
            (repo / "data" / "tag_catalog").mkdir(parents=True)
            (repo / "data" / "tag_catalog" / "danbooru_tags.csv").write_text("id,name\n", encoding="utf-8")

            try:
                os.chdir(exe_dir)
                with patch.object(sys, "frozen", True, create=True), patch.object(
                    sys, "executable", str(exe_dir / "BatchBench.exe")
                ), patch.dict(os.environ, {}, clear=True):
                    paths._ENV_LOADED = True
                    self.addCleanup(setattr, paths, "_ENV_LOADED", False)
                    self.assertEqual(paths.user_data_root(), repo.resolve())
            finally:
                os.chdir(self.old_cwd)

    def test_frozen_dist_layout_does_not_prefer_bundled_glossary_over_catalog(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "BatchBench"
            exe_dir = repo / "dist" / "BatchBench"
            internal = exe_dir / "_internal"
            exe_dir.mkdir(parents=True)
            internal.mkdir(parents=True)
            (exe_dir / "tag_editor_glossary.json").write_text("{}", encoding="utf-8")
            (internal / "tag_editor_glossary.json").write_text("{}", encoding="utf-8")
            (repo / "data" / "tag_catalog").mkdir(parents=True)
            (repo / "data" / "tag_catalog" / "danbooru_tags.sqlite3").write_bytes(b"sqlite")

            try:
                os.chdir(exe_dir)
                with patch.object(sys, "frozen", True, create=True), patch.object(
                    sys, "executable", str(exe_dir / "BatchBench.exe")
                ), patch.object(sys, "_MEIPASS", str(internal), create=True), patch.dict(os.environ, {}, clear=True):
                    paths._ENV_LOADED = True
                    self.addCleanup(setattr, paths, "_ENV_LOADED", False)
                    self.assertEqual(paths.user_data_root(), repo.resolve())
            finally:
                os.chdir(self.old_cwd)

    def test_frozen_without_external_data_uses_exe_directory_not_internal_resources(self):
        with tempfile.TemporaryDirectory() as td:
            exe_dir = Path(td) / "BatchBench"
            internal = exe_dir / "_internal"
            exe_dir.mkdir(parents=True)
            internal.mkdir(parents=True)
            (internal / "tag_editor_glossary.json").write_text("{}", encoding="utf-8")

            try:
                os.chdir(exe_dir)
                with patch.object(sys, "frozen", True, create=True), patch.object(
                    sys, "executable", str(exe_dir / "BatchBench.exe")
                ), patch.object(sys, "_MEIPASS", str(internal), create=True), patch.dict(os.environ, {}, clear=True):
                    paths._ENV_LOADED = True
                    self.addCleanup(setattr, paths, "_ENV_LOADED", False)
                    self.assertEqual(paths.user_data_root(), exe_dir.resolve())
            finally:
                os.chdir(self.old_cwd)

    def test_frozen_layout_loads_env_before_resolving_data_root(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "BatchBench"
            exe_dir = repo / "dist" / "BatchBench"
            shared = Path(td) / "shared-data"
            exe_dir.mkdir(parents=True)
            repo.mkdir(exist_ok=True)
            (repo / ".env").write_text(f"BATCHBENCH_DATA_DIR={shared}\n", encoding="utf-8")

            try:
                os.chdir(exe_dir)
                with patch.object(sys, "frozen", True, create=True), patch.object(
                    sys, "executable", str(exe_dir / "BatchBench.exe")
                ), patch.dict(os.environ, {}, clear=True):
                    paths._ENV_LOADED = False
                    self.addCleanup(setattr, paths, "_ENV_LOADED", False)
                    self.assertEqual(paths.user_data_root(), shared.resolve())
            finally:
                os.chdir(self.old_cwd)


if __name__ == "__main__":
    unittest.main()
