import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import normalizer


class NormalizerSecurityTests(unittest.TestCase):
    def test_load_preset_blocks_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "anime").mkdir(parents=True, exist_ok=True)
            outside = root / "outside.json"
            outside.write_text("{}", encoding="utf-8")

            with self.assertRaises(ValueError):
                normalizer.load_preset(root, "anime", "../../outside.json")

    def test_load_preset_valid(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "anime" / "ok.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {"rules": {"trim": True}}
            target.write_text(json.dumps(payload), encoding="utf-8")

            loaded = normalizer.load_preset(root, "anime", "ok.json")
            self.assertEqual(loaded, payload)

    def test_apply_normalization_backup_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img = root / "sample.png"
            txt = root / "sample.txt"
            img.write_bytes(b"")
            txt.write_text("old_tag\n", encoding="utf-8")

            preset_root = root / "presets"
            (preset_root / "anime").mkdir(parents=True, exist_ok=True)
            (preset_root / "anime" / "rules.json").write_text(
                json.dumps(
                    {
                        "rules": {
                            "replace_map": {"old_tag": "new_tag"},
                            "sort": {"enabled": False, "priority_groups": []},
                        }
                    }
                ),
                encoding="utf-8",
            )

            opts = normalizer.NormalizeOptions(
                dataset_path=root,
                recursive=False,
                include_missing_txt=False,
                preset_type="anime",
                preset_file="rules.json",
                backup_enabled=True,
                image_exts=[".png"],
            )

            with patch("services.normalizer._make_backup", return_value=(False, "disk full")):
                result = normalizer.apply_normalization(opts, preset_root)

            self.assertFalse(result.get("ok"))
            self.assertIn("Backup failed", result.get("error", ""))
            self.assertEqual(txt.read_text(encoding="utf-8").strip(), "old_tag")


if __name__ == "__main__":
    unittest.main()
