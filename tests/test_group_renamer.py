import tempfile
import unittest
from pathlib import Path

from services import group_renamer


class GroupRenamerTests(unittest.TestCase):
    def test_natural_order_handles_mixed_numeric_and_text_names(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "10.png").write_bytes(b"img")
            (root / "cover.png").write_bytes(b"img")
            (root / "2.png").write_bytes(b"img")
            (root / "01 extras").mkdir()
            (root / "chapter").mkdir()
            (root / "01 extras" / "1.png").write_bytes(b"img")
            (root / "chapter" / "page.png").write_bytes(b"img")

            _, log, meta = group_renamer.handle(
                {
                    "rn_root": str(root),
                    "rn_exts": ".png",
                    "rn_dry_run": "on",
                    "rn_top_order": "name",
                    "rn_folder_order": "name",
                    "rn_inside_order": "name",
                },
                {},
            )

            self.assertTrue(meta.get("ok"), msg=log)
            self.assertIn("Planned renames: 5", log)

    def test_invalid_number_fields_fall_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "cover.png").write_bytes(b"img")

            _, log, meta = group_renamer.handle(
                {
                    "rn_root": str(root),
                    "rn_exts": ".png",
                    "rn_dry_run": "on",
                    "rn_start": "bad",
                    "rn_pad": "",
                    "rn_suffix_pad": "nope",
                },
                {},
            )

            self.assertTrue(meta.get("ok"), msg=log)
            self.assertIn("Numbering: pad=3, suffix_pad=0, start=1", log)


if __name__ == "__main__":
    unittest.main()
