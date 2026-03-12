import tempfile
import unittest
from pathlib import Path

from services import combine_datasets


class CombineDatasetSafetyTests(unittest.TestCase):
    def _make_pair(self, root: Path, stem: str):
        (root / f"{stem}.png").write_bytes(b"img")
        (root / f"{stem}.txt").write_text("tag\n", encoding="utf-8")

    def test_requires_output_folder(self):
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a"
            b = Path(td) / "b"
            a.mkdir(parents=True, exist_ok=True)
            b.mkdir(parents=True, exist_ok=True)
            self._make_pair(a, "one")
            self._make_pair(b, "two")

            _, _, meta = combine_datasets.handle(
                {
                    "source_folders": f"{a}\n{b}",
                    "out_dir": "",
                    "exts_combine": ".png",
                },
                {},
            )

            self.assertFalse(meta.get("ok"), msg=meta)
            self.assertIn("Output folder is required", meta.get("error", ""))


if __name__ == "__main__":
    unittest.main()
