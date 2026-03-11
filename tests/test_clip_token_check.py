import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.clip_token_check import estimate_token_count, handle, scan_caption_files


class ClipTokenCheckTests(unittest.TestCase):
    def test_estimate_token_count_counts_basic_chunks(self):
        self.assertEqual(estimate_token_count("legs_folded, brick wall"), 6)

    def test_recursive_scan_excludes_temp_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            temp = root / "_temp"
            root.mkdir(parents=True, exist_ok=True)
            temp.mkdir(parents=True, exist_ok=True)

            (root / "a.png").write_bytes(b"img")
            (root / "a.txt").write_text("cat dog", encoding="utf-8")

            (temp / "b.png").write_bytes(b"img")
            (temp / "b.txt").write_text("cat dog bird fish", encoding="utf-8")

            result = scan_caption_files(
                folder=root,
                exts=[".png"],
                recursive=True,
                limit=3,
                mode="estimate",
                include_temp=False,
            )

            self.assertTrue(result.get("ok"))
            self.assertEqual(result.get("scanned_captions"), 1)
            self.assertEqual(result.get("over_limit"), 1)

    def test_handle_exact_falls_back_to_estimate_when_tokenizer_unavailable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            root.mkdir(parents=True, exist_ok=True)
            (root / "a.png").write_bytes(b"img")
            (root / "a.txt").write_text("cat dog", encoding="utf-8")

            with patch("services.clip_token_check.try_get_clip_tokenizer", return_value=None):
                active_tab, log, meta = handle(
                    {
                        "folder_clip_tokens": str(root),
                        "exts_clip_tokens": ".png",
                        "recursive_clip_tokens": "1",
                        "limit_clip_tokens": "77",
                        "mode_clip_tokens": "exact",
                        "topn_clip_tokens": "20",
                    },
                    {},
                )

            self.assertEqual(active_tab, "clip_tokens")
            self.assertTrue(meta.get("ok"), msg=meta)
            self.assertIn("fallback", log.lower())


if __name__ == "__main__":
    unittest.main()
