import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from services import webtoon_splitter


class WebtoonSplitterOrderingTests(unittest.TestCase):
    def test_name_order_can_descend_naturally(self):
        paths = [Path("1.png"), Path("10.png"), Path("2.png")]

        ordered = webtoon_splitter._sort_paths(paths, "name", "desc")

        self.assertEqual([p.name for p in ordered], ["10.png", "2.png", "1.png"])

    def test_created_order_can_descend(self):
        paths = [Path("old.png"), Path("new.png"), Path("middle.png")]
        timestamps = {"old.png": 1.0, "middle.png": 2.0, "new.png": 3.0}

        with patch("services.webtoon_splitter._created_ts", side_effect=lambda p: timestamps[p.name]):
            ordered = webtoon_splitter._sort_paths(paths, "ctime", "desc")

        self.assertEqual([p.name for p in ordered], ["new.png", "middle.png", "old.png"])

    def test_handle_passes_selected_sort_order_to_merge(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("1.png", "10.png", "2.png"):
                (root / name).write_bytes(b"placeholder")

            seen = []

            def fake_merge_many(images, *_args):
                seen.append([p.name for p in images])
                return Image.new("RGB", (8, 8), "white")

            with patch("services.webtoon_splitter.merge_many", side_effect=fake_merge_many):
                _tab, log, meta = webtoon_splitter.handle(
                    {
                        "wt_folder": str(root),
                        "wt_exts": ".png",
                        "wt_sort_by": "name",
                        "wt_sort_dir": "desc",
                        "wt_dry_run": "on",
                    },
                    {},
                )

        self.assertTrue(meta.get("ok"), msg=log)
        self.assertIn("Order: name desc", log)
        self.assertEqual(seen, [["10.png", "2.png", "1.png"]])


class WebtoonSplitterStripeColorTests(unittest.TestCase):
    def test_parse_stripe_colors_accepts_multiple_hex_formats(self):
        colors, invalid = webtoon_splitter._parse_stripe_colors("#fff, 000000; #336699 nope")

        self.assertEqual(colors, [(255, 255, 255), (0, 0, 0), (51, 102, 153)])
        self.assertEqual(invalid, ["nope"])

    def test_find_stripes_can_match_white_and_black_rows(self):
        img = Image.new("RGB", (10, 20), "#336699")
        for y in range(2, 5):
            for x in range(10):
                img.putpixel((x, y), (255, 255, 255))
        for y in range(10, 13):
            for x in range(10):
                img.putpixel((x, y), (0, 0, 0))

        stripes = webtoon_splitter._find_stripes(
            img,
            threshold=245,
            min_height=2,
            ratio=1.0,
            max_gap=0,
            stripe_colors=[(255, 255, 255), (0, 0, 0)],
        )

        self.assertEqual(stripes, [(2, 4), (10, 12)])

    def test_empty_stripe_colors_falls_back_to_white(self):
        colors, invalid = webtoon_splitter._parse_stripe_colors("")

        self.assertEqual(colors, [(255, 255, 255)])
        self.assertEqual(invalid, [])


if __name__ == "__main__":
    unittest.main()
