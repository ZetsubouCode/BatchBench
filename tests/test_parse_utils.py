import unittest

from utils.parse import (
    parse_bool,
    parse_int,
    parse_float,
    parse_exts,
    parse_tag_list,
    parse_line_list,
)


class ParseUtilsTests(unittest.TestCase):
    def test_parse_bool(self):
        self.assertTrue(parse_bool("true"))
        self.assertFalse(parse_bool("no"))
        self.assertTrue(parse_bool("unknown", default=True))

    def test_parse_int(self):
        self.assertEqual(parse_int("42", 1), 42)
        self.assertEqual(parse_int("", 9), 9)
        self.assertEqual(parse_int("bad", 3), 3)

    def test_parse_float(self):
        self.assertAlmostEqual(parse_float("3.5", 1.0), 3.5)
        self.assertIsNone(parse_float("", None))
        self.assertEqual(parse_float("bad", 2.5), 2.5)

    def test_parse_exts(self):
        self.assertEqual(parse_exts(".jpg,png", default=[".webp"]), [".jpg", ".png"])
        self.assertEqual(parse_exts("", default=[".webp"]), [".webp"])

    def test_parse_tag_list(self):
        raw = "a, b\nc\n, d"
        self.assertEqual(parse_tag_list(raw), ["a", "b", "c", "d"])
        self.assertEqual(parse_tag_list(["a, b", "a"], dedupe=True), ["a", "b"])

    def test_parse_line_list(self):
        raw = "one\n\n two \nthree"
        self.assertEqual(parse_line_list(raw), ["one", "two", "three"])
        self.assertEqual(parse_line_list(["x", "x"], dedupe=True), ["x"])


if __name__ == "__main__":
    unittest.main()
