import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from services.epub_extractor import handle


def _write_test_epub(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>""",
        )
        zf.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata>
    <meta name="cover" content="cover-image"/>
  </metadata>
  <manifest>
    <item id="cover-image" href="Images/cover.jpg" media-type="image/jpeg"/>
    <item id="chap1" href="Text/chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="chap2" href="Text/chapter2.xhtml" media-type="application/xhtml+xml"/>
    <item id="page1" href="Images/page001.jpg" media-type="image/jpeg"/>
    <item id="page2" href="Images/page002.png" media-type="image/png"/>
  </manifest>
  <spine>
    <itemref idref="chap2"/>
    <itemref idref="chap1"/>
  </spine>
</package>""",
        )
        zf.writestr(
            "OEBPS/Text/chapter1.xhtml",
            """<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body><img src="../Images/page001.jpg"/></body>
</html>""",
        )
        zf.writestr(
            "OEBPS/Text/chapter2.xhtml",
            """<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body><svg><image href="../Images/page002.png"/></svg></body>
</html>""",
        )
        zf.writestr("OEBPS/Images/cover.jpg", b"cover")
        zf.writestr("OEBPS/Images/page001.jpg", b"page1")
        zf.writestr("OEBPS/Images/page002.png", b"page2")


class EpubExtractorTests(unittest.TestCase):
    def test_extracts_in_reading_order_with_cover_first_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            epub = root / "Book One.epub"
            out = root / "out"
            _write_test_epub(epub)

            active_tab, log, meta = handle(
                {
                    "epub_input": str(epub),
                    "output_dir": str(out),
                    "use_reading_order": "on",
                    "extract_cover": "on",
                    "create_report": "on",
                },
                {},
            )

            self.assertEqual(active_tab, "epub_extractor")
            self.assertTrue(meta["ok"], log)
            book_dir = out / "Book One"
            self.assertEqual((book_dir / "001.jpg").read_bytes(), b"cover")
            self.assertEqual((book_dir / "002.png").read_bytes(), b"page2")
            self.assertEqual((book_dir / "003.jpg").read_bytes(), b"page1")
            report = json.loads((book_dir / "extract_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["mode"], "reading_order")
            self.assertEqual(report["found_images"], 3)
            self.assertEqual(report["extracted"], 3)

    def test_dry_run_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            epub = root / "book.epub"
            out = root / "out"
            _write_test_epub(epub)

            _, log, meta = handle(
                {
                    "epub_input": str(epub),
                    "output_dir": str(out),
                    "dry_run": "on",
                    "use_reading_order": "on",
                    "extract_cover": "on",
                },
                {},
            )

            self.assertTrue(meta["ok"], log)
            self.assertIn("Dry run finished", log)
            self.assertFalse(out.exists())

    def test_cover_can_be_removed_when_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            epub = root / "book.epub"
            out = root / "out"
            _write_test_epub(epub)

            _, log, meta = handle(
                {
                    "epub_input": str(epub),
                    "output_dir": str(out),
                    "use_reading_order": "on",
                    "extract_cover": "off",
                    "create_report": "off",
                },
                {},
            )

            self.assertTrue(meta["ok"], log)
            book_dir = out / "book"
            self.assertEqual((book_dir / "001.png").read_bytes(), b"page2")
            self.assertEqual((book_dir / "002.jpg").read_bytes(), b"page1")
            self.assertFalse((book_dir / "003.jpg").exists())


if __name__ == "__main__":
    unittest.main()
