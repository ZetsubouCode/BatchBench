import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from app import app


class BatchAdjustPreviewApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_list_images_returns_supported_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            nested = root / "nested"
            nested.mkdir(parents=True, exist_ok=True)

            Image.new("RGB", (20, 20), (255, 0, 0)).save(root / "a.jpg", "JPEG")
            Image.new("RGB", (20, 20), (0, 255, 0)).save(nested / "b.png", "PNG")
            (root / "note.txt").write_text("ignore me", encoding="utf-8")

            resp = self.client.get(
                "/api/batch-adjust/list-images",
                query_string={
                    "folder": str(root),
                    "recursive": "1",
                    "limit": "10",
                },
            )

            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("ok"), msg=data)
            self.assertEqual(data.get("total"), 2)
            self.assertEqual(data.get("images"), ["a.jpg", "nested/b.png"])

    def test_preview_returns_jpeg_and_resizes_to_max_side(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "sample.png"
            img = Image.new("RGB", (260, 130), (255, 0, 0))
            for x in range(130, 260):
                for y in range(130):
                    img.putpixel((x, y), (0, 255, 0))
            img.save(src, "PNG")

            resp = self.client.post(
                "/api/batch-adjust/preview",
                json={
                    "folder": str(root),
                    "rel": "sample.png",
                    "preview_max_side": 128,
                    "cfg": {"saturation": -1.0, "contrast": 0.1},
                },
            )

            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.mimetype, "image/jpeg")
            with Image.open(BytesIO(resp.data)) as out:
                self.assertEqual(out.size, (128, 64))
                px = out.getpixel((10, 10))
                self.assertIsInstance(px, tuple)

    def test_preview_rejects_bad_relative_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            Image.new("RGB", (8, 8), (10, 20, 30)).save(root / "x.png", "PNG")

            resp = self.client.post(
                "/api/batch-adjust/preview",
                json={
                    "folder": str(root),
                    "rel": "../x.png",
                    "cfg": {"saturation": -1},
                },
            )

            self.assertEqual(resp.status_code, 400)
            data = resp.get_json()
            self.assertFalse(data.get("ok"))


if __name__ == "__main__":
    unittest.main()
