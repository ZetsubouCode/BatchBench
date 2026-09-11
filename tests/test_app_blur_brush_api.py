import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from app import app


class BlurBrushApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_image_endpoint_serves_image_without_txt_pair(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            Image.new("RGB", (24, 12), (12, 34, 56)).save(root / "solo.png", "PNG")

            resp = self.client.get(
                "/api/blur_brush/image",
                query_string={"folder": str(root), "path": "solo.png"},
            )
            try:
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.mimetype, "image/png")
                with Image.open(BytesIO(resp.data)) as out:
                    self.assertEqual(out.size, (24, 12))
            finally:
                resp.close()

    def test_image_endpoint_works_when_folder_name_is_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            root.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (200, 20, 30)).save(root / "sample.jpg", "JPEG")

            resp = self.client.get(
                "/api/blur_brush/image",
                query_string={"folder": str(root), "path": "sample.jpg"},
            )
            try:
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.mimetype, "image/jpeg")
            finally:
                resp.close()

    def test_image_endpoint_rejects_traversal_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            Image.new("RGB", (8, 8), (1, 2, 3)).save(root / "safe.png", "PNG")

            resp = self.client.get(
                "/api/blur_brush/image",
                query_string={"folder": str(root), "path": "../safe.png"},
            )

            self.assertEqual(resp.status_code, 400)
            data = resp.get_json()
            self.assertFalse(data.get("ok"))


if __name__ == "__main__":
    unittest.main()
