import base64
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from services import blur_brush


def _mask_data_url(mask: Image.Image) -> str:
    buf = BytesIO()
    mask.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class BlurBrushPreviewTests(unittest.TestCase):
    def test_preview_returns_png_bytes_and_applies_effect(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_path = root / "src.png"

            src = Image.new("RGB", (40, 20), (0, 0, 0))
            for x in range(20, 40):
                for y in range(20):
                    src.putpixel((x, y), (255, 255, 255))
            src.save(src_path, "PNG")

            mask = Image.new("L", (40, 20), 255)
            result = blur_brush.preview_brush_effect(
                image_path=src_path,
                mask_png_base64=_mask_data_url(mask),
                mode="blur",
                strength=6,
                feather=0,
                preview_size=(20, 10),
            )

            self.assertTrue(result.get("ok"))
            self.assertEqual(result.get("preview_mime"), "image/png")
            payload = result.get("preview_bytes")
            self.assertIsInstance(payload, (bytes, bytearray))
            self.assertGreater(len(payload), 10)

            with Image.open(BytesIO(payload)) as out:
                self.assertEqual(out.size, (20, 10))
                px = out.getpixel((9, 5))
                self.assertIsInstance(px, tuple)
                self.assertGreater(px[0], 0)
                self.assertLess(px[0], 255)

    def test_preview_rejects_empty_mask(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_path = root / "src.png"
            Image.new("RGB", (16, 16), (127, 127, 127)).save(src_path, "PNG")
            empty_mask = Image.new("L", (16, 16), 0)

            result = blur_brush.preview_brush_effect(
                image_path=src_path,
                mask_png_base64=_mask_data_url(empty_mask),
                mode="blur",
                strength=8,
                feather=0,
            )

            self.assertFalse(result.get("ok"))
            self.assertEqual(result.get("error"), "Mask empty")


if __name__ == "__main__":
    unittest.main()
