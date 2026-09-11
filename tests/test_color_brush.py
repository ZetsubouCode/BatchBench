from pathlib import Path

from PIL import Image

from services import color_brush


def _png_data_url(image: Image.Image) -> str:
    from base64 import b64encode
    from io import BytesIO

    buf = BytesIO()
    image.save(buf, "PNG")
    return "data:image/png;base64," + b64encode(buf.getvalue()).decode("ascii")


def test_color_paint_composites_rgba_layer(tmp_path: Path):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(source_path)

    layer = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
    layer.putpixel((0, 0), (255, 0, 0, 255))

    result = color_brush.apply_color_paint(
        source_path,
        _png_data_url(layer),
        backup=False,
        output_mode="overwrite",
        action_count=1,
    )

    assert result["ok"] is True
    with Image.open(source_path) as output:
        assert output.convert("RGBA").getpixel((0, 0)) == (255, 0, 0, 255)


def test_color_paint_rejects_transparent_layer(tmp_path: Path):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(source_path)

    result = color_brush.apply_color_paint(
        source_path,
        _png_data_url(Image.new("RGBA", (2, 2), (0, 0, 0, 0))),
        backup=False,
    )

    assert result["ok"] is False
    assert result["error"] == "Color layer empty"


def test_color_paint_copy_keeps_source_and_uses_paint_suffix(tmp_path: Path):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(source_path)
    layer = Image.new("RGBA", (2, 2), (0, 255, 0, 255))

    result = color_brush.apply_color_paint(source_path, _png_data_url(layer), backup=True, output_mode="copy")

    assert result["ok"] is True
    assert result["saved_path"].name == "source_paint.png"
    assert not Path(str(source_path) + ".bak").exists()
    with Image.open(source_path) as original:
        assert original.convert("RGBA").getpixel((0, 0)) == (0, 0, 0, 255)


def test_color_paint_overwrite_creates_backup(tmp_path: Path):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(source_path)
    layer = Image.new("RGBA", (2, 2), (0, 0, 255, 255))

    result = color_brush.apply_color_paint(source_path, _png_data_url(layer), backup=True, output_mode="overwrite")

    assert result["ok"] is True
    assert Path(str(source_path) + ".bak").exists()
