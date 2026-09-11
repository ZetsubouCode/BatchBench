from pathlib import Path

from PIL import Image

from services import webp_converter


def test_webp_converter_defaults_output_folder_to_source_output(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(source / "sample.jpg")

    active_tab, log, meta = webp_converter.handle({"src_png": str(source)}, {})

    assert active_tab == "webp"
    assert meta["ok"]
    assert "Output folder not set" in log
    assert (source / "output" / "sample.png").exists()
