from __future__ import annotations

import base64
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from utils.io import log_join

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

try:
    _R_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9.1
    _R_BILINEAR = Image.BILINEAR


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_exts(exts: Optional[Sequence[str]] = None) -> set[str]:
    if not exts:
        return set(IMG_EXTS)
    normalized: set[str] = set()
    for raw in exts:
        value = str(raw or "").strip().lower()
        if not value:
            continue
        normalized.add(value if value.startswith(".") else f".{value}")
    return normalized or set(IMG_EXTS)


def list_images(folder: Path, recursive: bool = True, exts: Optional[Sequence[str]] = None) -> List[str]:
    """Return safely-relative image paths for the selected editable folder."""
    extset = _normalize_exts(exts)
    if not folder.exists() or not folder.is_dir():
        return []

    iterator = folder.rglob("*") if recursive else folder.iterdir()
    images = sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in extset),
        key=lambda path: str(path).lower(),
    )
    return [path.relative_to(folder).as_posix() for path in images]


def _decode_paint_layer(raw_value: str) -> Image.Image:
    if not raw_value:
        raise ValueError("paint_png_base64 is required")

    raw = str(raw_value).strip()
    if raw.startswith("data:"):
        if "," not in raw:
            raise ValueError("Invalid data URL for paint layer")
        raw = raw.split(",", 1)[1]

    try:
        data = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise ValueError(f"Invalid paint layer base64: {exc}") from exc

    try:
        with Image.open(BytesIO(data)) as paint:
            return paint.convert("RGBA")
    except Exception as exc:
        raise ValueError(f"Invalid paint layer image: {exc}") from exc


def _copy_output_path(src: Path) -> Path:
    candidate = src.with_name(f"{src.stem}_paint{src.suffix}")
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = src.with_name(f"{src.stem}_paint_{counter}{src.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _save_image(image: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()

    if ext in {".jpg", ".jpeg"}:
        image.convert("RGB").save(out_path, "JPEG", quality=95)
        return
    if ext == ".png":
        image.save(out_path, "PNG", optimize=True)
        return
    if ext == ".webp":
        image.save(out_path, "WEBP", quality=95, method=6)
        return
    if ext == ".bmp":
        image.convert("RGB").save(out_path, "BMP")
        return

    image.convert("RGB").save(out_path.with_suffix(".jpg"), "JPEG", quality=95)


def _prepare_color_output(image_path: Path, paint_png_base64: str) -> Dict[str, Any]:
    lines: List[str] = [f"[{_ts()}] Loaded image: {image_path}"]
    if not image_path.exists() or not image_path.is_file():
        return {"ok": False, "error": "Image not found", "log": lines}

    try:
        with Image.open(image_path) as source_raw:
            source = source_raw.convert("RGBA")
        paint = _decode_paint_layer(paint_png_base64)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "log": lines}
    except Exception as exc:
        return {"ok": False, "error": f"Failed loading image: {exc}", "log": lines}

    if paint.size != source.size:
        paint = paint.resize(source.size, resample=_R_BILINEAR)
        lines.append(f"[{_ts()}] Resized paint layer to source size: {source.width}x{source.height}")

    if paint.getchannel("A").getbbox() is None:
        lines.append(f"[{_ts()}] Nothing to apply: color layer is empty.")
        return {"ok": False, "error": "Color layer empty", "log": lines}

    try:
        output = Image.alpha_composite(source, paint)
    except Exception as exc:
        return {"ok": False, "error": f"Failed compositing color layer: {exc}", "log": lines}

    lines.append(f"[{_ts()}] Prepared original-resolution color paint layer.")
    return {"ok": True, "image": output, "log": lines}


def apply_color_paint(
    image_path: Path,
    paint_png_base64: str,
    backup: bool = True,
    output_mode: str = "overwrite",
    action_count: int = 0,
) -> Dict[str, Any]:
    prepared = _prepare_color_output(image_path, paint_png_base64)
    if not prepared.get("ok"):
        return prepared

    lines = list(prepared.get("log") or [])
    output = prepared.get("image")
    if not isinstance(output, Image.Image):
        return {"ok": False, "error": "Paint output build failed", "log": lines}

    mode = str(output_mode or "overwrite").strip().lower()
    if mode not in {"overwrite", "copy"}:
        mode = "overwrite"

    backup_path: Optional[Path] = None
    try:
        if mode == "overwrite":
            save_path = image_path
            if backup:
                backup_path = Path(str(image_path) + ".bak")
                shutil.copy2(image_path, backup_path)
                lines.append(f"[{_ts()}] Backup created: {backup_path}")
        else:
            save_path = _copy_output_path(image_path)
            if backup:
                lines.append(f"[{_ts()}] Copy mode selected: backup skipped.")

        _save_image(output, save_path)
    except Exception as exc:
        return {"ok": False, "error": f"Save failed: {exc}", "log": lines}

    lines.append(f"[{_ts()}] Applied Color Paint | actions={max(0, int(action_count or 0))} | output={mode}")
    lines.append(f"[{_ts()}] Saved: {save_path}")
    return {
        "ok": True,
        "saved_path": save_path,
        "backup_path": backup_path,
        "output_mode": mode,
        "log": lines,
    }


def handle(form: Any, ctx: Dict[str, Any]) -> Tuple[str, str]:
    return "color_brush", log_join([
        "Color Brush uses async API endpoints.",
        "Use tab: Color Brush.",
    ])
