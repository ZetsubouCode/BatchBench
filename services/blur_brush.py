from __future__ import annotations

import base64
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageFilter

from utils.io import log_join

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODE_BLUR = "blur"
MODE_MOSAIC = "mosaic"
MODE_BOX = "box"
VALID_MODES = {MODE_BLUR, MODE_MOSAIC, MODE_BOX}

try:
    _R_BILINEAR = Image.Resampling.BILINEAR
    _R_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    _R_BILINEAR = Image.BILINEAR
    _R_NEAREST = Image.NEAREST


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _norm_mode(raw: Any) -> str:
    mode = str(raw or MODE_BLUR).strip().lower()
    if mode not in VALID_MODES:
        return MODE_BLUR
    return mode


def _clamp_int(raw: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        val = int(str(raw).strip())
    except Exception:
        val = default
    return max(min_value, min(max_value, val))


def _normalize_exts(exts: Optional[Sequence[str]] = None) -> set:
    if not exts:
        return set(IMG_EXTS)
    out = set()
    for raw in exts:
        val = str(raw or "").strip().lower()
        if not val:
            continue
        if not val.startswith("."):
            val = "." + val
        out.add(val)
    return out or set(IMG_EXTS)


def list_images(folder: Path, recursive: bool = True, exts: Optional[Sequence[str]] = None) -> List[str]:
    extset = _normalize_exts(exts)
    if not folder.exists() or not folder.is_dir():
        return []
    files: List[Path] = []
    if recursive:
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in extset:
                files.append(p)
    else:
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in extset:
                files.append(p)
    files.sort(key=lambda p: str(p).lower())
    return [p.relative_to(folder).as_posix() for p in files]


def _decode_mask(mask_png_base64: str) -> Image.Image:
    if not mask_png_base64:
        raise ValueError("mask_png_base64 is required")

    raw = mask_png_base64.strip()
    if raw.startswith("data:"):
        if "," not in raw:
            raise ValueError("Invalid data URL for mask")
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise ValueError(f"Invalid mask base64: {exc}") from exc

    try:
        with Image.open(BytesIO(data)) as m:
            return m.convert("L")
    except Exception as exc:
        raise ValueError(f"Invalid mask image: {exc}") from exc


def _mosaic(img: Image.Image, block_size: int) -> Image.Image:
    w, h = img.size
    block = max(1, int(block_size))
    small_w = max(1, w // block)
    small_h = max(1, h // block)
    return img.resize((small_w, small_h), resample=_R_BILINEAR).resize((w, h), resample=_R_NEAREST)


def _build_effect(img: Image.Image, mode: str, strength: int) -> Image.Image:
    if mode == MODE_MOSAIC:
        return _mosaic(img, strength)
    if mode == MODE_BOX:
        return img.filter(ImageFilter.BoxBlur(radius=max(1, strength)))
    return img.filter(ImageFilter.GaussianBlur(radius=max(1, strength)))


def _save_image(img: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        img.convert("RGB").save(out_path, "JPEG", quality=95)
        return
    if ext == ".png":
        img.save(out_path, "PNG", optimize=True)
        return
    if ext == ".webp":
        img.save(out_path, "WEBP", quality=95, method=6)
        return
    if ext == ".bmp":
        img.save(out_path, "BMP")
        return
    img.convert("RGB").save(out_path.with_suffix(".jpg"), "JPEG", quality=95)


def _copy_output_path(src: Path) -> Path:
    base = src.with_name(f"{src.stem}_brush{src.suffix}")
    if not base.exists():
        return base
    bump = 1
    while True:
        candidate = src.with_name(f"{src.stem}_brush_{bump}{src.suffix}")
        if not candidate.exists():
            return candidate
        bump += 1


def _prepare_brush_output(
    image_path: Path,
    mask_png_base64: str,
    mode: str,
    strength: int,
    feather: int = 0,
) -> Dict[str, Any]:
    lines: List[str] = []
    lines.append(f"[{_ts()}] Loaded image: {image_path}")

    if not image_path.exists() or not image_path.is_file():
        return {"ok": False, "error": "Image not found", "log": lines}

    mode_n = _norm_mode(mode)
    strength_n = _clamp_int(strength, default=12, min_value=1, max_value=120)
    feather_n = _clamp_int(feather, default=0, min_value=0, max_value=64)

    try:
        mask = _decode_mask(mask_png_base64)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "log": lines}

    try:
        with Image.open(image_path) as source_raw:
            has_alpha = "A" in source_raw.getbands()
            source = source_raw.convert("RGBA" if has_alpha else "RGB")

        if mask.size != source.size:
            resample = _R_NEAREST if mode_n == MODE_MOSAIC else _R_BILINEAR
            mask = mask.resize(source.size, resample=resample)
            lines.append(f"[{_ts()}] Resized mask: {mask.size[0]}x{mask.size[1]}")

        if feather_n > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_n))

        if mask.getbbox() is None:
            lines.append(f"[{_ts()}] Nothing to apply: mask is empty.")
            return {"ok": False, "error": "Mask empty", "log": lines}

        effect = _build_effect(source, mode_n, strength_n)
        out = Image.composite(effect, source, mask)
    except Exception as exc:
        return {"ok": False, "error": f"Failed preparing effect: {exc}", "log": lines}

    lines.append(f"[{_ts()}] Prepared mode={mode_n} strength={strength_n} feather={feather_n}")
    return {
        "ok": True,
        "image": out,
        "mode": mode_n,
        "strength": strength_n,
        "feather": feather_n,
        "log": lines,
    }


def apply_brush_effect(
    image_path: Path,
    mask_png_base64: str,
    mode: str,
    strength: int,
    feather: int = 0,
    backup: bool = True,
    output_mode: str = "overwrite",
) -> Dict[str, Any]:
    prepared = _prepare_brush_output(
        image_path=image_path,
        mask_png_base64=mask_png_base64,
        mode=mode,
        strength=strength,
        feather=feather,
    )
    if not prepared.get("ok"):
        return prepared

    lines = list(prepared.get("log") or [])
    mode_n = str(prepared.get("mode") or MODE_BLUR)
    strength_n = int(prepared.get("strength") or 12)
    feather_n = int(prepared.get("feather") or 0)
    out = prepared.get("image")
    if not isinstance(out, Image.Image):
        return {"ok": False, "error": "Preview image build failed", "log": lines}

    output_mode_n = str(output_mode or "overwrite").strip().lower()
    if output_mode_n not in {"overwrite", "copy"}:
        output_mode_n = "overwrite"

    backup_path: Optional[Path] = None
    try:
        if output_mode_n == "overwrite":
            save_path = image_path
            if backup:
                backup_path = Path(str(image_path) + ".bak")
                shutil.copy2(image_path, backup_path)
                lines.append(f"[{_ts()}] Backup created: {backup_path}")
        else:
            save_path = _copy_output_path(image_path)
            if backup:
                lines.append(f"[{_ts()}] Backup skipped in copy mode.")

        _save_image(out, save_path)
    except Exception as exc:
        return {"ok": False, "error": f"Save failed: {exc}", "log": lines}

    lines.append(
        f"[{_ts()}] Applied mode={mode_n} strength={strength_n} feather={feather_n} output={output_mode_n}"
    )
    lines.append(f"[{_ts()}] Saved: {save_path}")
    return {
        "ok": True,
        "saved_path": save_path,
        "backup_path": backup_path,
        "mode": mode_n,
        "strength": strength_n,
        "feather": feather_n,
        "output_mode": output_mode_n,
        "log": lines,
    }


def preview_brush_effect(
    image_path: Path,
    mask_png_base64: str,
    mode: str,
    strength: int,
    feather: int = 0,
    preview_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    prepared = _prepare_brush_output(
        image_path=image_path,
        mask_png_base64=mask_png_base64,
        mode=mode,
        strength=strength,
        feather=feather,
    )
    if not prepared.get("ok"):
        return prepared

    lines = list(prepared.get("log") or [])
    mode_n = str(prepared.get("mode") or MODE_BLUR)
    strength_n = int(prepared.get("strength") or 12)
    feather_n = int(prepared.get("feather") or 0)
    out = prepared.get("image")
    if not isinstance(out, Image.Image):
        return {"ok": False, "error": "Preview image build failed", "log": lines}

    if preview_size and len(preview_size) == 2:
        w = _clamp_int(preview_size[0], default=out.width, min_value=1, max_value=4096)
        h = _clamp_int(preview_size[1], default=out.height, min_value=1, max_value=4096)
        if (w, h) != out.size:
            out = out.resize((w, h), resample=_R_BILINEAR)
            lines.append(f"[{_ts()}] Preview resized: {w}x{h}")

    try:
        buf = BytesIO()
        out.save(buf, "PNG", optimize=True)
        preview_bytes = buf.getvalue()
    except Exception as exc:
        return {"ok": False, "error": f"Preview encode failed: {exc}", "log": lines}

    lines.append(f"[{_ts()}] Preview ready mode={mode_n} strength={strength_n} feather={feather_n}")
    return {
        "ok": True,
        "preview_bytes": preview_bytes,
        "preview_mime": "image/png",
        "mode": mode_n,
        "strength": strength_n,
        "feather": feather_n,
        "log": lines,
    }


def handle(form, ctx) -> Tuple[str, str]:
    lines = [
        "Brush Blur tool uses async API endpoints.",
        "Use tab: Brush Blur.",
    ]
    return "blur_brush", log_join(lines)
