from __future__ import annotations

from typing import Tuple, List, Dict, Any, Optional, Iterable
from pathlib import Path

from PIL import Image

from utils.io import readable_path, ensure_out_dir, log_join
from utils.image_ops import apply_preset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PARAM_KEYS = [
    "exposure_ev",
    "brightness",
    "contrast",
    "highlights",
    "shadows",
    "saturation",
    "warmth",
    "tint",
    "sharpness",
    "vignette",
]


def _pfloat(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        s = str(val).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _pint(val: Any, default: int = 0) -> int:
    try:
        if val is None:
            return default
        return int(str(val).strip() or default)
    except Exception:
        return default


def _pbool(val: Any, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def _iter_images(src: Path, recursive: bool, skip_root: Optional[Path]) -> Iterable[Path]:
    if recursive:
        for p in src.rglob("*"):
            if not p.is_file():
                continue
            if skip_root is not None:
                try:
                    if skip_root == p or skip_root in p.parents:
                        continue
                except Exception:
                    pass
            if p.suffix.lower() in IMG_EXTS:
                yield p
        return

    for p in src.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def _pick_out_ext(input_ext: str, output_format: str) -> str:
    fmt = (output_format or "same").strip().lower()
    if fmt in {"same", ""}:
        return input_ext if input_ext in IMG_EXTS else ".jpg"
    if fmt in {"jpg", "jpeg"}:
        return ".jpg"
    if fmt == "png":
        return ".png"
    if fmt == "webp":
        return ".webp"
    return input_ext if input_ext in IMG_EXTS else ".jpg"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    base = path.with_suffix("")
    ext = path.suffix
    bump = 1
    while True:
        candidate = Path(f"{base}_{bump}{ext}")
        if not candidate.exists():
            return candidate
        bump += 1


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


def handle(form, ctx) -> Tuple[str, str]:
    """
    Batch color/lighting adjustment.

    - Choose preset (base params).
    - UI loads preset values into sliders (as input params).
    - Tweak sliders, then process images into output folder (non-destructive).
    """
    active_tab = "batch"
    presets: Dict[str, Dict[str, Any]] = ctx.get("presets", {}) or {}

    src_raw = (form.get("src_batch") or "").strip()
    dst_raw = (form.get("dst_batch") or "").strip()
    preset_name = (form.get("preset") or "").strip()

    suffix = (form.get("suffix") or "_adj").strip() or "_adj"
    limit = max(0, _pint(form.get("limit"), 0))
    recursive = _pbool(form.get("recursive"), True)
    output_format = (form.get("output_format") or "same").strip().lower()

    work_dir = readable_path(ctx.get("work_dir", "")) if ctx.get("work_dir") else None
    src = readable_path(src_raw) if src_raw else (work_dir if work_dir else readable_path(""))
    dst = readable_path(dst_raw) if dst_raw else Path("")

    lines: List[str] = []

    if not preset_name:
        lines.append("Preset is required.")
        return active_tab, log_join(lines)
    if preset_name not in presets:
        lines.append(f"Preset not found: {preset_name}")
        return active_tab, log_join(lines)

    if not src or not src.exists() or not src.is_dir():
        lines.append("Source folder not found.")
        if work_dir:
            lines.append(f"Hint: leave Source empty to use WORK_DIR: {work_dir}")
        return active_tab, log_join(lines)

    # Default output under source if user didn't fill it.
    if not dst_raw:
        base_out = src / "_batch_adjust_out"
        dst = base_out
        bump = 1
        while dst.exists():
            dst = Path(f"{str(base_out)}_{bump}")
            bump += 1

    ensure_out_dir(dst)

    # Final cfg = preset base + overrides (if provided)
    cfg: Dict[str, Any] = dict(presets[preset_name])
    for k in PARAM_KEYS:
        v = _pfloat(form.get(k))
        if v is not None:
            cfg[k] = v

    # Avoid re-processing output folder if dst is inside src and recursive is ON.
    skip_root = None
    try:
        if recursive:
            src_res = src.resolve()
            dst_res = dst.resolve()
            if dst_res == src_res or src_res in dst_res.parents or dst_res in src_res.parents:
                skip_root = dst_res
    except Exception:
        skip_root = None

    lines.append("Batch Adjust")
    lines.append(f"- src: {src}")
    lines.append(f"- dst: {dst}")
    lines.append(f"- preset: {preset_name}")
    lines.append(f"- recursive: {recursive}")
    lines.append(f"- output_format: {output_format}")
    lines.append(f"- suffix: {suffix}")
    if limit:
        lines.append(f"- limit: {limit}")
    lines.append("-")

    count = 0
    errors = 0
    shown = 0
    max_show = 30

    for p in _iter_images(src, recursive=recursive, skip_root=skip_root):
        try:
            with Image.open(p) as im:
                input_ext = p.suffix.lower()
                out_ext = _pick_out_ext(input_ext, output_format)

                alpha = None
                if im.mode in ("RGBA", "LA"):
                    try:
                        alpha = im.getchannel("A")
                    except Exception:
                        alpha = None

                rgb = im.convert("RGB")
                out_rgb = apply_preset(rgb, cfg)

                # restore alpha if output supports it
                if alpha is not None and out_ext in {".png", ".webp"}:
                    out_img = out_rgb.convert("RGBA")
                    out_img.putalpha(alpha)
                else:
                    out_img = out_rgb

                try:
                    rel = p.relative_to(src)
                except Exception:
                    rel = Path(p.name)

                out_dir = dst / rel.parent
                out_name = f"{rel.stem}{suffix}{out_ext}"
                out_path = _unique_path(out_dir / out_name)

                _save_image(out_img, out_path)

            count += 1
            if shown < max_show:
                try:
                    lines.append(f"[OK] {out_path.relative_to(dst).as_posix()}")
                except Exception:
                    lines.append(f"[OK] {out_path.name}")
                shown += 1

            if limit and count >= limit:
                break

        except Exception as e:
            errors += 1
            lines.append(f"[ERROR] {p.name}: {e}")

    if count > shown:
        lines.append(f"... ({count - shown} more files)")
    lines.append(f"Done. {count} files processed. Errors: {errors}.")
    return active_tab, log_join(lines)
