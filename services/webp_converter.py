from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from PIL import Image
from utils.io import readable_path, ensure_out_dir
from utils.tool_result import build_tool_result

COMMON_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def _supported_image_exts() -> set[str]:
    exts = {ext.lower() for ext in Image.registered_extensions().keys()}
    if not exts:
        return set(COMMON_IMAGE_EXTS)
    return exts | COMMON_IMAGE_EXTS


def _build_output_names(files: List[Path]) -> Dict[Path, str]:
    out_names: Dict[Path, str] = {}
    used: set[str] = set()
    for p in sorted(files, key=lambda x: x.name.lower()):
        candidate = f"{p.stem}.png"
        if candidate in used:
            ext_tag = p.suffix.lower().lstrip(".") or "img"
            candidate = f"{p.stem}_{ext_tag}.png"
        idx = 2
        while candidate in used:
            candidate = f"{p.stem}_{idx}.png"
            idx += 1
        used.add(candidate)
        out_names[p] = candidate
    return out_names


def handle(form, ctx):
    active_tab = "webp"
    src_raw = (form.get("src_png", "") or form.get("src_webp", "") or "").strip()
    dst_raw = (form.get("dst_png", "") or form.get("dst_webp", "") or "").strip()
    src = readable_path(src_raw)
    dst = readable_path(dst_raw)
    lines: List[str] = []

    def _done(ok: bool, error: str = ""):
        return build_tool_result(active_tab, lines, ok=ok, error=error)

    if not src_raw:
        lines.append("Source folder is required.")
        return _done(False, "Source folder is required.")
    elif not dst_raw:
        lines.append("Output folder is required.")
        return _done(False, "Output folder is required.")
    elif not src.exists() or not src.is_dir():
        lines.append("Source folder not found.")
        return _done(False, "Source folder not found.")
    else:
        ensure_out_dir(dst)
        supported_exts = _supported_image_exts()
        files = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in supported_exts]
        out_name_map = _build_output_names(files)
        count = 0
        skipped = 0
        errors = 0
        workers = max(1, min(8, (os.cpu_count() or 4)))

        if not files:
            lines.append("No image files found in source folder.")
            return _done(False, "No image files found in source folder.")

        def _convert(path: Path, out_name: str):
            out = dst / out_name
            try:
                if path.resolve() == out.resolve():
                    return "SKIP", path.name, out.name, "Source and output are the same file."
            except Exception:
                pass
            with Image.open(path) as im:
                if im.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16", "F"}:
                    im = im.convert("RGB")
                im.save(out, "PNG")
            return "OK", path.name, out.name, ""

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_convert, p, out_name_map[p]): p for p in files}
            for fut in as_completed(futures):
                src_path = futures[fut]
                try:
                    status, src_name, out_name, message = fut.result()
                    if status == "SKIP":
                        lines.append(f"[SKIP] {src_name}: {message}")
                        skipped += 1
                    else:
                        lines.append(f"Converted: {src_name} -> {out_name}")
                        count += 1
                except Exception as e:
                    lines.append(f"[ERROR] {src_path.name}: {e}")
                    errors += 1
        lines.append(f"Done. {count} files converted, {skipped} files skipped, {errors} errors.")
        return _done(errors == 0, "" if errors == 0 else f"{errors} conversion(s) failed.")
    return _done(False, "Invalid input.")
