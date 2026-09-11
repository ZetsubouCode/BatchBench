from typing import List, Optional, Tuple
from pathlib import Path
import re
from PIL import Image

from utils.io import readable_path, ensure_out_dir
from utils.merge_groups_core import merge_many
from utils.tool_result import build_tool_result

Color = Tuple[int, int, int]
DEFAULT_STRIPE_COLORS: List[Color] = [(255, 255, 255)]


def _natural_key(name: str):
    parts = []
    for token in re.findall(r"\d+|\D+", name):
        if token.isdigit():
            parts.append((0, int(token)))
        else:
            parts.append((1, token.lower()))
    return parts


def _created_ts(path: Path) -> float:
    stat = path.stat()
    return float(getattr(stat, "st_birthtime", stat.st_ctime))


def _sort_paths(paths: List[Path], sort_by: str, sort_dir: str) -> List[Path]:
    sort_by = sort_by if sort_by in ("name", "ctime") else "name"
    sort_dir = sort_dir if sort_dir in ("asc", "desc") else "asc"
    if sort_by == "ctime":
        multiplier = -1.0 if sort_dir == "desc" else 1.0
        return sorted(paths, key=lambda p: (_created_ts(p) * multiplier, _natural_key(p.name)))
    return sorted(paths, key=lambda p: _natural_key(p.name), reverse=(sort_dir == "desc"))


def _list_images(folder: Path, glob_pat: str, exts: List[str], sort_by: str, sort_dir: str) -> List[Path]:
    extset = {(e if e.startswith(".") else f".{e}").lower().strip() for e in exts if e.strip()}
    return _sort_paths([p for p in folder.glob(glob_pat) if p.is_file() and p.suffix.lower() in extset], sort_by, sort_dir)


def _fill_small_gaps(rows: List[bool], max_gap: int) -> List[bool]:
    if max_gap <= 0:
        return rows
    out = rows[:]
    i = 0
    n = len(rows)
    while i < n:
        if out[i]:
            i += 1
            continue
        start = i
        while i < n and not out[i]:
            i += 1
        end = i - 1
        if start > 0 and i < n and (end - start + 1) <= max_gap:
            for k in range(start, end + 1):
                out[k] = True
    return out


def _parse_stripe_colors(raw: str) -> Tuple[List[Color], List[str]]:
    colors: List[Color] = []
    invalid: List[str] = []
    seen = set()
    for token in re.split(r"[\s,;]+", str(raw or "")):
        value = token.strip()
        if not value:
            continue
        if value.startswith("#"):
            value = value[1:]
        if len(value) == 3 and re.fullmatch(r"[0-9a-fA-F]{3}", value):
            value = "".join(ch * 2 for ch in value)
        if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
            invalid.append(token.strip())
            continue
        color = (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
        if color not in seen:
            seen.add(color)
            colors.append(color)
    return colors or DEFAULT_STRIPE_COLORS[:], invalid


def _format_stripe_colors(colors: List[Color]) -> str:
    return ", ".join(f"#{r:02X}{g:02X}{b:02X}" for r, g, b in colors)


def _find_stripes(
    img: Image.Image,
    threshold: int,
    min_height: int,
    ratio: float,
    max_gap: int,
    stripe_colors: Optional[List[Color]] = None,
) -> List[Tuple[int, int]]:
    import numpy as np

    rgb = np.asarray(img.convert("RGB"), dtype=np.int16)
    if rgb.size == 0:
        return []
    h = rgb.shape[0]
    tolerance = 255 - max(0, min(255, int(threshold)))
    matches = np.zeros(rgb.shape[:2], dtype=bool)
    for color in stripe_colors or DEFAULT_STRIPE_COLORS:
        target = np.asarray(color, dtype=np.int16)
        matches |= (np.abs(rgb - target).max(axis=2) <= tolerance)
    rows = (matches.mean(axis=1) >= ratio).tolist()
    rows = _fill_small_gaps(rows, max_gap)

    stripes: List[Tuple[int, int]] = []
    y = 0
    while y < h:
        if not rows[y]:
            y += 1
            continue
        start = y
        while y < h and rows[y]:
            y += 1
        end = y - 1
        if (end - start + 1) >= min_height:
            stripes.append((start, end))
    return stripes


def _segments_from_stripes(img_height: int, stripes: List[Tuple[int, int]], min_panel: int):
    segments: List[Tuple[int, int]] = []
    last = 0
    for start, end in stripes:
        if start - last >= min_panel:
            segments.append((last, start))
        last = end + 1
    if img_height - last >= min_panel:
        segments.append((last, img_height))
    return segments


def _chapter_dirs(folder: Path, sort_by: str, sort_dir: str) -> List[Path]:
    pat = re.compile(r"^(?:chapter\s*)?\d+", re.IGNORECASE)
    subs = [d for d in folder.iterdir() if d.is_dir()]
    numbered = [d for d in subs if pat.match(d.name)]
    return _sort_paths(numbered or subs, sort_by, sort_dir)


def _targets(folder: Path, glob_pat: str, exts: List[str], sort_by: str, sort_dir: str) -> List[Tuple[Path, List[Path]]]:
    targets: List[Tuple[Path, List[Path]]] = []
    here = _list_images(folder, glob_pat, exts, sort_by, sort_dir)
    if here:
        targets.append((folder, here))
    for sub in _chapter_dirs(folder, sort_by, sort_dir):
        imgs = _list_images(sub, glob_pat, exts, sort_by, sort_dir)
        if imgs:
            targets.append((sub, imgs))
    return targets


def handle(form, ctx):
    active_tab = "webtoon"
    folder_raw = (form.get("wt_folder", "") or "").strip()
    folder = readable_path(folder_raw)
    out_dir_raw = (form.get("wt_out_dir", "") or "").strip()
    out_dir = readable_path(out_dir_raw) if out_dir_raw else None
    glob_pat = form.get("wt_glob", "*.*").strip() or "*.*"
    sort_by = (form.get("wt_sort_by", "name") or "name").strip().lower()
    sort_dir = (form.get("wt_sort_dir", "asc") or "asc").strip().lower()
    sort_by = sort_by if sort_by in ("name", "ctime") else "name"
    sort_dir = sort_dir if sort_dir in ("asc", "desc") else "asc"

    exts_raw = (form.get("wt_exts", ".png,.jpg,.jpeg,.webp") or ".png,.jpg,.jpeg,.webp").strip()
    exts = [e.strip().lower() for e in exts_raw.split(",") if e.strip()] or [".png"]

    resize_mode = form.get("wt_resize", "match-width")
    white_threshold = int(form.get("wt_white_threshold", "245") or 245)
    stripe_colors, invalid_stripe_colors = _parse_stripe_colors(form.get("wt_stripe_colors", "#FFFFFF"))
    row_ratio = float(form.get("wt_row_ratio", "98") or 98)
    min_stripe = int(form.get("wt_min_stripe", "12") or 12)
    max_gap = int(form.get("wt_max_gap", "2") or 2)
    min_panel = int(form.get("wt_min_panel", "128") or 128)

    save_strip = bool(form.get("wt_save_strip"))
    overwrite = bool(form.get("wt_overwrite"))
    dry_run = bool(form.get("wt_dry_run"))

    # Clamp inputs to sane ranges
    white_threshold = max(0, min(255, white_threshold))
    row_ratio = max(0.0, min(100.0, row_ratio)) / 100.0
    min_stripe = max(1, min_stripe)
    max_gap = max(0, max_gap)
    min_panel = max(1, min_panel)
    resize_mode = resize_mode if resize_mode in ("match-width", "none") else "match-width"

    lines: List[str] = []
    def _done(ok: bool, error: str = ""):
        return build_tool_result(active_tab, lines, ok=ok, error=error)
    if not folder_raw:
        lines.append("Source folder is required.")
        return _done(False, "Source folder is required.")
    if not folder.exists() or not folder.is_dir():
        lines.append("Source folder not found.")
        return _done(False, "Source folder not found.")

    targets = _targets(folder, glob_pat, exts, sort_by, sort_dir)
    if not targets:
        lines.append(f"No images found with pattern '{glob_pat}' and extensions {', '.join(exts)}.")
        return _done(False, "No images found for webtoon split.")

    lines.append(f"Root: {folder} | Chapters detected: {len(targets)}")
    lines.append(f"Order: {sort_by} {sort_dir}")
    lines.append(f"Stripe colors: {_format_stripe_colors(stripe_colors)}")
    if invalid_stripe_colors:
        lines.append(f"Ignored invalid stripe color(s): {', '.join(invalid_stripe_colors)}")
    errors = 0
    for i, (chap_path, pages) in enumerate(targets, start=1):
        lines.append(f"[{i}/{len(targets)}] {chap_path.name}: {len(pages)} page(s)")
        target_out = (out_dir / chap_path.name) if out_dir else (chap_path / "_panels")
        if not dry_run:
            ensure_out_dir(target_out)

        try:
            merged = merge_many(pages, "v", "center", 0, "#FFFFFF", resize_mode)
            lines.append(f"  Merged size: {merged.width}x{merged.height}px (gapless stack)")

            stripes = _find_stripes(merged, white_threshold, min_stripe, row_ratio, max_gap, stripe_colors)
            segments = _segments_from_stripes(merged.height, stripes, min_panel)

            lines.append(
                f"  Stripe rule: >= {min_stripe}px tall, row >= {int(row_ratio*100)}% matching selected colors "
                f"(threshold {white_threshold}, tolerance +/-{255 - white_threshold}/channel)"
            )
            lines.append(f"  Stripes found: {len(stripes)} | Planned slices: {len(segments)} (>= {min_panel}px)")
            for idx, (start, end) in enumerate(stripes, start=1):
                lines.append(f"    Stripe {idx}: y={start}..{end} (h={end-start+1})")

            if dry_run:
                merged.close()
                lines.append("  Dry run: no files written.")
                continue

            if save_strip:
                strip_path = target_out / f"{chap_path.name}_strip.png"
                if strip_path.exists() and not overwrite:
                    lines.append(f"  Skip strip (exists): {strip_path.name}")
                else:
                    merged.save(strip_path)
                    lines.append(f"  Saved strip -> {strip_path.name}")

            pad = max(3, len(str(len(segments)))) if segments else 3
            saved = 0
            for idx, (top, bottom) in enumerate(segments, start=1):
                if bottom <= top:
                    continue
                out_path = target_out / f"{chap_path.name}_{idx:0{pad}d}.png"
                if out_path.exists() and not overwrite:
                    lines.append(f"  Skip slice (exists): {out_path.name}")
                    continue
                try:
                    piece = merged.crop((0, top, merged.width, bottom))
                    piece.save(out_path)
                    piece.close()
                    saved += 1
                    lines.append(f"  Saved slice {idx} -> {out_path.name} (y={top}..{bottom-1})")
                except Exception as e:
                    errors += 1
                    lines.append(f"  [ERROR] slice {idx} ({out_path.name}): {e}")
            merged.close()
            lines.append(f"  Done: {saved} slice(s) into {target_out}")
        except Exception as e:
            errors += 1
            lines.append(f"[ERROR] {chap_path.name}: {e}")

    if dry_run:
        lines.append("Dry run finished: no files were saved.")
        return _done(True)
    return _done(errors == 0, "" if errors == 0 else f"{errors} webtoon split operation(s) failed.")
