from __future__ import annotations

import json
import posixpath
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET

from utils.io import readable_path
from utils.parse import parse_bool, parse_int
from utils.tool_result import build_tool_result


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
SVG_EXT = ".svg"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")
XHTML_HREF_RE = re.compile(
    r"""(?:src|href|xlink:href)\s*=\s*["']([^"']+)["']""",
    re.IGNORECASE,
)


def _tag_name(elem: ET.Element) -> str:
    return str(elem.tag).rsplit("}", 1)[-1].lower()


def _clean_href(href: str) -> str:
    value = str(href or "").strip()
    if not value:
        return ""
    parts = urlsplit(value)
    if parts.scheme or parts.netloc:
        return ""
    return unquote(parts.path or value.split("#", 1)[0]).replace("\\", "/")


def _safe_zip_path(path: str) -> str:
    value = str(path or "").replace("\\", "/").strip("/")
    if not value:
        return ""
    normalized = posixpath.normpath(value)
    if normalized in {"", "."}:
        return ""
    if normalized.startswith("../") or normalized == ".." or posixpath.isabs(normalized):
        return ""
    return normalized


def _resolve_member(base_file: str, href: str) -> str:
    clean = _clean_href(href)
    if not clean:
        return ""
    if clean.startswith("/"):
        return _safe_zip_path(clean)
    base_dir = posixpath.dirname(base_file)
    return _safe_zip_path(posixpath.join(base_dir, clean))


def _allowed_ext(path: str, extract_svg: bool) -> bool:
    ext = Path(path).suffix.lower()
    return ext in IMAGE_EXTS or (extract_svg and ext == SVG_EXT)


def _dedupe(paths: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _sanitize_stem(stem: str) -> str:
    clean = UNSAFE_NAME_RE.sub("_", str(stem or "").strip())
    clean = clean.strip(" ._")
    return clean or "epub"


def _sanitize_filename(name: str) -> str:
    clean = UNSAFE_NAME_RE.sub("_", Path(name).name)
    clean = clean.strip(" ._")
    return clean or "image"


def _unique_path(out_dir: Path, filename: str, used_names: Set[str], overwrite: bool) -> Path:
    filename = _sanitize_filename(filename)
    stem = Path(filename).stem or "image"
    suffix = Path(filename).suffix
    candidate = filename
    idx = 1
    while candidate.lower() in used_names or ((out_dir / candidate).exists() and not overwrite):
        candidate = f"{stem}_{idx}{suffix}"
        idx += 1
    used_names.add(candidate.lower())
    return out_dir / candidate


def _zip_image_order(zf: zipfile.ZipFile, extract_svg: bool) -> List[str]:
    paths: List[str] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        member = _safe_zip_path(info.filename)
        if member and member == info.filename.replace("\\", "/") and _allowed_ext(member, extract_svg):
            paths.append(member)
    return _dedupe(paths)


def _read_xml(zf: zipfile.ZipFile, member: str) -> ET.Element:
    with zf.open(member) as fh:
        return ET.fromstring(fh.read())


def _find_rootfile_path(zf: zipfile.ZipFile) -> str:
    root = _read_xml(zf, "META-INF/container.xml")
    for elem in root.iter():
        if _tag_name(elem) == "rootfile":
            full_path = _safe_zip_path(elem.attrib.get("full-path", ""))
            if full_path:
                return full_path
    raise ValueError("OPF package path not found in META-INF/container.xml")


def _parse_manifest_and_spine(
    zf: zipfile.ZipFile,
    opf_path: str,
) -> Tuple[Dict[str, Dict[str, str]], List[str], Optional[str]]:
    opf_root = _read_xml(zf, opf_path)
    opf_dir = posixpath.dirname(opf_path)
    manifest: Dict[str, Dict[str, str]] = {}
    spine_ids: List[str] = []
    cover_id: Optional[str] = None

    for elem in opf_root.iter():
        name = _tag_name(elem)
        if name == "item":
            item_id = elem.attrib.get("id", "")
            href = elem.attrib.get("href", "")
            if not item_id or not href:
                continue
            full_path = _safe_zip_path(posixpath.join(opf_dir, _clean_href(href)))
            if not full_path:
                continue
            manifest[item_id] = {
                "path": full_path,
                "media_type": elem.attrib.get("media-type", ""),
                "properties": elem.attrib.get("properties", ""),
            }
            if "cover-image" in elem.attrib.get("properties", "").split():
                cover_id = item_id
        elif name == "itemref":
            idref = elem.attrib.get("idref", "")
            if idref:
                spine_ids.append(idref)
        elif name == "meta":
            if elem.attrib.get("name", "").lower() == "cover":
                content = elem.attrib.get("content", "")
                if content:
                    cover_id = content

    return manifest, spine_ids, cover_id


def _image_refs_from_xhtml(zf: zipfile.ZipFile, xhtml_path: str) -> List[str]:
    try:
        with zf.open(xhtml_path) as fh:
            raw = fh.read()
    except Exception:
        return []

    refs: List[str] = []
    try:
        root = ET.fromstring(raw)
        for elem in root.iter():
            if _tag_name(elem) not in {"img", "image"}:
                continue
            for attr in ("src", "href", XLINK_HREF):
                value = elem.attrib.get(attr)
                if value:
                    refs.append(value)
    except ET.ParseError:
        text = raw.decode("utf-8", errors="ignore")
        refs.extend(match.group(1) for match in XHTML_HREF_RE.finditer(text))
    return refs


def _reading_order(
    zf: zipfile.ZipFile,
    name_set: Set[str],
    *,
    extract_cover: bool,
    extract_svg: bool,
) -> Tuple[List[str], str, Optional[str]]:
    opf_path = _find_rootfile_path(zf)
    manifest, spine_ids, cover_id = _parse_manifest_and_spine(zf, opf_path)

    cover_path: Optional[str] = None
    if cover_id and cover_id in manifest:
        candidate = manifest[cover_id]["path"]
        if candidate in name_set and _allowed_ext(candidate, extract_svg):
            cover_path = candidate

    ordered: List[str] = []
    for idref in spine_ids:
        item = manifest.get(idref)
        if not item:
            continue
        xhtml_path = item["path"]
        if xhtml_path not in name_set:
            continue
        for href in _image_refs_from_xhtml(zf, xhtml_path):
            member = _resolve_member(xhtml_path, href)
            if member in name_set and _allowed_ext(member, extract_svg):
                ordered.append(member)

    ordered = _dedupe(ordered)
    if cover_path:
        if extract_cover:
            ordered = [cover_path] + [path for path in ordered if path != cover_path]
        else:
            ordered = [path for path in ordered if path != cover_path]

    if not ordered:
        raise ValueError("No images found through OPF spine")
    return ordered, "reading_order", cover_path


def _output_filename(member: str, index: int, rename_mode: str, start_number: int, pad: int) -> str:
    suffix = Path(member).suffix.lower()
    if rename_mode == "original":
        return Path(member).name
    return f"{start_number + index - 1:0{pad}d}{suffix}"


def _scan_epubs(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".epub" else []
    globber = input_path.rglob if recursive else input_path.glob
    return sorted((p for p in globber("*.epub") if p.is_file()), key=lambda p: str(p).lower())


def _write_report(path: Path, report: Dict[str, Any], errors: List[str]) -> None:
    try:
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        errors.append(f"Could not write report JSON: {exc}")


def _process_epub(
    epub_path: Path,
    output_root: Path,
    *,
    use_reading_order: bool,
    extract_cover: bool,
    extract_svg: bool,
    rename_mode: str,
    start_number: int,
    pad: int,
    dry_run: bool,
    overwrite: bool,
    create_report: bool,
) -> Dict[str, Any]:
    out_dir = output_root / _sanitize_stem(epub_path.stem)
    errors: List[str] = []
    mode = "zip_order_fallback"
    cover_path: Optional[str] = None

    try:
        with zipfile.ZipFile(epub_path) as zf:
            name_set = {
                _safe_zip_path(info.filename)
                for info in zf.infolist()
                if not info.is_dir() and _safe_zip_path(info.filename)
            }
            images: List[str]
            if use_reading_order:
                try:
                    images, mode, cover_path = _reading_order(
                        zf,
                        name_set,
                        extract_cover=extract_cover,
                        extract_svg=extract_svg,
                    )
                except Exception as exc:
                    errors.append(f"Reading-order parse failed; used ZIP order fallback: {exc}")
                    images = _zip_image_order(zf, extract_svg)
            else:
                images = _zip_image_order(zf, extract_svg)

            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)

            report_files: List[Dict[str, Any]] = []
            used_names: Set[str] = set()
            extracted = 0
            skipped = 0
            for idx, member in enumerate(images, start=1):
                filename = _output_filename(member, idx, rename_mode, start_number, pad)
                out_path = _unique_path(out_dir, filename, used_names, overwrite)
                report_files.append({
                    "index": idx,
                    "source": member,
                    "output": out_path.name,
                })
                if dry_run:
                    continue
                try:
                    with zf.open(member) as src, out_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                    extracted += 1
                except KeyError:
                    skipped += 1
                    errors.append(f"Missing image member: {member}")
                except Exception as exc:
                    errors.append(f"{member}: {exc}")

            report = {
                "epub": str(epub_path),
                "output_dir": str(out_dir),
                "mode": mode,
                "found_images": len(images),
                "extracted": extracted,
                "skipped": skipped,
                "errors": errors,
                "files": report_files,
            }
            if create_report and not dry_run:
                _write_report(out_dir / "extract_report.json", report, errors)
            report["errors"] = errors
            return report
    except zipfile.BadZipFile:
        errors.append("Invalid/corrupt EPUB: not a readable ZIP archive.")
    except Exception as exc:
        errors.append(str(exc))

    return {
        "epub": str(epub_path),
        "output_dir": str(out_dir),
        "mode": mode,
        "found_images": 0,
        "extracted": 0,
        "skipped": 0,
        "errors": errors,
        "files": [],
        "cover": cover_path,
    }


def handle(form, ctx):
    active_tab = "epub_extractor"
    lines: List[str] = []

    def _done(ok: bool, error: str = ""):
        return build_tool_result(active_tab, lines, ok=ok, error=error)

    input_raw = (form.get("epub_input", "") or "").strip()
    output_raw = (form.get("output_dir", "") or "").strip()
    recursive = parse_bool(form.get("recursive"), default=False)
    use_reading_order = parse_bool(form.get("use_reading_order"), default=True)
    extract_cover = parse_bool(form.get("extract_cover"), default=True)
    extract_svg = parse_bool(form.get("extract_svg"), default=False)
    dry_run = parse_bool(form.get("dry_run"), default=False)
    overwrite = parse_bool(form.get("overwrite"), default=False)
    create_report = parse_bool(form.get("create_report"), default=True)
    rename_mode = str(form.get("rename_mode", "sequential") or "sequential").strip().lower()
    start_number = max(0, parse_int(form.get("start_number"), default=1))
    pad = max(1, parse_int(form.get("pad"), default=3))
    limit = max(0, parse_int(form.get("limit"), default=0))

    if rename_mode not in {"sequential", "original"}:
        rename_mode = "sequential"

    if not input_raw:
        lines.append("EPUB input is required.")
        return _done(False, "EPUB input is required.")
    if not output_raw:
        lines.append("Output folder is required.")
        return _done(False, "Output folder is required.")

    epub_input = readable_path(input_raw)
    output_dir = readable_path(output_raw)

    if not epub_input.exists():
        lines.append(f"EPUB input not found: {epub_input}")
        return _done(False, "EPUB input not found.")
    if epub_input.is_file() and epub_input.suffix.lower() != ".epub":
        lines.append("EPUB input file must use the .epub extension.")
        return _done(False, "EPUB input file must use the .epub extension.")
    if not epub_input.is_file() and not epub_input.is_dir():
        lines.append("EPUB input must be a .epub file or a folder.")
        return _done(False, "EPUB input must be a .epub file or a folder.")

    epub_files = _scan_epubs(epub_input, recursive)
    if limit:
        epub_files = epub_files[:limit]
    if not epub_files:
        lines.append("No EPUB files found.")
        return _done(False, "No EPUB files found.")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    lines.append("EPUB Image Extractor")
    lines.append(f"Input: {epub_input}")
    lines.append(f"Output: {output_dir}")
    lines.append(f"EPUB files: {len(epub_files)}")
    lines.append(
        "Options: "
        f"reading_order={'on' if use_reading_order else 'off'}, "
        f"cover={'on' if extract_cover else 'off'}, "
        f"svg={'on' if extract_svg else 'off'}, "
        f"rename={rename_mode}, "
        f"dry_run={'on' if dry_run else 'off'}"
    )
    lines.append("")

    processed = 0
    total_extracted = 0
    total_skipped = 0
    total_errors = 0

    for epub_path in epub_files:
        result = _process_epub(
            epub_path,
            output_dir,
            use_reading_order=use_reading_order,
            extract_cover=extract_cover,
            extract_svg=extract_svg,
            rename_mode=rename_mode,
            start_number=start_number,
            pad=pad,
            dry_run=dry_run,
            overwrite=overwrite,
            create_report=create_report,
        )
        processed += 1
        error_count = len(result["errors"])
        total_errors += error_count
        total_extracted += int(result["extracted"])
        total_skipped += int(result["skipped"])

        lines.append(f"[{epub_path.name}]")
        lines.append(f"Mode: {result['mode']}")
        lines.append(f"Images found: {result['found_images']}")
        if dry_run:
            lines.append(f"Would extract: {result['found_images']}")
        else:
            lines.append(f"Extracted: {result['extracted']}")
        lines.append(f"Skipped existing: {result['skipped']}")
        lines.append(f"Errors: {error_count}")
        for error in result["errors"]:
            lines.append(f"  [ERROR] {error}")
        lines.append(f"Output: {result['output_dir']}")
        lines.append("")

    if dry_run:
        lines.append("Dry run finished: no files were written.")
    lines.append(
        "Done. "
        f"EPUB processed: {processed}/{len(epub_files)} | "
        f"Images extracted: {total_extracted} | skipped: {total_skipped} | errors: {total_errors}"
    )
    ok = total_errors == 0
    return _done(ok, "" if ok else f"{total_errors} EPUB extraction error(s).")
