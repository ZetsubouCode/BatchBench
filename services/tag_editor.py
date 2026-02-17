from collections import Counter, OrderedDict
from pathlib import Path
from typing import Tuple, List, Set, Optional
from utils.io import readable_path
from utils.dataset import split_tags, join_tags
from utils.parse import parse_bool, parse_exts
from utils.tool_result import build_tool_result
from utils.text_io import read_text_best_effort
import shutil

EDIT_MODES = {"insert", "delete", "replace", "dedup"}  # non-move, non-undo
_TXT_TAG_CACHE: "OrderedDict[str, Tuple[Tuple[int, int], List[str]]]" = OrderedDict()
_TXT_TAG_CACHE_MAX = 4096


def _cache_put(key: str, value: Tuple[Tuple[int, int], List[str]]):
    _TXT_TAG_CACHE[key] = value
    _TXT_TAG_CACHE.move_to_end(key)
    while len(_TXT_TAG_CACHE) > _TXT_TAG_CACHE_MAX:
        _TXT_TAG_CACHE.popitem(last=False)


def _read_text_tags(path: Path) -> List[str]:
    try:
        text, _, _ = read_text_best_effort(path)
    except Exception:
        return []
    return split_tags(text)


def _read_tags_cached(txt_path: Path) -> List[str]:
    key = str(txt_path)
    try:
        st = txt_path.stat()
    except Exception:
        return []
    sig = (st.st_mtime_ns, st.st_size)
    cached = _TXT_TAG_CACHE.get(key)
    if cached and cached[0] == sig:
        _TXT_TAG_CACHE.move_to_end(key)
        return list(cached[1])
    tags = _read_text_tags(txt_path)
    _cache_put(key, (sig, list(tags)))
    return tags


def _invalidate_cache(txt_path: Path):
    _TXT_TAG_CACHE.pop(str(txt_path), None)

def _normalize_exts(exts: List[str]) -> Set[str]:
    out: Set[str] = set()
    for raw in exts or []:
        val = (raw or "").strip().lower()
        if not val:
            continue
        if not val.startswith("."):
            val = "." + val
        out.add(val)
    return out

def _list_images(
    folder_path: Path,
    exts: List[str],
    recursive: bool = False,
    exclude_dir: Optional[Path] = None,
) -> List[Path]:
    if not folder_path.exists() or not folder_path.is_dir():
        return []
    extset = _normalize_exts(exts)
    if not extset:
        return []
    if recursive:
        images: List[Path] = []
        for p in folder_path.rglob("*"):
            if not p.is_file():
                continue
            if exclude_dir and exclude_dir in p.parents:
                continue
            if p.suffix.lower() in extset:
                images.append(p)
    else:
        images = [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in extset]
    return sorted(images, key=lambda p: str(p).lower())


def _is_within(path: Path, root: Path) -> bool:
    try:
        path_resolved = path.resolve()
        root_resolved = root.resolve()
    except Exception:
        return False
    return path_resolved == root_resolved or root_resolved in path_resolved.parents


def _next_pair_target(folder: Path, stem: str, image_suffix: str) -> Tuple[Path, Path]:
    bump = 0
    while True:
        suffix = "" if bump == 0 else f"_{bump}"
        candidate_stem = f"{stem}{suffix}"
        dst_img = folder / f"{candidate_stem}{image_suffix}"
        dst_txt = folder / f"{candidate_stem}.txt"
        if not dst_img.exists() and not dst_txt.exists():
            return dst_img, dst_txt
        bump += 1


def _move_pair_transaction(
    src_img: Path,
    src_txt: Path,
    dst_img: Path,
    dst_txt: Path,
    require_txt: bool,
) -> Tuple[bool, str]:
    moved: List[Tuple[Path, Path]] = []
    had_txt = src_txt.exists()
    if require_txt and not had_txt:
        return False, f"Missing paired .txt for {src_img.name}"

    try:
        shutil.move(str(src_img), str(dst_img))
        moved.append((dst_img, src_img))
        if had_txt:
            shutil.move(str(src_txt), str(dst_txt))
            moved.append((dst_txt, src_txt))
        return True, ""
    except Exception as exc:
        rollback_errors: List[str] = []
        for moved_path, original_path in reversed(moved):
            try:
                if not moved_path.exists():
                    continue
                if original_path.exists():
                    rollback_errors.append(
                        f"cannot rollback {moved_path.name} because source already exists: {original_path}"
                    )
                    continue
                shutil.move(str(moved_path), str(original_path))
            except Exception as rollback_exc:
                rollback_errors.append(f"{moved_path.name}: {rollback_exc}")
        message = str(exc)
        if rollback_errors:
            message = f"{message} | rollback issues: {'; '.join(rollback_errors)}"
        return False, message

def scan_tags(folder: Path, exts: List[str], recursive: bool = False) -> dict:
    if not folder or not folder.exists() or not folder.is_dir():
        return {"ok": False, "error": f"Folder not found: {folder}"}

    images = _list_images(folder, exts, recursive=recursive)
    counts: Counter = Counter()
    tag_files = 0

    for img in images:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            continue
        tags = _read_tags_cached(txt)
        tag_files += 1
        if tags:
            counts.update(set(tags))

    items = [
        {"tag": tag, "count": int(cnt)}
        for tag, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    ]
    return {
        "ok": True,
        "folder": str(folder),
        "total_images": len(images),
        "tag_files": tag_files,
        "tags": items,
    }

def list_images_with_tags(
    folder: Path,
    exts: List[str],
    recursive: bool = False,
    limit: int = 80,
    tag_limit: int = 200,
) -> dict:
    if not folder or not folder.exists() or not folder.is_dir():
        return {"ok": False, "error": f"Folder not found: {folder}"}

    exclude_dir = None
    if recursive and folder.name.lower() != "_temp":
        temp = folder / "_temp"
        if temp.exists():
            exclude_dir = temp
    images = _list_images(folder, exts, recursive=recursive, exclude_dir=exclude_dir)
    total = len(images)
    if limit and limit > 0:
        images = images[:limit]

    items = []
    for img in images:
        txt = img.with_suffix(".txt")
        tags: List[str] = []
        has_txt = False
        if txt.exists():
            has_txt = True
            tags = _read_tags_cached(txt)
        tag_count = len(tags)
        if tag_limit > 0 and len(tags) > tag_limit:
            tags = tags[:tag_limit]
        rel = img.relative_to(folder).as_posix()
        items.append(
            {
                "name": img.name,
                "rel": rel,
                "tags": tags,
                "tag_count": tag_count,
                "tags_truncated": tag_count > len(tags),
                "has_txt": has_txt,
            }
        )

    return {"ok": True, "folder": str(folder), "total": total, "images": items}

def remove_tag(txt_path: Path, tag: str, backup: bool = True) -> dict:
    if not txt_path or not txt_path.exists() or not txt_path.is_file():
        return {"ok": False, "error": "Missing .txt"}
    try:
        src, _, _ = read_text_best_effort(txt_path)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    taglist = split_tags(src)
    newtags = [t for t in taglist if t != tag]
    removed = len(newtags) != len(taglist)
    if not removed:
        return {"ok": True, "removed": False, "tags": taglist}
    if backup:
        try:
            txt_path.with_suffix(txt_path.suffix + ".bak").write_text(src, encoding="utf-8")
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
    try:
        txt_path.write_text(join_tags(newtags), encoding="utf-8")
        _invalidate_cache(txt_path)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "removed": True, "tags": newtags}

def handle(form, ctx):
    active_tab = "tags"

    # Raw inputs
    raw_folder_text = (form.get("folder", "") or "").strip()
    raw_folder = readable_path(raw_folder_text)
    mode = (form.get("mode", "insert") or "insert").strip().lower()
    edit_target = (form.get("edit_target", "recursive") or "recursive").strip().lower()
    tags_field = form.get("tags", "")
    exts = parse_exts(form.get("exts"), default=[".jpg", ".jpeg", ".png", ".webp"])
    backup = parse_bool(form.get("backup"), default=False)
    stage_only = parse_bool(form.get("stage_only"), default=False)
    temp_dir_raw = (form.get("temp_dir", "") or "").strip()

    lines: List[str] = []

    def _done(ok: bool, error: str = ""):
        return build_tool_result(active_tab, lines, ok=ok, error=error)

    # Derive base and temp from whatever the user passed:
    # If they give <base>/_temp, base = parent; else base = given.
    if not raw_folder_text:
        lines.append("Folder not provided.")
        return _done(False, "Folder not provided.")

    if raw_folder.name.lower() == "_temp":
        base_folder = raw_folder.parent
        inferred_temp = raw_folder
    else:
        base_folder = raw_folder
        inferred_temp = base_folder / "_temp"
    temp_folder = readable_path(temp_dir_raw) if temp_dir_raw else inferred_temp
    if stage_only and mode in EDIT_MODES:
        edit_target = "temp"
    if edit_target not in {"recursive", "temp", "base"}:
        lines.append(f"Unknown edit_target: {edit_target}")
        return _done(False, f"Unknown edit_target: {edit_target}")

    def _parse_tag_list(raw: str) -> List[str]:
        return [t.strip() for t in (raw or "").split(",") if t.strip()]

    def _parse_replace_mapping(raw: str) -> dict:
        mapping = {}
        parts = [p.strip() for p in (raw or "").split(";") if p.strip()]
        for p in parts:
            if "->" not in p:
                continue
            old, new = [x.strip() for x in p.split("->", 1)]
            if old:
                mapping[old] = new
        return mapping

    add: List[str] = []
    deltags: Set[str] = set()
    mapping = {}
    if mode == "insert":
        add = _parse_tag_list(tags_field)
    elif mode == "delete":
        deltags = set(_parse_tag_list(tags_field))
    elif mode == "replace":
        mapping = _parse_replace_mapping(tags_field)

    # Ensure folders exist as appropriate per mode
    if mode == "move":
        if not base_folder.exists() or not base_folder.is_dir():
            lines.append(f"Base folder not found: {base_folder}")
            return _done(False, f"Base folder not found: {base_folder}")
        if edit_target == "temp":
            lines.append("Move mode does not support edit_target=temp. Use undo to restore from temp.")
            return _done(False, "Move mode does not support edit_target=temp.")
        recursive_scan = edit_target == "recursive"
        exclude_dir = temp_folder if recursive_scan and _is_within(temp_folder, base_folder) else None
        scan_folder = base_folder
        temp_folder.mkdir(parents=True, exist_ok=True)
    elif mode == "undo":
        if not temp_folder.exists() or not temp_folder.is_dir():
            lines.append(f"Temp folder not found: {temp_folder}")
            return _done(False, f"Temp folder not found: {temp_folder}")
        recursive_scan = edit_target == "recursive"
        exclude_dir = None
        scan_folder = temp_folder
    elif mode in EDIT_MODES:
        if not base_folder.exists() or not base_folder.is_dir():
            lines.append(f"Base folder not found: {base_folder}")
            return _done(False, f"Base folder not found: {base_folder}")
        if edit_target == "temp":
            if not temp_folder.exists() or not temp_folder.is_dir():
                lines.append(f"Temp folder not found: {temp_folder}")
                return _done(False, f"Temp folder not found: {temp_folder}")
            scan_folder = temp_folder
            recursive_scan = False
            exclude_dir = None
        elif edit_target == "base":
            scan_folder = base_folder
            recursive_scan = False
            exclude_dir = None
        else:
            scan_folder = base_folder
            recursive_scan = True
            exclude_dir = temp_folder if _is_within(temp_folder, base_folder) else None
    else:
        lines.append(f"Unknown mode: {mode}")
        return _done(False, f"Unknown mode: {mode}")

    # ---------- MOVE ----------
    if mode == "move":
        images = _list_images(scan_folder, exts, recursive=recursive_scan, exclude_dir=exclude_dir)
        processed = 0
        errors = 0
        want: Set[str] = set([t.strip() for t in tags_field.split(",") if t.strip()])
        if not want:
            lines.append("Move skipped: no tags provided.")
            return _done(False, "Move skipped: no tags provided.")
        if not images:
            lines.append(f"No image files with {exts} found in: {scan_folder}")
            return _done(False, f"No images found in: {scan_folder}")

        for img in images:
            txt = img.with_suffix(".txt")
            if not txt.exists():
                continue
            try:
                src, _, _ = read_text_best_effort(txt)
            except Exception as exc:
                lines.append(f"[ERROR] reading {txt.name}: {exc}")
                continue
            taglist = split_tags(src)

            matches = sorted([t for t in taglist if t in want])
            if matches:
                dest_img, dest_txt = _next_pair_target(temp_folder, img.stem, img.suffix.lower())
                ok, error = _move_pair_transaction(
                    img, txt, dest_img, dest_txt, require_txt=True
                )
                if not ok:
                    lines.append(f"[ERROR] moving {img.name}: {error}")
                    errors += 1
                    continue
                lines.append(f"{img.name}: moved to {temp_folder} (matched: {', '.join(matches)})")
                processed += 1

        lines.append(f"Done. {processed} file(s) moved to {temp_folder}. Errors: {errors}.")
        return _done(errors == 0, "" if errors == 0 else f"{errors} move operation(s) failed.")

    # ---------- UNDO ----------
    if mode == "undo":
        images_in_temp = _list_images(scan_folder, exts, recursive=recursive_scan)
        if not images_in_temp:
            lines.append(f"No image files with {exts} found in temp folder: {scan_folder}")
            return _done(False, f"No image files found in temp folder: {scan_folder}")

        restored = 0
        errors = 0
        for img in sorted(images_in_temp, key=lambda p: p.name.lower()):
            txt = img.with_suffix(".txt")
            had_txt = txt.exists()
            dest_img, dest_txt = _next_pair_target(base_folder, img.stem, img.suffix.lower())
            ok, error = _move_pair_transaction(
                img, txt, dest_img, dest_txt, require_txt=False
            )
            if not ok:
                lines.append(f"[ERROR] restoring {img.name}: {error}")
                errors += 1
                continue
            restored += 1
            lines.append(f"Restored: {dest_img.name}{' (+ .txt)' if had_txt else ''}")

        lines.append(f"Done. {restored} file(s) restored from {scan_folder} to {base_folder}. Errors: {errors}.")
        return _done(errors == 0, "" if errors == 0 else f"{errors} restore operation(s) failed.")

    # ---------- EDIT MODES (insert/delete/replace/dedup) ----------
    if mode in EDIT_MODES:
        if mode == "insert" and not add:
            lines.append("Insert skipped: no tags provided.")
            return _done(False, "Insert skipped: no tags provided.")
        if mode == "delete" and not deltags:
            lines.append("Delete skipped: no tags provided.")
            return _done(False, "Delete skipped: no tags provided.")
        if mode == "replace" and not mapping:
            lines.append("Replace skipped: no mappings provided.")
            return _done(False, "Replace skipped: no mappings provided.")
        images = _list_images(scan_folder, exts, recursive=recursive_scan, exclude_dir=exclude_dir)
    if not images:
        lines.append(f"No image files with {exts} found in: {scan_folder}")
        return _done(False, f"No image files found in: {scan_folder}")

    processed = 0
    errors = 0
    for img in images:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            continue

        try:
            src, _, _ = read_text_best_effort(txt)
        except Exception as exc:
            lines.append(f"[ERROR] reading {txt.name}: {exc}")
            errors += 1
            continue
        taglist = split_tags(src)

        newtags = list(taglist)
        action_desc = ""
        if mode == "insert":
            for t in add:
                if t not in newtags:
                    newtags.append(t)
            action_desc = f"insert -> {add}"
        elif mode == "delete":
            newtags = [t for t in taglist if t not in deltags]
            action_desc = f"delete -> {sorted(deltags)}"
        elif mode == "replace":
            newtags = [mapping.get(t, t) for t in taglist]
            action_desc = f"replace -> {mapping}"
        elif mode == "dedup":
            seen: Set[str] = set(); out: List[str] = []
            for t in taglist:
                if t not in seen:
                    out.append(t); seen.add(t)
            newtags = out
            action_desc = f"dedup -> {len(taglist)-len(out)} removed"

        content = join_tags(newtags)
        if content.strip() == src.strip():
            continue
        try:
            if backup:
                txt.with_suffix(txt.suffix + ".bak").write_text(src, encoding="utf-8")
            txt.write_text(content, encoding="utf-8")
            _invalidate_cache(txt)
            lines.append(f"{img.name}: {action_desc}")
        except Exception as exc:
            lines.append(f"[ERROR] writing {txt.name}: {exc}")
            errors += 1
            continue

        processed += 1

    lines.append(f"Done. {processed} file(s) updated in {scan_folder}. Errors: {errors}.")
    return _done(errors == 0, "" if errors == 0 else f"{errors} file update(s) failed.")
