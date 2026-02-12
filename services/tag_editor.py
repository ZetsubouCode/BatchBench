from collections import Counter
from pathlib import Path
from typing import Tuple, List, Set, Optional
from utils.io import readable_path, log_join
from utils.dataset import split_tags, join_tags
import shutil

EDIT_MODES = {"insert", "delete", "replace", "dedup"}  # non-move, non-undo

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
        try:
            text = txt.read_text(encoding="utf-8")
        except Exception:
            continue
        tags = split_tags(text)
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

def list_images_with_tags(folder: Path, exts: List[str], recursive: bool = False, limit: int = 80) -> dict:
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
            try:
                tags = split_tags(txt.read_text(encoding="utf-8"))
            except Exception:
                tags = []
        rel = img.relative_to(folder).as_posix()
        items.append(
            {
                "name": img.name,
                "rel": rel,
                "tags": tags,
                "has_txt": has_txt,
            }
        )

    return {"ok": True, "folder": str(folder), "total": total, "images": items}

def remove_tag(txt_path: Path, tag: str, backup: bool = True) -> dict:
    if not txt_path or not txt_path.exists() or not txt_path.is_file():
        return {"ok": False, "error": "Missing .txt"}
    try:
        src = txt_path.read_text(encoding="utf-8")
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
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "removed": True, "tags": newtags}

def handle(form, ctx) -> Tuple[str, str]:
    active_tab = "tags"

    # Raw inputs
    raw_folder = readable_path(form.get("folder",""))
    mode = (form.get("mode","insert") or "insert").lower()
    edit_target = (form.get("edit_target","recursive") or "recursive").lower()
    tags_field = form.get("tags","")
    exts = [e.strip().lower() for e in form.get("exts",".jpg,.jpeg,.png,.webp").split(",") if e.strip()]
    backup = bool(form.get("backup"))
    stage_only = str(form.get("stage_only", "")).strip().lower() in {"1", "true", "yes", "on"}

    lines: List[str] = []

    # Derive base and temp from whatever the user passed:
    # If they give <base>/_temp, base = parent; else base = given.
    if not raw_folder:
        lines.append("Folder not provided.")
        return active_tab, log_join(lines)

    base_folder = raw_folder.parent if raw_folder.name.lower() == "_temp" else raw_folder
    temp_folder = base_folder / "_temp"
    edit_target = "recursive"

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
            return active_tab, log_join(lines)
        # Lazily create temp on first move
        temp_folder.mkdir(parents=True, exist_ok=True)
        scan_folder = base_folder
    elif mode == "undo":
        # For undo we require temp to exist; restore to base
        if not temp_folder.exists() or not temp_folder.is_dir():
            lines.append(f"Temp folder not found: {temp_folder}")
            return active_tab, log_join(lines)
        scan_folder = temp_folder
    elif mode in EDIT_MODES:
        if not base_folder.exists() or not base_folder.is_dir():
            lines.append(f"Base folder not found: {base_folder}")
            return active_tab, log_join(lines)
        if stage_only:
            lines.append("Stage-only is not supported for edit modes. Use Move mode to populate _temp.")
            return active_tab, log_join(lines)
        if not temp_folder.exists() or not temp_folder.is_dir():
            lines.append(f"Temp folder not found: {temp_folder}")
            return active_tab, log_join(lines)
        scan_folder = temp_folder
    else:
        lines.append(f"Unknown mode: {mode}")
        return active_tab, log_join(lines)

    # ---------- MOVE ----------
    if mode == "move":
        images = _list_images(scan_folder, exts)
        processed = 0
        want: Set[str] = set([t.strip() for t in tags_field.split(",") if t.strip()])
        if not want:
            lines.append("Move skipped: no tags provided.")
            return active_tab, log_join(lines)

        for img in images:
            txt = img.with_suffix(".txt")
            if not txt.exists():
                continue
            src = txt.read_text(encoding="utf-8")
            taglist = split_tags(src)

            matches = sorted([t for t in taglist if t in want])
            if matches:
                stem, ext = img.stem, img.suffix
                dest_img = temp_folder / img.name
                dest_txt = temp_folder / txt.name
                bump = 1
                while dest_img.exists() or dest_txt.exists():
                    new_stem = f"{stem}_{bump}"
                    dest_img = temp_folder / f"{new_stem}{ext}"
                    dest_txt = temp_folder / f"{new_stem}.txt"
                    bump += 1
                shutil.move(str(img), str(dest_img))
                shutil.move(str(txt), str(dest_txt))
                lines.append(f"{img.name}: moved to {temp_folder} (matched: {', '.join(matches)})")
                processed += 1

        lines.append(f"Done. {processed} file(s) moved to {temp_folder}.")
        return active_tab, log_join(lines)

    # ---------- UNDO ----------
    if mode == "undo":
        images_in_temp = _list_images(scan_folder, exts)
        if not images_in_temp:
            lines.append(f"No image files with {exts} found in temp folder: {scan_folder}")
            return active_tab, log_join(lines)

        restored = 0
        for img in sorted(images_in_temp, key=lambda p: p.name.lower()):
            txt = img.with_suffix(".txt")
            stem, ext = img.stem, img.suffix
            dest_img = base_folder / img.name
            dest_txt = base_folder / (stem + ".txt")

            bump = 1
            while dest_img.exists() or dest_txt.exists():
                new_stem = f"{stem}_{bump}"
                dest_img = base_folder / f"{new_stem}{ext}"
                dest_txt = base_folder / f"{new_stem}.txt"
                bump += 1

            try:
                shutil.move(str(img), str(dest_img))
                if txt.exists():
                    shutil.move(str(txt), str(dest_txt))
                restored += 1
                lines.append(f"Restored: {dest_img.name}{' (+ .txt)' if dest_txt.exists() else ''}")
            except Exception as e:
                lines.append(f"[ERROR] restoring {img.name}: {e}")

        lines.append(f"Done. {restored} file(s) restored from {scan_folder} to {base_folder}.")
        return active_tab, log_join(lines)

    # ---------- EDIT MODES (insert/delete/replace/dedup) ----------
    if mode in EDIT_MODES:
        if mode == "insert" and not add:
            lines.append("Insert skipped: no tags provided.")
            return active_tab, log_join(lines)
        if mode == "delete" and not deltags:
            lines.append("Delete skipped: no tags provided.")
            return active_tab, log_join(lines)
        if mode == "replace" and not mapping:
            lines.append("Replace skipped: no mappings provided.")
            return active_tab, log_join(lines)
        images = _list_images(scan_folder, exts)
    if not images:
        lines.append(f"No image files with {exts} found in: {scan_folder}")
        return active_tab, log_join(lines)

    processed = 0
    for img in images:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            continue

        src = txt.read_text(encoding="utf-8")
        taglist = split_tags(src)

        if mode == "insert":
            for t in add:
                if t not in taglist:
                    taglist.append(t)
            if backup:
                txt.with_suffix(txt.suffix + ".bak").write_text(src, encoding="utf-8")
            txt.write_text(join_tags(taglist), encoding="utf-8")
            lines.append(f"{img.name}: insert -> {add}")

        elif mode == "delete":
            newtags = [t for t in taglist if t not in deltags]
            if backup:
                txt.with_suffix(txt.suffix + ".bak").write_text(src, encoding="utf-8")
            txt.write_text(join_tags(newtags), encoding="utf-8")
            lines.append(f"{img.name}: delete -> {sorted(deltags)}")

        elif mode == "replace":
            newtags = [mapping.get(t, t) for t in taglist]
            if backup:
                txt.with_suffix(txt.suffix + ".bak").write_text(src, encoding="utf-8")
            txt.write_text(join_tags(newtags), encoding="utf-8")
            lines.append(f"{img.name}: replace -> {mapping}")

        elif mode == "dedup":
            seen: Set[str] = set(); out: List[str] = []
            for t in taglist:
                if t not in seen:
                    out.append(t); seen.add(t)
            if backup:
                txt.with_suffix(txt.suffix + ".bak").write_text(src, encoding="utf-8")
            txt.write_text(join_tags(out), encoding="utf-8")
            lines.append(f"{img.name}: dedup -> {len(taglist)-len(out)} removed")

        processed += 1

    lines.append(f"Done. {processed} files checked in {scan_folder}.")
    return active_tab, log_join(lines)
