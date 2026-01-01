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
    temp_dir_input = (form.get("temp_dir","") or "").strip()

    lines: List[str] = []

    # Derive base and temp from whatever the user passed:
    # If they give <base>/_temp, base = parent; else base = given.
    if not raw_folder:
        lines.append("Folder not provided.")
        return active_tab, log_join(lines)

    base_folder = raw_folder.parent if raw_folder.name.lower() == "_temp" else raw_folder
    temp_folder = readable_path(temp_dir_input) if temp_dir_input else (base_folder / "_temp")

    if edit_target == "auto":
        edit_target = "recursive"
    if edit_target not in {"temp", "base", "recursive"}:
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
        if edit_target == "base":
            if not base_folder.exists() or not base_folder.is_dir():
                lines.append(f"Base folder not found: {base_folder}")
                return active_tab, log_join(lines)
            scan_folder = base_folder
        elif edit_target == "recursive":
            if not base_folder.exists() or not base_folder.is_dir():
                lines.append(f"Base folder not found: {base_folder}")
                return active_tab, log_join(lines)
            do_stage = stage_only or mode in {"delete", "replace"}
            if do_stage:
                temp_folder.mkdir(parents=True, exist_ok=True)

                if mode == "insert" and not add:
                    lines.append("Recursive staging skipped: no tags provided for insert.")
                    return active_tab, log_join(lines)
                if mode == "delete" and not deltags:
                    lines.append("Recursive staging skipped: no tags provided for delete.")
                    return active_tab, log_join(lines)
                if mode == "replace" and not mapping:
                    lines.append("Recursive staging skipped: no mappings provided for replace.")
                    return active_tab, log_join(lines)

                staged_images: List[Path] = []
                candidates = _list_images(base_folder, exts, recursive=True, exclude_dir=temp_folder)
                if not candidates:
                    lines.append(f"No image files with {exts} found in: {base_folder}")
                    return active_tab, log_join(lines)

                for img in candidates:
                    txt = img.with_suffix(".txt")
                    if not txt.exists():
                        continue
                    src = txt.read_text(encoding="utf-8")
                    taglist = split_tags(src)

                    should_move = False
                    if mode == "insert":
                        should_move = any(t not in taglist for t in add)
                    elif mode == "delete":
                        should_move = any(t in taglist for t in deltags)
                    elif mode == "replace":
                        should_move = any(t in taglist for t in mapping.keys())
                    elif mode == "dedup":
                        should_move = len(taglist) != len(set(taglist))

                    if should_move:
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
                        staged_images.append(dest_img)
                        rel = img.relative_to(base_folder)
                        lines.append(f"{rel}: staged to {temp_folder}")

                lines.append(f"Recursive staging: moved {len(staged_images)} file(s) to {temp_folder}.")
                if stage_only:
                    return active_tab, log_join(lines)

            if not temp_folder.exists() or not temp_folder.is_dir():
                lines.append(f"Temp folder not found: {temp_folder}")
                return active_tab, log_join(lines)
            scan_folder = temp_folder
        else:
            # Edit inside temp; create it if missing so we don't hard fail
            temp_folder.mkdir(parents=True, exist_ok=True)
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
