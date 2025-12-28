from typing import Tuple, List, Set
from utils.io import readable_path, log_join
from utils.dataset import split_tags, join_tags
import shutil

EDIT_MODES = {"insert", "delete", "replace", "dedup"}  # non-move, non-undo

def handle(form, ctx) -> Tuple[str, str]:
    active_tab = "tags"

    # Raw inputs
    raw_folder = readable_path(form.get("folder",""))
    mode = (form.get("mode","insert") or "insert").lower()
    tags_field = form.get("tags","")
    exts = [e.strip().lower() for e in form.get("exts",".jpg,.jpeg,.png,.webp").split(",") if e.strip()]
    backup = bool(form.get("backup"))
    temp_dir_input = (form.get("temp_dir","") or "").strip()

    lines: List[str] = []

    # Derive base and temp from whatever the user passed:
    # If they give <base>/_temp, base = parent; else base = given.
    if not raw_folder:
        lines.append("Folder not provided.")
        return active_tab, log_join(lines)

    base_folder = raw_folder.parent if raw_folder.name.lower() == "_temp" else raw_folder
    temp_folder = readable_path(temp_dir_input) if temp_dir_input else (base_folder / "_temp")

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
        # Edit inside temp; create it if missing so we don't hard fail
        temp_folder.mkdir(parents=True, exist_ok=True)
        scan_folder = temp_folder
    else:
        lines.append(f"Unknown mode: {mode}")
        return active_tab, log_join(lines)

    # Helper to enumerate images inside a folder
    def list_images(folder_path):
        if not folder_path.exists() or not folder_path.is_dir():
            return []
        return [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in exts]

    # ---------- MOVE ----------
    if mode == "move":
        images = list_images(scan_folder)
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
        images_in_temp = list_images(scan_folder)
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

    # ---------- EDIT MODES IN TEMP (insert/delete/replace/dedup) ----------
    images = list_images(scan_folder)
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
            add = [t.strip() for t in tags_field.split(",") if t.strip()]
            for t in add:
                if t not in taglist:
                    taglist.append(t)
            if backup:
                txt.with_suffix(txt.suffix + ".bak").write_text(src, encoding="utf-8")
            txt.write_text(join_tags(taglist), encoding="utf-8")
            lines.append(f"{img.name}: insert -> {add}")

        elif mode == "delete":
            deltags = set([t.strip() for t in tags_field.split(",") if t.strip()])
            newtags = [t for t in taglist if t not in deltags]
            if backup:
                txt.with_suffix(txt.suffix + ".bak").write_text(src, encoding="utf-8")
            txt.write_text(join_tags(newtags), encoding="utf-8")
            lines.append(f"{img.name}: delete -> {sorted(deltags)}")

        elif mode == "replace":
            mapping = {}
            parts = [p.strip() for p in tags_field.split(";") if p.strip()]
            for p in parts:
                if "->" in p:
                    old, new = [x.strip() for x in p.split("->", 1)]
                    mapping[old] = new
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
