from typing import Tuple, List
import os
import re
import shutil
import string

from utils.io import readable_path, ensure_out_dir, log_join
from utils.dataset import collect_pairs

def _label_for_index(idx: int) -> str:
    if idx < 26:
        return chr(ord("A") + idx)
    return f"Set {idx + 1}"

def _auto_suffix(base: str, index: int) -> str:
    if index <= 1:
        return base
    if re.match(r".*_[A-Za-z]$", base):
        prefix = base[:-1]
        letter = base[-1]
        alphabet = string.ascii_uppercase if letter.isupper() else string.ascii_lowercase
        pos = alphabet.find(letter)
        if pos >= 0:
            new_pos = pos + (index - 1)
            if new_pos < len(alphabet):
                return prefix + alphabet[new_pos]
    return f"{base}{index}"

def _build_suffixes(base_suffix: str, count: int) -> List[str]:
    base = (base_suffix or "").strip() or "_B"
    return [_auto_suffix(base, i + 1) for i in range(count)]

def handle(form, ctx) -> Tuple[str, str]:
    active_tab = "combine"
    folder_a = (form.get("folder_a","") or "").strip()
    folder_b = (form.get("folder_b","") or "").strip()
    extra_folders = form.get("extra_folders","") or ""
    out_dir = readable_path(form.get("out_dir",""))
    suffix = form.get("suffix_combine","_B")
    exts = [e.strip().lower() for e in form.get("exts_combine",".jpg,.jpeg,.png,.webp").split(",") if e.strip()]
    move_instead = bool(form.get("move_instead"))
    lines: List[str] = []

    raw_folders = []
    if folder_a:
        raw_folders.append(folder_a)
    if folder_b:
        raw_folders.append(folder_b)
    for line in str(extra_folders).splitlines():
        val = line.strip()
        if val:
            raw_folders.append(val)

    folders = []
    seen = set()
    for raw in raw_folders:
        path = readable_path(raw)
        key = str(path)
        if os.name == "nt":
            key = key.lower()
        if key in seen:
            continue
        seen.add(key)
        folders.append(path)

    if len(folders) < 2:
        lines.append("Please provide at least 2 dataset folders.")
        return active_tab, log_join(lines)

    invalid = []
    for idx, folder in enumerate(folders):
        if not folder.exists() or not folder.is_dir():
            invalid.append(f"Folder {_label_for_index(idx)}: {folder}")
    if invalid:
        lines.append("Folder(s) not found:")
        lines.extend(invalid)
        return active_tab, log_join(lines)

    ensure_out_dir(out_dir)
    datasets = []
    for idx, folder in enumerate(folders):
        pairs = collect_pairs(folder, exts)
        datasets.append({"folder": folder, "pairs": pairs, "label": _label_for_index(idx)})

    base_idx = max(range(len(datasets)), key=lambda i: len(datasets[i]["pairs"]))
    others = [(idx, d) for idx, d in enumerate(datasets) if idx != base_idx]
    suffixes = _build_suffixes(suffix, len(others))

    lines.append(f"Datasets: {len(datasets)}")
    for d in datasets:
        lines.append(f"Folder {d['label']}: {d['folder']} (pairs: {len(d['pairs'])})")
    lines.append(f"Base set (no suffix): {datasets[base_idx]['label']}")
    for (_, d), suf in zip(others, suffixes):
        lines.append(f"Suffix for {d['label']}: '{suf}'")

    copied = 0
    for img, txt in datasets[base_idx]["pairs"]:
        target_img = out_dir / img.name
        target_txt = out_dir / txt.name
        try:
            if move_instead:
                shutil.move(str(img), str(target_img))
                shutil.move(str(txt), str(target_txt))
            else:
                shutil.copy2(str(img), str(target_img))
                shutil.copy2(str(txt), str(target_txt))
            copied += 1
        except Exception as e:
            lines.append(f"[ERROR] copying {img.name}: {e}")

    renamed = 0
    for (_, d), suf in zip(others, suffixes):
        for img, txt in d["pairs"]:
            new_stem = img.stem + suf
            target_img = out_dir / (new_stem + img.suffix)
            target_txt = out_dir / (new_stem + ".txt")
            bump = 1
            while target_img.exists() or target_txt.exists():
                new_stem_b = f"{new_stem}_{bump}"
                target_img = out_dir / (new_stem_b + img.suffix)
                target_txt = out_dir / (new_stem_b + ".txt")
                bump += 1
            try:
                if move_instead:
                    shutil.move(str(img), str(target_img))
                    shutil.move(str(txt), str(target_txt))
                else:
                    shutil.copy2(str(img), str(target_img))
                    shutil.copy2(str(txt), str(target_txt))
                renamed += 1
            except Exception as e:
                lines.append(f"[ERROR] renaming {img.name}: {e}")

    lines.append(f"Done. Copied {copied} unchanged and {renamed} renamed into '{out_dir}'.")
    return active_tab, log_join(lines)
