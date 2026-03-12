from typing import List
import os
import re
import shutil
import string

from utils.io import readable_path, ensure_out_dir
from utils.dataset import collect_pairs
from utils.parse import parse_bool, parse_exts
from utils.tool_result import build_tool_result

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


def _path_within(path: str, root: str) -> bool:
    try:
        path_resolved = readable_path(path).resolve()
        root_resolved = readable_path(root).resolve()
    except Exception:
        return False
    return path_resolved == root_resolved or root_resolved in path_resolved.parents


def _next_pair_target(out_dir, stem: str, image_suffix: str):
    bump = 0
    while True:
        suffix = "" if bump == 0 else f"_{bump}"
        candidate_stem = f"{stem}{suffix}"
        dst_img = out_dir / f"{candidate_stem}{image_suffix}"
        dst_txt = out_dir / f"{candidate_stem}.txt"
        if not dst_img.exists() and not dst_txt.exists():
            return dst_img, dst_txt
        bump += 1


def _transfer_pair(src_img, src_txt, dst_img, dst_txt, move: bool):
    if move:
        moved = []
        try:
            shutil.move(str(src_img), str(dst_img))
            moved.append((dst_img, src_img))
            shutil.move(str(src_txt), str(dst_txt))
            moved.append((dst_txt, src_txt))
            return True, ""
        except Exception as exc:
            rollback_errors = []
            for moved_path, original_path in reversed(moved):
                try:
                    if not moved_path.exists():
                        continue
                    if original_path.exists():
                        rollback_errors.append(
                            f"cannot rollback {moved_path.name}; source exists: {original_path}"
                        )
                        continue
                    shutil.move(str(moved_path), str(original_path))
                except Exception as rollback_exc:
                    rollback_errors.append(f"{moved_path.name}: {rollback_exc}")
            message = str(exc)
            if rollback_errors:
                message = f"{message} | rollback issues: {'; '.join(rollback_errors)}"
            return False, message

    created = []
    try:
        shutil.copy2(str(src_img), str(dst_img))
        created.append(dst_img)
        shutil.copy2(str(src_txt), str(dst_txt))
        created.append(dst_txt)
        return True, ""
    except Exception as exc:
        cleanup_errors = []
        for created_path in reversed(created):
            try:
                if created_path.exists():
                    created_path.unlink()
            except Exception as cleanup_exc:
                cleanup_errors.append(f"{created_path.name}: {cleanup_exc}")
        message = str(exc)
        if cleanup_errors:
            message = f"{message} | cleanup issues: {'; '.join(cleanup_errors)}"
        return False, message


def handle(form, ctx):
    active_tab = "combine"
    source_folders = form.get("source_folders", "") or ""
    folder_a = (form.get("folder_a","") or "").strip()
    folder_b = (form.get("folder_b","") or "").strip()
    extra_folders = form.get("extra_folders","") or ""
    out_dir_raw = (form.get("out_dir", "") or "").strip()
    suffix = form.get("suffix_combine","_B")
    exts = parse_exts(form.get("exts_combine"), default=[".jpg", ".jpeg", ".png", ".webp"])
    move_instead = parse_bool(form.get("move_instead"), default=False)
    lines: List[str] = []

    def _done(ok: bool, error: str = ""):
        return build_tool_result(active_tab, lines, ok=ok, error=error)

    if not out_dir_raw:
        lines.append("Output folder is required.")
        return _done(False, "Output folder is required.")
    out_dir = readable_path(out_dir_raw)

    raw_folders = []
    if folder_a:
        raw_folders.append(folder_a)
    if folder_b:
        raw_folders.append(folder_b)
    for block in (source_folders, extra_folders):
        for line in str(block).splitlines():
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
        lines.append("Please provide at least 2 source dataset folders.")
        return _done(False, "Please provide at least 2 source dataset folders.")

    invalid = []
    for idx, folder in enumerate(folders):
        if not folder.exists() or not folder.is_dir():
            invalid.append(f"Source {_label_for_index(idx)}: {folder}")
    if invalid:
        lines.append("Source folder(s) not found:")
        lines.extend(invalid)
        return _done(False, "One or more source folders were not found.")

    if move_instead:
        for folder in folders:
            if _path_within(str(out_dir), str(folder)):
                lines.append(
                    "Move mode requires output outside all source folders to avoid self-move conflicts."
                )
                lines.append(f"Invalid output: {out_dir}")
                lines.append(f"Inside source: {folder}")
                return _done(
                    False,
                    "Output folder must be outside source folders when move_instead is enabled.",
                )

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
        lines.append(f"Source {d['label']}: {d['folder']} (pairs: {len(d['pairs'])})")
    lines.append(f"Base set (no suffix): {datasets[base_idx]['label']}")
    for (_, d), suf in zip(others, suffixes):
        lines.append(f"Suffix for {d['label']}: '{suf}'")

    copied = 0
    renamed = 0
    errors = 0
    for img, txt in datasets[base_idx]["pairs"]:
        target_img, target_txt = _next_pair_target(out_dir, img.stem, img.suffix.lower())
        ok, error = _transfer_pair(img, txt, target_img, target_txt, move_instead)
        if ok:
            copied += 1
        else:
            errors += 1
            lines.append(f"[ERROR] {img.name}: {error}")

    for (_, d), suf in zip(others, suffixes):
        for img, txt in d["pairs"]:
            target_img, target_txt = _next_pair_target(
                out_dir, img.stem + suf, img.suffix.lower()
            )
            ok, error = _transfer_pair(img, txt, target_img, target_txt, move_instead)
            if ok:
                renamed += 1
            else:
                errors += 1
                lines.append(f"[ERROR] {img.name}: {error}")

    lines.append(
        f"Done. Copied {copied} unchanged and {renamed} renamed into '{out_dir}'. Errors: {errors}."
    )
    return _done(errors == 0, "" if errors == 0 else f"{errors} pair transfer(s) failed.")
