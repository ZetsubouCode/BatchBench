# services/group_renamer.py
from typing import List, Iterable
from pathlib import Path
import shutil
import re

from utils.io import readable_path
from utils.parse import parse_bool, parse_exts
from utils.tool_result import build_tool_result


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]


def _list_images(folder: Path, exts: set) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]


def _order(items: Iterable[Path], how: str) -> List[Path]:
    how = (how or "name").lower()
    if how == "mtime":
        return sorted(items, key=lambda p: (p.stat().st_mtime, p.name.lower()))
    if how == "ctime":
        return sorted(items, key=lambda p: (p.stat().st_ctime, p.name.lower()))
    return sorted(items, key=lambda p: _natural_key(p.name))


def _next_pair_target(out_dir: Path, stem: str, image_suffix: str, include_txt: bool):
    bump = 0
    while True:
        suffix = "" if bump == 0 else f"-{bump}"
        candidate_stem = f"{stem}{suffix}"
        dst_img = out_dir / f"{candidate_stem}{image_suffix}"
        dst_txt = (out_dir / f"{candidate_stem}.txt") if include_txt else None
        if not dst_img.exists() and (dst_txt is None or not dst_txt.exists()):
            return dst_img, dst_txt
        bump += 1


def _transfer_pair(src_img: Path, dst_img: Path, txt_src: Path, txt_dst: Path, move: bool):
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    if txt_dst:
        txt_dst.parent.mkdir(parents=True, exist_ok=True)

    if move:
        moved = []
        try:
            shutil.move(str(src_img), str(dst_img))
            moved.append((dst_img, src_img))
            if txt_src and txt_dst:
                shutil.move(str(txt_src), str(txt_dst))
                moved.append((txt_dst, txt_src))
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
        if txt_src and txt_dst:
            shutil.copy2(str(txt_src), str(txt_dst))
            created.append(txt_dst)
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
    active_tab = "rename"

    root_raw = (form.get("rn_root", "") or "").strip()
    root = readable_path(root_raw)
    out_dir_in = (form.get("rn_out", "") or "").strip()
    out_dir = readable_path(out_dir_in) if out_dir_in else (root / "_renamed")

    exts = set(parse_exts(form.get("rn_exts"), default=[".png", ".jpg", ".jpeg", ".webp"]))
    include_txt = parse_bool(form.get("rn_include_txt"), default=False)
    move_instead = parse_bool(form.get("rn_move_instead"), default=False)
    dry_run = parse_bool(form.get("rn_dry_run"), default=False)

    pad = max(1, int(form.get("rn_pad", "3")))
    suffix_pad = max(0, int(form.get("rn_suffix_pad", "0")))
    sep = (form.get("rn_sep", "_") or "_")
    start = max(1, int(form.get("rn_start", "1")))

    order_top = form.get("rn_top_order", "name")
    order_folder = form.get("rn_folder_order", "name")
    order_inside = form.get("rn_inside_order", "name")

    lines: List[str] = []

    def _done(ok: bool, error: str = ""):
        return build_tool_result(active_tab, lines, ok=ok, error=error)

    if not root_raw:
        lines.append("Root folder is required.")
        return _done(False, "Root folder is required.")
    if not root.exists() or not root.is_dir():
        lines.append("Root folder not found.")
        return _done(False, "Root folder not found.")

    top_images = _order(_list_images(root, exts), order_top)
    subfolders = _order([p for p in root.iterdir() if p.is_dir()], order_folder)

    seq = start
    planned = []

    def pref(n: int) -> str:
        return f"{n:0{pad}d}"

    def part(n: int) -> str:
        return f"{n:0{suffix_pad}d}" if suffix_pad > 0 else str(n)

    for img in top_images:
        dst_img = out_dir / f"{pref(seq)}{img.suffix.lower()}"
        if include_txt:
            txt_src = img.with_suffix(".txt")
            txt_dst = out_dir / f"{pref(seq)}.txt" if txt_src.exists() else None
        else:
            txt_src = txt_dst = None
        planned.append(("top", img, dst_img, txt_src, txt_dst))
        seq += 1

    for folder in subfolders:
        imgs = _order(_list_images(folder, exts), order_inside)
        if not imgs:
            continue
        prefix = pref(seq)
        k = 1
        for img in imgs:
            stem = f"{prefix}{sep}{part(k)}"
            dst_img = out_dir / f"{stem}{img.suffix.lower()}"
            if include_txt:
                txt_src = img.with_suffix(".txt")
                txt_dst = out_dir / f"{stem}.txt" if txt_src.exists() else None
            else:
                txt_src = txt_dst = None
            planned.append(("group", img, dst_img, txt_src, txt_dst))
            k += 1
        seq += 1

    lines.append(f"Top-level images: {len(top_images)} | Subfolders: {len(subfolders)}")
    lines.append(f"Output: {out_dir}")
    lines.append(f"Numbering: pad={pad}, suffix_pad={suffix_pad}, start={start}, sep='{sep}'")
    lines.append(f"Order: top={order_top}, folder={order_folder}, inside={order_inside}")
    lines.append(f"Mode: {'MOVE' if move_instead else 'COPY'} | Include .txt: {include_txt}")
    lines.append(f"Planned renames: {len(planned)}")
    for kind, src, dst, txt_src, txt_dst in planned[:20]:
        lines.append(f"  {kind:5s}  {src.name} -> {dst.name}" + ("  (+txt)" if txt_dst else ""))

    if dry_run:
        lines.append("Dry run: no files written.")
        return _done(True)

    done = 0
    errors = 0
    for _, src, dst, txt_src, txt_dst in planned:
        wants_txt = bool(txt_src and txt_dst and txt_src.exists())
        dst_u, txt_dst_u = _next_pair_target(
            out_dir,
            dst.stem,
            dst.suffix.lower(),
            include_txt=wants_txt,
        )
        ok, error = _transfer_pair(
            src,
            dst_u,
            txt_src if wants_txt else None,
            txt_dst_u if wants_txt else None,
            move_instead,
        )
        if ok:
            done += 1
        else:
            errors += 1
            lines.append(f"[ERROR] {src.name}: {error}")

    lines.append(f"Done. {done} image(s) processed into: {out_dir}. Errors: {errors}.")
    return _done(errors == 0, "" if errors == 0 else f"{errors} rename operation(s) failed.")
