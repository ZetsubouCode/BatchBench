from typing import Tuple, List
from utils.io import readable_path, ensure_out_dir, log_join
from utils.dataset import collect_pairs
import shutil

def handle(form, ctx) -> Tuple[str, str]:
    active_tab = "combine"
    folder_a = readable_path(form.get("folder_a",""))
    folder_b = readable_path(form.get("folder_b",""))
    out_dir = readable_path(form.get("out_dir",""))
    suffix = form.get("suffix_combine","_B")
    exts = [e.strip().lower() for e in form.get("exts_combine",".jpg,.jpeg,.png,.webp").split(",") if e.strip()]
    move_instead = bool(form.get("move_instead"))
    lines: List[str] = []

    if not folder_a.exists() or not folder_a.is_dir():
        lines.append("Folder A not found.")
        return active_tab, log_join(lines)
    if not folder_b.exists() or not folder_b.is_dir():
        lines.append("Folder B not found.")
        return active_tab, log_join(lines)

    ensure_out_dir(out_dir)
    pairs_a = collect_pairs(folder_a, exts)
    pairs_b = collect_pairs(folder_b, exts)
    len_a, len_b = len(pairs_a), len(pairs_b)

    small_pairs, small_name = (pairs_a, "A") if len_a <= len_b else (pairs_b, "B")
    big_pairs, big_name = (pairs_b, "B") if len_a <= len_b else (pairs_a, "A")

    lines.append(f"Pairs A: {len_a}, Pairs B: {len_b}")
    lines.append(f"Renaming smaller set: {small_name} using suffix '{suffix}'")

    copied = 0
    for img, txt in big_pairs:
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
    for img, txt in small_pairs:
        new_stem = img.stem + suffix
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
