from pathlib import Path
from typing import Tuple, List
from PIL import Image
from utils.io import readable_path, ensure_out_dir, log_join

def handle(form, ctx) -> Tuple[str, str]:
    active_tab = "webp"
    src = readable_path(form.get("src_webp",""))
    dst = readable_path(form.get("dst_webp",""))
    lines: List[str] = []

    if not src.exists() or not src.is_dir():
        lines.append("Source folder not found.")
    else:
        ensure_out_dir(dst)
        count=0
        for p in src.iterdir():
            if p.suffix.lower()==".webp":
                out = dst/(p.stem+".png")
                try:
                    with Image.open(p) as im:
                        im.save(out,"PNG")
                    lines.append(f"Converted: {p.name} -> {out.name}"); count+=1
                except Exception as e:
                    lines.append(f"[ERROR] {p.name}: {e}")
        lines.append(f"Done. {count} files converted.")
    return active_tab, log_join(lines)
