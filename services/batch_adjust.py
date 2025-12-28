from typing import Tuple, List
from pathlib import Path
from PIL import Image
from utils.io import readable_path, ensure_out_dir, log_join
from utils.image_ops import apply_preset

def handle(form, ctx) -> Tuple[str, str]:
    active_tab = "batch"
    src = readable_path(form.get("src_batch",""))
    dst = readable_path(form.get("dst_batch",""))
    preset_name = form.get("preset","")
    suffix = form.get("suffix","_adj")
    limit = int(form.get("limit","0"))
    presets = ctx.get("presets", {})
    cfg = presets.get(preset_name)
    lines: List[str] = []

    if not cfg:
        lines.append("Preset not found.")
    elif not src.exists() or not src.is_dir():
        lines.append("Source folder not found.")
    else:
        ensure_out_dir(dst)
        count=0
        for p in src.iterdir():
            if p.suffix.lower() in (".jpg",".jpeg",".png",".webp",".bmp"):
                try:
                    with Image.open(p) as im:
                        im=im.convert("RGB")
                        out=apply_preset(im,cfg)
                        out_path = dst/f"{p.stem}{suffix}.jpg"
                        out.save(out_path,"JPEG",quality=95)
                        lines.append(f"Saved {out_path.name}"); count+=1
                        if limit and count>=limit: break
                except Exception as e:
                    lines.append(f"[ERROR] {p.name}: {e}")
        lines.append(f"Done. {count} files processed.")
    return active_tab, log_join(lines)
