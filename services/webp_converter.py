from pathlib import Path
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
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
        files = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() == ".webp"]
        count = 0
        workers = max(1, min(8, (os.cpu_count() or 4)))

        def _convert(path: Path):
            out = dst / (path.stem + ".png")
            with Image.open(path) as im:
                im.save(out, "PNG")
            return path.name, out.name

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_convert, p): p for p in files}
            for fut in as_completed(futures):
                src_path = futures[fut]
                try:
                    src_name, out_name = fut.result()
                    lines.append(f"Converted: {src_name} -> {out_name}")
                    count += 1
                except Exception as e:
                    lines.append(f"[ERROR] {src_path.name}: {e}")
        lines.append(f"Done. {count} files converted.")
    return active_tab, log_join(lines)
