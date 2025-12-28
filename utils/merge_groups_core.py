import re
from typing import Dict, List, Tuple
from pathlib import Path
from PIL import Image

PAT = re.compile(r"^(?P<prefix>.+)_(?P<order>\d+)$")

def list_pngs(folder: Path, glob_pat: str) -> List[Path]:
    return sorted([p for p in folder.glob(glob_pat) if p.suffix.lower()=='.png'], key=lambda p: p.name.lower())

def group_by_prefix(files: List[Path]) -> Dict[str, List[Tuple[int, Path]]]:
    groups: Dict[str, List[Tuple[int, Path]]] = {}
    for p in files:
        m = PAT.match(p.stem)
        if not m: 
            continue
        prefix = m.group("prefix")
        order = int(m.group("order"))
        groups.setdefault(prefix, []).append((order, p))
    for k in list(groups.keys()):
        groups[k].sort(key=lambda t: t[0])
    return groups

def scale_to_height(img: Image.Image, height: int) -> Image.Image:
    if img.height == height: return img
    ratio = height / img.height
    return img.resize((int(img.width * ratio), height), Image.Resampling.LANCZOS)

def scale_to_width(img: Image.Image, width: int) -> Image.Image:
    if img.width == width: return img
    ratio = width / img.width
    return img.resize((width, int(img.height * ratio)), Image.Resampling.LANCZOS)

def merge_many(images: List[Path], orientation: str, align: str, gap: int, bg: str, resize: str) -> Image.Image:
    ims = [Image.open(p).convert("RGB") for p in images]
    try:
        if resize == "auto":
            resize = "match-height" if orientation == "h" else "match-width"

        if orientation == "h":
            if resize == "match-height":
                target_h = max(im.height for im in ims)
                ims = [scale_to_height(im, target_h) for im in ims]
            w = sum(im.width for im in ims) + gap * (len(ims)-1)
            h = max(im.height for im in ims)
            canvas = Image.new("RGB", (w, h), bg)
            x = 0
            for im in ims:
                if align == "top": y = 0
                elif align == "bottom": y = h - im.height
                else: y = (h - im.height) // 2
                canvas.paste(im, (x, y)); x += im.width + gap
        else:
            if resize == "match-width":
                target_w = max(im.width for im in ims)
                ims = [scale_to_width(im, target_w) for im in ims]
            w = max(im.width for im in ims)
            h = sum(im.height for im in ims) + gap * (len(ims)-1)
            canvas = Image.new("RGB", (w, h), bg)
            y = 0
            for im in ims:
                if align == "left": x = 0
                elif align == "right": x = w - im.width
                else: x = (w - im.width) // 2
                canvas.paste(im, (x, y)); y += im.height + gap
        return canvas
    finally:
        for im in ims:
            try: im.close()
            except Exception: pass
