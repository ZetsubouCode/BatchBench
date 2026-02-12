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
    if not images:
        raise ValueError("No input images provided.")

    if resize == "auto":
        resize = "match-height" if orientation == "h" else "match-width"

    # Pass 1: collect source sizes only (low memory)
    src_sizes: List[Tuple[int, int]] = []
    for p in images:
        with Image.open(p) as im:
            src_sizes.append((im.width, im.height))

    planned_sizes: List[Tuple[int, int]] = []
    if orientation == "h":
        if resize == "match-height":
            target_h = max(h for _, h in src_sizes)
            for w, h in src_sizes:
                if h == target_h:
                    planned_sizes.append((w, h))
                else:
                    planned_sizes.append((max(1, int(round(w * (target_h / h)))), target_h))
        else:
            planned_sizes = list(src_sizes)
        canvas_w = sum(w for w, _ in planned_sizes) + gap * (len(planned_sizes) - 1)
        canvas_h = max(h for _, h in planned_sizes)
    else:
        if resize == "match-width":
            target_w = max(w for w, _ in src_sizes)
            for w, h in src_sizes:
                if w == target_w:
                    planned_sizes.append((w, h))
                else:
                    planned_sizes.append((target_w, max(1, int(round(h * (target_w / w))))))
        else:
            planned_sizes = list(src_sizes)
        canvas_w = max(w for w, _ in planned_sizes)
        canvas_h = sum(h for _, h in planned_sizes) + gap * (len(planned_sizes) - 1)

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)

    # Pass 2: open-paste-close one-by-one
    if orientation == "h":
        x = 0
        for path, (target_w, target_h) in zip(images, planned_sizes):
            with Image.open(path) as src:
                im = src.convert("RGB")
                if im.width != target_w or im.height != target_h:
                    im = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
                if align == "top":
                    y = 0
                elif align == "bottom":
                    y = canvas_h - target_h
                else:
                    y = (canvas_h - target_h) // 2
                canvas.paste(im, (x, y))
                if im is not src:
                    try:
                        im.close()
                    except Exception:
                        pass
            x += target_w + gap
    else:
        y = 0
        for path, (target_w, target_h) in zip(images, planned_sizes):
            with Image.open(path) as src:
                im = src.convert("RGB")
                if im.width != target_w or im.height != target_h:
                    im = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
                if align == "left":
                    x = 0
                elif align == "right":
                    x = canvas_w - target_w
                else:
                    x = (canvas_w - target_w) // 2
                canvas.paste(im, (x, y))
                if im is not src:
                    try:
                        im.close()
                    except Exception:
                        pass
            y += target_h + gap

    return canvas
