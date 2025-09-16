import os, io, json, glob, re, math
from pathlib import Path
from typing import List, Tuple
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image, ImageEnhance, ImageOps
from dotenv import load_dotenv

load_dotenv()
APP_NAME = os.getenv("APP_NAME", "BatchBench")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-dev-dev")
app.config["APP_NAME"] = APP_NAME

# Work dir
WORK_DIR = os.getenv("WORK_DIR", "").strip() or str(Path(__file__).parent.joinpath("_work"))
Path(WORK_DIR).mkdir(parents=True, exist_ok=True)

PRESET_FILES = {}
for name in ["preset_keep_warm_balanced.json", "preset_neutral_daylight.json", "custom.json"]:
    p = Path(__file__).parent / name
    if p.exists():
        try:
            PRESET_FILES[name] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass

# ---------- helpers ----------
def list_presets():
    return list(PRESET_FILES.keys())

def readable_path(p: str) -> Path:
    path = Path(p).expanduser()
    return path

def ensure_out_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def image_paths(folder: Path, exts: List[str]) -> List[Path]:
    exts = [e.lower() for e in exts]
    return [p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in exts]

def log_line(lines: List[str], msg: str):
    print(msg)
    lines.append(msg)

# ---------- routes ----------
@app.route("/")
def index():
    return render_template("index.html", presets=PRESET_FILES)

@app.route("/webp-to-png", methods=["GET", "POST"])
def webp_to_png():
    log = None
    if request.method == "POST":
        src = readable_path(request.form.get("src", ""))
        dst = readable_path(request.form.get("dst", ""))
        lines = []
        if not src.exists() or not src.is_dir():
            flash("Source folder not found.")
        else:
            ensure_out_dir(dst)
            count = 0
            for p in src.iterdir():
                if p.suffix.lower() == ".webp":
                    out = dst / (p.stem + ".png")
                    try:
                        with Image.open(p) as im:
                            im.save(out, "PNG")
                        log_line(lines, f"Converted: {p.name} -> {out.name}")
                        count += 1
                    except Exception as e:
                        log_line(lines, f"[ERROR] {p.name}: {e}")
            log_line(lines, f"Done. {count} files converted.")
            log = "\n".join(lines)
    return render_template("webp_to_png.html", log=log)

# ---- Batch Adjust ----
def apply_preset(img: Image.Image, cfg: dict) -> Image.Image:
    # exposure_ev: multiply lightness roughly by 2**ev
    ev = float(cfg.get("exposure_ev", 0))
    if ev != 0:
        img = ImageEnhance.Brightness(img).enhance(pow(2.0, ev))

    # brightness [-1..1] -> factor 1 + value
    b = float(cfg.get("brightness", 0))
    if b != 0:
        img = ImageEnhance.Brightness(img).enhance(max(0.0, 1.0 + b))

    # contrast [-1..1]
    c = float(cfg.get("contrast", 0))
    if c != 0:
        img = ImageEnhance.Contrast(img).enhance(max(0.0, 1.0 + c))

    # saturation [-1..1]
    s = float(cfg.get("saturation", 0))
    if s != 0:
        img = ImageEnhance.Color(img).enhance(max(0.0, 1.0 + s))

    # sharpness [-1..1]
    sh = float(cfg.get("sharpness", 0))
    if sh != 0:
        img = ImageEnhance.Sharpness(img).enhance(max(0.0, 1.0 + sh))

    # warmth [-1..1] : push R vs B
    w = float(cfg.get("warmth", 0))
    if w != 0:
        r,g,b = img.split()
        r = ImageEnhance.Brightness(r).enhance(1.0 + 0.5*max(0,w))
        bch = ImageEnhance.Brightness(b).enhance(1.0 - 0.5*max(0,w))
        if w < 0:
            r = ImageEnhance.Brightness(r).enhance(1.0 + 0.5*w)  # w negative reduces
            bch = ImageEnhance.Brightness(b).enhance(1.0 - 0.5*w) # increase blue
        img = Image.merge("RGB", (r,g,bch))

    # tint [-1..1] : push towards magenta (-) or green (+)
    t = float(cfg.get("tint", 0))
    if t != 0:
        r,g,b = img.split()
        g = ImageEnhance.Brightness(g).enhance(1.0 + 0.5*t)
        img = Image.merge("RGB", (r,g,b))

    # highlights/shadows simple curves
    def apply_curve(channel, curve):
        lut = [min(255, max(0, int(curve(i)))) for i in range(256)]
        return channel.point(lut)

    hi = float(cfg.get("highlights", 0))
    shd = float(cfg.get("shadows", 0))
    if hi != 0 or shd != 0:
        r,g,b = img.split()
        if shd != 0:
            def lift(x):  # lift lows
                xf = x/255.0
                return 255.0 * (xf + shd * (1 - xf) * 0.5)
            r = apply_curve(r, lift)
            g = apply_curve(g, lift)
            b = apply_curve(b, lift)
        if hi != 0:
            def tame(x): # reduce highs
                xf = x/255.0
                return 255.0 * (xf - hi * (xf**2) * 0.5)
            r = apply_curve(r, tame)
            g = apply_curve(g, tame)
            b = apply_curve(b, tame)
        img = Image.merge("RGB", (r,g,b))

    # vignette [0..1] simple radial darken
    vig = float(cfg.get("vignette", 0))
    if vig > 0:
        w,h = img.size
        cx, cy = w/2, h/2
        maxd = math.hypot(cx, cy)
        px = img.load()
        for y in range(h):
            for x in range(w):
                d = math.hypot(x-cx, y-cy) / maxd
                factor = 1 - vig * (d**2)
                r,g,b = px[x,y]
                px[x,y] = (int(r*factor), int(g*factor), int(b*factor))

    return img

@app.route("/batch-adjust", methods=["GET","POST"])
def batch_adjust():
    log = None
    preset_names = list_presets()
    if request.method == "POST":
        src = readable_path(request.form.get("src", ""))
        dst = readable_path(request.form.get("dst", ""))
        preset_name = request.form.get("preset", "")
        suffix = request.form.get("suffix", "_adj")
        limit = int(request.form.get("limit", "0"))

        lines = []
        cfg = PRESET_FILES.get(preset_name)
        if not cfg:
            flash("Preset not found.")
        elif not src.exists() or not src.is_dir():
            flash("Source folder not found.")
        else:
            ensure_out_dir(dst)
            count = 0
            for p in src.iterdir():
                if p.suffix.lower() in (".jpg",".jpeg",".png",".webp",".bmp"):
                    try:
                        with Image.open(p) as im:
                            im = im.convert("RGB")
                            out = apply_preset(im, cfg)
                            out_path = dst / f"{p.stem}{suffix}.jpg"
                            out.save(out_path, "JPEG", quality=95)
                            lines.append(f"Saved {out_path.name}")
                            count += 1
                            if limit and count >= limit:
                                break
                    except Exception as e:
                        lines.append(f"[ERROR] {p.name}: {e}")
            lines.append(f"Done. {count} files processed.")
            log = "\n".join(lines)
    return render_template("batch_adjust.html", preset_names=preset_names, log=log)

# ---- Dataset tagging ----
def parse_tags_field(text: str) -> List[str]:
    return [t.strip() for t in text.split(",") if t.strip()]

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""

def write_txt(path: Path, text: str, backup: bool):
    if backup and path.exists():
        path.with_suffix(path.suffix + ".bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text, encoding="utf-8")

def split_tags(text: str) -> List[str]:
    return [t.strip() for t in text.split(",") if t.strip()]

def join_tags(tags: List[str]) -> str:
    return ", ".join(tags)

@app.route("/dataset-labeling", methods=["GET","POST"])
def dataset_labeling():
    log = None
    if request.method == "POST":
        folder = readable_path(request.form.get("folder",""))
        mode = request.form.get("mode","insert")
        tags_field = request.form.get("tags","")
        exts = [e.strip().lower() for e in request.form.get("exts",".jpg,.jpeg,.png,.webp").split(",") if e.strip()]
        backup = bool(request.form.get("backup"))

        lines = []
        if not folder.exists() or not folder.is_dir():
            flash("Folder not found.")
        else:
            images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
            processed = 0
            for img in images:
                txt = img.with_suffix(".txt")
                if not txt.exists():
                    continue
                src = read_txt(txt)
                taglist = split_tags(src)

                if mode == "insert":
                    add = parse_tags_field(tags_field)
                    for t in add:
                        if t not in taglist:
                            taglist.append(t)
                    write_txt(txt, join_tags(taglist), backup)
                    lines.append(f"{img.name}: insert -> {add}")

                elif mode == "delete":
                    deltags = set(parse_tags_field(tags_field))
                    taglist = [t for t in taglist if t not in deltags]
                    write_txt(txt, join_tags(taglist), backup)
                    lines.append(f"{img.name}: delete -> {sorted(deltags)}")

                elif mode == "replace":
                    mapping = {}
                    parts = [p.strip() for p in tags_field.split(";") if p.strip()]
                    for p in parts:
                        if "->" in p:
                            old, new = [x.strip() for x in p.split("->",1)]
                            mapping[old] = new
                    taglist = [mapping.get(t, t) for t in taglist]
                    write_txt(txt, join_tags(taglist), backup)
                    lines.append(f"{img.name}: replace -> {mapping}")

                elif mode == "move":
                    pair = parse_tags_field(tags_field)
                    if len(pair) == 2:
                        A, B = pair
                        if A in taglist and B in taglist:
                            taglist = [t for t in taglist if t != A]
                            idx = taglist.index(B)
                            taglist.insert(idx, A)
                            write_txt(txt, join_tags(taglist), backup)
                            lines.append(f"{img.name}: move {A} before {B}")
                        else:
                            lines.append(f"{img.name}: skip (A or B not in list)")
                    else:
                        lines.append("Invalid move format. Use: tagA,tagB")

                elif mode == "dedup":
                    seen = set()
                    out = []
                    for t in taglist:
                        if t not in seen:
                            out.append(t)
                            seen.add(t)
                    write_txt(txt, join_tags(out), backup)
                    lines.append(f"{img.name}: dedup -> {len(taglist)-len(out)} removed")
                processed += 1
            lines.append(f"Done. {processed} files checked.")
            log = "\n".join(lines)
    return render_template("dataset_labeling.html", log=log)

# ---- Rename suffix ----
@app.route("/rename-suffix", methods=["GET","POST"])
def rename_suffix():
    log = None
    if request.method == "POST":
        folder = readable_path(request.form.get("folder",""))
        suffix = request.form.get("suffix","_2")
        lines = []
        if not folder.exists() or not folder.is_dir():
            flash("Folder not found.")
        else:
            count = 0
            for p in folder.iterdir():
                if p.is_file():
                    new = p.with_name(p.stem + suffix + p.suffix)
                    try:
                        p.rename(new)
                        lines.append(f"Renamed: {p.name} -> {new.name}")
                        count += 1
                    except Exception as e:
                        lines.append(f"[ERROR] {p.name}: {e}")
            lines.append(f"Done. {count} files renamed.")
            log = "\n".join(lines)
    return render_template("rename_suffix.html", log=log)

# ---- Reorder pairs ----
def collect_numeric_subfolders(root: Path):
    subs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    subs.sort(key=lambda p: int(p.name))
    return subs

def collect_pairs(folder: Path, image_exts):
    files = list(folder.iterdir())
    stems_txt = {p.stem for p in files if p.suffix.lower() == ".txt"}
    out = []
    for p in files:
        if p.suffix.lower() in image_exts and p.stem in stems_txt:
            out.append((p, p.with_suffix(".txt")))
    out.sort(key=lambda t: t[0].stem.lower())
    return out

@app.route("/reorder-pairs", methods=["GET","POST"])
def reorder_pairs():
    log = None
    if request.method == "POST":
        root = readable_path(request.form.get("root",""))
        digits = int(request.form.get("digits","3"))
        start = int(request.form.get("start","1"))
        image_exts = [e.strip().lower() for e in request.form.get("exts",".jpg,.jpeg,.png,.webp").split(",") if e.strip()]
        lines = []
        if not root.exists() or not root.is_dir():
            flash("Root folder not found.")
        else:
            seq = start
            renamed = 0
            for sub in collect_numeric_subfolders(root):
                pairs = collect_pairs(sub, image_exts)
                for img, txt in pairs:
                    new_stem = str(seq).zfill(digits)
                    new_img = img.with_name(new_stem + img.suffix)
                    new_txt = txt.with_name(new_stem + ".txt")
                    try:
                        img.rename(new_img)
                        txt.rename(new_txt)
                        lines.append(f"{sub.name}: {img.name} + {txt.name} -> {new_img.name} + {new_txt.name}")
                        renamed += 1
                        seq += 1
                    except Exception as e:
                        lines.append(f"[ERROR] {img.name}: {e}")
            lines.append(f"Done. {renamed} pairs renamed (global sequence).")
            log = "\n".join(lines)
    return render_template("reorder_pairs.html", log=log)

if __name__ == "__main__":
    app.run(debug=True)
