import os, json
from pathlib import Path
from typing import List, Tuple
from flask import Flask, render_template, request, flash, jsonify
from PIL import Image, ImageEnhance
from dotenv import load_dotenv
import shutil
import math

load_dotenv()
APP_NAME = os.getenv("APP_NAME", "BatchBench")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-dev-dev")
app.config["APP_NAME"] = APP_NAME

# Work dir
WORK_DIR = os.getenv("WORK_DIR", "").strip() or str(Path(__file__).parent.joinpath("_work"))
Path(WORK_DIR).mkdir(parents=True, exist_ok=True)

# Load presets if present
PRESET_FILES = {}
for name in ["preset_keep_warm_balanced.json", "preset_neutral_daylight.json", "custom.json"]:
    p = Path(__file__).parent / name
    if p.exists():
        try:
            PRESET_FILES[name] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass

# ---------- helpers ----------
def list_presets() -> List[str]:
    return list(PRESET_FILES.keys())

def readable_path(p: str) -> Path:
    return Path(p).expanduser()

def ensure_out_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def log_join(lines: List[str]) -> str:
    return "\n".join(lines)

def apply_preset(img: Image.Image, cfg: dict) -> Image.Image:
    ev = float(cfg.get("exposure_ev", 0))
    if ev != 0: img = ImageEnhance.Brightness(img).enhance(pow(2.0, ev))
    b = float(cfg.get("brightness", 0))
    if b != 0: img = ImageEnhance.Brightness(img).enhance(max(0.0, 1.0 + b))
    c = float(cfg.get("contrast", 0))
    if c != 0: img = ImageEnhance.Contrast(img).enhance(max(0.0, 1.0 + c))
    s = float(cfg.get("saturation", 0))
    if s != 0: img = ImageEnhance.Color(img).enhance(max(0.0, 1.0 + s))
    sh = float(cfg.get("sharpness", 0))
    if sh != 0: img = ImageEnhance.Sharpness(img).enhance(max(0.0, 1.0 + sh))

    # warmth [-1..1] via channel balance
    w = float(cfg.get("warmth", 0))
    if w != 0:
        r,g,b = img.split()
        r = ImageEnhance.Brightness(r).enhance(1.0 + 0.5*max(0,w))
        bch = ImageEnhance.Brightness(b).enhance(1.0 - 0.5*max(0,w))
        if w < 0:
            r = ImageEnhance.Brightness(r).enhance(1.0 + 0.5*w)
            bch = ImageEnhance.Brightness(b).enhance(1.0 - 0.5*w)
        img = Image.merge("RGB", (r,g,bch))

    # tint [-1..1] (towards green for +, magenta for -)
    t = float(cfg.get("tint", 0))
    if t != 0:
        r,g,b = img.split()
        g = ImageEnhance.Brightness(g).enhance(1.0 + 0.5*t)
        img = Image.merge("RGB", (r,g,b))

    # simple highlights/shadows curves
    hi = float(cfg.get("highlights", 0))
    shd = float(cfg.get("shadows", 0))
    if hi != 0 or shd != 0:
        def apply_curve(channel, curve):
            lut = [min(255, max(0, int(curve(i)))) for i in range(256)]
            return channel.point(lut)
        r,g,b = img.split()
        if shd != 0:
            def lift(x): xf = x/255.0; return 255.0 * (xf + shd * (1 - xf) * 0.5)
            r = apply_curve(r, lift); g = apply_curve(g, lift); b = apply_curve(b, lift)
        if hi != 0:
            def tame(x): xf = x/255.0; return 255.0 * (xf - hi * (xf**2) * 0.5)
            r = apply_curve(r, tame); g = apply_curve(g, tame); b = apply_curve(b, tame)
        img = Image.merge("RGB", (r,g,b))
    # optional vignette (0..1)
    vig = float(cfg.get("vignette", 0))
    if vig > 0:
        w,h = img.size; cx,cy = w/2, h/2; maxd = math.hypot(cx, cy); px = img.load()
        for y in range(h):
            for x in range(w):
                d = math.hypot(x-cx, y-cy) / maxd
                factor = 1 - vig * (d**2)
                R,G,B = px[x,y]; px[x,y] = (int(R*factor), int(G*factor), int(B*factor))
    return img

def split_tags(text: str): return [t.strip() for t in text.split(",") if t.strip()]
def join_tags(tags: List[str]): return ", ".join(tags)

# ---------- dataset helpers ----------
def collect_pairs(folder: Path, image_exts: List[str]) -> List[Tuple[Path, Path]]:
    files = list(folder.iterdir())
    stems_txt = {p.stem for p in files if p.suffix.lower() == ".txt"}
    out = []
    for p in files:
        if p.is_file() and p.suffix.lower() in image_exts and p.stem in stems_txt:
            out.append((p, p.with_suffix(".txt")))
    out.sort(key=lambda t: t[0].stem.lower())
    return out

# ---------- folder picker API ----------
def _windows_drives():
    drives = []
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        root = f"{c}:\\"
        if os.path.exists(root):
            drives.append(root)
    return drives

@app.get("/api/drives")
def api_drives():
    if os.name == "nt":
        return jsonify({"drives": _windows_drives()})
    return jsonify({"drives": ["/"]})

@app.get("/api/list-dir")
def api_list_dir():
    path = request.args.get("path", "")
    p = readable_path(path) if path else Path("/")
    out = {"path": str(p), "parent": str(p.parent) if p != p.parent else str(p)}
    dirs = []
    try:
        for d in sorted([x for x in p.iterdir() if x.is_dir()], key=lambda x: x.name.lower()):
            dirs.append(str(d))
    except Exception:
        pass
    out["dirs"] = dirs
    return jsonify(out)

# ---------- TABS PAGE ----------
@app.route("/", methods=["GET", "POST"])
def index():
    preset_names = list_presets()
    active_tab = request.values.get("tab", "webp")
    log = ""

    if request.method == "POST":
        tool = request.form.get("tool", "")

        # 1) WebP -> PNG
        if tool == "webp":
            active_tab = "webp"
            src = readable_path(request.form.get("src_webp",""))
            dst = readable_path(request.form.get("dst_webp",""))
            lines = []
            if not src.exists() or not src.is_dir():
                flash("Source folder not found.")
            else:
                ensure_out_dir(dst)
                count=0
                for p in src.iterdir():
                    if p.suffix.lower()==".webp":
                        out = dst/(p.stem+".png")
                        try:
                            with Image.open(p) as im: im.save(out,"PNG")
                            lines.append(f"Converted: {p.name} -> {out.name}"); count+=1
                        except Exception as e:
                            lines.append(f"[ERROR] {p.name}: {e}")
                lines.append(f"Done. {count} files converted.")
            log = log_join(lines)

        # 2) Batch Adjust
        elif tool == "batch":
            active_tab = "batch"
            src = readable_path(request.form.get("src_batch",""))
            dst = readable_path(request.form.get("dst_batch",""))
            preset_name = request.form.get("preset","")
            suffix = request.form.get("suffix","_adj")
            limit = int(request.form.get("limit","0"))
            cfg = PRESET_FILES.get(preset_name)
            lines=[]
            if not cfg:
                flash("Preset not found.")
            elif not src.exists() or not src.is_dir():
                flash("Source folder not found.")
            else:
                ensure_out_dir(dst)
                count=0
                for p in src.iterdir():
                    if p.suffix.lower() in (".jpg",".jpeg",".png",".webp",".bmp"):
                        try:
                            with Image.open(p) as im:
                                im=im.convert("RGB"); out=apply_preset(im,cfg)
                                out_path = dst/f"{p.stem}{suffix}.jpg"
                                out.save(out_path,"JPEG",quality=95)
                                lines.append(f"Saved {out_path.name}"); count+=1
                                if limit and count>=limit: break
                        except Exception as e:
                            lines.append(f"[ERROR] {p.name}: {e}")
                lines.append(f"Done. {count} files processed.")
            log = log_join(lines)

        # 3) Dataset Tag Editor
        elif tool == "tags":
            active_tab = "tags"
            folder = readable_path(request.form.get("folder",""))
            mode = request.form.get("mode","insert")
            tags_field = request.form.get("tags","")
            exts = [e.strip().lower() for e in request.form.get("exts",".jpg,.jpeg,.png,.webp").split(",") if e.strip()]
            backup = bool(request.form.get("backup"))
            lines=[]
            if not folder.exists() or not folder.is_dir():
                flash("Folder not found.")
            else:
                images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
                processed=0
                for img in images:
                    txt = img.with_suffix(".txt")
                    if not txt.exists(): continue
                    src = txt.read_text(encoding="utf-8") if txt.exists() else ""
                    taglist = split_tags(src)

                    if mode=="insert":
                        add = [t.strip() for t in tags_field.split(",") if t.strip()]
                        for t in add:
                            if t not in taglist: taglist.append(t)
                        if backup and txt.exists():
                            txt.with_suffix(txt.suffix+".bak").write_text(txt.read_text(encoding="utf-8"), encoding="utf-8")
                        txt.write_text(join_tags(taglist), encoding="utf-8")
                        lines.append(f"{img.name}: insert -> {add}")

                    elif mode=="delete":
                        deltags=set([t.strip() for t in tags_field.split(",") if t.strip()])
                        taglist=[t for t in taglist if t not in deltags]
                        if backup and txt.exists():
                            txt.with_suffix(txt.suffix+".bak").write_text(txt.read_text(encoding="utf-8"), encoding="utf-8")
                        txt.write_text(join_tags(taglist), encoding="utf-8")
                        lines.append(f"{img.name}: delete -> {sorted(deltags)}")

                    elif mode=="replace":
                        mapping={}
                        parts=[p.strip() for p in tags_field.split(";") if p.strip()]
                        for p in parts:
                            if "->" in p:
                                old,new=[x.strip() for x in p.split("->",1)]
                                mapping[old]=new
                        taglist=[mapping.get(t,t) for t in taglist]
                        if backup and txt.exists():
                            txt.with_suffix(txt.suffix+".bak").write_text(txt.read_text(encoding="utf-8"), encoding="utf-8")
                        txt.write_text(join_tags(taglist), encoding="utf-8")
                        lines.append(f"{img.name}: replace -> {mapping}")

                    elif mode=="move":
                        pair=[t.strip() for t in tags_field.split(",") if t.strip()]
                        if len(pair)==2:
                            A,B=pair
                            if A in taglist and B in taglist:
                                taglist=[t for t in taglist if t!=A]
                                idx=taglist.index(B); taglist.insert(idx,A)
                                if backup and txt.exists():
                                    txt.with_suffix(txt.suffix+".bak").write_text(txt.read_text(encoding="utf-8"), encoding="utf-8")
                                txt.write_text(join_tags(taglist), encoding="utf-8")
                                lines.append(f"{img.name}: move {A} before {B}")
                            else:
                                lines.append(f"{img.name}: skip (A or B not in list)")
                        else:
                            lines.append("Invalid move format. Use: tagA,tagB")

                    elif mode=="dedup":
                        seen=set(); out=[]
                        for t in taglist:
                            if t not in seen: out.append(t); seen.add(t)
                        if backup and txt.exists():
                            txt.with_suffix(txt.suffix+".bak").write_text(txt.read_text(encoding="utf-8"), encoding="utf-8")
                        txt.write_text(join_tags(out), encoding="utf-8")
                        lines.append(f"{img.name}: dedup -> {len(taglist)-len(out)} removed")
                    processed+=1
                lines.append(f"Done. {processed} files checked.")
            log = log_join(lines)

        # 4) Combine Datasets (NEW)
        elif tool == "combine":
            active_tab = "combine"
            folder_a = readable_path(request.form.get("folder_a",""))
            folder_b = readable_path(request.form.get("folder_b",""))
            out_dir = readable_path(request.form.get("out_dir",""))
            suffix = request.form.get("suffix_combine","_B")
            exts = [e.strip().lower() for e in request.form.get("exts_combine",".jpg,.jpeg,.png,.webp").split(",") if e.strip()]
            move_instead = bool(request.form.get("move_instead"))
            lines = []

            if not folder_a.exists() or not folder_a.is_dir():
                flash("Folder A not found.")
            elif not folder_b.exists() or not folder_b.is_dir():
                flash("Folder B not found.")
            else:
                ensure_out_dir(out_dir)
                pairs_a = collect_pairs(folder_a, exts)
                pairs_b = collect_pairs(folder_b, exts)
                len_a, len_b = len(pairs_a), len(pairs_b)

                # Choose smaller set to rename
                small_pairs, small_name = (pairs_a, "A") if len_a <= len_b else (pairs_b, "B")
                big_pairs, big_name = (pairs_b, "B") if len_a <= len_b else (pairs_a, "A")

                lines.append(f"Pairs A: {len_a}, Pairs B: {len_b}")
                lines.append(f"Renaming smaller set: {small_name} using suffix '{suffix}'")

                # Copy/Move big pairs unchanged
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

                # Copy/Move small pairs with suffix
                renamed = 0
                for img, txt in small_pairs:
                    new_stem = img.stem + suffix
                    target_img = out_dir / (new_stem + img.suffix)
                    target_txt = out_dir / (new_stem + ".txt")

                    # collision safety: if exists, add numeric bump
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
            log = log_join(lines)

    return render_template(
        "index.html",
        presets=PRESET_FILES,
        preset_names=preset_names,
        active_tab=active_tab,
        log=log,
    )

if __name__ == "__main__":
    app.run(debug=True)
