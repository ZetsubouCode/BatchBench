import os, json
from pathlib import Path
from typing import List
from flask import Flask, render_template, request, flash, jsonify
from dotenv import load_dotenv

from utils.io import readable_path, windows_drives

from services.registry import TOOL_REGISTRY

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

def list_presets() -> List[str]:
    return list(PRESET_FILES.keys())

# -------- Folder picker APIs --------
@app.get("/api/drives")
def api_drives():
    if os.name == "nt":
        return jsonify({"drives": windows_drives()})
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

# -------- Main page (dispatch via registry) --------
@app.route("/", methods=["GET", "POST"])
def index():
    active_tab = request.values.get("tab", "webp")
    log = ""

    is_ajax = request.headers.get("X-Requested-With", "").lower() in ("xmlhttprequest", "fetch")

    if request.method == "POST":
        tool = request.form.get("tool", "")
        handler = TOOL_REGISTRY.get(tool)
        if not handler:
            flash(f"Unknown tool: {tool}")
        else:
            ctx = {"presets": PRESET_FILES}
            try:
                active_tab, log = handler(request.form, ctx)
            except Exception as e:
                flash(f"Error running tool '{tool}': {e}")

        if is_ajax:
            return jsonify({"ok": True, "active_tab": active_tab, "log": log})

    return render_template(
        "index.html",
        preset_names=list_presets(),
        active_tab=active_tab,
        log=log,
        presets=PRESET_FILES,
    )


if __name__ == "__main__":
    app.run(debug=True)