import os, json, sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from flask import Flask, render_template, request, flash, jsonify
from dotenv import load_dotenv

from utils.io import readable_path, windows_drives

from services.registry import TOOL_REGISTRY
from services import normalizer
from services.pipeline import PIPELINE_MANAGER

load_dotenv()
APP_NAME = os.getenv("APP_NAME", "BatchBench")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-dev-dev")
app.config["APP_NAME"] = APP_NAME

# Work dir
WORK_DIR = os.getenv("WORK_DIR", "").strip() or str(Path(__file__).parent.joinpath("_work"))
Path(WORK_DIR).mkdir(parents=True, exist_ok=True)

# Dataset normalization preset root
NORMALIZE_PRESET_ROOT = Path(__file__).parent / "presets"
NORMALIZE_PRESET_ROOT.mkdir(parents=True, exist_ok=True)

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


# -------- Normalization helpers --------
def _parse_exts(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if not raw:
        return normalizer.DEFAULT_IMAGE_EXTS
    parts = []
    for token in str(raw).split(","):
        val = token.strip()
        if val:
            if not val.startswith("."):
                val = "." + val
            parts.append(val)
    return parts or normalizer.DEFAULT_IMAGE_EXTS


def _parse_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        s = str(val).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def build_normalize_options(payload: Dict[str, Any]) -> Tuple[Optional[normalizer.NormalizeOptions], Optional[str]]:
    dataset_path = (payload.get("dataset_path") or "").strip()
    if not dataset_path:
        return None, "dataset_path is required"

    preset_type = (payload.get("preset_type") or "anime").strip()
    preset_file = (payload.get("preset_file") or "").strip()
    if not preset_file:
        return None, "preset_file is required"

    opts = normalizer.NormalizeOptions(
        dataset_path=readable_path(dataset_path),
        recursive=_parse_bool(payload.get("recursive")),
        include_missing_txt=_parse_bool(payload.get("include_missing_txt")),
        preset_type=preset_type or "default",
        preset_file=preset_file,
        extra_remove=normalizer.clean_input_list(payload.get("extra_remove") or ""),
        extra_keep=normalizer.clean_input_list(payload.get("extra_keep") or ""),
        move_unknown_background_to_optional=_parse_bool(payload.get("move_unknown_background_to_optional")),
        background_threshold=_parse_float(payload.get("background_threshold")),
        normalize_order=_parse_bool(payload.get("normalize_order", True), True),
        preview_limit=int(payload.get("preview_limit") or 30),
        backup_enabled=_parse_bool(payload.get("backup_enabled", True), True),
        image_exts=_parse_exts(payload.get("image_exts")),
        identity_tags=normalizer.clean_input_list(payload.get("identity_tags") or ""),
    )
    opts.preview_limit = max(1, min(500, opts.preview_limit))
    return opts, None


def _serialize_scan_result(data: Dict[str, Any]) -> Dict[str, Any]:
    if not data.get("ok"):
        return data
    out = dict(data)
    counts = out.get("tag_counts")
    if counts:
        out["tag_counts"] = {k: int(v) for k, v in dict(counts).items()}
    out["top_tags"] = [{"tag": tag, "count": int(cnt)} for tag, cnt in (out.get("top_tags") or [])]
    return out
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


# -------- Normalization APIs --------
@app.get("/api/presets")
def api_preset_list():
    ptype = request.args.get("type", "anime")
    files = normalizer.list_preset_files(NORMALIZE_PRESET_ROOT, ptype)
    return jsonify({"ok": True, "presets": files})


@app.get("/api/preset")
def api_preset_get():
    ptype = request.args.get("type", "anime")
    fname = request.args.get("file", "")
    try:
        data = normalizer.load_preset(NORMALIZE_PRESET_ROOT, ptype, fname)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "preset": data})


@app.get("/api/preset-types")
def api_preset_types():
    types = []
    try:
        for p in NORMALIZE_PRESET_ROOT.iterdir():
            if p.is_dir():
                types.append(p.name)
    except Exception:
        pass
    return jsonify({"ok": True, "types": sorted(types)})


@app.post("/api/normalize/scan")
def api_normalize_scan():
    payload = request.get_json(silent=True) or {}
    opts, err = build_normalize_options(payload)
    if err:
        return jsonify({"ok": False, "error": err}), 400
    result = normalizer.scan_dataset(opts, NORMALIZE_PRESET_ROOT)
    return jsonify(_serialize_scan_result(result))


@app.post("/api/normalize/dryrun")
def api_normalize_dryrun():
    payload = request.get_json(silent=True) or {}
    opts, err = build_normalize_options(payload)
    if err:
        return jsonify({"ok": False, "error": err}), 400
    result = normalizer.dry_run(opts, NORMALIZE_PRESET_ROOT)
    return jsonify(result)


@app.post("/api/normalize/apply")
def api_normalize_apply():
    payload = request.get_json(silent=True) or {}
    opts, err = build_normalize_options(payload)
    if err:
        return jsonify({"ok": False, "error": err}), 400
    result = normalizer.apply_normalization(opts, NORMALIZE_PRESET_ROOT)
    return jsonify(result)


# -------- Pipeline APIs --------
@app.get("/api/pipeline/presets")
def api_pipeline_presets():
    types = []
    files = {}
    try:
        for p in NORMALIZE_PRESET_ROOT.iterdir():
            if p.is_dir():
                types.append(p.name)
                files[p.name] = normalizer.list_preset_files(NORMALIZE_PRESET_ROOT, p.name)
    except Exception:
        pass
    return jsonify({"ok": True, "types": sorted(types), "files": files})


@app.post("/api/pipeline/start")
def api_pipeline_start():
    payload = request.get_json(silent=True) or {}
    payload["preset_root"] = str(NORMALIZE_PRESET_ROOT)
    payload["dataset_path"] = (payload.get("dataset_path") or "").strip()
    payload["working_dir"] = (payload.get("working_dir") or "").strip()
    payload["output_dir"] = (payload.get("output_dir") or "").strip()
    payload["preset_type"] = (payload.get("preset_type") or "anime").strip()
    payload["preset_file"] = (payload.get("preset_file") or "").strip()
    payload["image_exts"] = _parse_exts(payload.get("image_exts"))
    payload["recursive"] = _parse_bool(payload.get("recursive"))
    payload["run_autotag"] = _parse_bool(payload.get("run_autotag", True), True)
    payload["auto_detect_download"] = _parse_bool(payload.get("auto_detect_download", True), True)
    payload["move_unknown_background_to_optional"] = _parse_bool(
        payload.get("move_unknown_background_to_optional")
    )
    payload["download_timeout"] = int(payload.get("download_timeout") or 300)
    payload["download_poll_interval"] = int(payload.get("download_poll_interval") or 2)
    payload["extra_remove"] = payload.get("extra_remove") or ""
    payload["extra_keep"] = payload.get("extra_keep") or ""
    payload["identity_tags"] = payload.get("identity_tags") or ""
    ok, job_id, err = PIPELINE_MANAGER.start_job(payload)
    if not ok:
        return jsonify({"ok": False, "error": err or "start failed"}), 400
    return jsonify({"ok": True, "job_id": job_id})


@app.post("/api/pipeline/pause")
def api_pipeline_pause():
    payload = request.get_json(silent=True) or {}
    job_id = payload.get("job_id", "")
    ok, msg = PIPELINE_MANAGER.pause_job(job_id)
    code = 200 if ok else 400
    return jsonify({"ok": ok, "message": msg}), code


@app.post("/api/pipeline/resume")
def api_pipeline_resume():
    payload = request.get_json(silent=True) or {}
    job_id = payload.get("job_id", "")
    ok, msg = PIPELINE_MANAGER.resume_job(job_id)
    code = 200 if ok else 400
    return jsonify({"ok": ok, "message": msg}), code


@app.post("/api/pipeline/stop")
def api_pipeline_stop():
    payload = request.get_json(silent=True) or {}
    job_id = payload.get("job_id", "")
    ok, msg = PIPELINE_MANAGER.stop_job(job_id)
    code = 200 if ok else 400
    return jsonify({"ok": ok, "message": msg}), code


@app.get("/api/pipeline/status")
def api_pipeline_status():
    job_id = request.args.get("job_id")
    ok, data, err = PIPELINE_MANAGER.get_status(job_id)
    code = 200 if ok else 404
    return jsonify({"ok": ok, "job": data, "error": err if not ok else ""}), code


@app.post("/api/pipeline/open-path")
def api_pipeline_open_path():
    payload = request.get_json(silent=True) or {}
    target = readable_path(payload.get("path", ""))
    if not target.exists():
        return jsonify({"ok": False, "error": f"Path not found: {target}"}), 400
    try:
        if os.name == "nt":
            os.startfile(str(target))  # type: ignore
        else:
            import subprocess

            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(target)])
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True})

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
