import os, json, sys, shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from flask import Flask, render_template, request, flash, jsonify, redirect, url_for, session, send_file
from dotenv import load_dotenv

from utils.io import readable_path, windows_drives, default_browse_root

from services.registry import TOOL_REGISTRY
from services import normalizer
from services import tag_editor
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


def _parse_int(val: Any, default: int) -> int:
    if val is None:
        return default
    try:
        return int(str(val).strip())
    except Exception:
        return default


def _safe_child(root: Path, rel: str) -> Optional[Path]:
    if not rel:
        return None
    rel_norm = str(rel).replace("\\", "/")
    rel_path = Path(rel_norm)
    if rel_path.is_absolute() or rel_path.drive:
        return None
    try:
        candidate = (root / rel_path).resolve()
        root_res = root.resolve()
    except Exception:
        return None
    if candidate == root_res or root_res in candidate.parents:
        return candidate
    return None


def _safe_child_or_root(root: Path, rel: str) -> Optional[Path]:
    if not rel or str(rel).strip() in {".", "./", ".\\"}:
        try:
            return root.resolve()
        except Exception:
            return None
    return _safe_child(root, rel)


def _bad_rel(rel: str) -> bool:
    rel_norm = str(rel).replace("\\", "/")
    parts = [p for p in Path(rel_norm).parts if p not in ("", ".", "./", ".\\")]
    return any(p == ".." for p in parts)


def _parse_tag_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        chunks = [str(x) for x in raw]
    else:
        chunks = [str(raw)]
    out: List[str] = []
    seen = set()
    for chunk in chunks:
        for line in chunk.replace("\r", "\n").split("\n"):
            for token in line.split(","):
                val = token.strip()
                if val and val not in seen:
                    out.append(val)
                    seen.add(val)
    return out


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
        pinned_tags=_parse_tag_list(payload.get("pinned_tags") or payload.get("trigger_tag") or ""),
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
        drives = windows_drives()
        return jsonify({"drives": drives or [str(default_browse_root())]})
    return jsonify({"drives": [str(default_browse_root())]})

@app.get("/api/list-dir")
def api_list_dir():
    path = request.args.get("path", "")
    p = readable_path(path) if path else default_browse_root()
    out = {"path": str(p), "parent": str(p.parent) if p != p.parent else str(p)}
    dirs = []
    try:
        for d in sorted([x for x in p.iterdir() if x.is_dir()], key=lambda x: x.name.lower()):
            dirs.append(str(d))
    except Exception:
        pass
    out["dirs"] = dirs
    return jsonify(out)

@app.post("/api/tags/scan")
def api_tags_scan():
    payload = request.get_json(silent=True) or {}
    folder = (payload.get("folder") or "").strip()
    if not folder:
        return jsonify({"ok": False, "error": "folder is required"}), 400
    exts = _parse_exts(payload.get("exts"))
    recursive = _parse_bool(payload.get("recursive"))
    result = tag_editor.scan_tags(readable_path(folder), exts, recursive=recursive)
    code = 200 if result.get("ok") else 400
    return jsonify(result), code


@app.post("/api/tags/images")
def api_tags_images():
    payload = request.get_json(silent=True) or {}
    folder = (payload.get("folder") or "").strip()
    if not folder:
        return jsonify({"ok": False, "error": "folder is required"}), 400
    exts = _parse_exts(payload.get("exts"))
    recursive = _parse_bool(payload.get("recursive"))
    limit = _parse_int(payload.get("limit"), 80)
    if limit <= 0:
        limit = 80
    limit = max(1, min(500, limit))
    result = tag_editor.list_images_with_tags(readable_path(folder), exts, recursive=recursive, limit=limit)
    code = 200 if result.get("ok") else 400
    return jsonify(result), code


@app.post("/api/tags/tag-remove")
def api_tags_tag_remove():
    payload = request.get_json(silent=True) or {}
    folder = (payload.get("folder") or "").strip()
    rel = (payload.get("rel") or "").strip()
    tag = (payload.get("tag") or "").strip()
    backup = _parse_bool(payload.get("backup"), True)
    if not folder or not rel or not tag:
        return jsonify({"ok": False, "error": "folder, rel, and tag are required"}), 400
    if _bad_rel(rel):
        return jsonify({"ok": False, "error": "Invalid path"}), 400
    root = readable_path(folder)
    if not root.exists() or not root.is_dir():
        return jsonify({"ok": False, "error": f"Folder not found: {root}"}), 400
    target = _safe_child(root, rel)
    if not target or not target.exists() or not target.is_file():
        return jsonify({"ok": False, "error": "File not found"}), 404
    txt_path = target if target.suffix.lower() == ".txt" else target.with_suffix(".txt")
    if not txt_path.exists():
        return jsonify({"ok": False, "error": "Missing .txt"}), 400
    result = tag_editor.remove_tag(txt_path, tag, backup=backup)
    if not result.get("ok"):
        return jsonify({"ok": False, "error": result.get("error") or "Remove failed"}), 400
    return jsonify(
        {
            "ok": True,
            "rel": target.relative_to(root).as_posix(),
            "tag": tag,
            "removed": bool(result.get("removed")),
            "tags": result.get("tags") or [],
        }
    )


@app.get("/api/tags/image")
def api_tags_image():
    folder = (request.args.get("folder") or "").strip()
    rel = (request.args.get("path") or "").strip()
    if not folder or not rel:
        return jsonify({"ok": False, "error": "folder and path are required"}), 400
    root = readable_path(folder)
    if not root.exists() or not root.is_dir():
        return jsonify({"ok": False, "error": f"Folder not found: {root}"}), 400
    target = _safe_child(root, rel)
    if not target or not target.exists() or not target.is_file():
        return jsonify({"ok": False, "error": "File not found"}), 404
    return send_file(target, conditional=True)


@app.post("/api/tags/upload")
def api_tags_upload():
    folder = (request.form.get("folder") or "").strip()
    if not folder:
        return jsonify({"ok": False, "error": "folder is required"}), 400
    root = readable_path(folder)
    if not root.exists() or not root.is_dir():
        return jsonify({"ok": False, "error": f"Folder not found: {root}"}), 400

    exts = _parse_exts(request.form.get("exts"))
    allowed = set([e.lower() for e in exts])
    allowed.add(".txt")

    saved = []
    skipped = []
    errors = []

    def _unique_dest(name: str) -> Path:
        base = Path(name).stem
        suffix = Path(name).suffix
        candidate = root / name
        bump = 1
        while candidate.exists():
            candidate = root / f"{base}_{bump}{suffix}"
            bump += 1
        return candidate

    for f in request.files.getlist("files"):
        filename = Path(f.filename or "").name
        if not filename:
            continue
        suffix = Path(filename).suffix.lower()
        if suffix not in allowed:
            skipped.append(filename)
            continue
        try:
            dest = _unique_dest(filename)
            f.save(str(dest))
            saved.append(dest.name)
        except Exception as exc:
            errors.append(f"{filename}: {exc}")

    return jsonify({"ok": True, "saved": saved, "skipped": skipped, "errors": errors})


@app.post("/api/tags/dirs")
def api_tags_dirs():
    payload = request.get_json(silent=True) or {}
    folder = (payload.get("folder") or "").strip()
    rel = (payload.get("rel") or "").strip()
    hide_temp = _parse_bool(payload.get("hide_temp"), True)
    if not folder:
        return jsonify({"ok": False, "error": "folder is required"}), 400
    root = readable_path(folder)
    if not root.exists() or not root.is_dir():
        return jsonify({"ok": False, "error": f"Folder not found: {root}"}), 400
    if _bad_rel(rel):
        return jsonify({"ok": False, "error": "Invalid path"}), 400
    target = _safe_child_or_root(root, rel)
    if not target or not target.exists() or not target.is_dir():
        return jsonify({"ok": False, "error": "Path not found"}), 404

    dirs = []
    try:
        for d in sorted([x for x in target.iterdir() if x.is_dir()], key=lambda x: x.name.lower()):
            if hide_temp and d.name.lower() == "_temp":
                continue
            try:
                has_children = any(p.is_dir() for p in d.iterdir())
            except Exception:
                has_children = False
            dirs.append({"name": d.name, "rel": d.relative_to(root).as_posix(), "has_children": has_children})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "root": str(root), "rel": rel, "dirs": dirs})


@app.post("/api/tags/mkdir")
def api_tags_mkdir():
    payload = request.get_json(silent=True) or {}
    folder = (payload.get("folder") or "").strip()
    rel = (payload.get("rel") or "").strip()
    if not folder or not rel:
        return jsonify({"ok": False, "error": "folder and rel are required"}), 400
    if _bad_rel(rel):
        return jsonify({"ok": False, "error": "Invalid path"}), 400
    root = readable_path(folder)
    if not root.exists() or not root.is_dir():
        return jsonify({"ok": False, "error": f"Folder not found: {root}"}), 400
    target = _safe_child(root, rel)
    if not target:
        return jsonify({"ok": False, "error": "Invalid path"}), 400
    if target.exists():
        return jsonify({"ok": False, "error": "Folder already exists"}), 400
    try:
        target.mkdir(parents=True, exist_ok=False)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "rel": target.relative_to(root).as_posix()})


@app.post("/api/tags/rmdir")
def api_tags_rmdir():
    payload = request.get_json(silent=True) or {}
    folder = (payload.get("folder") or "").strip()
    rel = (payload.get("rel") or "").strip()
    force = _parse_bool(payload.get("force"))
    if not folder or not rel:
        return jsonify({"ok": False, "error": "folder and rel are required"}), 400
    if _bad_rel(rel):
        return jsonify({"ok": False, "error": "Invalid path"}), 400
    root = readable_path(folder)
    if not root.exists() or not root.is_dir():
        return jsonify({"ok": False, "error": f"Folder not found: {root}"}), 400
    target = _safe_child(root, rel)
    if not target or not target.exists() or not target.is_dir():
        return jsonify({"ok": False, "error": "Folder not found"}), 404
    if target == root:
        return jsonify({"ok": False, "error": "Cannot delete root folder"}), 400

    try:
        if not force:
            if any(target.iterdir()):
                return jsonify({"ok": False, "error": "Folder not empty", "needs_force": True}), 400
            target.rmdir()
        else:
            shutil.rmtree(target)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True})


@app.post("/api/tags/move")
def api_tags_move():
    payload = request.get_json(silent=True) or {}
    folder = (payload.get("folder") or "").strip()
    src_rel = (payload.get("src") or "").strip()
    dst_rel = (payload.get("dst") or "").strip()
    if not folder or not src_rel:
        return jsonify({"ok": False, "error": "folder and src are required"}), 400
    if _bad_rel(src_rel) or _bad_rel(dst_rel):
        return jsonify({"ok": False, "error": "Invalid path"}), 400
    root = readable_path(folder)
    if not root.exists() or not root.is_dir():
        return jsonify({"ok": False, "error": f"Folder not found: {root}"}), 400

    src = _safe_child(root, src_rel)
    dst_parent = _safe_child_or_root(root, dst_rel)
    if not src or not src.exists() or not src.is_dir():
        return jsonify({"ok": False, "error": "Source folder not found"}), 404
    if not dst_parent or not dst_parent.exists() or not dst_parent.is_dir():
        return jsonify({"ok": False, "error": "Target folder not found"}), 404

    try:
        src_res = src.resolve()
        dst_res = dst_parent.resolve()
    except Exception:
        return jsonify({"ok": False, "error": "Path resolution failed"}), 400

    if dst_res == src_res or src_res in dst_res.parents:
        return jsonify({"ok": False, "error": "Cannot move a folder into itself"}), 400

    if dst_res == src_res.parent:
        return jsonify({"ok": True, "moved": False, "rel": src_rel})

    target = dst_parent / src.name
    if target.exists():
        bump = 1
        while True:
            candidate = dst_parent / f"{src.name}_{bump}"
            if not candidate.exists():
                target = candidate
                break
            bump += 1
    try:
        shutil.move(str(src), str(target))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "moved": True, "rel": target.relative_to(root).as_posix()})


@app.post("/api/tags/move-file")
def api_tags_move_file():
    payload = request.get_json(silent=True) or {}
    folder = (payload.get("folder") or "").strip()
    src_rel = (payload.get("src") or "").strip()
    dst_rel = (payload.get("dst") or "").strip()
    if not folder or not src_rel:
        return jsonify({"ok": False, "error": "folder and src are required"}), 400
    if _bad_rel(src_rel) or _bad_rel(dst_rel):
        return jsonify({"ok": False, "error": "Invalid path"}), 400
    root = readable_path(folder)
    if not root.exists() or not root.is_dir():
        return jsonify({"ok": False, "error": f"Folder not found: {root}"}), 400

    src = _safe_child(root, src_rel)
    dst_parent = _safe_child_or_root(root, dst_rel)
    if not src or not src.exists() or not src.is_file():
        return jsonify({"ok": False, "error": "Source file not found"}), 404
    if not dst_parent or not dst_parent.exists() or not dst_parent.is_dir():
        return jsonify({"ok": False, "error": "Target folder not found"}), 404

    try:
        src_parent_res = src.parent.resolve()
        dst_parent_res = dst_parent.resolve()
    except Exception:
        return jsonify({"ok": False, "error": "Path resolution failed"}), 400

    if dst_parent_res == src_parent_res:
        return jsonify({"ok": True, "moved": False, "rel": src_rel})

    src_txt = src.with_suffix(".txt")
    src_txt_bak = Path(str(src_txt) + ".bak")
    src_bak = src.with_suffix(".bak")
    src_img_bak = Path(str(src) + ".bak")

    sidecars = []
    warnings = []
    if src_txt.exists():
        sidecars.append((src_txt, ".txt"))
    else:
        warnings.append(f"Missing sidecar: {src_txt.name}")
    if src_txt_bak.exists():
        sidecars.append((src_txt_bak, ".txt.bak"))
    if src_bak.exists():
        sidecars.append((src_bak, ".bak"))
    if src_img_bak.exists() and src_img_bak != src_bak:
        sidecars.append((src_img_bak, f"{src.suffix}.bak"))

    stem = src.stem
    suffix = src.suffix
    bump = 1
    while True:
        candidate_stem = stem if bump == 1 else f"{stem}_{bump-1}"
        dest_img = dst_parent / f"{candidate_stem}{suffix}"
        conflicts = dest_img.exists()
        if not conflicts:
            for _, dest_suffix in sidecars:
                if (dst_parent / f"{candidate_stem}{dest_suffix}").exists():
                    conflicts = True
                    break
        if not conflicts:
            break
        bump += 1

    try:
        shutil.move(str(src), str(dest_img))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    for src_path, dest_suffix in sidecars:
        dest_path = dst_parent / f"{dest_img.stem}{dest_suffix}"
        try:
            shutil.move(str(src_path), str(dest_path))
        except Exception as exc:
            warnings.append(f"Failed moving {src_path.name}: {exc}")

    return jsonify(
        {
            "ok": True,
            "moved": True,
            "rel": dest_img.relative_to(root).as_posix(),
            "warnings": warnings,
        }
    )


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
    payload["preset_library"] = PRESET_FILES
    payload["dataset_path"] = (payload.get("dataset_path") or "").strip()
    payload["working_dir"] = (payload.get("working_dir") or "").strip()
    payload["output_dir"] = (payload.get("output_dir") or "").strip()
    payload["preset_type"] = (payload.get("preset_type") or "anime").strip()
    payload["preset_file"] = (payload.get("preset_file") or "").strip()
    payload["image_exts"] = _parse_exts(payload.get("image_exts"))
    payload["recursive"] = _parse_bool(payload.get("recursive"))
    payload["run_autotag"] = _parse_bool(payload.get("run_autotag", True), True)
    payload["move_unknown_background_to_optional"] = _parse_bool(
        payload.get("move_unknown_background_to_optional")
    )
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
            ctx = {
                "presets": PRESET_FILES,
                "normalize_preset_root": str(NORMALIZE_PRESET_ROOT),
                "work_dir": str(WORK_DIR),
            }
            try:
                active_tab, log = handler(request.form, ctx)
            except Exception as e:
                flash(f"Error running tool '{tool}': {e}")

        if is_ajax:
            return jsonify({"ok": True, "active_tab": active_tab, "log": log})
        session["last_log"] = log
        session["last_tab"] = active_tab
        return redirect(url_for("index", tab=active_tab))

    if "last_log" in session:
        log = session.pop("last_log", "")
        last_tab = session.pop("last_tab", None)
        if "tab" not in request.values and last_tab:
            active_tab = last_tab

    return render_template(
        "index.html",
        preset_names=list_presets(),
        active_tab=active_tab,
        log=log,
        presets=PRESET_FILES,
        work_dir=str(WORK_DIR)
    )


if __name__ == "__main__":
    app.run(debug=True)
