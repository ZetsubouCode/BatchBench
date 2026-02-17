import os, json, re, sys, shutil
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from flask import Flask, render_template, request, flash, jsonify, redirect, url_for, session, send_file
from dotenv import load_dotenv

from utils.io import readable_path, readable_path_or_none, windows_drives, default_browse_root
from utils.parse import parse_bool, parse_int, parse_float, parse_exts, parse_tag_list
from utils.tool_result import unpack_tool_result

from services.registry import TOOL_REGISTRY
from services import normalizer
from services import tag_editor
from services import danbooru_client
from services import blur_brush
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
SERVER_UPLOAD_IMAGE_EXTS = {ext.lower() for ext in normalizer.DEFAULT_IMAGE_EXTS}
BROWSE_STRICT_MODE = parse_bool(os.getenv("BROWSE_STRICT_MODE"), default=False)


def _parse_path_list(raw: str) -> List[Path]:
    if not raw:
        return []
    out: List[Path] = []
    for token in re.split(r"[,\n;]+", raw):
        value = token.strip()
        if not value:
            continue
        path = readable_path(value)
        try:
            out.append(path.resolve())
        except Exception:
            out.append(path)
    return out


_browse_roots = _parse_path_list(os.getenv("BROWSE_ALLOWED_ROOTS", ""))
BROWSE_ALLOWED_ROOTS = _browse_roots


def _is_allowed_path(path: Path) -> bool:
    if not BROWSE_STRICT_MODE:
        return True
    if not BROWSE_ALLOWED_ROOTS:
        return False
    try:
        target = path.resolve()
    except Exception:
        target = path
    for root in BROWSE_ALLOWED_ROOTS:
        try:
            root_resolved = root.resolve()
        except Exception:
            root_resolved = root
        if target == root_resolved or root_resolved in target.parents:
            return True
    return False

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
    return parse_exts(raw, default=normalizer.DEFAULT_IMAGE_EXTS)


def _parse_bool(val: Any, default: bool = False) -> bool:
    return parse_bool(val, default=default)


def _parse_float(val: Any) -> Optional[float]:
    return parse_float(val)


def _parse_int(val: Any, default: int) -> int:
    return parse_int(val, default=default)


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
    return parse_tag_list(raw, dedupe=True)


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
        preview_limit=_parse_int(payload.get("preview_limit"), 30),
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
    if BROWSE_STRICT_MODE:
        allowed = [str(root) for root in BROWSE_ALLOWED_ROOTS]
        return jsonify({"drives": allowed or [str(default_browse_root())]})
    if os.name == "nt":
        drives = windows_drives()
        return jsonify({"drives": drives or [str(default_browse_root())]})
    return jsonify({"drives": [str(default_browse_root())]})

@app.get("/api/list-dir")
def api_list_dir():
    path = request.args.get("path", "")
    if path:
        p = readable_path(path)
    elif BROWSE_STRICT_MODE and BROWSE_ALLOWED_ROOTS:
        p = BROWSE_ALLOWED_ROOTS[0]
    else:
        p = default_browse_root()
    if not _is_allowed_path(p):
        return jsonify({"ok": False, "error": f"Path is outside allowed roots: {p}"}), 403
    if not p.exists() or not p.is_dir():
        return jsonify({"ok": False, "error": f"Path not found: {p}"}), 404

    out = {"ok": True, "path": str(p), "parent": str(p.parent) if p != p.parent else str(p)}
    dirs = []
    try:
        for d in sorted([x for x in p.iterdir() if x.is_dir()], key=lambda x: x.name.lower()):
            if not _is_allowed_path(d):
                continue
            dirs.append(str(d))
    except Exception as exc:
        return jsonify({**out, "ok": False, "error": str(exc)}), 500
    out["dirs"] = dirs
    return jsonify(out)


@app.post("/api/danbooru/taginfo")
def api_danbooru_taginfo():
    payload = request.get_json(silent=True) or {}
    include_related = _parse_bool(payload.get("include_related", True), True)
    include_preview = _parse_bool(payload.get("include_preview", True), True)
    preview_limit = _parse_int(payload.get("preview_limit"), 8)
    preview_limit = max(1, min(20, preview_limit))
    result = danbooru_client.lookup_tag_info(
        payload.get("tag"),
        include_related=include_related,
        include_preview=include_preview,
        preview_limit=preview_limit,
    )
    error_code = result.get("error_code")
    if error_code:
        result = {k: v for k, v in result.items() if k != "error_code"}
    if result.get("ok"):
        return jsonify(result)
    code = 400 if error_code == "invalid_input" else 502
    return jsonify(result), code

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
    tag_limit = _parse_int(payload.get("tag_limit"), 200)
    tag_limit = max(10, min(2000, tag_limit))
    result = tag_editor.list_images_with_tags(
        readable_path(folder),
        exts,
        recursive=recursive,
        limit=limit,
        tag_limit=tag_limit,
    )
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


@app.get("/api/blur_brush/list-images")
def api_blur_brush_list_images():
    folder = (request.args.get("folder") or "").strip()
    recursive = _parse_bool(request.args.get("recursive"), True)
    exts = _parse_exts(request.args.get("exts"))
    if not folder:
        return jsonify({"ok": False, "error": "folder is required"}), 400
    root = readable_path(folder)
    if not root.exists() or not root.is_dir():
        return jsonify({"ok": False, "error": f"Folder not found: {root}"}), 400
    images = blur_brush.list_images(root, recursive=recursive, exts=exts)
    return jsonify({"ok": True, "folder": str(root), "total": len(images), "images": images})


@app.post("/api/blur_brush/apply")
def api_blur_brush_apply():
    payload = request.get_json(silent=True) or {}
    folder = (payload.get("folder") or "").strip()
    rel = (payload.get("rel") or payload.get("rel_image_path") or "").strip()
    mask_png_base64 = (payload.get("mask_png_base64") or "").strip()
    mode = (payload.get("mode") or "blur").strip().lower()
    strength = _parse_int(payload.get("strength"), 12)
    feather = _parse_int(payload.get("feather"), 0)
    backup = _parse_bool(payload.get("backup"), True)
    output_mode = (payload.get("output_mode") or "overwrite").strip().lower()

    if not folder or not rel:
        return jsonify({"ok": False, "error": "folder and rel are required"}), 400
    if not mask_png_base64:
        return jsonify({"ok": False, "error": "mask_png_base64 is required"}), 400
    if _bad_rel(rel):
        return jsonify({"ok": False, "error": "Invalid path"}), 400

    root = readable_path(folder)
    if not root.exists() or not root.is_dir():
        return jsonify({"ok": False, "error": f"Folder not found: {root}"}), 400
    target = _safe_child(root, rel)
    if not target or not target.exists() or not target.is_file():
        return jsonify({"ok": False, "error": "File not found"}), 404

    result = blur_brush.apply_brush_effect(
        image_path=target,
        mask_png_base64=mask_png_base64,
        mode=mode,
        strength=strength,
        feather=feather,
        backup=backup,
        output_mode=output_mode,
    )
    if not result.get("ok"):
        return jsonify({"ok": False, "error": result.get("error") or "Apply failed", "log": result.get("log") or []}), 400

    saved_path = result.get("saved_path")
    backup_path = result.get("backup_path")
    saved_rel = ""
    backup_rel = ""
    if isinstance(saved_path, Path):
        try:
            saved_rel = saved_path.relative_to(root).as_posix()
        except Exception:
            saved_rel = str(saved_path)
    if isinstance(backup_path, Path):
        try:
            backup_rel = backup_path.relative_to(root).as_posix()
        except Exception:
            backup_rel = str(backup_path)

    return jsonify(
        {
            "ok": True,
            "saved": saved_rel,
            "backup": backup_rel,
            "mode": result.get("mode"),
            "strength": result.get("strength"),
            "feather": result.get("feather"),
            "output_mode": result.get("output_mode"),
            "log": result.get("log") or [],
        }
    )


@app.post("/api/blur_brush/preview")
def api_blur_brush_preview():
    payload = request.get_json(silent=True) or {}
    folder = (payload.get("folder") or "").strip()
    rel = (payload.get("rel") or payload.get("rel_image_path") or "").strip()
    mask_png_base64 = (payload.get("mask_png_base64") or "").strip()
    mode = (payload.get("mode") or "blur").strip().lower()
    strength = _parse_int(payload.get("strength"), 12)
    feather = _parse_int(payload.get("feather"), 0)
    preview_width = _parse_int(payload.get("preview_width"), 0)
    preview_height = _parse_int(payload.get("preview_height"), 0)

    if not folder or not rel:
        return jsonify({"ok": False, "error": "folder and rel are required"}), 400
    if not mask_png_base64:
        return jsonify({"ok": False, "error": "mask_png_base64 is required"}), 400
    if _bad_rel(rel):
        return jsonify({"ok": False, "error": "Invalid path"}), 400

    root = readable_path(folder)
    if not root.exists() or not root.is_dir():
        return jsonify({"ok": False, "error": f"Folder not found: {root}"}), 400
    target = _safe_child(root, rel)
    if not target or not target.exists() or not target.is_file():
        return jsonify({"ok": False, "error": "File not found"}), 404

    preview_size = None
    if preview_width > 0 and preview_height > 0:
        preview_size = (preview_width, preview_height)

    result = blur_brush.preview_brush_effect(
        image_path=target,
        mask_png_base64=mask_png_base64,
        mode=mode,
        strength=strength,
        feather=feather,
        preview_size=preview_size,
    )
    if not result.get("ok"):
        return jsonify({"ok": False, "error": result.get("error") or "Preview failed", "log": result.get("log") or []}), 400

    preview_bytes = result.get("preview_bytes")
    if not isinstance(preview_bytes, (bytes, bytearray)) or not preview_bytes:
        return jsonify({"ok": False, "error": "Invalid preview bytes"}), 500

    preview_mime = str(result.get("preview_mime") or "image/png")
    resp = send_file(BytesIO(preview_bytes), mimetype=preview_mime, conditional=False, download_name="brush_preview.png")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.post("/api/tags/upload")
def api_tags_upload():
    folder = (request.form.get("folder") or "").strip()
    if not folder:
        return jsonify({"ok": False, "error": "folder is required"}), 400
    root = readable_path(folder)
    if not root.exists() or not root.is_dir():
        return jsonify({"ok": False, "error": f"Folder not found: {root}"}), 400

    requested_exts = _parse_exts(request.form.get("exts"))
    allowed = {e.lower() for e in requested_exts if e.lower() in SERVER_UPLOAD_IMAGE_EXTS}
    if not allowed:
        allowed = set(SERVER_UPLOAD_IMAGE_EXTS)
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

    moved_pairs = []
    try:
        # Move image + sidecars as one transaction; rollback on failure.
        move_plan = [(src, dest_img)]
        for src_path, dest_suffix in sidecars:
            move_plan.append((src_path, dst_parent / f"{dest_img.stem}{dest_suffix}"))

        for src_path, dest_path in move_plan:
            shutil.move(str(src_path), str(dest_path))
            moved_pairs.append((src_path, dest_path))
    except Exception as exc:
        rollback_errors = []
        for src_path, dest_path in reversed(moved_pairs):
            try:
                if dest_path.exists() and not src_path.exists():
                    shutil.move(str(dest_path), str(src_path))
            except Exception as rb_exc:
                rollback_errors.append(f"{dest_path.name}: {rb_exc}")
        error_msg = str(exc)
        if rollback_errors:
            error_msg = f"{error_msg} | rollback errors: {'; '.join(rollback_errors)}"
        return jsonify({"ok": False, "error": error_msg}), 500
    except BaseException:
        # Keep default exception chain for non-Exception errors.
        raise

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
    payload["copy_mode"] = (payload.get("copy_mode") or "copy")
    payload["incremental_copy"] = _parse_bool(payload.get("incremental_copy"))
    payload["clean_working"] = _parse_bool(payload.get("clean_working", True), True)
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
    since_log_id_raw = request.args.get("since_log_id")
    since_log_id: Optional[int] = None
    if since_log_id_raw not in (None, ""):
        try:
            since_log_id = int(since_log_id_raw)
        except Exception:
            since_log_id = None
    ok, data, err = PIPELINE_MANAGER.get_status(job_id, since_log_id=since_log_id)
    code = 200 if ok else 404
    return jsonify({"ok": ok, "job": data, "error": err if not ok else ""}), code


@app.post("/api/pipeline/open-path")
def api_pipeline_open_path():
    payload = request.get_json(silent=True) or {}
    target = readable_path_or_none(payload.get("path", ""))
    if not target:
        return jsonify({"ok": False, "error": "path is required"}), 400
    if not _is_allowed_path(target):
        return jsonify({"ok": False, "error": f"Path is outside allowed roots: {target}"}), 403
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
        request_ok = True
        request_error = ""
        if not handler:
            request_ok = False
            request_error = f"Unknown tool: {tool}"
            flash(request_error)
        else:
            ctx = {
                "presets": PRESET_FILES,
                "normalize_preset_root": str(NORMALIZE_PRESET_ROOT),
                "work_dir": str(WORK_DIR),
            }
            try:
                raw_result = handler(request.form, ctx)
                active_tab, log, meta = unpack_tool_result(raw_result)
                if not meta.get("ok", True):
                    request_ok = False
                    request_error = str(meta.get("error") or "Tool failed")
            except Exception as e:
                request_ok = False
                request_error = f"Error running tool '{tool}': {e}"
                flash(request_error)
            if not request_ok and request_error:
                flash(request_error)

        if is_ajax:
            code = 200 if request_ok else 400
            return jsonify({"ok": request_ok, "active_tab": active_tab, "log": log, "error": request_error}), code
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
