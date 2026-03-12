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
TAG_EDITOR_GLOSSARY_PATH = Path(__file__).parent / "tag_editor_glossary.json"

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


def _rel_to_root(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        try:
            return path.relative_to(root).as_posix()
        except Exception:
            return str(path)


def _project_paths(folder_raw: str) -> Dict[str, Any]:
    return tag_editor.resolve_project_paths(readable_path(folder_raw))


def _json_error(message: str, code: int = 400, **extra):
    payload = {"ok": False, "error": message, "warnings": [], "info": [], **extra}
    return jsonify(payload), code


def _resolve_project_root_from_payload(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Any]]:
    folder = (payload.get("folder") or "").strip()
    if not folder:
        return None, _json_error("folder is required", 400)
    paths = _project_paths(folder)
    root = paths["project_root"]
    if not root.exists() or not root.is_dir():
        return None, _json_error(
            f"Project root not found: {root}",
            400,
            normalized_root=str(root),
            needs_init=True,
        )
    return paths, None


def _resolve_rel_under(root: Path, rel: str) -> Optional[Path]:
    rel_path = Path(str(rel or "").replace("\\", "/").lstrip("/"))
    if not str(rel_path) or any(part == ".." for part in rel_path.parts):
        return None
    target = root / rel_path
    try:
        target_res = target.resolve()
        root_res = root.resolve()
    except Exception:
        return None
    if target_res != root_res and root_res not in target_res.parents:
        return None
    return target


def _parse_cheatsheet_content(content: str) -> Dict[str, Any]:
    raw_lines = str(content or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [line.strip() for line in raw_lines]
    trigger = ""
    sections: List[Dict[str, Any]] = []
    all_tags: List[str] = []

    def _extend_unique(items: List[str]):
        for item in items:
            if item and item not in all_tags:
                all_tags.append(item)

    idx = 0
    while idx < len(lines) and not lines[idx]:
        idx += 1
    if idx < len(lines) and ":" not in lines[idx]:
        trigger = lines[idx]
        idx += 1

    current: Optional[Dict[str, Any]] = None
    while idx < len(lines):
        line = lines[idx]
        idx += 1
        if not line:
            continue
        if ":" in line:
            left, right = line.split(":", 1)
            category = left.strip()
            if category:
                base_tags = parse_tag_list(right or "")
                current = {
                    "category": category,
                    "tags": base_tags,
                    "conditionals": [],
                }
                sections.append(current)
                _extend_unique(base_tags)
                continue
        conditional_tags = parse_tag_list(line)
        if current and conditional_tags:
            current["conditionals"].append(conditional_tags)
            _extend_unique(conditional_tags)
        elif conditional_tags:
            _extend_unique(conditional_tags)

    return {
        "trigger": trigger,
        "sections": sections,
        "tags": all_tags,
    }


def _move_file_with_sidecars(root: Path, src_rel: str, dst_rel: str) -> Tuple[Dict[str, Any], int]:
    src = _safe_child(root, src_rel)
    dst_parent = _safe_child_or_root(root, dst_rel)
    if not src or not src.exists() or not src.is_file():
        return {"ok": False, "error": "Source file not found"}, 404
    if not dst_parent or not dst_parent.exists() or not dst_parent.is_dir():
        return {"ok": False, "error": "Target folder not found"}, 404

    try:
        src_parent_res = src.parent.resolve()
        dst_parent_res = dst_parent.resolve()
    except Exception:
        return {"ok": False, "error": "Path resolution failed"}, 400

    if dst_parent_res == src_parent_res:
        return {"ok": True, "moved": False, "rel": src_rel, "warnings": []}, 200

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
        return {"ok": False, "error": error_msg}, 500
    except BaseException:
        raise

    return {
        "ok": True,
        "moved": True,
        "rel": _rel_to_root(root, dest_img),
        "warnings": warnings,
    }, 200


def _remove_empty_dirs(root: Path, keep: Optional[Path] = None) -> None:
    if not root.exists() or not root.is_dir():
        return
    try:
        keep_res = keep.resolve() if keep else None
    except Exception:
        keep_res = None
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            if keep_res and path.resolve() == keep_res:
                continue
        except Exception:
            pass
        try:
            path.rmdir()
        except OSError:
            continue


def _bad_rel(rel: str) -> bool:
    rel_norm = str(rel).replace("\\", "/")
    parts = [p for p in Path(rel_norm).parts if p not in ("", ".", "./", ".\\")]
    return any(p == ".." for p in parts)


def _parse_tag_list(raw: Any) -> List[str]:
    return parse_tag_list(raw, dedupe=True)


def _default_glossary_payload() -> Dict[str, Any]:
    return {"version": 1, "categories": {"Unsorted": []}, "updated_at": 0}


def _normalize_glossary_payload(payload: Any) -> Dict[str, Any]:
    src = payload if isinstance(payload, dict) else {}
    categories = src.get("categories") if isinstance(src.get("categories"), dict) else {}
    normalized_categories: Dict[str, List[str]] = {}
    for raw_name, raw_tags in categories.items():
        name = str(raw_name).strip() or "Unsorted"
        seen = set()
        tags: List[str] = []
        if isinstance(raw_tags, list):
            for raw_tag in raw_tags:
                tag = str(raw_tag or "").strip().lower()
                tag = re.sub(r"\s+", "_", tag).strip("_")
                if not tag or tag in seen:
                    continue
                seen.add(tag)
                tags.append(tag)
        normalized_categories[name] = tags
    if "Unsorted" not in normalized_categories:
        normalized_categories["Unsorted"] = []
    updated_at = src.get("updated_at", 0)
    try:
        updated_at = int(updated_at)
    except Exception:
        updated_at = 0
    return {"version": 1, "categories": normalized_categories, "updated_at": max(0, updated_at)}


def _load_tag_editor_glossary() -> Dict[str, Any]:
    if not TAG_EDITOR_GLOSSARY_PATH.exists():
        return _default_glossary_payload()
    try:
        payload = json.loads(TAG_EDITOR_GLOSSARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _default_glossary_payload()
    return _normalize_glossary_payload(payload)


def _save_tag_editor_glossary(payload: Any) -> Dict[str, Any]:
    normalized = _normalize_glossary_payload(payload)
    TAG_EDITOR_GLOSSARY_PATH.write_text(json.dumps(normalized, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return normalized


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


@app.post("/api/tags/project-state")
def api_tags_project_state():
    payload = request.get_json(silent=True) or {}
    folder = (payload.get("folder") or "").strip()
    if not folder:
        return _json_error("folder is required", 400)
    exts = _parse_exts(payload.get("exts"))
    result = tag_editor.inspect_project_layout(readable_path(folder), exts)
    code = 200 if result.get("ok") else 400
    result.setdefault("normalized_root", result.get("project_root", ""))
    result.setdefault("warnings", [])
    result.setdefault("info", [])
    result.setdefault("needs_init", not bool(result.get("ready")))
    return jsonify(result), code


@app.post("/api/tags/project-init")
def api_tags_project_init():
    payload = request.get_json(silent=True) or {}
    folder = (payload.get("folder") or "").strip()
    if not folder:
        return _json_error("folder is required", 400)
    exts = _parse_exts(payload.get("exts"))
    apply_changes = _parse_bool(payload.get("apply"))
    if not apply_changes:
        result = tag_editor.inspect_project_layout(readable_path(folder), exts)
        code = 200 if result.get("ok") else 400
        result.setdefault("normalized_root", result.get("project_root", ""))
        result.setdefault("warnings", [])
        result.setdefault("info", [])
        result["apply"] = False
        return jsonify(result), code
    result = tag_editor.initialize_project_layout(readable_path(folder), exts, create_prompt=True)
    code = 200 if result.get("ok") else 400
    result.setdefault("normalized_root", result.get("project_root", ""))
    result.setdefault("warnings", [])
    result.setdefault("info", [])
    result["apply"] = True
    return jsonify(result), code


@app.post("/api/tags/database-to-temp")
def api_tags_database_to_temp():
    payload = request.get_json(silent=True) or {}
    folder = (payload.get("folder") or "").strip()
    srcs = payload.get("srcs")
    if not folder:
        return _json_error("folder is required", 400)
    if not isinstance(srcs, list) or not srcs:
        return _json_error("srcs must be a non-empty list", 400)
    result = tag_editor.move_database_files_to_temp(readable_path(folder), [str(item or "") for item in srcs])
    code = 200 if result.get("ok") or result.get("moved") else 400
    result.setdefault("normalized_root", result.get("project_root", ""))
    return jsonify(result), code

@app.post("/api/tags/scan")
def api_tags_scan():
    payload = request.get_json(silent=True) or {}
    paths, error = _resolve_project_root_from_payload(payload)
    if error:
        return error
    exts = _parse_exts(payload.get("exts"))
    recursive = _parse_bool(payload.get("recursive"))
    area = (payload.get("area") or "temp").strip().lower()
    if area == "database":
        folder = paths["database_root"]
    elif area == "dataset":
        folder = paths["dataset_root"]
    else:
        folder = paths["temp_root"]
    result = tag_editor.scan_tags(folder, exts, recursive=recursive)
    result.update(
        {
            "area": area,
            "normalized_root": str(paths["project_root"]),
            "needs_init": False,
            "warnings": [],
            "info": [],
        }
    )
    code = 200 if result.get("ok") else 400
    return jsonify(result), code


@app.post("/api/tags/images")
def api_tags_images():
    payload = request.get_json(silent=True) or {}
    paths, error = _resolve_project_root_from_payload(payload)
    if error:
        return error
    exts = _parse_exts(payload.get("exts"))
    recursive = _parse_bool(payload.get("recursive"))
    area = (payload.get("area") or "database").strip().lower()
    include_temp = _parse_bool(payload.get("include_temp"), True)
    limit = _parse_int(payload.get("limit"), 80)
    if limit <= 0:
        limit = 80
    limit = max(1, min(500, limit))
    tag_limit = _parse_int(payload.get("tag_limit"), 200)
    tag_limit = max(10, min(2000, tag_limit))
    if area == "dataset":
        result = tag_editor.list_dataset_images(
            paths["project_root"],
            exts,
            recursive=recursive,
            limit=limit,
            tag_limit=tag_limit,
            include_temp=include_temp,
        )
    elif area == "temp":
        result = tag_editor.list_temp_images(paths["project_root"], exts, recursive=recursive, limit=limit, tag_limit=tag_limit)
    else:
        result = tag_editor.list_database_images(paths["project_root"], exts, recursive=recursive, limit=limit, tag_limit=tag_limit)
    result.setdefault("normalized_root", str(paths["project_root"]))
    result.setdefault("needs_init", False)
    result.setdefault("warnings", [])
    result.setdefault("info", [])
    code = 200 if result.get("ok") else 400
    return jsonify(result), code


@app.post("/api/tags/tag-remove")
def api_tags_tag_remove():
    payload = request.get_json(silent=True) or {}
    rel = (payload.get("rel") or "").strip()
    tag = (payload.get("tag") or "").strip()
    backup = _parse_bool(payload.get("backup"), True)
    area = (payload.get("area") or "temp").strip().lower()
    if not rel or not tag:
        return _json_error("folder, rel, and tag are required", 400)
    paths, error = _resolve_project_root_from_payload(payload)
    if error:
        return error
    base_root = paths["temp_root"] if area != "dataset" else paths["dataset_root"]
    target = _resolve_rel_under(base_root, rel)
    if not target or not target.exists() or not target.is_file():
        return _json_error("File not found", 404, normalized_root=str(paths["project_root"]))
    txt_path = target if target.suffix.lower() == ".txt" else target.with_suffix(".txt")
    if not txt_path.exists():
        return _json_error("Missing .txt", 400, normalized_root=str(paths["project_root"]))
    result = tag_editor.remove_tag(txt_path, tag, backup=backup)
    if not result.get("ok"):
        return _json_error(result.get("error") or "Remove failed", 400, normalized_root=str(paths["project_root"]))
    return jsonify(
        {
            "ok": True,
            "rel": _rel_to_root(base_root, target),
            "tag": tag,
            "removed": bool(result.get("removed")),
            "tags": result.get("tags") or [],
            "normalized_root": str(paths["project_root"]),
            "needs_init": False,
            "warnings": [],
            "info": [],
        }
    )


@app.post("/api/tags/tag-add")
def api_tags_tag_add():
    payload = request.get_json(silent=True) or {}
    rel = (payload.get("rel") or "").strip()
    raw_tags = payload.get("tags")
    backup = _parse_bool(payload.get("backup"), True)
    create_missing_txt = _parse_bool(payload.get("create_missing_txt"), True)
    area = (payload.get("area") or "temp").strip().lower()
    if not rel:
        return _json_error("folder and rel are required", 400)
    paths, error = _resolve_project_root_from_payload(payload)
    if error:
        return error
    base_root = paths["temp_root"] if area != "dataset" else paths["dataset_root"]
    target = _resolve_rel_under(base_root, rel)
    if not target or not target.exists() or not target.is_file():
        return _json_error("File not found", 404, normalized_root=str(paths["project_root"]))
    txt_path = target if target.suffix.lower() == ".txt" else target.with_suffix(".txt")

    tags = parse_tag_list(raw_tags) if raw_tags is not None else []
    result = tag_editor.add_tags(
        txt_path,
        tags,
        backup=backup,
        create_missing_txt=create_missing_txt,
    )
    if not result.get("ok"):
        return _json_error(result.get("error") or "Add failed", 400, normalized_root=str(paths["project_root"]))
    return jsonify(
        {
            "ok": True,
            "rel": _rel_to_root(base_root, target),
            "tags": result.get("tags") or [],
            "added": result.get("added") or [],
            "created": bool(result.get("created")),
            "changed": bool(result.get("changed")),
            "normalized_root": str(paths["project_root"]),
            "needs_init": False,
            "warnings": [],
            "info": [],
        }
    )


@app.post("/api/tags/cheatsheet")
def api_tags_cheatsheet():
    payload = request.get_json(silent=True) or {}
    rel = (payload.get("rel") or "").strip()
    if not rel:
        return _json_error("folder and rel are required", 400)
    paths, error = _resolve_project_root_from_payload(payload)
    if error:
        return error
    root = paths["project_root"]
    target = _resolve_rel_under(root, rel)
    if not target or not target.exists() or not target.is_file():
        return _json_error("Cheat sheet file not found", 404, normalized_root=str(root))
    if target.suffix.lower() != ".txt":
        return _json_error("Cheat sheet must be a .txt file", 400, normalized_root=str(root))
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = target.read_text(encoding="utf-8-sig")
        except Exception as exc:
            return _json_error(str(exc), 500, normalized_root=str(root))
    except Exception as exc:
        return _json_error(str(exc), 500, normalized_root=str(root))
    parsed = _parse_cheatsheet_content(content)
    return jsonify(
        {
            "ok": True,
            "rel": _rel_to_root(root, target),
            "content": content,
            "trigger": parsed.get("trigger") or "",
            "sections": parsed.get("sections") or [],
            "tags": parsed.get("tags") or [],
            "normalized_root": str(root),
            "needs_init": False,
            "warnings": [],
            "info": [],
        }
    )


@app.get("/api/tags/image")
def api_tags_image():
    folder = (request.args.get("folder") or "").strip()
    rel = (request.args.get("path") or "").strip()
    area = (request.args.get("area") or "database").strip().lower()
    if not folder or not rel:
        return _json_error("folder and path are required", 400)
    resolved = tag_editor.resolve_image_path(readable_path(folder), rel, area=area)
    if not resolved.get("ok"):
        return _json_error(resolved.get("error") or "Invalid path", 400, normalized_root=resolved.get("project_root", ""))
    root = resolved["root"]
    target = resolved["target"]
    if not root.exists() or not root.is_dir():
        return _json_error(f"Folder not found: {root}", 400, normalized_root=resolved.get("project_root", ""))
    if not target or not target.exists() or not target.is_file():
        return _json_error("File not found", 404, normalized_root=resolved.get("project_root", ""))
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
            saved_rel = _rel_to_root(root, saved_path)
        except Exception:
            saved_rel = str(saved_path)
    if isinstance(backup_path, Path):
        try:
            backup_rel = _rel_to_root(root, backup_path)
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


@app.get("/api/tags/glossary")
def api_tags_glossary_get():
    return jsonify({"ok": True, "glossary": _load_tag_editor_glossary()})


@app.post("/api/tags/glossary")
def api_tags_glossary_save():
    payload = request.get_json(silent=True) or {}
    try:
        saved = _save_tag_editor_glossary(payload.get("glossary"))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "glossary": saved})


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
        return _json_error("folder is required", 400)
    paths = _project_paths(folder)
    root = paths["dataset_root"]
    if not paths["project_root"].exists() or not paths["project_root"].is_dir():
        return _json_error(f"Project root not found: {paths['project_root']}", 400, normalized_root=str(paths["project_root"]), needs_init=True)
    if not root.exists() or not root.is_dir():
        return _json_error(f"Dataset folder not found: {root}", 400, normalized_root=str(paths["project_root"]), needs_init=True)

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

    return jsonify({"ok": True, "saved": saved, "skipped": skipped, "errors": errors, "normalized_root": str(paths["project_root"]), "needs_init": False})


@app.post("/api/tags/dirs")
def api_tags_dirs():
    payload = request.get_json(silent=True) or {}
    rel = (payload.get("rel") or "").strip()
    hide_temp = _parse_bool(payload.get("hide_temp"), True)
    paths, error = _resolve_project_root_from_payload(payload)
    if error:
        return error
    root = paths["dataset_root"]
    if not root.exists() or not root.is_dir():
        return _json_error(f"Dataset folder not found: {root}", 400, normalized_root=str(paths["project_root"]), needs_init=True)
    if _bad_rel(rel):
        return _json_error("Invalid path", 400, normalized_root=str(paths["project_root"]))
    target = _safe_child_or_root(root, rel)
    if not target or not target.exists() or not target.is_dir():
        return _json_error("Path not found", 404, normalized_root=str(paths["project_root"]))

    dirs = []
    try:
        for d in sorted([x for x in target.iterdir() if x.is_dir()], key=lambda x: x.name.lower()):
            if hide_temp and d.name.lower() == "_temp":
                continue
            try:
                has_children = any(p.is_dir() for p in d.iterdir())
            except Exception:
                has_children = False
            dirs.append({"name": d.name, "rel": _rel_to_root(root, d), "has_children": has_children})
    except Exception as exc:
        return _json_error(str(exc), 500, normalized_root=str(paths["project_root"]))
    return jsonify({"ok": True, "root": str(root), "rel": rel, "dirs": dirs, "normalized_root": str(paths["project_root"]), "needs_init": False, "warnings": [], "info": []})


@app.post("/api/tags/mkdir")
def api_tags_mkdir():
    payload = request.get_json(silent=True) or {}
    rel = (payload.get("rel") or "").strip()
    if not rel:
        return _json_error("folder and rel are required", 400)
    paths, error = _resolve_project_root_from_payload(payload)
    if error:
        return error
    if _bad_rel(rel):
        return _json_error("Invalid path", 400, normalized_root=str(paths["project_root"]))
    root = paths["dataset_root"]
    if not root.exists() or not root.is_dir():
        return _json_error(f"Dataset folder not found: {root}", 400, normalized_root=str(paths["project_root"]), needs_init=True)
    target = _safe_child(root, rel)
    if not target:
        return _json_error("Invalid path", 400, normalized_root=str(paths["project_root"]))
    if target.exists():
        return _json_error("Folder already exists", 400, normalized_root=str(paths["project_root"]))
    try:
        target.mkdir(parents=True, exist_ok=False)
    except Exception as exc:
        return _json_error(str(exc), 500, normalized_root=str(paths["project_root"]))
    return jsonify({"ok": True, "rel": _rel_to_root(root, target), "normalized_root": str(paths["project_root"]), "needs_init": False, "warnings": [], "info": []})


@app.post("/api/tags/rmdir")
def api_tags_rmdir():
    payload = request.get_json(silent=True) or {}
    rel = (payload.get("rel") or "").strip()
    force = _parse_bool(payload.get("force"))
    if not rel:
        return _json_error("folder and rel are required", 400)
    paths, error = _resolve_project_root_from_payload(payload)
    if error:
        return error
    if _bad_rel(rel):
        return _json_error("Invalid path", 400, normalized_root=str(paths["project_root"]))
    root = paths["dataset_root"]
    if not root.exists() or not root.is_dir():
        return _json_error(f"Dataset folder not found: {root}", 400, normalized_root=str(paths["project_root"]), needs_init=True)
    target = _safe_child(root, rel)
    if not target or not target.exists() or not target.is_dir():
        return _json_error("Folder not found", 404, normalized_root=str(paths["project_root"]))
    if target == root:
        return _json_error("Cannot delete root folder", 400, normalized_root=str(paths["project_root"]))

    try:
        if not force:
            if any(target.iterdir()):
                return _json_error("Folder not empty", 400, normalized_root=str(paths["project_root"]), needs_force=True)
            target.rmdir()
        else:
            shutil.rmtree(target)
    except Exception as exc:
        return _json_error(str(exc), 500, normalized_root=str(paths["project_root"]))
    return jsonify({"ok": True, "normalized_root": str(paths["project_root"]), "needs_init": False, "warnings": [], "info": []})


@app.post("/api/tags/move")
def api_tags_move():
    payload = request.get_json(silent=True) or {}
    src_rel = (payload.get("src") or "").strip()
    dst_rel = (payload.get("dst") or "").strip()
    if not src_rel:
        return _json_error("folder and src are required", 400)
    paths, error = _resolve_project_root_from_payload(payload)
    if error:
        return error
    if _bad_rel(src_rel) or _bad_rel(dst_rel):
        return _json_error("Invalid path", 400, normalized_root=str(paths["project_root"]))
    root = paths["dataset_root"]
    if not root.exists() or not root.is_dir():
        return _json_error(f"Dataset folder not found: {root}", 400, normalized_root=str(paths["project_root"]), needs_init=True)

    src = _safe_child(root, src_rel)
    dst_parent = _safe_child_or_root(root, dst_rel)
    if not src or not src.exists() or not src.is_dir():
        return _json_error("Source folder not found", 404, normalized_root=str(paths["project_root"]))
    if not dst_parent or not dst_parent.exists() or not dst_parent.is_dir():
        return _json_error("Target folder not found", 404, normalized_root=str(paths["project_root"]))

    try:
        src_res = src.resolve()
        dst_res = dst_parent.resolve()
    except Exception:
        return _json_error("Path resolution failed", 400, normalized_root=str(paths["project_root"]))

    if dst_res == src_res or src_res in dst_res.parents:
        return _json_error("Cannot move a folder into itself", 400, normalized_root=str(paths["project_root"]))

    if dst_res == src_res.parent:
        return jsonify({"ok": True, "moved": False, "rel": src_rel, "normalized_root": str(paths["project_root"]), "needs_init": False, "warnings": [], "info": []})

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
        return _json_error(str(exc), 500, normalized_root=str(paths["project_root"]))
    return jsonify({"ok": True, "moved": True, "rel": _rel_to_root(root, target), "normalized_root": str(paths["project_root"]), "needs_init": False, "warnings": [], "info": []})


@app.post("/api/tags/move-file")
def api_tags_move_file():
    payload = request.get_json(silent=True) or {}
    src_rel = (payload.get("src") or "").strip()
    dst_rel = (payload.get("dst") or "").strip()
    if not src_rel:
        return _json_error("folder and src are required", 400)
    paths, error = _resolve_project_root_from_payload(payload)
    if error:
        return error
    if _bad_rel(src_rel) or _bad_rel(dst_rel):
        return _json_error("Invalid path", 400, normalized_root=str(paths["project_root"]))
    root = paths["dataset_root"]
    if not root.exists() or not root.is_dir():
        return _json_error(f"Dataset folder not found: {root}", 400, normalized_root=str(paths["project_root"]), needs_init=True)
    result, code = _move_file_with_sidecars(root, src_rel, dst_rel)
    result.setdefault("normalized_root", str(paths["project_root"]))
    result.setdefault("needs_init", False)
    result.setdefault("warnings", [])
    result.setdefault("info", [])
    return jsonify(result), code

@app.post("/api/tags/move-files")
def api_tags_move_files():
    payload = request.get_json(silent=True) or {}
    dst_rel = (payload.get("dst") or "").strip()
    srcs_raw = payload.get("srcs")
    paths, error = _resolve_project_root_from_payload(payload)
    if error:
        return error
    if _bad_rel(dst_rel):
        return _json_error("Invalid path", 400, normalized_root=str(paths["project_root"]))
    if not isinstance(srcs_raw, list) or not srcs_raw:
        return _json_error("srcs must be a non-empty list", 400, normalized_root=str(paths["project_root"]))
    srcs = []
    seen = set()
    for raw in srcs_raw:
        src_rel = (str(raw or "")).strip()
        if not src_rel or src_rel in seen:
            continue
        if _bad_rel(src_rel):
            return _json_error(f"Invalid path: {src_rel}", 400, normalized_root=str(paths["project_root"]))
        seen.add(src_rel)
        srcs.append(src_rel)
    if not srcs:
        return _json_error("No valid source files provided", 400, normalized_root=str(paths["project_root"]))
    root = paths["dataset_root"]
    if not root.exists() or not root.is_dir():
        return _json_error(f"Dataset folder not found: {root}", 400, normalized_root=str(paths["project_root"]), needs_init=True)

    moved = []
    errors = []
    for src_rel in srcs:
        result, code = _move_file_with_sidecars(root, src_rel, dst_rel)
        if code == 200 and result.get("ok"):
            moved.append(
                {
                    "src": src_rel,
                    "rel": result.get("rel") or src_rel,
                    "moved": bool(result.get("moved")),
                    "warnings": result.get("warnings") or [],
                }
            )
        else:
            errors.append({"src": src_rel, "error": result.get("error") or f"HTTP {code}"})

    return jsonify(
        {
            "ok": len(errors) == 0,
            "moved": moved,
            "errors": errors,
            "normalized_root": str(paths["project_root"]),
            "needs_init": False,
            "warnings": [],
            "info": [],
        }
    )


@app.post("/api/tags/return-temp")
def api_tags_return_temp():
    payload = request.get_json(silent=True) or {}
    paths, error = _resolve_project_root_from_payload(payload)
    if error:
        return error
    root = paths["dataset_root"]
    temp_dir = paths["temp_root"]
    source_scope = str(payload.get("source") or "temp").strip().lower()
    if source_scope not in {"temp", "all"}:
        source_scope = "temp"
    if source_scope == "temp" and (not temp_dir.exists() or not temp_dir.is_dir()):
        return _json_error("_temp folder not found", 404, normalized_root=str(paths["project_root"]), needs_init=True)

    exts = _parse_exts(payload.get("exts"))
    extset = {ext.lower() for ext in exts or []}
    source_images: List[Path] = []
    image_iter = root.rglob("*") if source_scope == "all" else temp_dir.rglob("*")
    for path in image_iter:
        if not path.is_file():
            continue
        if extset and path.suffix.lower() not in extset:
            continue
        if source_scope == "all" and path.parent == root:
            # Root-level files are already in the destination folder.
            continue
        source_images.append(path)
    source_images.sort(key=lambda p: _rel_to_root(root, p).lower())

    moved = []
    errors = []
    for path in source_images:
        src_rel = _rel_to_root(root, path)
        result, code = _move_file_with_sidecars(root, src_rel, "")
        if code == 200 and result.get("ok"):
            moved.append(
                {
                    "src": src_rel,
                    "rel": result.get("rel") or src_rel,
                    "moved": bool(result.get("moved", True)),
                    "warnings": result.get("warnings") or [],
                }
            )
        else:
            errors.append({"src": src_rel, "error": result.get("error") or f"HTTP {code}"})

    if temp_dir.exists() and temp_dir.is_dir():
        _remove_empty_dirs(temp_dir, keep=temp_dir)
    return jsonify(
        {
            "ok": len(errors) == 0,
            "moved": moved,
            "errors": errors,
            "total": len(source_images),
            "source": source_scope,
            "normalized_root": str(paths["project_root"]),
            "needs_init": False,
            "warnings": [],
            "info": [],
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
    code = 200 if ok or err == "job not found" else 404
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
    active_tab = request.values.get("tab", "guide")
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
