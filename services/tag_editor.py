from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import json
import os
import re
import shutil
import zipfile

from utils.dataset import join_tags
from utils.io import readable_path
from utils.parse import parse_bool, parse_exts, parse_tag_list
from utils.text_io import read_text_best_effort
from utils.tool_result import build_tool_result

EDIT_MODES = {"insert", "delete", "replace", "dedup", "move"}
DEFAULT_PROJECT_PROMPT = """trigger_word

appearance:
hair_color, eye_color, hairstyle

accessory:
hair_ornament, necklace, earrings

outfit:
top, bottom, footwear

optional:
expression, pose, background
"""
DEFAULT_QUIZ_SEGMENTS = [
    {"id": "identity_appearance", "label": "Identity / Appearance", "order": 1},
    {"id": "outfit", "label": "Outfit", "order": 2},
    {"id": "expression", "label": "Expression", "order": 3},
    {"id": "body_composition", "label": "Body Composition / Pose", "order": 4},
    {"id": "camera_angle", "label": "POV / Camera Angle", "order": 5},
    {"id": "lighting", "label": "Lighting", "order": 6},
    {"id": "background", "label": "Background Details", "order": 7},
]
DEFAULT_TAGGING_QUIZ_SETTINGS = {
    "segments": DEFAULT_QUIZ_SEGMENTS,
    "prompt_init": {
        "mode": "quiz_segment_template",
        "overwrite_existing": False,
        "backup_before_overwrite": True,
        "append_missing_sections": False,
        "custom_template": "",
    },
}
TAGGING_QUIZ_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings" / "tagging_quiz.json"
TAGGING_SESSION_REL = Path("dataset") / "_temp" / "tagging_session.json"
TAGGING_SESSION_STORE_REL = Path("dataset") / "_temp" / "tagging_sessions"
_TXT_TAG_CACHE: "OrderedDict[str, Tuple[Tuple[int, int], List[str]]]" = OrderedDict()
_TXT_TAG_CACHE_MAX = 4096


def _sanitize_tag(t: str) -> str:
    t = (t or "").strip()
    if not t:
        return ""
    return re.sub(r"\s+", "_", t)


def _dedup_tags(tags: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for raw in tags:
        tag = _sanitize_tag(raw)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def _normalize_tags(raw_tags: List[str], dedupe: bool = False) -> List[str]:
    tags = [tag for tag in (_sanitize_tag(x) for x in raw_tags) if tag]
    return _dedup_tags(tags) if dedupe else tags


def _cache_put(key: str, value: Tuple[Tuple[int, int], List[str]]):
    _TXT_TAG_CACHE[key] = value
    _TXT_TAG_CACHE.move_to_end(key)
    while len(_TXT_TAG_CACHE) > _TXT_TAG_CACHE_MAX:
        _TXT_TAG_CACHE.popitem(last=False)


def _read_text_tags(path: Path) -> List[str]:
    try:
        text, _, _ = read_text_best_effort(path)
    except Exception:
        return []
    return _normalize_tags(parse_tag_list(text, dedupe=False), dedupe=False)


def _read_tags_cached(txt_path: Path) -> List[str]:
    key = str(txt_path)
    try:
        st = txt_path.stat()
    except Exception:
        return []
    sig = (st.st_mtime_ns, st.st_size)
    cached = _TXT_TAG_CACHE.get(key)
    if cached and cached[0] == sig:
        _TXT_TAG_CACHE.move_to_end(key)
        return list(cached[1])
    tags = _read_text_tags(txt_path)
    _cache_put(key, (sig, list(tags)))
    return tags


def _invalidate_cache(txt_path: Path):
    _TXT_TAG_CACHE.pop(str(txt_path), None)


def _normalize_exts(exts: List[str]) -> Set[str]:
    out: Set[str] = set()
    for raw in exts or []:
        val = (raw or "").strip().lower()
        if not val:
            continue
        if not val.startswith("."):
            val = "." + val
        out.add(val)
    return out


def _list_images(
    folder_path: Path,
    exts: List[str],
    recursive: bool = False,
    exclude_dir: Optional[Path] = None,
) -> List[Path]:
    if not folder_path.exists() or not folder_path.is_dir():
        return []
    extset = _normalize_exts(exts)
    if not extset:
        return []
    if recursive:
        images: List[Path] = []
        for p in folder_path.rglob("*"):
            if not p.is_file():
                continue
            if exclude_dir and exclude_dir in p.parents:
                continue
            if p.suffix.lower() in extset:
                images.append(p)
    else:
        images = [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in extset]
    return sorted(images, key=lambda p: str(p).lower())


def _next_pair_target(folder: Path, stem: str, image_suffix: str) -> Tuple[Path, Path, bool]:
    bump = 0
    while True:
        suffix = "" if bump == 0 else f"_{bump}"
        candidate_stem = f"{stem}{suffix}"
        dst_img = folder / f"{candidate_stem}{image_suffix}"
        dst_txt = folder / f"{candidate_stem}.txt"
        if not dst_img.exists() and not dst_txt.exists():
            return dst_img, dst_txt, bump > 0
        bump += 1


def _move_pair_transaction(
    src_img: Path,
    src_txt: Path,
    dst_img: Path,
    dst_txt: Path,
    require_txt: bool,
) -> Tuple[bool, str]:
    moved: List[Tuple[Path, Path]] = []
    had_txt = src_txt.exists()
    if require_txt and not had_txt:
        return False, f"Missing paired .txt for {src_img.name}"

    try:
        shutil.move(str(src_img), str(dst_img))
        moved.append((dst_img, src_img))
        if had_txt:
            shutil.move(str(src_txt), str(dst_txt))
            moved.append((dst_txt, src_txt))
        return True, ""
    except Exception as exc:
        rollback_errors: List[str] = []
        for moved_path, original_path in reversed(moved):
            try:
                if not moved_path.exists():
                    continue
                if original_path.exists():
                    rollback_errors.append(
                        f"cannot rollback {moved_path.name} because source already exists: {original_path}"
                    )
                    continue
                shutil.move(str(moved_path), str(original_path))
            except Exception as rollback_exc:
                rollback_errors.append(f"{moved_path.name}: {rollback_exc}")
        message = str(exc)
        if rollback_errors:
            message = f"{message} | rollback issues: {'; '.join(rollback_errors)}"
        return False, message


def _copy_pair_with_txt(src_img: Path, dst_img: Path, dst_txt: Path, txt_content: str) -> Tuple[bool, str]:
    copied: List[Path] = []
    try:
        shutil.copy2(src_img, dst_img)
        copied.append(dst_img)
        dst_txt.write_text(txt_content, encoding="utf-8")
        copied.append(dst_txt)
        return True, ""
    except Exception as exc:
        for path in reversed(copied):
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        return False, str(exc)


def _copy_image_only(src_img: Path, dst_img: Path) -> Tuple[bool, str]:
    try:
        shutil.copy2(src_img, dst_img)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _next_unique_file(path: Path) -> Tuple[Path, bool]:
    if not path.exists():
        return path, False
    bump = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{bump}{path.suffix}")
        if not candidate.exists():
            return candidate, True
        bump += 1


def _created_ts(path: Path) -> float:
    try:
        return float(path.stat().st_ctime)
    except Exception:
        return 0.0


def _is_temp_relative(rel: Path) -> bool:
    parts = rel.parts
    return len(parts) >= 2 and parts[0].lower() == "_temp"


def _serialize_image_item(root_folder: Path, img: Path, tag_limit: int, area: str = "dataset") -> dict:
    txt = img.with_suffix(".txt")
    tags: List[str] = []
    has_txt = False
    if txt.exists():
        has_txt = True
        tags = _read_tags_cached(txt)
    tag_count = len(tags)
    if tag_limit > 0 and len(tags) > tag_limit:
        tags = tags[:tag_limit]
    rel_path = img.relative_to(root_folder)
    rel = rel_path.as_posix()
    parent_rel = rel_path.parent.as_posix() if rel_path.parent != Path(".") else ""
    return {
        "name": img.name,
        "rel": rel,
        "parent_rel": parent_rel,
        "area": area,
        "tags": tags,
        "tag_count": tag_count,
        "tags_truncated": tag_count > len(tags),
        "has_txt": has_txt,
        "in_temp": _is_temp_relative(rel_path),
        "created_ts": _created_ts(img),
    }


def _make_serializable_path_info(paths: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "project_root": str(paths["project_root"]),
        "database_root": str(paths["database_root"]),
        "dataset_root": str(paths["dataset_root"]),
        "temp_root": str(paths["temp_root"]),
        "prompt_path": str(paths["prompt_path"]),
        "normalized_from_legacy": bool(paths["normalized_from_legacy"]),
        "normalized_input": str(paths["normalized_input"]),
    }


def _clean_messages(*items: str) -> List[str]:
    return [str(item).strip() for item in items if str(item or "").strip()]


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _settings_path() -> Path:
    return TAGGING_QUIZ_SETTINGS_PATH


def _normalize_quiz_segments(raw_segments: Any) -> List[Dict[str, Any]]:
    src = raw_segments if isinstance(raw_segments, list) else DEFAULT_QUIZ_SEGMENTS
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for idx, raw in enumerate(src):
        if not isinstance(raw, dict):
            continue
        seg_id = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("id") or "").strip().lower()).strip("_")
        label = re.sub(r"\s+", " ", str(raw.get("label") or "").strip())
        if not seg_id or seg_id in seen:
            continue
        seen.add(seg_id)
        try:
            order = int(raw.get("order", idx + 1))
        except Exception:
            order = idx + 1
        out.append({"id": seg_id, "label": label or seg_id.replace("_", " ").title(), "order": order})
    if not out:
        out = _json_clone(DEFAULT_QUIZ_SEGMENTS)
    return sorted(out, key=lambda item: (int(item.get("order") or 0), str(item.get("id") or "")))


def _segment_id_from_label(label: str) -> str:
    seg_id = re.sub(r"[^a-z0-9_]+", "_", str(label or "").strip().lower()).strip("_")
    return seg_id or "segment"


def _segment_label_from_cheatsheet_name(name: str) -> str:
    label = re.sub(r"[_\s]+", " ", str(name or "").strip())
    return label.title() or "Segment"


def default_tagging_quiz_settings() -> Dict[str, Any]:
    return _json_clone(DEFAULT_TAGGING_QUIZ_SETTINGS)


def normalize_tagging_quiz_settings(payload: Any) -> Dict[str, Any]:
    src = payload if isinstance(payload, dict) else {}
    default = default_tagging_quiz_settings()
    prompt_src = src.get("prompt_init") if isinstance(src.get("prompt_init"), dict) else {}
    prompt_default = default["prompt_init"]
    mode = str(prompt_src.get("mode") or prompt_default["mode"]).strip().lower()
    if mode not in {"default_template", "quiz_segment_template", "custom_template", "blank", "none"}:
        mode = prompt_default["mode"]
    return {
        "segments": _normalize_quiz_segments(src.get("segments")),
        "prompt_init": {
            "mode": mode,
            "overwrite_existing": bool(prompt_src.get("overwrite_existing", prompt_default["overwrite_existing"])),
            "backup_before_overwrite": bool(
                prompt_src.get("backup_before_overwrite", prompt_default["backup_before_overwrite"])
            ),
            "append_missing_sections": bool(
                prompt_src.get("append_missing_sections", prompt_default["append_missing_sections"])
            ),
            "custom_template": str(prompt_src.get("custom_template") or "")[:20000],
        },
    }


def load_tagging_quiz_settings() -> Dict[str, Any]:
    path = _settings_path()
    if not path.exists():
        return default_tagging_quiz_settings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_tagging_quiz_settings()
    return normalize_tagging_quiz_settings(data)


def save_tagging_quiz_settings(payload: Any) -> Dict[str, Any]:
    settings = normalize_tagging_quiz_settings(payload)
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, settings)
    return settings


def _prompt_section_name(label: str) -> str:
    label = re.sub(r"\s+", " ", str(label or "").strip())
    return label.lower() or "tags"


def _quiz_segment_prompt_tags(seg_id: str) -> List[str]:
    defaults = {
        "identity_appearance": ["hair_color", "eye_color", "hairstyle"],
        "outfit": ["top", "bottom", "footwear"],
        "expression": ["smile", "serious", "angry"],
        "body_composition": ["standing", "sitting", "cowboy_shot", "full_body"],
        "camera_angle": ["looking_at_viewer", "from_above", "from_below"],
        "lighting": ["soft_lighting", "backlighting", "dim_lighting"],
        "background": ["indoors", "outdoors", "classroom"],
    }
    return defaults.get(seg_id, ["tag_a", "tag_b", "tag_c"])


def _build_quiz_segment_prompt(segments: List[Dict[str, Any]]) -> str:
    blocks = ["trigger_word", ""]
    for segment in _normalize_quiz_segments(segments):
        blocks.append(f"{_prompt_section_name(segment.get('label') or segment.get('id'))}:")
        blocks.append(join_tags(_quiz_segment_prompt_tags(str(segment.get("id") or ""))))
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def _build_prompt_content(settings: Dict[str, Any]) -> str:
    prompt_init = settings.get("prompt_init") or {}
    mode = str(prompt_init.get("mode") or "quiz_segment_template").strip().lower()
    if mode == "none":
        return ""
    if mode == "blank":
        return "trigger_word\n"
    if mode == "custom_template":
        return str(prompt_init.get("custom_template") or "")
    if mode == "default_template":
        return DEFAULT_PROJECT_PROMPT
    return _build_quiz_segment_prompt(settings.get("segments") or DEFAULT_QUIZ_SEGMENTS)


def parse_cheatsheet_text(content: str) -> Dict[str, Any]:
    raw_lines = str(content or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [line.strip() for line in raw_lines]
    trigger = ""
    sections: List[Dict[str, Any]] = []
    all_tags: List[str] = []

    def _extend_unique(items: List[str]):
        for item in items:
            tag = _sanitize_tag(item)
            if tag and tag not in all_tags:
                all_tags.append(tag)

    idx = 0
    while idx < len(lines) and not lines[idx]:
        idx += 1
    if idx < len(lines) and ":" not in lines[idx]:
        trigger = lines[idx].strip().strip(",")
        idx += 1

    current: Optional[Dict[str, Any]] = None
    while idx < len(lines):
        line = lines[idx]
        idx += 1
        if not line:
            continue
        if ":" in line:
            left, right = line.split(":", 1)
            name = re.sub(r"\s+", " ", left.strip())
            if name:
                base_tags = _dedup_tags(parse_tag_list(right or "", dedupe=False))
                current = {"name": name, "category": name, "tags": base_tags, "conditionals": []}
                sections.append(current)
                _extend_unique(base_tags)
                continue
        tags = _dedup_tags(parse_tag_list(line, dedupe=False))
        if current and tags:
            current["conditionals"].append(tags)
            _extend_unique(tags)
        elif tags:
            _extend_unique(tags)
    return {"trigger": trigger, "sections": sections, "tags": all_tags}


def parse_cheatsheet_file(project_root: Path, prompt_file: str = "prompt.txt") -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    rel_path = Path(str(prompt_file or "prompt.txt").replace("\\", "/").lstrip("/"))
    if rel_path.is_absolute() or any(part == ".." for part in rel_path.parts):
        return {"ok": False, "error": "Invalid prompt file", **_make_serializable_path_info(paths)}
    target = paths["project_root"] / rel_path
    if not target.exists() or not target.is_file():
        return {
            "ok": False,
            "error": f"Cheatsheet not found: {rel_path.as_posix()}",
            "trigger": "",
            "sections": [],
            "tags": [],
            **_make_serializable_path_info(paths),
        }
    try:
        text, _, _ = read_text_best_effort(target)
    except Exception as exc:
        return {"ok": False, "error": str(exc), **_make_serializable_path_info(paths)}
    parsed = parse_cheatsheet_text(text)
    return {"ok": True, "rel": rel_path.as_posix(), **parsed, **_make_serializable_path_info(paths)}


def _existing_prompt_sections(content: str) -> Set[str]:
    sections: Set[str] = set()
    for raw in str(content or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line or ":" not in line:
            continue
        left, _ = line.split(":", 1)
        name = re.sub(r"\s+", " ", left.strip().lower())
        if name:
            sections.add(name)
    return sections


def _section_blocks_from_prompt(content: str) -> Dict[str, str]:
    blocks: Dict[str, str] = {}
    lines = str(content or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or ":" not in line:
            idx += 1
            continue
        left, _ = line.split(":", 1)
        name = re.sub(r"\s+", " ", left.strip().lower())
        start = idx
        idx += 1
        while idx < len(lines):
            next_line = lines[idx].strip()
            if next_line and ":" in next_line:
                break
            idx += 1
        if name:
            blocks[name] = "\n".join(lines[start:idx]).strip()
    return blocks


def handle_prompt_init(project_root: Path, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    prompt_path = paths["prompt_path"]
    settings = normalize_tagging_quiz_settings(settings or load_tagging_quiz_settings())
    prompt_init = settings.get("prompt_init") or {}
    mode = str(prompt_init.get("mode") or "quiz_segment_template").strip().lower()
    logs: List[str] = [f"prompt mode used: {mode}"]
    warnings: List[str] = []
    errors: List[str] = []
    action = "skipped"

    if mode == "none":
        logs.append("prompt.txt skipped: mode none")
        return {"ok": True, "action": action, "logs": logs, "warnings": warnings, "errors": errors}

    if prompt_path.exists():
        logs.append("prompt.txt skipped: existing file protected")
    else:
        try:
            desired = _build_prompt_content(settings)
            prompt_path.write_text(desired, encoding="utf-8")
            action = "created"
            logs.append("prompt.txt created")
        except Exception as exc:
            errors.append(f"prompt.txt: {exc}")

    trigger = extract_trigger_word(prompt_path)
    logs.append(f"trigger detected: {trigger}" if trigger else "trigger not detected")
    return {"ok": len(errors) == 0, "action": action, "logs": logs, "warnings": warnings, "errors": errors}


def _is_image_file(path: Path, extset: Set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in extset


def _image_name_index(images: List[Path]) -> Set[str]:
    return {img.name.lower() for img in images if img and img.is_file()}


def resolve_project_paths(raw_folder: Path) -> Dict[str, Any]:
    normalized_input = readable_path(str(raw_folder or "")).expanduser()
    project_root = normalized_input
    normalized_from_legacy = False
    if project_root.name.lower() in {"dataset", "database"} and project_root.parent != project_root:
        project_root = project_root.parent
        normalized_from_legacy = True
    return {
        "project_root": project_root,
        "database_root": project_root / "database",
        "dataset_root": project_root / "dataset",
        "temp_root": project_root / "dataset" / "_temp",
        "prompt_path": project_root / "prompt.txt",
        "normalized_from_legacy": normalized_from_legacy,
        "normalized_input": str(normalized_input),
    }


def extract_trigger_word(prompt_path: Path) -> str:
    if not prompt_path.exists() or not prompt_path.is_file():
        return ""
    try:
        text, _, _ = read_text_best_effort(prompt_path)
    except Exception:
        return ""
    lines = [line.strip() for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    non_empty = [line for line in lines if line]
    if not non_empty:
        return ""
    trigger_keys = {"trigger", "trigger_word", "trigger word", "activation", "activation_word"}
    for line in non_empty:
        if ":" not in line:
            continue
        left, right = line.split(":", 1)
        if left.strip().lower() in trigger_keys:
            return right.strip().strip(",")
    first = non_empty[0]
    if ":" not in first:
        return first.strip().strip(",")
    left, right = first.split(":", 1)
    if left.strip().lower() in trigger_keys:
        return right.strip().strip(",")
    return ""


def inspect_project_layout(project_root: Path, exts: List[str]) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    root = paths["project_root"]
    extset = _normalize_exts(exts)
    if not str(root).strip():
        return {"ok": False, "error": "Project root is required", **_make_serializable_path_info(paths)}
    if root.exists() and not root.is_dir():
        return {"ok": False, "error": f"Project root is not a folder: {root}", **_make_serializable_path_info(paths)}

    database_exists = paths["database_root"].is_dir()
    dataset_exists = paths["dataset_root"].is_dir()
    temp_exists = paths["temp_root"].is_dir()
    prompt_exists = paths["prompt_path"].exists()
    missing = []
    if not database_exists:
        missing.append("database")
    if not dataset_exists:
        missing.append("dataset")
    if not temp_exists:
        missing.append("dataset/_temp")
    if not prompt_exists:
        missing.append("prompt.txt")

    root_images: List[Path] = []
    root_non_images = 0
    if root.exists() and root.is_dir() and extset:
        for path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if _is_image_file(path, extset):
                root_images.append(path)
            elif path.is_file():
                root_non_images += 1

    database_images = _list_images(paths["database_root"], exts, recursive=True)
    dataset_images = _list_images(paths["dataset_root"], exts, recursive=True, exclude_dir=paths["temp_root"])
    database_names = _image_name_index(database_images)
    dataset_names = _image_name_index(dataset_images)
    missing_root_in_database = [img.name for img in root_images if img.name.lower() not in database_names]
    missing_root_in_dataset = [img.name for img in root_images if img.name.lower() not in dataset_names]
    dataset_missing_txt_files = [
        img.relative_to(paths["dataset_root"]).with_suffix(".txt").as_posix()
        for img in dataset_images
        if not img.with_suffix(".txt").exists()
    ]
    dataset_missing_txt = len(dataset_missing_txt_files)
    dataset_existing_txt = max(0, len(dataset_images) - dataset_missing_txt)

    if missing_root_in_database:
        missing.append(f"database(root images:{len(missing_root_in_database)})")
    if missing_root_in_dataset:
        missing.append(f"dataset(root images:{len(missing_root_in_dataset)})")
    if dataset_missing_txt:
        missing.append(f"dataset(txt pairs:{dataset_missing_txt})")

    trigger_word = extract_trigger_word(paths["prompt_path"])
    preview_examples: List[Dict[str, str]] = []
    conflict_count = 0
    missing_database_name_index = {name.lower() for name in missing_root_in_database}
    for img in root_images:
        if img.name.lower() not in missing_database_name_index:
            continue
        if paths["database_root"].exists():
            dst_img = paths["database_root"] / img.name
            if paths["database_root"].joinpath(img.name).exists():
                conflict_count += 1
            preview_examples.append({"source": img.name, "image": dst_img.relative_to(root).as_posix()})
        else:
            preview_examples.append({"source": img.name, "image": f"database/{img.name}"})

    ready = root.exists() and root.is_dir() and not missing
    warnings: List[str] = []
    if root.exists() and root.is_dir() and not extset:
        warnings.append("No valid image extensions supplied; root-level image scan skipped.")
    if root.exists() and root.is_dir() and not root_images and not ready:
        warnings.append("No root-level source images detected for initialization preview.")
    if dataset_missing_txt:
        warnings.append(f"Dataset has {dataset_missing_txt} image(s) without paired .txt.")
    return {
        "ok": True,
        **_make_serializable_path_info(paths),
        "exists": root.exists(),
        "is_dir": root.is_dir() if root.exists() else False,
        "ready": ready,
        "needs_init": not ready,
        "status": "ready" if ready else ("partial" if root.exists() else "missing"),
        "missing": missing,
        "database_exists": database_exists,
        "dataset_exists": dataset_exists,
        "temp_exists": temp_exists,
        "prompt_exists": prompt_exists,
        "root_image_count": len(root_images),
        "root_non_image_count": root_non_images,
        "database_image_count": len(database_images),
        "dataset_image_count": len(dataset_images),
        "root_images": [img.name for img in root_images[:20]],
        "missing_root_in_database": missing_root_in_database[:50],
        "missing_root_in_dataset": missing_root_in_dataset[:50],
        "missing_dataset_txt": dataset_missing_txt_files[:100],
        "trigger_word": trigger_word,
        "trigger_detected": bool(trigger_word),
        "init_preview": {
            "create_dirs": [part for part in ["database", "dataset", "dataset/_temp"] if part in missing],
            "create_prompt": not prompt_exists,
            "root_images_found": len(root_images),
            "database_images_found": len(database_images),
            "dataset_images_found": len(dataset_images),
            "copy_images": len(missing_root_in_database),
            "copy_dataset_images": len(missing_root_in_dataset),
            "generate_txt": dataset_missing_txt + len(missing_root_in_dataset),
            "existing_txt": dataset_existing_txt,
            "skip_images": 0,
            "conflict_rename": conflict_count,
            "examples": preview_examples[:5],
        },
        "warnings": warnings,
        "info": _clean_messages(
            f"Normalized legacy input to project root: {root}" if paths["normalized_from_legacy"] else "",
            f"Trigger word detected: {trigger_word}" if trigger_word else "",
            (
                f"Missing root images -> database: {len(missing_root_in_database)}, "
                f"dataset: {len(missing_root_in_dataset)}"
                if root_images
                else ""
            ),
        ),
    }


def initialize_project_layout(
    project_root: Path,
    exts: List[str],
    create_prompt: bool = True,
    tagging_quiz_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    root = paths["project_root"]
    inspect = inspect_project_layout(root, exts)
    if not inspect.get("ok"):
        return inspect
    if root.exists() and not root.is_dir():
        return {"ok": False, "error": f"Project root is not a folder: {root}", **_make_serializable_path_info(paths)}
    if not root.exists():
        return {"ok": False, "error": f"Project root not found: {root}", **_make_serializable_path_info(paths)}

    logs: List[str] = []
    errors: List[str] = []
    created_dirs: List[str] = []
    moved_database: List[Dict[str, str]] = []
    copied_dataset: List[Dict[str, str]] = []
    generated_txt: List[str] = []
    skipped: List[str] = []
    renamed: List[Dict[str, str]] = []

    for name, path in (("database", paths["database_root"]), ("dataset", paths["dataset_root"]), ("dataset/_temp", paths["temp_root"])):
        if path.exists():
            if not path.is_dir():
                errors.append(f"{name} exists but is not a folder: {path}")
            else:
                skipped.append(f"{name}: already exists")
            continue
        try:
            path.mkdir(parents=True, exist_ok=False)
            created_dirs.append(name)
            logs.append(f"[mkdir] {name}")
            logs.append(f"folder created: {name}/")
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    prompt_result = {"ok": True, "action": "skipped", "logs": ["prompt mode used: disabled"], "errors": []}
    if create_prompt:
        prompt_result = handle_prompt_init(root, tagging_quiz_settings)
        logs.extend(prompt_result.get("logs") or [])
        errors.extend(prompt_result.get("errors") or [])
        if prompt_result.get("action") == "skipped" and paths["prompt_path"].exists():
            skipped.append("prompt.txt: already exists")

    trigger_word = extract_trigger_word(paths["prompt_path"])
    extset = _normalize_exts(exts)
    root_images = [path for path in sorted(root.iterdir(), key=lambda p: p.name.lower()) if _is_image_file(path, extset)]
    txt_content = trigger_word.strip()

    dataset_images_before = _list_images(paths["dataset_root"], exts, recursive=True, exclude_dir=paths["temp_root"])
    dataset_name_index = _image_name_index(dataset_images_before)

    # Initialization first ensures root-level source images exist in dataset/.
    dataset_ready_source_keys: Set[str] = set()
    for src_img in root_images:
        src_key = src_img.name.lower()
        if src_key in dataset_name_index:
            skipped.append(f"dataset/{src_img.name}: image already exists")
            logs.append(f"[skip] dataset/{src_img.name} image already exists")
            dataset_ready_source_keys.add(src_key)
            continue
        dst_img = paths["dataset_root"] / src_img.name
        renamed_flag = False
        if dst_img.exists():
            dst_img, renamed_flag = _next_unique_file(dst_img)
        if renamed_flag:
            renamed.append({"source": src_img.name, "target": dst_img.name, "scope": "dataset"})
            logs.append(f"[rename][dataset] {src_img.name} -> {dst_img.name}")
        ok, error = _copy_image_only(src_img, dst_img)
        if not ok:
            errors.append(f"dataset/{src_img.name}: {error}")
            logs.append(f"[error] copy dataset/{src_img.name}: {error}")
            continue
        dataset_name_index.add(dst_img.name.lower())
        dataset_ready_source_keys.add(src_key)
        rel_img = dst_img.relative_to(paths["dataset_root"]).as_posix()
        copied_dataset.append({"source": src_img.name, "image": rel_img})
        logs.append(f"[copy] {src_img.name} -> dataset/{rel_img}")

    # Captions are generated for dataset images (excluding _temp).
    dataset_images = _list_images(paths["dataset_root"], exts, recursive=True, exclude_dir=paths["temp_root"])
    for img in dataset_images:
        txt_path = img.with_suffix(".txt")
        rel_txt = txt_path.relative_to(paths["dataset_root"]).as_posix()
        if txt_path.exists():
            skipped.append(f"dataset/{rel_txt}: already exists")
            logs.append(f"[skip] dataset/{rel_txt} already exists")
            continue
        try:
            txt_path.write_text(txt_content, encoding="utf-8")
            generated_txt.append(rel_txt)
            logs.append(f"[txt] dataset/{rel_txt}" + (f" (trigger: {trigger_word})" if trigger_word else " (empty)"))
        except Exception as exc:
            errors.append(f"dataset/{rel_txt}: {exc}")
            logs.append(f"[error] txt dataset/{rel_txt}: {exc}")

    # After dataset copy + txt generation, move original root images to database/.
    for src_img in root_images:
        src_key = src_img.name.lower()
        if src_key not in dataset_ready_source_keys:
            skipped.append(f"database/{src_img.name}: skipped move because dataset copy failed")
            logs.append(f"[skip] move {src_img.name} -> database (dataset copy not ready)")
            continue
        if not src_img.exists():
            continue
        dst_img = paths["database_root"] / src_img.name
        renamed_flag = False
        if dst_img.exists():
            dst_img, renamed_flag = _next_unique_file(dst_img)
        if renamed_flag:
            renamed.append({"source": src_img.name, "target": dst_img.name, "scope": "database"})
            logs.append(f"[rename][database] {src_img.name} -> {dst_img.name}")
        try:
            shutil.move(str(src_img), str(dst_img))
            moved_database.append({"source": src_img.name, "image": dst_img.name})
            logs.append(f"[move] {src_img.name} -> database/{dst_img.name}")
        except Exception as exc:
            errors.append(f"database/{src_img.name}: {exc}")
            logs.append(f"[error] move {src_img.name} -> database: {exc}")

    logs.append(f"copied images count: {len(copied_dataset) + len(moved_database)}")
    logs.append(f"created txt count: {len(generated_txt)}")
    logs.append(f"skipped existing count: {len(skipped)}")
    logs.append(f"conflict renamed count: {len(renamed)}")
    logs.append(f"error count: {len(errors)}")

    return {
        "ok": len(errors) == 0,
        **_make_serializable_path_info(paths),
        "trigger_word": trigger_word,
        "created_dirs": created_dirs,
        "moved_database": moved_database,
        "copied": moved_database,
        "copied_dataset": copied_dataset,
        "generated_txt": generated_txt,
        "skipped": skipped,
        "renamed": renamed,
        "warnings": _clean_messages(
            f"Trigger word detected: {trigger_word}" if trigger_word else "",
            "Some files were skipped or renamed during initialization." if skipped or renamed else "",
            *(prompt_result.get("warnings") or []),
        ),
        "errors": errors,
        "logs": logs,
        "summary": {
            "created_dirs": len(created_dirs),
            "copied_images": len(moved_database) + len(copied_dataset),
            "moved_database_images": len(moved_database),
            "copied_database_images": len(moved_database),
            "copied_dataset_images": len(copied_dataset),
            "generated_txt": len(generated_txt),
            "skipped": len(skipped),
            "renamed": len(renamed),
            "errors": len(errors),
            "prompt_action": prompt_result.get("action") or "skipped",
        },
    }


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(tmp), str(path))


def tagging_session_path(project_root: Path) -> Path:
    paths = resolve_project_paths(project_root)
    return paths["project_root"] / TAGGING_SESSION_REL


def tagging_session_store_dir(project_root: Path) -> Path:
    paths = resolve_project_paths(project_root)
    return paths["project_root"] / TAGGING_SESSION_STORE_REL


def _stable_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_mtime(paths: Dict[str, Any]) -> int:
    try:
        return int(paths["prompt_path"].stat().st_mtime)
    except Exception:
        return 0


def _image_list_hash(images: List[Path], root: Path) -> str:
    rels = []
    for img in images:
        try:
            rels.append(img.relative_to(root).as_posix())
        except Exception:
            rels.append(str(img))
    return _stable_hash(rels)


def _session_image_list_hash(session: Dict[str, Any]) -> str:
    fp = session.get("source_fingerprint") if isinstance(session.get("source_fingerprint"), dict) else {}
    return re.sub(r"[^a-fA-F0-9]+", "", str(fp.get("image_list_hash") or "").strip().lower())


def _session_slot_path(project_root: Path, session: Dict[str, Any]) -> Optional[Path]:
    image_hash = _session_image_list_hash(session)
    if not image_hash:
        return None
    return tagging_session_store_dir(project_root) / f"tagging_session_{image_hash[:32]}.json"


def _read_json_file(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_session_slot(project_root: Path, session: Dict[str, Any]) -> Optional[Path]:
    if not isinstance(session, dict):
        return None
    slot = _session_slot_path(project_root, session)
    if not slot:
        return None
    payload = _json_clone(session)
    payload["project_root"] = str(resolve_project_paths(project_root)["project_root"])
    payload["updated_at"] = payload.get("updated_at") or _utc_now_iso()
    atomic_write_json(slot, payload)
    return slot


def _find_session_slot_by_image_hash(project_root: Path, image_hash: str) -> Optional[Tuple[Path, Dict[str, Any]]]:
    clean_hash = re.sub(r"[^a-fA-F0-9]+", "", str(image_hash or "").strip().lower())
    if not clean_hash:
        return None
    store = tagging_session_store_dir(project_root)
    if not store.exists() or not store.is_dir():
        return None
    matches: List[Tuple[float, Path, Dict[str, Any]]] = []
    for path in store.glob("tagging_session_*.json"):
        try:
            session = _read_json_file(path)
        except Exception:
            continue
        if _session_image_list_hash(session) != clean_hash:
            continue
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0
        matches.append((mtime, path, session))
    if not matches:
        return None
    _, path, session = sorted(matches, key=lambda item: item[0], reverse=True)[0]
    return path, session


def _normalize_mapping_rows(rows: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        left: List[str] = []
        seen: Set[str] = set()
        for raw in row.get("left_sections") or []:
            name = re.sub(r"\s+", " ", str(raw or "").strip())
            if not name or name in seen:
                continue
            seen.add(name)
            left.append(name)
        right_raw = row.get("right_segment")
        right = str(right_raw).strip() if right_raw not in (None, "", "null") else None
        if not left and not right:
            out.append({"left_sections": [], "right_segment": None})
        else:
            out.append({"left_sections": left, "right_segment": right})
    return out


def _materialize_mapping_row_segments(
    settings_segments: Any,
    rows: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    segments = _normalize_quiz_segments(settings_segments)
    segment_by_id: Dict[str, Dict[str, Any]] = {str(segment.get("id") or ""): dict(segment) for segment in segments}
    mapping_rows = _normalize_mapping_rows(rows)
    known_ids = set(segment_by_id)
    generated_ids: Dict[str, str] = {}
    used_ids = set(known_ids)
    ordered_ids: List[str] = []

    def add_ordered_id(seg_id: Optional[str]) -> None:
        clean = str(seg_id or "").strip()
        if clean and clean not in ordered_ids:
            ordered_ids.append(clean)

    def unique_id(base: str) -> str:
        clean = _segment_id_from_label(base)
        candidate = clean
        suffix = 2
        while candidate in used_ids:
            candidate = f"{clean}_{suffix}"
            suffix += 1
        used_ids.add(candidate)
        return candidate

    materialized_rows: List[Dict[str, Any]] = []
    for row in mapping_rows:
        left = list(row.get("left_sections") or [])
        right = row.get("right_segment")
        if right and right in known_ids:
            materialized_rows.append({"left_sections": left, "right_segment": right})
            add_ordered_id(right)
            continue
        if left:
            first_section = left[0]
            label = _segment_label_from_cheatsheet_name(first_section)
            if right:
                generated_id = str(right)
                if generated_id not in used_ids:
                    used_ids.add(generated_id)
                elif generated_id not in generated_ids.values():
                    generated_id = unique_id(generated_id)
            else:
                generated_id = generated_ids.get(first_section)
                if not generated_id:
                    generated_id = unique_id(first_section)
                    generated_ids[first_section] = generated_id
            if generated_id not in known_ids and generated_id not in segment_by_id:
                segment_by_id[generated_id] = {"id": generated_id, "label": label, "order": len(segment_by_id) + 1}
                known_ids.add(generated_id)
            materialized_rows.append({"left_sections": left, "right_segment": generated_id})
            add_ordered_id(generated_id)
            continue
        materialized_rows.append({"left_sections": left, "right_segment": right})

    ordered_segments: List[Dict[str, Any]] = []
    seen_segment_ids: Set[str] = set()
    for seg_id in ordered_ids:
        segment = segment_by_id.get(seg_id)
        if not segment or seg_id in seen_segment_ids:
            continue
        seen_segment_ids.add(seg_id)
        ordered_segments.append({**segment, "order": len(ordered_segments) + 1})
    for segment in segments:
        seg_id = str(segment.get("id") or "")
        if not seg_id or seg_id in seen_segment_ids:
            continue
        seen_segment_ids.add(seg_id)
        ordered_segments.append({**segment, "order": len(ordered_segments) + 1})
    for seg_id, segment in segment_by_id.items():
        if not seg_id or seg_id in seen_segment_ids:
            continue
        seen_segment_ids.add(seg_id)
        ordered_segments.append({**segment, "order": len(ordered_segments) + 1})
    return _normalize_quiz_segments(ordered_segments), materialized_rows


def _section_tag_index(parsed: Dict[str, Any]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for section in parsed.get("sections") or []:
        name = re.sub(r"\s+", " ", str(section.get("name") or section.get("category") or "").strip())
        if not name:
            continue
        tags: List[str] = []
        tags.extend(section.get("tags") or [])
        for cond in section.get("conditionals") or []:
            tags.extend(cond or [])
        index[name.lower()] = _dedup_tags(tags)
    return index


def build_tagging_quiz_recommendations(
    project_root: Path,
    mapping_rows: List[Dict[str, Any]],
    prompt_file: str = "prompt.txt",
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    settings = normalize_tagging_quiz_settings(settings or load_tagging_quiz_settings())
    parsed = parse_cheatsheet_file(paths["project_root"], prompt_file)
    warnings: List[str] = []
    if not parsed.get("ok"):
        warnings.append(parsed.get("error") or "Cheatsheet not available; recommendations are empty.")
        parsed = {"sections": [], "trigger": "", "tags": []}
    index = _section_tag_index(parsed)
    segments, materialized_rows = _materialize_mapping_row_segments(settings.get("segments"), mapping_rows)
    recommendations: Dict[str, List[str]] = {segment["id"]: [] for segment in segments}
    for row in materialized_rows:
        segment_id = row.get("right_segment")
        if not segment_id:
            continue
        if segment_id not in recommendations:
            warnings.append(f"Mapped segment no longer exists: {segment_id}")
            recommendations.setdefault(segment_id, [])
        tags: List[str] = []
        for section_name in row.get("left_sections") or []:
            found = index.get(str(section_name).lower())
            if found is None:
                warnings.append(f"Mapped cheatsheet section no longer exists: {section_name}")
                continue
            tags.extend(found)
        recommendations[segment_id] = _dedup_tags((recommendations.get(segment_id) or []) + tags)
    logs = [f"Built recommendations: {len(recommendations)} segments"]
    for seg_id, tags in recommendations.items():
        if not tags:
            logs.append(f"Warning: {seg_id} segment has no connected cheatsheet section")
    return {
        "ok": True,
        "recommendations": recommendations,
        "segments": segments,
        "mapping_rows": materialized_rows,
        "trigger": parsed.get("trigger") or "",
        "warnings": warnings,
        "logs": logs,
        **_make_serializable_path_info(paths),
    }


def _image_rel_for_session(dataset_root: Path, img: Path) -> str:
    return "dataset/" + img.relative_to(dataset_root).as_posix()


def _dataset_path_from_image_rel(project_root: Path, image_rel: str) -> Optional[Path]:
    paths = resolve_project_paths(project_root)
    rel = str(image_rel or "").replace("\\", "/").lstrip("/")
    if rel.lower().startswith("dataset/"):
        rel = rel[8:]
    elif rel.lower().startswith("database/"):
        rel = rel[9:]
    rel_path = Path(rel)
    if not str(rel_path) or rel_path.is_absolute() or any(part == ".." for part in rel_path.parts):
        return None
    target = paths["dataset_root"] / rel_path
    try:
        target_res = target.resolve()
        root_res = paths["dataset_root"].resolve()
    except Exception:
        return None
    if target_res != root_res and root_res not in target_res.parents:
        return None
    return target


def _default_segment_state(default_tags: List[str]) -> Dict[str, Any]:
    return {
        "selected": _dedup_tags(default_tags),
        "manual": [],
        "removed_defaults": [],
        "skipped": False,
        "updated_at": _utc_now_iso(),
    }


def _map_existing_tags_to_segments(
    tags: List[str],
    segments: List[Dict[str, Any]],
    recommendations: Dict[str, List[str]],
    session_defaults: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    segment_states: Dict[str, Dict[str, Any]] = {}
    pools = {
        seg["id"]: set((recommendations.get(seg["id"]) or []) + (session_defaults.get(seg["id"]) or []))
        for seg in segments
    }
    unsorted: List[str] = []
    for tag in _dedup_tags(tags):
        placed = False
        for seg in segments:
            seg_id = seg["id"]
            if tag not in pools.get(seg_id, set()):
                continue
            state = segment_states.setdefault(seg_id, _default_segment_state([]))
            state["selected"] = _dedup_tags((state.get("selected") or []) + [tag])
            placed = True
            break
        if not placed:
            unsorted.append(tag)
    if unsorted:
        segment_states["__unsorted__"] = _default_segment_state([])
        segment_states["__unsorted__"]["selected"] = unsorted
    return segment_states


def _ensure_session_image_entry(session: Dict[str, Any], image_rel: str) -> Dict[str, Any]:
    images = session.setdefault("images", {})
    entry = images.setdefault(
        image_rel,
        {"status": "pending", "segments": {}, "final_tags_written": False, "missing": False},
    )
    entry.setdefault("segments", {})
    entry.setdefault("status", "pending")
    entry.setdefault("final_tags_written", False)
    return entry


def load_tagging_session(project_root: Path) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    session_path = tagging_session_path(paths["project_root"])
    warnings: List[str] = []
    exts = [".jpg", ".jpeg", ".png", ".webp"]
    images = _list_images(paths["dataset_root"], exts, recursive=True, exclude_dir=paths["temp_root"])
    image_rels = [_image_rel_for_session(paths["dataset_root"], img) for img in images]
    current_image_hash = _image_list_hash(images, paths["dataset_root"])
    session: Optional[Dict[str, Any]] = None
    restored_from = ""
    if session_path.exists():
        try:
            session = _read_json_file(session_path)
        except Exception as exc:
            return {"ok": False, "error": f"Could not read tagging session: {exc}", **_make_serializable_path_info(paths)}
        active_hash = _session_image_list_hash(session)
        if active_hash and active_hash != current_image_hash:
            _save_session_slot(paths["project_root"], session)
            found = _find_session_slot_by_image_hash(paths["project_root"], current_image_hash)
            if found:
                restored_from, session = str(found[0].name), found[1]
                atomic_write_json(session_path, session)
                warnings.append(f"Restored saved tagging session for this dataset: {restored_from}")
            else:
                warnings.append("Dataset image list changed since this session was saved; loading the active session and updating image entries.")
    else:
        found = _find_session_slot_by_image_hash(paths["project_root"], current_image_hash)
        if found:
            restored_from, session = str(found[0].name), found[1]
            atomic_write_json(session_path, session)
            warnings.append(f"Restored saved tagging session for this dataset: {restored_from}")
        else:
            return {"ok": True, "session": None, "exists": False, **_make_serializable_path_info(paths)}
    if not isinstance(session, dict):
        return {"ok": True, "session": None, "exists": False, **_make_serializable_path_info(paths)}
    fp = session.get("source_fingerprint") if isinstance(session.get("source_fingerprint"), dict) else {}
    current_prompt_mtime = _prompt_mtime(paths)
    if fp.get("prompt_txt_mtime") and int(fp.get("prompt_txt_mtime") or 0) != current_prompt_mtime:
        warnings.append("Cheatsheet changed since this session started.")

    existing_set = set(image_rels)
    for rel in image_rels:
        _ensure_session_image_entry(session, rel)
    for rel, entry in (session.get("images") or {}).items():
        if rel not in existing_set:
            if isinstance(entry, dict):
                entry["missing"] = True
            warnings.append(f"Image missing: {rel}")

    current = session.setdefault("current", {})
    current_rel = current.get("image_rel")
    if current_rel and current_rel not in existing_set:
        for idx, rel in enumerate(image_rels):
            entry = session.get("images", {}).get(rel) or {}
            if entry.get("status") != "completed":
                current["image_index"] = idx
                current["image_rel"] = rel
                current["segment_index"] = 0
                segments = session.get("quiz_segments") or []
                current["segment_id"] = segments[0]["id"] if segments else ""
                warnings.append("Current image was missing; moved to nearest unfinished image.")
                break

    session["updated_at"] = session.get("updated_at") or _utc_now_iso()
    return {
        "ok": True,
        "exists": True,
        "session": session,
        "warnings": warnings,
        "logs": ["Loaded tagging session"],
        **_make_serializable_path_info(paths),
    }


def save_tagging_session(project_root: Path, session: Dict[str, Any]) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    if not isinstance(session, dict):
        return {"ok": False, "error": "session must be an object", **_make_serializable_path_info(paths)}
    incoming_root = str(session.get("project_root") or "").strip()
    if incoming_root:
        try:
            incoming_paths = resolve_project_paths(Path(incoming_root))
            if incoming_paths["project_root"].resolve() != paths["project_root"].resolve():
                return {
                    "ok": False,
                    "error": "Session belongs to a different project root; refusing to overwrite this dataset session.",
                    **_make_serializable_path_info(paths),
                }
        except Exception:
            return {"ok": False, "error": "Invalid session project_root", **_make_serializable_path_info(paths)}
    session["project_root"] = str(paths["project_root"])
    session["updated_at"] = _utc_now_iso()
    atomic_write_json(tagging_session_path(paths["project_root"]), session)
    _save_session_slot(paths["project_root"], session)
    return {"ok": True, "session": session, "logs": ["Autosaved session"], **_make_serializable_path_info(paths)}


def start_tagging_session(
    project_root: Path,
    exts: List[str],
    mapping_rows: Optional[List[Dict[str, Any]]] = None,
    session_defaults: Optional[Dict[str, List[str]]] = None,
    recommendations: Optional[Dict[str, List[str]]] = None,
    settings: Optional[Dict[str, Any]] = None,
    replace: bool = True,
) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    settings = normalize_tagging_quiz_settings(settings or load_tagging_quiz_settings())
    segments, mapping_rows = _materialize_mapping_row_segments(settings.get("segments"), mapping_rows or [])
    session_defaults = {
        str(k): _dedup_tags(v if isinstance(v, list) else parse_tag_list(v))
        for k, v in (session_defaults or {}).items()
    }
    if recommendations is None:
        recommendations = build_tagging_quiz_recommendations(
            paths["project_root"],
            mapping_rows,
            settings=settings,
        ).get("recommendations") or {}
    images = _list_images(paths["dataset_root"], exts, recursive=True, exclude_dir=paths["temp_root"])
    image_rels = [_image_rel_for_session(paths["dataset_root"], img) for img in images]
    now = _utc_now_iso()
    session: Dict[str, Any] = {
        "version": 1,
        "project_root": str(paths["project_root"]),
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "current": {
            "image_index": 0,
            "image_rel": image_rels[0] if image_rels else "",
            "segment_index": 0,
            "segment_id": segments[0]["id"] if segments else "",
        },
        "quiz_segments": segments,
        "mapping_rows": mapping_rows,
        "recommendations": recommendations or {},
        "session_defaults": session_defaults,
        "images": {},
        "source_fingerprint": {
            "prompt_txt_mtime": _prompt_mtime(paths),
            "segment_settings_hash": _stable_hash(segments),
            "image_list_hash": _image_list_hash(images, paths["dataset_root"]),
        },
    }
    for img, image_rel in zip(images, image_rels):
        txt = img.with_suffix(".txt")
        existing = _read_tags_cached(txt) if txt.exists() else []
        segment_states = _map_existing_tags_to_segments(existing, segments, recommendations or {}, session_defaults)
        for segment in segments:
            seg_id = segment["id"]
            state = segment_states.setdefault(seg_id, _default_segment_state(session_defaults.get(seg_id, [])))
            state["selected"] = _dedup_tags((session_defaults.get(seg_id, []) or []) + (state.get("selected") or []))
        session["images"][image_rel] = {
            "status": "pending",
            "segments": segment_states,
            "final_tags_written": False,
            "missing": False,
        }
    if replace:
        existing_path = tagging_session_path(paths["project_root"])
        if existing_path.exists():
            try:
                existing_session = _read_json_file(existing_path)
                _save_session_slot(paths["project_root"], existing_session)
            except Exception:
                pass
        save_tagging_session(paths["project_root"], session)
    return {
        "ok": True,
        "session": session,
        "logs": [f"Started tagging flow: {len(images)} images, {len(segments)} segments"],
        **_make_serializable_path_info(paths),
    }


def _tags_for_segment_state(state: Dict[str, Any]) -> List[str]:
    return _dedup_tags((state.get("selected") or []) + (state.get("manual") or []))


def final_tags_from_segments(
    segments: Dict[str, Any],
    quiz_segments: List[Dict[str, Any]],
) -> List[str]:
    tags: List[str] = []
    unsorted = segments.get("__unsorted__") if isinstance(segments.get("__unsorted__"), dict) else None
    if unsorted:
        tags.extend(_tags_for_segment_state(unsorted))
    for segment in _normalize_quiz_segments(quiz_segments):
        state = segments.get(segment["id"])
        if isinstance(state, dict):
            tags.extend(_tags_for_segment_state(state))
    return _dedup_tags(tags)


def save_tagging_quiz_image(
    project_root: Path,
    image_rel: str,
    segments: Dict[str, Any],
    session_payload: Optional[Dict[str, Any]] = None,
    backup: bool = True,
) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    img = _dataset_path_from_image_rel(paths["project_root"], image_rel)
    if not img:
        return {"ok": False, "error": "Invalid image path", **_make_serializable_path_info(paths)}
    if not img.exists() or not img.is_file():
        return {"ok": False, "error": f"Image not found: {image_rel}", **_make_serializable_path_info(paths)}
    session = session_payload
    if not isinstance(session, dict):
        loaded = load_tagging_session(paths["project_root"])
        session = loaded.get("session") if loaded.get("ok") else None
    if not isinstance(session, dict):
        return {"ok": False, "error": "No tagging session loaded", **_make_serializable_path_info(paths)}
    quiz_segments = session.get("quiz_segments") or load_tagging_quiz_settings().get("segments") or []
    final_tags = final_tags_from_segments(segments or {}, quiz_segments)
    txt_path = img.with_suffix(".txt")
    old_text = ""
    had_txt = txt_path.exists()
    if had_txt:
        try:
            old_text, _, _ = read_text_best_effort(txt_path)
        except Exception:
            old_text = ""
    if backup and had_txt:
        try:
            txt_path.with_suffix(txt_path.suffix + ".bak").write_text(old_text, encoding="utf-8")
        except Exception as exc:
            return {"ok": False, "error": f"Could not back up txt: {exc}", **_make_serializable_path_info(paths)}
    try:
        txt_path.write_text(join_tags(final_tags), encoding="utf-8")
        _invalidate_cache(txt_path)
    except Exception as exc:
        return {"ok": False, "error": f"Could not write txt: {exc}", **_make_serializable_path_info(paths)}

    image_rel_norm = image_rel.replace("\\", "/").lstrip("/")
    if not image_rel_norm.lower().startswith("dataset/"):
        try:
            image_rel_norm = "dataset/" + img.relative_to(paths["dataset_root"]).as_posix()
        except Exception:
            pass
    entry = _ensure_session_image_entry(session, image_rel_norm)
    entry["segments"] = segments or {}
    entry["status"] = "completed"
    entry["final_tags_written"] = True
    entry["updated_at"] = _utc_now_iso()

    image_rels = list(session.get("images") or {})
    completed = 0
    for rel, item in (session.get("images") or {}).items():
        if isinstance(item, dict) and item.get("status") == "completed":
            completed += 1
    if image_rels and completed >= len([rel for rel, item in session.get("images", {}).items() if not item.get("missing")]):
        session["status"] = "completed"
    else:
        try:
            idx = image_rels.index(image_rel_norm)
        except ValueError:
            idx = -1
        next_idx = idx + 1
        while next_idx < len(image_rels):
            next_entry = session.get("images", {}).get(image_rels[next_idx]) or {}
            if next_entry.get("status") != "completed" and not next_entry.get("missing"):
                break
            next_idx += 1
        if next_idx < len(image_rels):
            first_seg = (session.get("quiz_segments") or [{}])[0].get("id") or ""
            session["current"] = {
                "image_index": next_idx,
                "image_rel": image_rels[next_idx],
                "segment_index": 0,
                "segment_id": first_seg,
            }

    save_tagging_session(paths["project_root"], session)
    logs = [f"Saved image {img.name}: {len(final_tags)} tags written"]
    if session.get("status") == "completed":
        logs.append(f"Session completed: {completed} / {len(image_rels)} images")
    return {
        "ok": True,
        "tags": final_tags,
        "txt_rel": txt_path.relative_to(paths["dataset_root"]).as_posix(),
        "session": session,
        "logs": logs,
        **_make_serializable_path_info(paths),
    }


def delete_tagging_session(project_root: Path) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    path = tagging_session_path(paths["project_root"])
    session: Optional[Dict[str, Any]] = None
    if path.exists():
        try:
            session = _read_json_file(path)
        except Exception:
            session = None
    if path.exists():
        path.unlink()
    slot = _session_slot_path(paths["project_root"], session or {}) if session else None
    if slot and slot.exists():
        slot.unlink()
    return {"ok": True, "deleted": True, "logs": ["Deleted tagging session"], **_make_serializable_path_info(paths)}


def archive_tagging_session(project_root: Path) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    path = tagging_session_path(paths["project_root"])
    if not path.exists():
        return {"ok": False, "error": "No tagging session to archive", **_make_serializable_path_info(paths)}
    try:
        session = _read_json_file(path)
    except Exception:
        session = {}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = path.with_name(f"tagging_session_{stamp}.json")
    target, _ = _next_unique_file(target)
    shutil.move(str(path), str(target))
    slot = _session_slot_path(paths["project_root"], session) if session else None
    if slot and slot.exists():
        slot.unlink()
    return {
        "ok": True,
        "archive": target.name,
        "logs": [f"Archived tagging session: {target.name}"],
        **_make_serializable_path_info(paths),
    }


def export_dataset_zip(project_root: Path) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    dataset_root = paths["dataset_root"]
    if not dataset_root.exists() or not dataset_root.is_dir():
        return {"ok": False, "error": f"Dataset folder not found: {dataset_root}", **_make_serializable_path_info(paths)}

    parent_dir = dataset_root.parent
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    zip_path, renamed = _next_unique_file(parent_dir / f"{dataset_root.name}__{stamp}.zip")
    zipped = 0
    excluded_temp = 0
    logs: List[str] = [
        f"[zip] source dataset: {dataset_root}",
        f"[zip] output zip: {zip_path}",
        "[zip] exclude rule: dataset/_temp/**",
    ]

    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(dataset_root.rglob("*"), key=lambda p: str(p).lower()):
                if not path.is_file():
                    continue
                rel = path.relative_to(dataset_root)
                rel_parts = [part.lower() for part in rel.parts]
                if rel_parts and rel_parts[0] == "_temp":
                    excluded_temp += 1
                    continue
                zf.write(path, rel.as_posix())
                zipped += 1
    except Exception as exc:
        try:
            if zip_path.exists():
                zip_path.unlink()
        except Exception:
            pass
        return {"ok": False, "error": f"Dataset zip failed: {exc}", **_make_serializable_path_info(paths)}

    if renamed:
        logs.append(f"[zip] output renamed to avoid collision: {zip_path.name}")
    logs.append(f"[zip] files added: {zipped}")
    logs.append(f"[zip] files excluded from _temp: {excluded_temp}")
    return {
        "ok": True,
        **_make_serializable_path_info(paths),
        "zip_path": str(zip_path),
        "zip_name": zip_path.name,
        "files_zipped": zipped,
        "excluded_temp_files": excluded_temp,
        "logs": logs,
    }


def list_database_images(project_root: Path, exts: List[str], recursive: bool, limit: int, tag_limit: int) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    result = list_images_with_tags(paths["database_root"], exts, recursive=recursive, limit=limit, tag_limit=tag_limit)
    if result.get("ok"):
        result.update({"area": "database", **_make_serializable_path_info(paths)})
    return result


def list_dataset_images(
    project_root: Path,
    exts: List[str],
    recursive: bool,
    limit: int,
    tag_limit: int,
    include_temp: bool = True,
) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    folder = paths["dataset_root"]
    if not folder.exists() or not folder.is_dir():
        return {"ok": False, "error": f"Folder not found: {folder}", **_make_serializable_path_info(paths)}
    images = _list_images(folder, exts, recursive=recursive, exclude_dir=None if include_temp else paths["temp_root"])
    total = len(images)
    if limit and limit > 0:
        images = images[:limit]
    items = [_serialize_image_item(folder, img, tag_limit, area="dataset") for img in images]
    return {"ok": True, "folder": str(folder), "total": total, "images": items, "area": "dataset", **_make_serializable_path_info(paths)}


def list_temp_images(project_root: Path, exts: List[str], recursive: bool, limit: int, tag_limit: int) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    folder = paths["temp_root"]
    if not folder.exists() or not folder.is_dir():
        return {"ok": False, "error": f"Folder not found: {folder}", **_make_serializable_path_info(paths)}
    images = _list_images(folder, exts, recursive=recursive, exclude_dir=None)
    total = len(images)
    if limit and limit > 0:
        images = images[:limit]
    items = [_serialize_image_item(folder, img, tag_limit, area="temp") for img in images]
    return {"ok": True, "folder": str(folder), "total": total, "images": items, "area": "temp", **_make_serializable_path_info(paths)}


def move_database_files_to_temp(project_root: Path, srcs: List[str]) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    database_root = paths["database_root"]
    temp_root = paths["temp_root"]
    if not database_root.exists() or not database_root.is_dir():
        return {"ok": False, "error": f"Database folder not found: {database_root}", **_make_serializable_path_info(paths)}
    temp_root.mkdir(parents=True, exist_ok=True)
    moved = []
    warnings = []
    errors = []
    seen: Set[str] = set()

    for raw_rel in srcs or []:
        rel = str(raw_rel or "").replace("\\", "/").strip().lstrip("/")
        if not rel or rel in seen:
            continue
        seen.add(rel)
        src_img = database_root / Path(rel)
        try:
            src_img_res = src_img.resolve()
            db_res = database_root.resolve()
        except Exception:
            errors.append({"src": rel, "error": "Path resolution failed"})
            continue
        if db_res not in src_img_res.parents and src_img_res != db_res:
            errors.append({"src": rel, "error": "Invalid relative path"})
            continue
        if not src_img.exists() or not src_img.is_file():
            errors.append({"src": rel, "error": "Source image not found"})
            continue
        src_txt = src_img.with_suffix(".txt")
        rel_parent = Path(rel).parent
        dst_dir = temp_root if rel_parent == Path(".") else temp_root / rel_parent
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            errors.append({"src": rel, "error": str(exc)})
            continue
        dst_img, dst_txt, renamed_flag = _next_pair_target(dst_dir, src_img.stem, src_img.suffix.lower())
        ok, error = _move_pair_transaction(src_img, src_txt, dst_img, dst_txt, require_txt=False)
        if not ok:
            errors.append({"src": rel, "error": error})
            continue
        item = {"src": rel, "rel": dst_img.relative_to(paths["dataset_root"]).as_posix(), "moved": True, "warnings": []}
        if renamed_flag:
            item["warnings"].append(f"Renamed on move: {src_img.name} -> {dst_img.name}")
        if not dst_txt.exists():
            item["warnings"].append(f"Missing sidecar: {src_img.stem}.txt")
        if item["warnings"]:
            warnings.extend(item["warnings"])
        moved.append(item)
    return {"ok": len(errors) == 0, **_make_serializable_path_info(paths), "moved": moved, "warnings": warnings, "errors": errors}


def resolve_image_path(project_root: Path, rel: str, area: str = "database") -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    area_key = (area or "database").strip().lower()
    if area_key == "database":
        root = paths["database_root"]
    elif area_key == "dataset":
        root = paths["dataset_root"]
    elif area_key == "temp":
        root = paths["temp_root"]
    else:
        return {"ok": False, "error": f"Unknown area: {area}", **_make_serializable_path_info(paths)}
    rel_path = Path(str(rel or "").replace("\\", "/").lstrip("/"))
    if not str(rel_path) or any(part == ".." for part in rel_path.parts):
        return {"ok": False, "error": "Invalid path", **_make_serializable_path_info(paths)}
    target = root / rel_path
    try:
        target_res = target.resolve()
        root_res = root.resolve()
    except Exception:
        return {"ok": False, "error": "Path resolution failed", **_make_serializable_path_info(paths)}
    if target_res != root_res and root_res not in target_res.parents:
        return {"ok": False, "error": "Invalid path", **_make_serializable_path_info(paths)}
    return {"ok": True, "root": root, "target": target, "area": area_key, **_make_serializable_path_info(paths)}


def scan_tags(folder: Path, exts: List[str], recursive: bool = False) -> dict:
    if not folder or not folder.exists() or not folder.is_dir():
        return {"ok": False, "error": f"Folder not found: {folder}"}
    images = _list_images(folder, exts, recursive=recursive)
    counts: Counter = Counter()
    tag_files = 0
    for img in images:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            continue
        tags = _read_tags_cached(txt)
        tag_files += 1
        if tags:
            counts.update(set(tags))
    items = [{"tag": tag, "count": int(cnt)} for tag, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    return {"ok": True, "folder": str(folder), "total_images": len(images), "tag_files": tag_files, "tags": items}


def list_images_with_tags(folder: Path, exts: List[str], recursive: bool = False, limit: int = 80, tag_limit: int = 200) -> dict:
    if not folder or not folder.exists() or not folder.is_dir():
        return {"ok": False, "error": f"Folder not found: {folder}"}
    images = _list_images(folder, exts, recursive=recursive, exclude_dir=None)
    total = len(images)
    if limit and limit > 0:
        images = images[:limit]
    items = [_serialize_image_item(folder, img, tag_limit) for img in images]
    return {"ok": True, "folder": str(folder), "total": total, "images": items}


def add_tags(txt_path: Path, raw_tags: List[str], backup: bool = True, create_missing_txt: bool = False) -> dict:
    tags_to_add = _dedup_tags(raw_tags or [])
    if not tags_to_add:
        return {"ok": False, "error": "No tags provided"}
    had_txt = txt_path.exists()
    src = ""
    current_tags: List[str] = []
    if had_txt:
        try:
            src, _, _ = read_text_best_effort(txt_path)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        current_tags = _normalize_tags(parse_tag_list(src, dedupe=False), dedupe=False)
    elif not create_missing_txt:
        return {"ok": False, "error": "Missing .txt"}
    next_tags = list(current_tags)
    added: List[str] = []
    for tag in tags_to_add:
        if tag in next_tags:
            continue
        next_tags.append(tag)
        added.append(tag)
    if not had_txt and not next_tags:
        return {"ok": False, "error": "No tags to write"}
    content = join_tags(next_tags)
    changed = (not had_txt) or (content.strip() != src.strip())
    if changed:
        if backup and had_txt:
            try:
                txt_path.with_suffix(txt_path.suffix + ".bak").write_text(src, encoding="utf-8")
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
        try:
            txt_path.write_text(content, encoding="utf-8")
            _invalidate_cache(txt_path)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
    return {"ok": True, "created": not had_txt, "changed": changed, "added": added, "tags": next_tags}


def remove_tag(txt_path: Path, tag: str, backup: bool = True) -> dict:
    if not txt_path or not txt_path.exists() or not txt_path.is_file():
        return {"ok": False, "error": "Missing .txt"}
    try:
        src, _, _ = read_text_best_effort(txt_path)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    tag = _sanitize_tag(tag)
    taglist = _normalize_tags(parse_tag_list(src, dedupe=False), dedupe=False)
    newtags = [t for t in taglist if t != tag]
    removed = len(newtags) != len(taglist)
    if not removed:
        return {"ok": True, "removed": False, "tags": taglist}
    if backup:
        try:
            txt_path.with_suffix(txt_path.suffix + ".bak").write_text(src, encoding="utf-8")
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
    try:
        txt_path.write_text(join_tags(newtags), encoding="utf-8")
        _invalidate_cache(txt_path)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "removed": True, "tags": newtags}


def handle(form, ctx):
    active_tab = "tags"
    raw_folder_text = (form.get("folder", "") or "").strip()
    mode = (form.get("mode", "insert") or "insert").strip().lower()
    tags_field = form.get("tags", "")
    exts = parse_exts(form.get("exts"), default=[".jpg", ".jpeg", ".png", ".webp"])
    backup = parse_bool(form.get("backup"), default=False)
    create_missing_txt = parse_bool(form.get("create_missing_txt"), default=False)
    lines: List[str] = []

    def _done(ok: bool, error: str = ""):
        return build_tool_result(active_tab, lines, ok=ok, error=error)

    if not raw_folder_text:
        lines.append("Project root not provided.")
        return _done(False, "Project root not provided.")
    if mode not in EDIT_MODES and mode != "undo":
        lines.append(f"Unknown mode: {mode}")
        return _done(False, f"Unknown mode: {mode}")

    raw_folder = readable_path(raw_folder_text)
    paths = resolve_project_paths(raw_folder)
    dataset_root = paths["dataset_root"]
    temp_folder = paths["temp_root"]
    compatibility_mode = False
    if (not dataset_root.exists() or not dataset_root.is_dir()) and raw_folder.exists() and raw_folder.is_dir():
        direct_dataset = None
        direct_temp = None
        if raw_folder.name.lower() == "_temp" and raw_folder.parent.exists() and raw_folder.parent.is_dir():
            direct_dataset = raw_folder.parent
            direct_temp = raw_folder
        elif (raw_folder / "_temp").exists() and (raw_folder / "_temp").is_dir():
            direct_dataset = raw_folder
            direct_temp = raw_folder / "_temp"
        if direct_dataset and direct_temp:
            dataset_root = direct_dataset
            temp_folder = direct_temp
            compatibility_mode = True
            lines.append(f"Compatibility mode: treating {raw_folder} as dataset root.")
    if not dataset_root.exists() or not dataset_root.is_dir():
        lines.append(f"Dataset folder not found: {dataset_root}")
        return _done(False, f"Dataset folder not found: {dataset_root}")
    if not temp_folder.exists() or not temp_folder.is_dir():
        if mode == "move":
            try:
                temp_folder.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                lines.append(f"Could not create temp folder: {temp_folder}: {exc}")
                return _done(False, f"Could not create temp folder: {temp_folder}")
        else:
            lines.append(f"Temp folder not found: {temp_folder}")
            return _done(False, f"Temp folder not found: {temp_folder}")
    if (not compatibility_mode) and (not paths.get("normalized_from_legacy")):
        project_state = inspect_project_layout(paths["project_root"], exts)
        if not project_state.get("ok"):
            error = project_state.get("error") or "Project setup validation failed."
            lines.append(error)
            return _done(False, error)
        if not project_state.get("ready"):
            missing_items = project_state.get("missing") or []
            missing_msg = ", ".join(missing_items) if missing_items else "project setup incomplete"
            lines.append(f"Project initialization required: {missing_msg}")
            return _done(False, "Project initialization required. Use Initialize Project first.")

    def _parse_tag_input(raw: str) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for line in (raw or "").replace("\r", "\n").split("\n"):
            for token in line.split(","):
                tag = _sanitize_tag(token)
                if not tag or tag in seen:
                    continue
                out.append(tag)
                seen.add(tag)
        return out

    def _parse_replace_mapping(raw: str) -> dict:
        mapping = {}
        parts = [p.strip() for p in (raw or "").split(";") if p.strip()]
        for p in parts:
            if "->" not in p:
                continue
            old, new = [x.strip() for x in p.split("->", 1)]
            old = _sanitize_tag(old)
            new = _sanitize_tag(new)
            if old:
                mapping[old] = new
        return mapping

    if mode == "move":
        move_tags = set(_parse_tag_input(tags_field))
        images = _list_images(dataset_root, exts, recursive=True, exclude_dir=temp_folder)
        if not images:
            lines.append(f"No image files with {exts} found in {dataset_root}.")
            return _done(False, f"No image files found in {dataset_root}.")
        moved = 0
        errors = 0
        for img in images:
            txt = img.with_suffix(".txt")
            if move_tags:
                if not txt.exists():
                    continue
                current_tags = set(_read_tags_cached(txt))
                if not move_tags.intersection(current_tags):
                    continue
            rel_parent = img.parent.relative_to(dataset_root)
            dst_parent = temp_folder if rel_parent == Path(".") else temp_folder / rel_parent
            try:
                dst_parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                lines.append(f"[ERROR] creating {dst_parent}: {exc}")
                errors += 1
                continue
            dst_img, dst_txt, renamed = _next_pair_target(dst_parent, img.stem, img.suffix.lower())
            ok, error = _move_pair_transaction(img, txt, dst_img, dst_txt, require_txt=False)
            if not ok:
                lines.append(f"[ERROR] moving {img.name}: {error}")
                errors += 1
                continue
            moved += 1
            rel_out = dst_img.relative_to(dataset_root)
            lines.append(f"Moved: {rel_out}{' (renamed)' if renamed else ''}")
        lines.append(f"Done. {moved} file(s) moved into {temp_folder}. Errors: {errors}.")
        return _done(errors == 0, "" if errors == 0 else f"{errors} move operation(s) failed.")

    if mode == "undo":
        images_in_temp = _list_images(temp_folder, exts, recursive=True)
        if not images_in_temp:
            lines.append(f"No image files with {exts} found in _temp.")
            return _done(False, "No image files found in _temp.")
        restored = 0
        errors = 0
        for img in sorted(images_in_temp, key=lambda p: p.name.lower()):
            txt = img.with_suffix(".txt")
            had_txt = txt.exists()
            rel_parent = img.parent.relative_to(temp_folder)
            dest_parent = dataset_root / rel_parent
            dest_parent.mkdir(parents=True, exist_ok=True)
            dest_img, dest_txt, _ = _next_pair_target(dest_parent, img.stem, img.suffix.lower())
            ok, error = _move_pair_transaction(img, txt, dest_img, dest_txt, require_txt=False)
            if not ok:
                lines.append(f"[ERROR] restoring {img.name}: {error}")
                errors += 1
                continue
            restored += 1
            lines.append(f"Restored: {dest_img.relative_to(dataset_root)}{' (+ .txt)' if had_txt else ''}")
        lines.append(f"Done. {restored} file(s) restored from {temp_folder} to {dataset_root}. Errors: {errors}.")
        return _done(errors == 0, "" if errors == 0 else f"{errors} restore operation(s) failed.")

    add: List[str] = []
    deltags: Set[str] = set()
    mapping = {}
    if mode == "insert":
        add = _parse_tag_input(tags_field)
    elif mode == "delete":
        deltags = set(_parse_tag_input(tags_field))
    elif mode == "replace":
        mapping = _parse_replace_mapping(tags_field)

    if mode == "insert" and not add:
        lines.append("Insert skipped: no tags provided.")
        return _done(False, "Insert skipped: no tags provided.")
    if mode == "delete" and not deltags:
        lines.append("Delete skipped: no tags provided.")
        return _done(False, "Delete skipped: no tags provided.")
    if mode == "replace" and not mapping:
        lines.append("Replace skipped: no mappings provided.")
        return _done(False, "Replace skipped: no mappings provided.")

    images = _list_images(temp_folder, exts, recursive=True, exclude_dir=None)
    if not images:
        lines.append(f"No image files with {exts} found in _temp.")
        return _done(False, "No image files found in _temp.")

    processed = 0
    errors = 0
    missing_txt_skipped = 0
    missing_txt_created = 0
    for img in images:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            if mode == "insert" and create_missing_txt:
                try:
                    txt.write_text(join_tags(add), encoding="utf-8")
                    _invalidate_cache(txt)
                    lines.append(f"{img.name}: created missing .txt with insert tags")
                    processed += 1
                    missing_txt_created += 1
                except Exception as exc:
                    lines.append(f"[ERROR] creating {txt.name}: {exc}")
                    errors += 1
            else:
                missing_txt_skipped += 1
            continue
        try:
            src, _, _ = read_text_best_effort(txt)
        except Exception as exc:
            lines.append(f"[ERROR] reading {txt.name}: {exc}")
            errors += 1
            continue
        taglist = _normalize_tags(parse_tag_list(src, dedupe=False), dedupe=False)
        newtags = list(taglist)
        action_desc = ""
        if mode == "insert":
            for t in add:
                if t not in newtags:
                    newtags.append(t)
            action_desc = f"insert -> {add}"
        elif mode == "delete":
            newtags = [t for t in taglist if t not in deltags]
            action_desc = f"delete -> {sorted(deltags)}"
        elif mode == "replace":
            newtags = [mapping.get(t, t) for t in taglist]
            action_desc = f"replace -> {mapping}"
        elif mode == "dedup":
            newtags = _dedup_tags(taglist)
            action_desc = f"dedup -> {len(taglist) - len(newtags)} removed"
        content = join_tags(newtags)
        if content.strip() == src.strip():
            continue
        try:
            if backup:
                txt.with_suffix(txt.suffix + ".bak").write_text(src, encoding="utf-8")
            txt.write_text(content, encoding="utf-8")
            _invalidate_cache(txt)
            lines.append(f"{img.name}: {action_desc}")
        except Exception as exc:
            lines.append(f"[ERROR] writing {txt.name}: {exc}")
            errors += 1
            continue
        processed += 1

    if mode == "insert" and missing_txt_skipped:
        lines.append(f"Skipped {missing_txt_skipped} image(s) without paired .txt in {temp_folder}.")
    if mode == "insert" and missing_txt_created:
        lines.append(f"Created {missing_txt_created} new .txt file(s) for images that had no paired tags.")
    lines.append(f"Done. {processed} file(s) updated in {temp_folder}. Errors: {errors}.")
    return _done(errors == 0, "" if errors == 0 else f"{errors} file update(s) failed.")
