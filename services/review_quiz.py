from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.dataset import join_tags
from utils.parse import parse_tag_list
from utils.text_io import read_text_best_effort

from . import tag_editor


REVIEW_QUIZ_CONFIG_PATH = Path(__file__).resolve().parent.parent / "_config" / "review_quiz.json"
DEFAULT_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]
QUEUE_MODES = {"missing_only", "all", "conflict_only", "not_reviewed", "uncertain_only"}
STEP_MODES = {"single", "multi", "manual"}
AREA_MODES = {"temp", "dataset"}
_CONFIG_CACHE: Dict[str, Any] = {"signature": None, "payload": None}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalize_tag(raw: Any) -> str:
    tag = str(raw or "").strip().lower()
    tag = re.sub(r"\s+", "_", tag)
    tag = re.sub(r"_+", "_", tag)
    return tag.strip("_")


def _normalize_tags(raw: Any) -> List[str]:
    tags: List[str] = []
    seen: Set[str] = set()
    chunks = raw if isinstance(raw, list) else parse_tag_list(raw)
    for item in chunks or []:
        tag = _normalize_tag(item)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def _slugify(raw: Any) -> str:
    value = _normalize_tag(raw)
    return re.sub(r"[^a-z0-9_]+", "_", value).strip("_")


def default_review_quiz_config() -> Dict[str, Any]:
    return {
        "version": 1,
        "quiz_review": {
            "enabled": True,
            "default_area": "temp",
            "image_fit": "contain",
            "auto_save": True,
            "show_current_tags": True,
            "show_progress": True,
            "keyboard_enabled": True,
            "thumbnail_preload_count": 3,
            "steps": [
                {
                    "id": "identity_appearance",
                    "label": "Identity / Appearance",
                    "mode": "multi",
                    "required": True,
                    "auto_advance": False,
                    "allow_not_applicable": False,
                    "queue_mode": "missing_only",
                    "preferred_tags": ["long_hair", "short_hair", "bangs"],
                    "mapped_glossary_categories": ["appearance"],
                    "prioritize_segment_suggestions": True,
                    "preferred_danbooru_categories": ["general"],
                    "show_broad_suggestions": True,
                    "tags": ["hair_color", "eye_color", "hairstyle", "long_hair", "short_hair", "bangs"],
                },
                {
                    "id": "outfit",
                    "label": "Outfit",
                    "mode": "multi",
                    "required": True,
                    "auto_advance": False,
                    "allow_not_applicable": True,
                    "queue_mode": "missing_only",
                    "preferred_tags": [],
                    "mapped_glossary_categories": ["outfit"],
                    "prioritize_segment_suggestions": True,
                    "preferred_danbooru_categories": ["general", "copyright"],
                    "show_broad_suggestions": True,
                    "tags": ["top", "bottom", "footwear", "dress", "shirt", "skirt"],
                },
                {
                    "id": "expression",
                    "label": "Expression",
                    "mode": "multi",
                    "required": False,
                    "auto_advance": False,
                    "allow_not_applicable": True,
                    "queue_mode": "missing_only",
                    "preferred_tags": [],
                    "mapped_glossary_categories": ["expression"],
                    "prioritize_segment_suggestions": True,
                    "preferred_danbooru_categories": ["general"],
                    "show_broad_suggestions": True,
                    "tags": ["smile", "serious", "angry", "closed_eyes", "open_mouth", "sweat"],
                },
                {
                    "id": "body_composition",
                    "label": "Body Composition / Pose",
                    "mode": "single",
                    "required": True,
                    "auto_advance": True,
                    "allow_not_applicable": False,
                    "queue_mode": "missing_only",
                    "preferred_tags": [],
                    "mapped_glossary_categories": ["pose", "body composition"],
                    "prioritize_segment_suggestions": True,
                    "preferred_danbooru_categories": ["general"],
                    "show_broad_suggestions": True,
                    "tags": ["full_body", "cowboy_shot", "upper_body", "portrait", "standing", "sitting", "clenched_hands"],
                },
                {
                    "id": "camera_angle",
                    "label": "POV / Camera Angle",
                    "mode": "multi",
                    "required": True,
                    "auto_advance": True,
                    "allow_not_applicable": True,
                    "queue_mode": "missing_only",
                    "preferred_tags": [],
                    "mapped_glossary_categories": ["camera", "pov"],
                    "prioritize_segment_suggestions": True,
                    "preferred_danbooru_categories": ["general"],
                    "show_broad_suggestions": True,
                    "tags": ["looking_at_viewer", "from_front", "from_side", "from_behind", "from_above", "from_below"],
                },
                {
                    "id": "lighting",
                    "label": "Lighting",
                    "mode": "multi",
                    "required": False,
                    "auto_advance": False,
                    "allow_not_applicable": True,
                    "queue_mode": "missing_only",
                    "preferred_tags": [],
                    "mapped_glossary_categories": ["lighting"],
                    "prioritize_segment_suggestions": True,
                    "preferred_danbooru_categories": ["general", "meta"],
                    "show_broad_suggestions": True,
                    "tags": ["soft_lighting", "backlighting", "dim_lighting", "sunlight", "shadow"],
                },
                {
                    "id": "background",
                    "label": "Background Details",
                    "mode": "multi",
                    "required": True,
                    "auto_advance": False,
                    "allow_not_applicable": True,
                    "queue_mode": "missing_only",
                    "preferred_tags": [],
                    "mapped_glossary_categories": ["background"],
                    "prioritize_segment_suggestions": True,
                    "preferred_danbooru_categories": ["general"],
                    "show_broad_suggestions": True,
                    "tags": ["simple_background", "indoors", "outdoors", "bedroom", "street"],
                },
            ],
        },
    }


def normalize_review_quiz_config(payload: Any) -> Dict[str, Any]:
    defaults = default_review_quiz_config()
    src = payload if isinstance(payload, dict) else {}
    quiz_src = src.get("quiz_review") if isinstance(src.get("quiz_review"), dict) else {}
    quiz_defaults = defaults["quiz_review"]
    legacy_manual_tagging = bool(quiz_src.get("manual_tagging", False))
    steps: List[Dict[str, Any]] = []
    raw_steps = quiz_src.get("steps") if isinstance(quiz_src.get("steps"), list) else []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            continue
        label = re.sub(r"\s+", " ", str(raw_step.get("label") or "").strip())
        step_id = _slugify(raw_step.get("id"))
        mode = str(raw_step.get("mode") or "single").strip().lower()
        if legacy_manual_tagging and mode in {"single", "multi"}:
            mode = "manual"
        required = bool(raw_step.get("required", False))
        queue_mode = str(raw_step.get("queue_mode") or ("missing_only" if required else "all")).strip().lower()
        steps.append(
            {
                "id": step_id,
                "label": label,
                "mode": mode,
                "required": required,
                "auto_advance": bool(raw_step.get("auto_advance", False)),
                "autosuggest_segment_only": bool(
                    raw_step.get("autosuggest_segment_only", quiz_src.get("autosuggest_segment_only", False))
                ),
                "danbooru_autosuggest": bool(raw_step.get("danbooru_autosuggest", False)),
                "preferred_tags": _normalize_tags(raw_step.get("preferred_tags")),
                "mapped_glossary_categories": [
                    re.sub(r"\s+", " ", str(value or "").strip()).lower()
                    for value in (raw_step.get("mapped_glossary_categories") if isinstance(raw_step.get("mapped_glossary_categories"), list) else [])
                    if str(value or "").strip()
                ],
                "prioritize_segment_suggestions": bool(raw_step.get("prioritize_segment_suggestions", True)),
                "preferred_danbooru_categories": [
                    _normalize_tag(value)
                    for value in (raw_step.get("preferred_danbooru_categories") if isinstance(raw_step.get("preferred_danbooru_categories"), list) else [])
                    if _normalize_tag(value)
                ],
                "show_broad_suggestions": bool(raw_step.get("show_broad_suggestions", True)),
                "allow_not_applicable": bool(raw_step.get("allow_not_applicable", False)),
                "queue_mode": queue_mode,
                "tags": _normalize_tags(raw_step.get("tags")),
            }
        )
    try:
        preload_count = int(quiz_src.get("thumbnail_preload_count", quiz_defaults["thumbnail_preload_count"]))
    except Exception:
        preload_count = quiz_defaults["thumbnail_preload_count"]
    return {
        "version": 1,
        "quiz_review": {
            "enabled": bool(quiz_src.get("enabled", quiz_defaults["enabled"])),
            "default_area": str(quiz_src.get("default_area") or quiz_defaults["default_area"]).strip().lower(),
            "image_fit": str(quiz_src.get("image_fit") or quiz_defaults["image_fit"]).strip().lower(),
            "auto_save": bool(quiz_src.get("auto_save", quiz_defaults["auto_save"])),
            "show_current_tags": bool(quiz_src.get("show_current_tags", quiz_defaults["show_current_tags"])),
            "show_progress": bool(quiz_src.get("show_progress", quiz_defaults["show_progress"])),
            "keyboard_enabled": bool(quiz_src.get("keyboard_enabled", quiz_defaults["keyboard_enabled"])),
            "thumbnail_preload_count": max(0, min(6, preload_count)),
            "steps": steps,
        },
    }


def validate_review_quiz_config(payload: Any) -> List[str]:
    config = normalize_review_quiz_config(payload)
    quiz = config["quiz_review"]
    errors: List[str] = []
    if quiz["default_area"] not in AREA_MODES:
        errors.append("Default target area must be temp or dataset.")
    if quiz["image_fit"] not in {"contain", "cover"}:
        errors.append("Image fit must be contain or cover.")
    seen_ids: Set[str] = set()
    for index, step in enumerate(quiz["steps"], start=1):
        prefix = f"Step {index}"
        if not step["id"]:
            errors.append(f"{prefix}: id is required.")
        elif step["id"] in seen_ids:
            errors.append(f"{prefix}: duplicate id '{step['id']}'.")
        seen_ids.add(step["id"])
        if not step["label"]:
            errors.append(f"{prefix}: label is required.")
        if step["mode"] not in STEP_MODES:
            errors.append(f"{prefix}: mode must be single, multi, or manual.")
        if step["queue_mode"] not in QUEUE_MODES:
            errors.append(f"{prefix}: queue mode is invalid.")
        if not step["tags"] and step["mode"] != "manual":
            errors.append(f"{prefix}: at least one tag is required.")
    return errors


def _config_signature() -> Optional[Tuple[int, int]]:
    try:
        stat = REVIEW_QUIZ_CONFIG_PATH.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def save_review_quiz_config(payload: Any) -> Dict[str, Any]:
    normalized = normalize_review_quiz_config(payload)
    errors = validate_review_quiz_config(normalized)
    if errors:
        raise ValueError(" ".join(errors))
    _write_json(REVIEW_QUIZ_CONFIG_PATH, normalized)
    _CONFIG_CACHE["signature"] = _config_signature()
    _CONFIG_CACHE["payload"] = deepcopy(normalized)
    return normalized


def load_review_quiz_config() -> Dict[str, Any]:
    signature = _config_signature()
    if signature is not None and signature == _CONFIG_CACHE.get("signature") and _CONFIG_CACHE.get("payload"):
        return deepcopy(_CONFIG_CACHE["payload"])
    if signature is None:
        return save_review_quiz_config(default_review_quiz_config())
    try:
        payload = json.loads(REVIEW_QUIZ_CONFIG_PATH.read_text(encoding="utf-8"))
        normalized = normalize_review_quiz_config(payload)
        errors = validate_review_quiz_config(normalized)
        if errors:
            raise ValueError(" ".join(errors))
    except Exception:
        normalized = default_review_quiz_config()
    _CONFIG_CACHE["signature"] = signature
    _CONFIG_CACHE["payload"] = deepcopy(normalized)
    return normalized


def reset_review_quiz_config() -> Dict[str, Any]:
    return save_review_quiz_config(default_review_quiz_config())


def _resolve_area_root(project_root: Path, area: str) -> Tuple[Path, str]:
    paths = tag_editor.resolve_project_paths(project_root)
    area_key = str(area or "temp").strip().lower()
    if area_key == "temp":
        root = paths["temp_root"]
    elif area_key == "dataset":
        root = paths["dataset_root"]
    else:
        raise ValueError("Area must be temp or dataset.")
    if not paths["project_root"].exists() or not paths["project_root"].is_dir():
        raise ValueError(f"Project root not found: {paths['project_root']}")
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Folder not found: {root}")
    return root, area_key


def _resolve_rel(root: Path, raw_rel: Any) -> Path:
    rel = Path(str(raw_rel or "").replace("\\", "/").lstrip("/"))
    if not str(rel) or rel.is_absolute() or rel.drive or any(part == ".." for part in rel.parts):
        raise ValueError("Invalid relative path.")
    target = root / rel
    try:
        root_resolved = root.resolve()
        target_resolved = target.resolve()
    except OSError as exc:
        raise ValueError("Path resolution failed.") from exc
    if target_resolved != root_resolved and root_resolved not in target_resolved.parents:
        raise ValueError("Invalid relative path.")
    return target


def _metadata_path(root: Path) -> Path:
    return root / ".bb_review.json"


def _load_metadata(root: Path) -> Dict[str, Any]:
    path = _metadata_path(root)
    if not path.exists():
        return {"version": 2, "items": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "items": {}}
    items = payload.get("items") if isinstance(payload, dict) and isinstance(payload.get("items"), dict) else {}
    return {"version": max(2, int(payload.get("version") or 1)) if isinstance(payload, dict) else 2, "items": items}


def _save_metadata(root: Path, payload: Dict[str, Any]) -> None:
    _write_json(_metadata_path(root), {"version": 2, "items": payload.get("items") or {}})


def _normalize_metadata_item(raw: Any) -> Dict[str, Any]:
    item = raw if isinstance(raw, dict) else {}
    reviewed = [str(value) for value in item.get("reviewed_steps", []) if str(value or "").strip()]
    not_applicable = [str(value) for value in item.get("not_applicable", []) if str(value or "").strip()]
    uncertain_src = item.get("uncertain_steps") if isinstance(item.get("uncertain_steps"), dict) else {}
    uncertain_steps: Dict[str, Dict[str, str]] = {}
    for step_id, detail in uncertain_src.items():
        clean_step = str(step_id or "").strip()
        if not clean_step:
            continue
        if isinstance(detail, dict):
            note = str(detail.get("note") or "").strip()
            updated_at = str(detail.get("updated_at") or "").strip()
        else:
            note = str(detail or "").strip()
            updated_at = ""
        uncertain_steps[clean_step] = {"note": note[:500], "updated_at": updated_at}
    return {
        "reviewed_steps": list(dict.fromkeys(reviewed)),
        "not_applicable": list(dict.fromkeys(not_applicable)),
        "uncertain_steps": uncertain_steps,
        "updated_at": str(item.get("updated_at") or ""),
    }


def _find_step(config: Dict[str, Any], step_id: Any) -> Dict[str, Any]:
    wanted = str(step_id or "").strip()
    for step in config["quiz_review"]["steps"]:
        if step["id"] == wanted:
            return step
    raise ValueError(f"Quiz step not found: {wanted or '(empty)'}")


def _image_exts(exts: Optional[List[str]]) -> List[str]:
    values = exts or DEFAULT_IMAGE_EXTS
    return [value if value.startswith(".") else f".{value}" for value in values]


def list_quiz_items(
    project_root: Path,
    area: str,
    step_id: str,
    queue: str,
    exts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    config = load_review_quiz_config()
    step = _find_step(config, step_id)
    root, area_key = _resolve_area_root(project_root, area)
    queue_mode = str(queue or step.get("queue_mode") or "missing_only").strip().lower()
    if queue_mode not in QUEUE_MODES:
        raise ValueError("Queue mode is invalid.")
    metadata = _load_metadata(root)
    step_tags = set(step["tags"])
    items: List[Dict[str, Any]] = []
    total = missing_count = conflict_count = not_reviewed_count = reviewed_count = 0
    exclude_dir = root / "_temp" if area_key == "dataset" else None
    for image in tag_editor._list_images(root, _image_exts(exts), recursive=True, exclude_dir=exclude_dir):
        total += 1
        rel = image.relative_to(root).as_posix()
        txt_path = image.with_suffix(".txt")
        tags = tag_editor._read_tags_cached(txt_path) if txt_path.exists() else []
        selected = [tag for tag in tags if tag in step_tags]
        meta = _normalize_metadata_item(metadata["items"].get(rel))
        reviewed = step["id"] in meta["reviewed_steps"]
        not_applicable = step["id"] in meta["not_applicable"]
        uncertain_detail = (meta.get("uncertain_steps") or {}).get(step["id"]) or {}
        uncertain = step["id"] in (meta.get("uncertain_steps") or {})
        missing = (not reviewed if step["mode"] == "manual" else not selected) and not not_applicable
        if step["required"] and uncertain and not not_applicable:
            missing = True
        conflict = step["mode"] != "manual" and len(set(selected)) > 1
        missing_count += int(missing)
        conflict_count += int(conflict)
        not_reviewed_count += int(not reviewed)
        reviewed_count += int(reviewed)
        include = (
            queue_mode == "all"
            or (queue_mode == "missing_only" and missing)
            or (queue_mode == "conflict_only" and conflict)
            or (queue_mode == "not_reviewed" and not reviewed)
            or (queue_mode == "uncertain_only" and uncertain)
        )
        if not include:
            continue
        items.append(
            {
                "rel": rel,
                "name": image.name,
                "tags": tags,
                "selected_step_tags": list(dict.fromkeys(selected)),
                "has_txt": txt_path.exists(),
                "missing": missing,
                "conflict": conflict,
                "reviewed": reviewed,
                "not_applicable": not_applicable,
                "uncertain": uncertain,
                "uncertain_note": uncertain_detail.get("note") or "",
            }
        )
    return {
        "ok": True,
        "area": area_key,
        "step": step,
        "queue": queue_mode,
        "items": items,
        "summary": {
            "total": total,
            "queue_count": len(items),
            "missing": missing_count,
            "conflict": conflict_count,
            "not_reviewed": not_reviewed_count,
            "reviewed": reviewed_count,
            "completed": total - missing_count,
        },
    }


def _replacement_tags(current_tags: List[str], step_tags: Set[str], selected_tags: List[str]) -> List[str]:
    deduped = tag_editor._dedup_tags(current_tags)
    first_step_index: Optional[int] = None
    kept: List[str] = []
    for tag in deduped:
        if tag in step_tags:
            if first_step_index is None:
                first_step_index = len(kept)
            continue
        kept.append(tag)
    insert_at = len(kept) if first_step_index is None else first_step_index
    return kept[:insert_at] + selected_tags + kept[insert_at:]


def save_quiz_item(
    project_root: Path,
    area: str,
    rel: str,
    step_id: str,
    selected_tags: Any,
    manual_tags: Any = None,
    not_applicable: bool = False,
    uncertain: bool = False,
    uncertain_note: str = "",
    mark_reviewed: bool = True,
    backup: bool = True,
) -> Dict[str, Any]:
    config = load_review_quiz_config()
    step = _find_step(config, step_id)
    manual_tagging = step["mode"] == "manual"
    has_manual_tags = manual_tags is not None
    if has_manual_tags and not manual_tagging:
        raise ValueError("Manual caption tags require a manual Quiz Review step.")
    if manual_tagging and not has_manual_tags:
        raise ValueError("Manual Quiz Review steps require the full caption tag list.")
    root, area_key = _resolve_area_root(project_root, area)
    image = _resolve_rel(root, rel)
    if not image.exists() or not image.is_file():
        raise ValueError("Image not found.")
    txt_path = image.with_suffix(".txt")
    had_txt = txt_path.exists() and txt_path.is_file()
    if not had_txt and not has_manual_tags:
        raise ValueError("Missing .txt. Quiz Review does not create caption files.")
    selected = _normalize_tags(selected_tags)
    if not has_manual_tags:
        invalid = [tag for tag in selected if tag not in set(step["tags"])]
        if invalid:
            raise ValueError(f"Tags are not part of step '{step['id']}': {', '.join(invalid)}")
        if step["mode"] == "single" and len(selected) > 1:
            raise ValueError("Single-choice steps accept at most one tag.")
    if not_applicable and not step["allow_not_applicable"]:
        raise ValueError("Not Applicable is not allowed for this step.")
    if not_applicable:
        selected = []
    uncertain_note = re.sub(r"\s+", " ", str(uncertain_note or "").strip())[:500]
    source_text = ""
    if had_txt:
        try:
            source_text, _, _ = read_text_best_effort(txt_path)
        except Exception as exc:
            raise ValueError(str(exc)) from exc
    previous_tags = tag_editor._normalize_tags(parse_tag_list(source_text, dedupe=False), dedupe=False)
    if has_manual_tags:
        next_tags = tag_editor._dedup_tags(_normalize_tags(manual_tags))
        if not had_txt and not next_tags:
            raise ValueError("Add at least one manual tag before saving a missing caption.")
    else:
        next_tags = _replacement_tags(previous_tags, set(step["tags"]), selected)
    next_text = join_tags(next_tags)
    changed = (not had_txt) or next_text.strip() != source_text.strip()
    backup_created = False
    if changed:
        if backup and had_txt:
            shutil.copy2(txt_path, Path(str(txt_path) + ".bak"))
            backup_created = True
        txt_path.write_text(next_text, encoding="utf-8")
        tag_editor._invalidate_cache(txt_path)

    metadata = _load_metadata(root)
    rel_key = image.relative_to(root).as_posix()
    previous_metadata = _normalize_metadata_item(metadata["items"].get(rel_key))
    current_metadata = _normalize_metadata_item(previous_metadata)
    reviewed_steps = current_metadata["reviewed_steps"]
    na_steps = current_metadata["not_applicable"]
    uncertain_steps = current_metadata.get("uncertain_steps") or {}
    if mark_reviewed and step["id"] not in reviewed_steps:
        reviewed_steps.append(step["id"])
    if not_applicable:
        if step["id"] not in na_steps:
            na_steps.append(step["id"])
    else:
        current_metadata["not_applicable"] = [value for value in na_steps if value != step["id"]]
    if uncertain:
        uncertain_steps[step["id"]] = {"note": uncertain_note, "updated_at": _now_iso()}
        if step["id"] in reviewed_steps and step["required"] and not not_applicable:
            current_metadata["reviewed_steps"] = [value for value in reviewed_steps if value != step["id"]]
    else:
        uncertain_steps.pop(step["id"], None)
    current_metadata["uncertain_steps"] = uncertain_steps
    current_metadata["updated_at"] = _now_iso()
    metadata["items"][rel_key] = current_metadata
    _save_metadata(root, metadata)
    added = [tag for tag in next_tags if tag not in previous_tags]
    removed = [tag for tag in previous_tags if tag not in next_tags]
    return {
        "ok": True,
        "area": area_key,
        "rel": rel_key,
        "tags": next_tags,
        "selected_step_tags": [tag for tag in next_tags if tag in set(step["tags"])],
        "added": added,
        "removed": removed,
        "changed": changed,
        "backup_created": backup_created,
        "reviewed": step["id"] in current_metadata["reviewed_steps"],
        "not_applicable": step["id"] in current_metadata["not_applicable"],
        "uncertain": step["id"] in current_metadata.get("uncertain_steps", {}),
        "uncertain_note": (current_metadata.get("uncertain_steps", {}).get(step["id"]) or {}).get("note", ""),
        "previous_has_txt": had_txt,
        "previous_tags": previous_tags,
        "previous_metadata": previous_metadata,
        "metadata": current_metadata,
        "warnings": [],
    }


def restore_quiz_item(
    project_root: Path,
    area: str,
    rel: str,
    tags: Any,
    metadata_item: Any,
    had_txt: bool = True,
) -> Dict[str, Any]:
    root, area_key = _resolve_area_root(project_root, area)
    image = _resolve_rel(root, rel)
    if not image.exists() or not image.is_file():
        raise ValueError("Image not found.")
    txt_path = image.with_suffix(".txt")
    if had_txt and (not txt_path.exists() or not txt_path.is_file()):
        raise ValueError("Missing .txt.")
    restored_tags = tag_editor._dedup_tags(_normalize_tags(tags))
    if had_txt:
        txt_path.write_text(join_tags(restored_tags), encoding="utf-8")
    elif txt_path.exists():
        txt_path.unlink()
    tag_editor._invalidate_cache(txt_path)
    metadata = _load_metadata(root)
    rel_key = image.relative_to(root).as_posix()
    restored_metadata = _normalize_metadata_item(metadata_item)
    if restored_metadata["reviewed_steps"] or restored_metadata["not_applicable"] or restored_metadata["updated_at"]:
        metadata["items"][rel_key] = restored_metadata
    else:
        metadata["items"].pop(rel_key, None)
    _save_metadata(root, metadata)
    return {
        "ok": True,
        "area": area_key,
        "rel": rel_key,
        "tags": restored_tags,
        "has_txt": bool(had_txt),
        "metadata": restored_metadata,
        "warnings": [],
    }


def steps_from_cheatsheet_sections(sections: Any) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    for index, section in enumerate(sections if isinstance(sections, list) else [], start=1):
        if not isinstance(section, dict):
            continue
        label = re.sub(r"\s+", " ", str(section.get("category") or f"Step {index}").strip())
        step_id = _slugify(label) or f"step_{index}"
        base_id = step_id
        bump = 2
        while step_id in seen_ids:
            step_id = f"{base_id}_{bump}"
            bump += 1
        seen_ids.add(step_id)
        tags = _normalize_tags(section.get("tags"))
        for conditional in section.get("conditionals") or []:
            for tag in _normalize_tags(conditional):
                if tag not in tags:
                    tags.append(tag)
        if not tags:
            continue
        steps.append(
            {
                "id": step_id,
                "label": label.title(),
                "mode": "multi",
                "required": False,
                "auto_advance": False,
                "allow_not_applicable": False,
                "queue_mode": "all",
                "tags": tags,
            }
        )
    return steps
