from __future__ import annotations

from collections import Counter
from datetime import datetime
import csv
import io
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from PIL import Image

from utils.dataset import join_tags
from utils.parse import parse_tag_list
from utils.text_io import read_text_best_effort

from . import tag_catalog
from . import tag_editor


ASSIST_REL = Path("database") / "tagging_assist.json"
VERSION = 1
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]
MALFORMED_RE = re.compile(r"[^a-z0-9_:+()\\.-]")
COLOR_TAG_RE = re.compile(r"(?:^|_)(black|white|red|blue|green|yellow|pink|purple|orange|brown|blonde|grey|gray)(?:_|$)")
MULTI_PERSON_TAGS = {"2girls", "2boys", "multiple_girls", "multiple_boys", "multiple_persons", "group"}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalize_tag(raw: Any) -> str:
    return tag_catalog.normalize_tag_name(raw)


def _normalize_tags(raw: Any) -> List[str]:
    chunks = raw if isinstance(raw, list) else parse_tag_list(str(raw or ""))
    out: List[str] = []
    seen: Set[str] = set()
    for item in chunks or []:
        tag = _normalize_tag(item)
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _assist_path(project_root: Path) -> Path:
    return Path(project_root) / ASSIST_REL


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def default_state() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "custom_tags": [],
        "required_trigger_tags": [],
        "rare_tag_threshold": 1,
        "packs": [],
        "sibling_groups": [],
        "machine_suggestions": {},
        "machine_ignored": {},
        "operation_history": [],
    }


def normalize_state(payload: Any) -> Dict[str, Any]:
    src = payload if isinstance(payload, dict) else {}
    out = default_state()
    out["custom_tags"] = _normalize_tags(src.get("custom_tags") or [])
    out["required_trigger_tags"] = _normalize_tags(src.get("required_trigger_tags") or [])
    try:
        out["rare_tag_threshold"] = max(0, int(src.get("rare_tag_threshold", out["rare_tag_threshold"])))
    except Exception:
        pass
    packs: List[Dict[str, Any]] = []
    for index, raw in enumerate(src.get("packs") if isinstance(src.get("packs"), list) else []):
        if not isinstance(raw, dict):
            continue
        pack_id = str(raw.get("id") or "").strip() or uuid.uuid4().hex
        name = re.sub(r"\s+", " ", str(raw.get("name") or "").strip())[:80] or "Tag pack"
        packs.append(
            {
                "id": pack_id,
                "name": name,
                "segment_id": _normalize_tag(raw.get("segment_id")),
                "tags": _normalize_tags(raw.get("tags") or []),
                "pinned": bool(raw.get("pinned", False)),
                "usage_count": max(0, int(raw.get("usage_count") or 0)),
                "last_used_at": str(raw.get("last_used_at") or ""),
                "note": str(raw.get("note") or "").strip()[:500],
                "order": int(raw.get("order", index)),
            }
        )
    out["packs"] = sorted(packs, key=lambda item: (not item.get("pinned"), int(item.get("order") or 0), item["name"].lower()))
    groups: List[Dict[str, Any]] = []
    for raw in src.get("sibling_groups") if isinstance(src.get("sibling_groups"), list) else []:
        if not isinstance(raw, dict):
            continue
        images = _normalize_rel_list(raw.get("images") or [])
        if len(images) < 2:
            continue
        groups.append(
            {
                "id": str(raw.get("id") or "").strip() or uuid.uuid4().hex,
                "name": re.sub(r"\s+", " ", str(raw.get("name") or "").strip())[:80] or "Sibling group",
                "images": images,
                "note": str(raw.get("note") or "").strip()[:500],
                "created_at": str(raw.get("created_at") or _now_iso()),
                "updated_at": str(raw.get("updated_at") or ""),
            }
        )
    out["sibling_groups"] = groups
    out["machine_suggestions"] = src.get("machine_suggestions") if isinstance(src.get("machine_suggestions"), dict) else {}
    out["machine_ignored"] = src.get("machine_ignored") if isinstance(src.get("machine_ignored"), dict) else {}
    out["operation_history"] = src.get("operation_history") if isinstance(src.get("operation_history"), list) else []
    return out


def load_state(project_root: Path) -> Dict[str, Any]:
    path = _assist_path(project_root)
    if not path.exists():
        return default_state()
    try:
        return normalize_state(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return default_state()


def save_state(project_root: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    state = normalize_state(payload)
    _atomic_write(_assist_path(project_root), state)
    return state


def set_custom_tags(project_root: Path, tags: Any) -> Dict[str, Any]:
    state = load_state(project_root)
    state["custom_tags"] = _normalize_tags(tags)
    return save_state(project_root, state)


def upsert_pack(project_root: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    state = load_state(project_root)
    raw_id = str(payload.get("id") or "").strip()
    packs = [pack for pack in state["packs"] if not raw_id or pack["id"] != raw_id]
    order = int(payload.get("order", len(packs)))
    pack = {
        "id": raw_id or uuid.uuid4().hex,
        "name": re.sub(r"\s+", " ", str(payload.get("name") or "").strip())[:80] or "Tag pack",
        "segment_id": _normalize_tag(payload.get("segment_id")),
        "tags": _normalize_tags(payload.get("tags") or []),
        "pinned": bool(payload.get("pinned", False)),
        "usage_count": max(0, int(payload.get("usage_count") or 0)),
        "last_used_at": str(payload.get("last_used_at") or ""),
        "note": str(payload.get("note") or "").strip()[:500],
        "order": order,
    }
    if not pack["tags"]:
        raise ValueError("A tag pack needs at least one tag.")
    packs.append(pack)
    state["packs"] = packs
    saved = save_state(project_root, state)
    return {"ok": True, "pack": pack, "assist": saved, "warnings": [], "info": []}


def delete_pack(project_root: Path, pack_id: str) -> Dict[str, Any]:
    state = load_state(project_root)
    before = len(state["packs"])
    state["packs"] = [pack for pack in state["packs"] if pack.get("id") != str(pack_id or "")]
    if len(state["packs"]) == before:
        raise ValueError("Tag pack not found.")
    return {"ok": True, "assist": save_state(project_root, state), "warnings": [], "info": []}


def list_packs(project_root: Path, segment_id: str = "", show_all: bool = False) -> Dict[str, Any]:
    state = load_state(project_root)
    segment = _normalize_tag(segment_id)
    packs = []
    for pack in state["packs"]:
        relevant = not pack.get("segment_id") or pack.get("segment_id") == segment
        if show_all or relevant:
            item = dict(pack)
            item["relevant"] = relevant
            packs.append(item)
    packs.sort(key=lambda item: (not item.get("relevant"), not item.get("pinned"), int(item.get("order") or 0), item["name"].lower()))
    return {"ok": True, "packs": packs, "custom_tags": state["custom_tags"], "warnings": [], "info": []}


def apply_pack(project_root: Path, pack_id: str, current_tags: Any) -> Dict[str, Any]:
    state = load_state(project_root)
    pack = next((item for item in state["packs"] if item.get("id") == str(pack_id or "")), None)
    if not pack:
        raise ValueError("Tag pack not found.")
    before = _normalize_tags(current_tags or [])
    merged = _normalize_tags(before + list(pack.get("tags") or []))
    pack["usage_count"] = int(pack.get("usage_count") or 0) + 1
    pack["last_used_at"] = _now_iso()
    save_state(project_root, state)
    return {"ok": True, "pack": pack, "tags": merged, "added": [tag for tag in merged if tag not in before], "warnings": [], "info": []}


def _normalize_rel(raw: Any) -> str:
    value = str(raw or "").replace("\\", "/").lstrip("/")
    rel = Path(value)
    if not value or rel.is_absolute() or rel.drive or any(part == ".." for part in rel.parts):
        return ""
    return rel.as_posix()


def _normalize_rel_list(raw: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in raw or []:
        rel = _normalize_rel(item)
        if rel and rel not in seen:
            seen.add(rel)
            out.append(rel)
    return out


def _area_root(project_root: Path, area: str) -> Path:
    paths = tag_editor.resolve_project_paths(project_root)
    return paths["dataset_root"] if str(area or "").lower() == "dataset" else paths["temp_root"]


def _resolve_rel(root: Path, rel: str) -> Path:
    clean = _normalize_rel(rel)
    if not clean:
        raise ValueError("Invalid relative path.")
    target = root / clean
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    if target_resolved != root_resolved and root_resolved not in target_resolved.parents:
        raise ValueError("Invalid relative path.")
    return target


def upsert_sibling_group(project_root: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    state = load_state(project_root)
    raw_id = str(payload.get("id") or "").strip()
    groups = [group for group in state["sibling_groups"] if not raw_id or group["id"] != raw_id]
    images = _normalize_rel_list(payload.get("images") or [])
    if len(images) < 2:
        raise ValueError("A sibling group needs at least two images.")
    group = {
        "id": raw_id or uuid.uuid4().hex,
        "name": re.sub(r"\s+", " ", str(payload.get("name") or "").strip())[:80] or "Sibling group",
        "images": images,
        "note": str(payload.get("note") or "").strip()[:500],
        "created_at": str(payload.get("created_at") or _now_iso()),
        "updated_at": _now_iso(),
    }
    groups.append(group)
    state["sibling_groups"] = groups
    return {"ok": True, "group": group, "assist": save_state(project_root, state), "warnings": [], "info": []}


def sibling_preview(project_root: Path, area: str, source_rel: str, dest_rels: Any, tags: Any, mode: str = "append") -> Dict[str, Any]:
    root = _area_root(project_root, area)
    selected = _normalize_tags(tags)
    if not selected:
        raise ValueError("Select at least one tag to propagate.")
    if str(mode or "append") not in {"append", "replace"}:
        raise ValueError("Unsupported propagation mode.")
    rows: List[Dict[str, Any]] = []
    for rel in _normalize_rel_list(dest_rels or []):
        image = _resolve_rel(root, rel)
        txt = image.with_suffix(".txt")
        before = tag_editor._read_tags_cached(txt) if txt.exists() else []
        after = _normalize_tags(selected if mode == "replace" else before + selected)
        rows.append({"rel": rel, "before": before, "after": after, "added": [tag for tag in after if tag not in before], "changed": before != after})
    return {"ok": True, "source_rel": _normalize_rel(source_rel), "mode": mode, "changes": rows, "warnings": [], "info": []}


def sibling_apply(project_root: Path, area: str, source_rel: str, dest_rels: Any, tags: Any, mode: str = "append") -> Dict[str, Any]:
    preview = sibling_preview(project_root, area, source_rel, dest_rels, tags, mode)
    root = _area_root(project_root, area)
    changed: List[Dict[str, Any]] = []
    for row in preview["changes"]:
        if not row["changed"]:
            continue
        txt = _resolve_rel(root, row["rel"]).with_suffix(".txt")
        if txt.exists():
            txt.write_text(join_tags(row["after"]), encoding="utf-8")
            tag_editor._invalidate_cache(txt)
            changed.append(row)
    state = load_state(project_root)
    history = list(state.get("operation_history") or [])
    history.append({"type": "sibling_propagation", "at": _now_iso(), "area": area, "source_rel": _normalize_rel(source_rel), "mode": mode, "changes": changed})
    state["operation_history"] = history[-100:]
    save_state(project_root, state)
    preview["applied"] = changed
    return preview


def store_machine_suggestions(project_root: Path, rel: str, suggestions: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    state = load_state(project_root)
    clean_rel = _normalize_rel(rel)
    if not clean_rel:
        raise ValueError("Invalid relative path.")
    rows = []
    for raw in suggestions if isinstance(suggestions, list) else []:
        if not isinstance(raw, dict):
            continue
        tag = _normalize_tag(raw.get("tag"))
        if not tag:
            continue
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0))))
        except Exception:
            confidence = 0.0
        rows.append({"tag": tag, "confidence": confidence, "category": str(raw.get("category") or ""), "source": "machine"})
    state["machine_suggestions"][clean_rel] = {"metadata": metadata or {}, "updated_at": _now_iso(), "suggestions": rows}
    return {"ok": True, "record": state["machine_suggestions"][clean_rel], "assist": save_state(project_root, state), "warnings": [], "info": []}


def list_machine_suggestions(project_root: Path, rel: str) -> Dict[str, Any]:
    state = load_state(project_root)
    clean_rel = _normalize_rel(rel)
    record = state["machine_suggestions"].get(clean_rel) or {"metadata": {}, "suggestions": []}
    ignored = set(state.get("machine_ignored", {}).get(clean_rel) or [])
    suggestions = [row for row in record.get("suggestions") or [] if row.get("tag") not in ignored]
    return {"ok": True, "metadata": record.get("metadata") or {}, "suggestions": suggestions, "ignored": sorted(ignored), "warnings": [], "info": []}


def ignore_machine_suggestion(project_root: Path, rel: str, tag: str) -> Dict[str, Any]:
    state = load_state(project_root)
    clean_rel = _normalize_rel(rel)
    clean_tag = _normalize_tag(tag)
    if not clean_rel or not clean_tag:
        raise ValueError("Invalid suggestion.")
    ignored = set(state.setdefault("machine_ignored", {}).get(clean_rel) or [])
    ignored.add(clean_tag)
    state["machine_ignored"][clean_rel] = sorted(ignored)
    return {"ok": True, "assist": save_state(project_root, state), "warnings": [], "info": []}


def _is_grayscaleish(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as image:
            sample = image.convert("RGB")
            sample.thumbnail((48, 48))
            pixels = list(sample.getdata())
    except Exception:
        return False
    if not pixels:
        return False
    diff = sum(max(r, g, b) - min(r, g, b) for r, g, b in pixels) / len(pixels)
    return diff < 8


def lint_captions(project_root: Path, area: str = "temp", severity: str = "", issue_type: str = "") -> Dict[str, Any]:
    root = _area_root(project_root, area)
    state = load_state(project_root)
    custom_tags = set(state.get("custom_tags") or [])
    required = set(state.get("required_trigger_tags") or [])
    rows: List[Dict[str, Any]] = []
    captions: List[Tuple[Path, str, List[str]]] = []
    exclude_dir = root / "_temp" if str(area or "").lower() == "dataset" else None
    for image in tag_editor._list_images(root, IMAGE_EXTS, recursive=True, exclude_dir=exclude_dir):
        rel = image.relative_to(root).as_posix()
        txt = image.with_suffix(".txt")
        tags = tag_editor._read_tags_cached(txt) if txt.exists() else []
        captions.append((image, rel, tags))
    tag_counts = Counter(tag for _, _, tags in captions for tag in set(tags))

    def add(level: str, rel: str, kind: str, explanation: str, tags: Iterable[str], action: str) -> None:
        if severity and level != severity:
            return
        if issue_type and kind != issue_type:
            return
        rows.append({"severity": level, "image_path": rel, "issue_type": kind, "explanation": explanation, "affected_tags": list(tags), "suggested_action": action})

    for image, rel, tags in captions:
        seen: Set[str] = set()
        for tag in tags:
            clean = _normalize_tag(tag)
            if tag in seen:
                add("warning", rel, "duplicate", f"Duplicate tag '{tag}'.", [tag], "Remove one duplicate manually.")
            seen.add(tag)
            if not clean or clean != tag or MALFORMED_RE.search(tag):
                add("warning", rel, "malformed", f"Tag '{tag}' has unusual formatting.", [tag], "Normalize spacing/underscores before saving.")
        for missing in sorted(required - set(tags)):
            add("error", rel, "missing_required_trigger", f"Required trigger tag '{missing}' is missing.", [missing], "Open the image and add the configured trigger if appropriate.")
        validations = tag_catalog.validate_tags(tags, custom_tags=custom_tags)
        for validation in validations:
            tag = validation["tag"]
            if validation.get("validation_status") == "Custom / unknown tag" and not validation.get("whitelisted"):
                add("warning", rel, "unknown", f"'{tag}' is not in the local catalog or custom whitelist.", [tag], "Verify it or add it to the project custom whitelist.")
            elif validation.get("validation_status") == "Alias that resolves to a canonical tag":
                add("warning", rel, "alias", f"'{validation.get('alias')}' resolves to '{validation.get('canonical')}'.", [validation.get("alias"), validation.get("canonical")], "Replace the alias with the canonical tag if desired.")
            elif validation.get("validation_status") == "Deprecated or invalid":
                add("error", rel, "deprecated", f"'{tag}' is explicitly deprecated in the local catalog.", [tag], "Choose a replacement before training.")
        for tag in set(tags):
            if 0 < int(state.get("rare_tag_threshold") or 0) >= tag_counts[tag]:
                add("advisory", rel, "rare_one_off", f"'{tag}' appears only {tag_counts[tag]} time(s).", [tag], "Check whether this is intentional.")
        tag_set = set(tags)
        if "solo" in tag_set and tag_set.intersection(MULTI_PERSON_TAGS):
            add("warning", rel, "conflict", "solo appears with a multiple-person tag.", ["solo", *sorted(tag_set.intersection(MULTI_PERSON_TAGS))], "Keep the tag that matches the image.")
        if "from_behind" in tag_set and {"looking_at_viewer", "from_front"}.intersection(tag_set):
            add("advisory", rel, "composition_conflict", "Back-view and front-facing/viewer-facing tags appear together.", ["from_behind"], "Review camera and face-direction tags.")
        if any(COLOR_TAG_RE.search(tag) for tag in tag_set) and _is_grayscaleish(image):
            add("advisory", rel, "grayscale_color", "Color-related tags appear on a mostly greyscale image.", [tag for tag in tag_set if COLOR_TAG_RE.search(tag)], "Keep only if the color is semantically useful.")
    return {"ok": True, "area": area, "rows": rows, "summary": {"images": len(captions), "issues": len(rows)}, "warnings": [], "info": []}


def export_lint(report: Dict[str, Any], fmt: str) -> Tuple[str, str]:
    rows = report.get("rows") or []
    if str(fmt).lower() == "csv":
        handle = io.StringIO()
        fieldnames = ["severity", "image_path", "issue_type", "explanation", "affected_tags", "suggested_action"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            item = {key: row.get(key, "") for key in fieldnames}
            item["affected_tags"] = ", ".join(row.get("affected_tags") or [])
            writer.writerow(item)
        return handle.getvalue(), "text/csv"
    return json.dumps(report, indent=2, ensure_ascii=True) + "\n", "application/json"
