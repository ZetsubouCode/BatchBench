"""Backward-compatible Guided Tagging Flow session reconciliation.

The original Guided Tagging Flow session was tied to an exact image-list hash.
That is useful for detecting a changed dataset, but it must not replace the
active session when images are added during an ongoing project.  This module
keeps the active session authoritative and reconciles it with the current
inventory by stable dataset-relative image path.

It is deliberately a small compatibility layer around ``services.tag_editor``
so historical sessions can be upgraded safely without changing their on-disk
location or requiring a manual migration.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from . import tag_editor


# Keep references before replacing the public functions below.
_ORIGINAL_START = tag_editor.start_tagging_session
_ORIGINAL_LOAD = tag_editor.load_tagging_session
_DEFAULT_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


def _now() -> str:
    return tag_editor._utc_now_iso()


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _segment_ids(segments: Any) -> List[str]:
    ids: List[str] = []
    for segment in _as_list(segments):
        if not isinstance(segment, dict):
            continue
        segment_id = str(segment.get("id") or "").strip()
        if segment_id and segment_id not in ids:
            ids.append(segment_id)
    return ids


def _legacy_segment_ids(session: Dict[str, Any]) -> List[str]:
    """Recover segment ids from older session payloads when possible."""
    from_config = _segment_ids(session.get("quiz_segments"))
    if from_config:
        return from_config

    discovered: List[str] = []
    for entry in _as_dict(session.get("images")).values():
        for segment_id in _as_dict(_as_dict(entry).get("segments")).keys():
            clean = str(segment_id or "").strip()
            if clean and clean != "__unsorted__" and clean not in discovered:
                discovered.append(clean)
    return discovered


def _normalize_segment_state(raw: Any) -> Dict[str, Any]:
    src = _as_dict(raw)
    selected = src.get("selected") if isinstance(src.get("selected"), list) else []
    manual = src.get("manual") if isinstance(src.get("manual"), list) else []
    removed_defaults = src.get("removed_defaults") if isinstance(src.get("removed_defaults"), list) else []
    return {
        "selected": tag_editor._dedup_tags(selected),
        "manual": tag_editor._dedup_tags(manual),
        "removed_defaults": tag_editor._dedup_tags(removed_defaults),
        "skipped": bool(src.get("skipped", False)),
        "updated_at": str(src.get("updated_at") or ""),
    }


def _merge_segments(candidate: Any, existing: Any, active_segment_ids: Iterable[str]) -> Tuple[Dict[str, Any], List[str]]:
    """Keep stored choices for known segments and add defaults for new ones."""
    merged = deepcopy(_as_dict(candidate))
    old_segments = _as_dict(existing)
    active_ids = set(active_segment_ids)
    retained: List[str] = []

    for segment_id in active_ids:
        state = old_segments.get(segment_id)
        if isinstance(state, dict):
            merged[segment_id] = _normalize_segment_state(state)
            retained.append(segment_id)

    # ``__unsorted__`` holds tags that could not be assigned to a guided step.
    # Preserve it regardless of settings changes so no historical tag choice is
    # silently lost while the session is being migrated.
    if isinstance(old_segments.get("__unsorted__"), dict):
        merged["__unsorted__"] = _normalize_segment_state(old_segments["__unsorted__"])

    return merged, retained


def _normalize_image_entry(raw: Any, fallback: Any) -> Dict[str, Any]:
    source = _as_dict(raw)
    out = deepcopy(_as_dict(fallback))
    out.setdefault("status", "pending")
    out.setdefault("segments", {})
    out.setdefault("final_tags_written", False)
    out.setdefault("missing", False)

    if source:
        out["status"] = str(source.get("status") or out["status"] or "pending")
        out["final_tags_written"] = bool(source.get("final_tags_written", out["final_tags_written"]))
        out["missing"] = bool(source.get("missing", False))
        if source.get("updated_at"):
            out["updated_at"] = str(source.get("updated_at"))
        if isinstance(source.get("legacy_segments"), dict):
            out["legacy_segments"] = deepcopy(source["legacy_segments"])
    return out


def _set_resume_cursor(
    session: Dict[str, Any],
    ordered_rels: List[str],
    previous_current: Any,
    preserve_current: bool,
) -> None:
    """Keep an ongoing cursor when possible; otherwise select first unfinished."""
    segments = _segment_ids(session.get("quiz_segments"))
    default_segment_id = segments[0] if segments else ""
    images = _as_dict(session.get("images"))

    if preserve_current:
        current = _as_dict(previous_current)
        current_rel = str(current.get("image_rel") or "")
        current_entry = _as_dict(images.get(current_rel))
        requested_segment = str(current.get("segment_id") or "")
        if (
            current_rel in ordered_rels
            and current_entry
            and not current_entry.get("missing")
            and current_entry.get("status") != "completed"
        ):
            segment_id = requested_segment if requested_segment in segments else default_segment_id
            session["current"] = {
                "image_index": ordered_rels.index(current_rel),
                "image_rel": current_rel,
                "segment_index": segments.index(segment_id) if segment_id in segments else 0,
                "segment_id": segment_id,
            }
            session["status"] = "active"
            return

    for index, rel in enumerate(ordered_rels):
        entry = _as_dict(images.get(rel))
        if entry.get("missing") or entry.get("status") == "completed":
            continue
        session["current"] = {
            "image_index": index,
            "image_rel": rel,
            "segment_index": 0,
            "segment_id": default_segment_id,
        }
        session["status"] = "active"
        return

    # Preserve a stable cursor after completion rather than pointing at a
    # removed image. The status remains the source of truth for the UI.
    if ordered_rels:
        last_rel = ordered_rels[-1]
        session["current"] = {
            "image_index": len(ordered_rels) - 1,
            "image_rel": last_rel,
            "segment_index": 0,
            "segment_id": default_segment_id,
        }
    else:
        session["current"] = {
            "image_index": 0,
            "image_rel": "",
            "segment_index": 0,
            "segment_id": default_segment_id,
        }
    session["status"] = "completed"


def reconcile_tagging_session(existing: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Merge a historical session into the current dataset inventory.

    Dataset-relative paths are intentionally the primary identity.  They are
    stable across application restarts and do not require hashing every image.
    For historical sessions that predate this reconciliation, unmatched paths
    are kept as ``missing`` history instead of being guessed as a different
    image. This avoids transferring a caption decision to the wrong file.
    """
    previous = _as_dict(existing)
    fresh = deepcopy(_as_dict(candidate))
    prior_images = _as_dict(previous.get("images"))
    fresh_images = _as_dict(fresh.get("images"))
    ordered_rels = list(fresh_images.keys())

    active_segment_ids = _segment_ids(fresh.get("quiz_segments"))
    previous_segment_ids = _legacy_segment_ids(previous)
    added_segment_ids = [segment_id for segment_id in active_segment_ids if segment_id not in previous_segment_ids]
    can_compare_segments = bool(previous_segment_ids)

    merged_images: Dict[str, Any] = {}
    retained_completed = 0
    retained_partial = 0
    new_images = 0
    requeued_for_new_segment = 0
    missing_history = 0

    for rel in ordered_rels:
        candidate_entry = _normalize_image_entry(fresh_images.get(rel), {})
        old_entry = _as_dict(prior_images.get(rel))
        if not old_entry:
            merged_images[rel] = candidate_entry
            new_images += 1
            continue

        entry = _normalize_image_entry(old_entry, candidate_entry)
        merged_segments, retained_segments = _merge_segments(
            candidate_entry.get("segments"),
            old_entry.get("segments"),
            active_segment_ids,
        )
        entry["segments"] = merged_segments
        entry["missing"] = False

        old_segment_payload = _as_dict(old_entry.get("segments"))
        stale_segments = {
            segment_id: deepcopy(state)
            for segment_id, state in old_segment_payload.items()
            if segment_id not in set(active_segment_ids) and segment_id != "__unsorted__"
        }
        if stale_segments:
            entry["legacy_segments"] = stale_segments

        was_completed = entry.get("status") == "completed"
        # New configured steps require a review of older images. Existing
        # selections remain present, but a final save is required to include
        # the newly introduced segment in the caption.
        if was_completed and can_compare_segments and added_segment_ids:
            entry["status"] = "pending"
            entry["final_tags_written"] = False
            entry["resume_reason"] = "new_guided_segment"
            requeued_for_new_segment += 1
            retained_partial += 1
        elif was_completed:
            retained_completed += 1
        else:
            retained_partial += 1

        # A partial historical entry may not have every current segment state.
        # Candidate defaults fill only the missing states; stored work wins.
        if retained_segments:
            entry["updated_at"] = str(old_entry.get("updated_at") or entry.get("updated_at") or "")
        merged_images[rel] = entry

    # Keep old entries that are no longer present as history. They never block
    # completion and give the UI/logs a transparent explanation of the count.
    for rel, raw_entry in prior_images.items():
        if rel in merged_images:
            continue
        entry = _normalize_image_entry(raw_entry, {})
        entry["missing"] = True
        merged_images[str(rel)] = entry
        missing_history += 1

    fresh["version"] = max(2, int(previous.get("version") or 1))
    fresh["created_at"] = str(previous.get("created_at") or fresh.get("created_at") or _now())
    fresh["images"] = merged_images
    fresh["updated_at"] = _now()
    _set_resume_cursor(
        fresh,
        ordered_rels,
        previous.get("current"),
        preserve_current=not added_segment_ids,
    )

    completed_now = sum(
        1
        for rel in ordered_rels
        if _as_dict(merged_images.get(rel)).get("status") == "completed"
        and not _as_dict(merged_images.get(rel)).get("missing")
    )
    summary = {
        "retained_completed": retained_completed,
        "retained_partial": retained_partial,
        "new_images": new_images,
        "missing_history": missing_history,
        "requeued_for_new_segment": requeued_for_new_segment,
        "added_segments": added_segment_ids,
        "completed": completed_now,
        "total": len(ordered_rels),
        "changed": bool(new_images or missing_history or added_segment_ids or previous.get("version") != fresh.get("version")),
    }
    return fresh, summary


def _active_session(project_root: Path) -> Optional[Dict[str, Any]]:
    path = tag_editor.tagging_session_path(project_root)
    if not path.exists():
        return None
    try:
        payload = tag_editor._read_json_file(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _resume_logs(summary: Dict[str, Any], session: Dict[str, Any]) -> List[str]:
    logs = [
        "[resume] active Guided Tagging Flow session reconciled",
        f"[resume] completed images retained: {summary['retained_completed']}",
        f"[resume] partial images retained: {summary['retained_partial']}",
        f"[resume] new images queued: {summary['new_images']}",
        f"[resume] missing images retained in history: {summary['missing_history']}",
    ]
    if summary["added_segments"]:
        logs.append("[resume] new segments require review: " + ", ".join(summary["added_segments"]))
    if summary["requeued_for_new_segment"]:
        logs.append(f"[resume] images requeued for new segments: {summary['requeued_for_new_segment']}")
    current = _as_dict(session.get("current"))
    if current.get("image_rel"):
        logs.append(f"[resume] next image: {current['image_rel']}")
    else:
        logs.append("[resume] no pending images")
    logs.append(f"[resume] progress: {summary['completed']} / {summary['total']} completed")
    return logs


def start_tagging_session(
    project_root: Path,
    exts: List[str],
    mapping_rows: Optional[List[Dict[str, Any]]] = None,
    session_defaults: Optional[Dict[str, List[str]]] = None,
    recommendations: Optional[Dict[str, List[str]]] = None,
    settings: Optional[Dict[str, Any]] = None,
    replace: bool = True,
) -> Dict[str, Any]:
    """Start or resume a Guided Tagging Flow without losing current progress."""
    existing = _active_session(project_root)
    fresh_result = _ORIGINAL_START(
        project_root,
        exts,
        mapping_rows=mapping_rows,
        session_defaults=session_defaults,
        recommendations=recommendations,
        settings=settings,
        replace=False,
    )
    if not fresh_result.get("ok") or not isinstance(fresh_result.get("session"), dict):
        return fresh_result

    session = fresh_result["session"]
    if not existing:
        if replace:
            tag_editor.save_tagging_session(project_root, session)
        fresh_result["logs"] = list(fresh_result.get("logs") or [])
        fresh_result["resume"] = {
            "resumed": False,
            "retained_completed": 0,
            "retained_partial": 0,
            "new_images": len(_as_dict(session.get("images"))),
            "missing_history": 0,
            "completed": 0,
            "total": len(_as_dict(session.get("images"))),
        }
        return fresh_result

    # Keep a durable historical copy under its old image-list hash before the
    # active pointer is replaced by the reconciled current inventory.
    try:
        tag_editor._save_session_slot(project_root, existing)
    except Exception:
        pass

    merged, summary = reconcile_tagging_session(existing, session)
    if replace:
        tag_editor.save_tagging_session(project_root, merged)
    fresh_result["session"] = merged
    fresh_result["resume"] = {"resumed": True, **summary}
    fresh_result["logs"] = list(fresh_result.get("logs") or []) + _resume_logs(summary, merged)
    fresh_result["warnings"] = list(fresh_result.get("warnings") or [])
    return fresh_result


def load_tagging_session(project_root: Path) -> Dict[str, Any]:
    """Load the active session and reconcile image additions/removals in place."""
    existing = _active_session(project_root)
    if not existing:
        return _ORIGINAL_LOAD(project_root)

    # The active pointer is authoritative. Do not replace it with a hash-slot
    # session just because new images changed the inventory fingerprint.
    fresh_result = _ORIGINAL_START(
        project_root,
        _DEFAULT_IMAGE_EXTS,
        mapping_rows=_as_list(existing.get("mapping_rows")),
        session_defaults=_as_dict(existing.get("session_defaults")),
        recommendations=_as_dict(existing.get("recommendations")),
        settings=tag_editor.load_tagging_quiz_settings(),
        replace=False,
    )
    if not fresh_result.get("ok") or not isinstance(fresh_result.get("session"), dict):
        return _ORIGINAL_LOAD(project_root)

    merged, summary = reconcile_tagging_session(existing, fresh_result["session"])
    should_save = summary["changed"] or merged.get("source_fingerprint") != existing.get("source_fingerprint")
    if should_save:
        try:
            tag_editor._save_session_slot(project_root, existing)
        except Exception:
            pass
        tag_editor.save_tagging_session(project_root, merged)

    return {
        "ok": True,
        "exists": True,
        "session": merged,
        "resume": {"resumed": True, **summary},
        "warnings": [],
        "logs": ["Loaded tagging session"] + _resume_logs(summary, merged),
        **tag_editor._make_serializable_path_info(tag_editor.resolve_project_paths(project_root)),
    }


def apply_patch() -> None:
    if getattr(tag_editor, "_guided_flow_resume_patch_applied", False):
        return
    tag_editor.start_tagging_session = start_tagging_session
    tag_editor.load_tagging_session = load_tagging_session
    tag_editor._guided_flow_resume_patch_applied = True


apply_patch()
