from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional

from PIL import Image

from services import jio7_tags
from services.offline_tagger_rules import (
    BUCKET_BACKGROUND_PLACE,
    BUCKET_CAMERA_COMPOSITION,
    BUCKET_LIGHTING_ENVIRONMENT,
    BUCKET_OBJECT_PROP,
    BUCKET_POSE_ACTION,
    COMPILED_REGEX_ALLOW_BACKGROUND,
    COMPILED_REGEX_ALLOW_CAMERA,
    COMPILED_REGEX_ALLOW_LIGHTING,
    COMPILED_REGEX_ALLOW_OBJECT,
    COMPILED_REGEX_ALLOW_POSE,
    EXACT_ALLOW_BACKGROUND,
    EXACT_ALLOW_CAMERA,
    EXACT_ALLOW_LIGHTING,
    EXACT_ALLOW_OBJECT,
    EXACT_ALLOW_POSE,
    normalize_rule_tag,
)
from services.tagger_model_manager import model_dir
from services.tagger_models.base import TagPrediction
from services.tagger_models.registry import CAFORMER_PROFILE, get_model_profile
from utils.tags import to_caption_tag


CACHE_SCHEMA_VERSION = 1
SUGGESTION_POLICY_VERSION = "jio7-context-v1"
SENSITIVITY_MULTIPLIERS = {"conservative": 1.10, "normal": 0.90, "broad": 0.75}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_JOBS: Dict[str, "InspectionJob"] = {}
_LATEST_BY_PROJECT: Dict[str, str] = {}
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cache_dir(project_root: Path) -> Path:
    return Path(project_root).resolve() / "dataset" / "_temp" / "tag_suggestions"


def cache_path(project_root: Path) -> Path:
    return cache_dir(project_root) / "cache.json"


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _load_cache(project_root: Path) -> Dict[str, Any]:
    path = cache_path(project_root)
    if not path.is_file():
        return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
    if data.get("schema_version") != CACHE_SCHEMA_VERSION or not isinstance(data.get("entries"), dict):
        return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
    return data


def _model_revision(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.is_dir():
        return "missing"
    for item in sorted(path.iterdir()):
        if item.is_file():
            stat = item.stat()
            digest.update(f"{item.name}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()[:16]


def _image_fingerprint(path: Path) -> Dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _segment_match(segments: Iterable[Dict[str, Any]], hints: Iterable[str]) -> str:
    hints = tuple(str(h).lower() for h in hints)
    segments = list(segments)
    for hint in hints:
        for segment in segments:
            seg_id = str(segment.get("id") or "")
            text = f"{seg_id} {segment.get('label') or ''}".lower()
            if hint in text:
                return seg_id
    return ""


def _rule_bucket(tag: str) -> str:
    key = normalize_rule_tag(tag)
    checks = (
        (BUCKET_CAMERA_COMPOSITION, EXACT_ALLOW_CAMERA, COMPILED_REGEX_ALLOW_CAMERA),
        (BUCKET_POSE_ACTION, EXACT_ALLOW_POSE, COMPILED_REGEX_ALLOW_POSE),
        (BUCKET_LIGHTING_ENVIRONMENT, EXACT_ALLOW_LIGHTING, COMPILED_REGEX_ALLOW_LIGHTING),
        (BUCKET_BACKGROUND_PLACE, EXACT_ALLOW_BACKGROUND, COMPILED_REGEX_ALLOW_BACKGROUND),
        (BUCKET_OBJECT_PROP, EXACT_ALLOW_OBJECT, COMPILED_REGEX_ALLOW_OBJECT),
    )
    for bucket, exact, patterns in checks:
        if key in exact or any(pattern.search(key) for pattern in patterns):
            return bucket
    return "unknown"


def route_segment(tag: str, semantic_category: str, segments: Iterable[Dict[str, Any]]) -> str:
    segments = list(segments)
    category = str(semantic_category or "").lower()
    if category == "expression":
        return _segment_match(segments, ("expression", "emotion", "face")) or "__other__"
    if category == "action":
        return _segment_match(segments, ("pose", "action", "body composition", "composition")) or "__other__"
    if category == "setting":
        return _segment_match(segments, ("background", "setting", "scene", "environment")) or "__other__"
    if category == "object":
        return _segment_match(segments, ("detail", "object", "background", "prop")) or "__other__"
    if category == "other":
        normalized = normalize_rule_tag(tag)
        if normalized in {"full_body", "upper_body", "lower_body", "portrait", "solo_focus"}:
            return _segment_match(segments, ("body composition", "composition", "pose", "body")) or "__other__"
        bucket = _rule_bucket(tag)
        hints = {
            BUCKET_CAMERA_COMPOSITION: ("camera", "angle", "composition", "view"),
            BUCKET_POSE_ACTION: ("pose", "action", "body composition", "composition"),
            BUCKET_LIGHTING_ENVIRONMENT: ("lighting", "light", "shadow"),
            BUCKET_BACKGROUND_PLACE: ("background", "setting", "scene", "environment"),
            BUCKET_OBJECT_PROP: ("detail", "object", "background", "prop"),
        }.get(bucket, ())
        return _segment_match(segments, hints) or "__other__"
    return "__other__"


def filter_and_route_predictions(
    predictions: Iterable[TagPrediction],
    classifier: jio7_tags.Jio7Classification,
    segments: Iterable[Dict[str, Any]],
    *,
    sensitivity: str = "normal",
    explicit_keep: Iterable[str] = (),
    fallback_threshold: float = 0.35,
) -> Dict[str, List[Dict[str, Any]]]:
    multiplier = SENSITIVITY_MULTIPLIERS.get(str(sensitivity).lower(), SENSITIVITY_MULTIPLIERS["normal"])
    routed: Dict[str, List[Dict[str, Any]]] = {}
    for prediction in predictions:
        threshold = prediction.recommended_threshold if prediction.recommended_threshold is not None else fallback_threshold
        threshold = max(0.01, min(0.99, float(threshold) * multiplier))
        if prediction.score < threshold:
            continue
        category = classifier.category_for(prediction.tag_key)
        if category is None or not classifier.is_allowed(
            prediction.tag_key,
            strict=True,
            model_category=prediction.model_category,
            explicit_keep=explicit_keep,
        ):
            continue
        segment_id = route_segment(prediction.tag_key, category, segments)
        routed.setdefault(segment_id, []).append(
            {
                "tag": to_caption_tag(prediction.tag_key),
                "tag_key": normalize_rule_tag(prediction.tag_key),
                "score": round(float(prediction.score), 6),
                "threshold": round(threshold, 6),
                "semantic_category": category,
                "model_category": prediction.model_category,
            }
        )
    for rows in routed.values():
        rows.sort(key=lambda row: (-row["score"], row["tag"]))
    return routed


@dataclass
class InspectionJob:
    job_id: str
    project_root: Path
    profile_key: str
    sensitivity: str
    segments: List[Dict[str, Any]]
    status: str = "queued"
    phase: str = "queued"
    total: int = 0
    processed: int = 0
    cached: int = 0
    inferred: int = 0
    eligible: int = 0
    current: str = ""
    error: str = ""
    logs: List[str] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def payload(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "project_root": str(self.project_root),
            "profile": self.profile_key,
            "sensitivity": self.sensitivity,
            "status": self.status,
            "phase": self.phase,
            "total": self.total,
            "processed": self.processed,
            "cached": self.cached,
            "inferred": self.inferred,
            "eligible": self.eligible,
            "current": self.current,
            "error": self.error,
            "logs": list(self.logs),
        }


def inspect_dataset_sync(
    job: InspectionJob,
    *,
    predictor: Optional[Callable[[Path], List[TagPrediction]]] = None,
    classifier: Optional[jio7_tags.Jio7Classification] = None,
) -> None:
    job.status = "running"
    job.phase = "discovering"
    profile = get_model_profile(job.profile_key)
    dataset = job.project_root / "dataset"
    images = sorted(
        (path for path in dataset.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and "_temp" not in path.parts),
        key=lambda path: path.as_posix().lower(),
    )
    job.total = len(images)
    job.logs.append(f"[inspect] {job.total} images found")
    if not images:
        job.status = "failed"
        job.phase = "failed"
        job.error = (
            f"No supported images found in {dataset}. "
            f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}."
        )
        job.logs.append(f"[inspect] [ERROR] {job.error}")
        return

    job.phase = "loading_classifier"
    classification = classifier or jio7_tags.load()
    if classification is None:
        job.status = "failed"
        job.phase = "failed"
        job.error = "Jio7 classification data is not installed. Manual tagging is still available."
        return
    managed_dir = model_dir(profile.key)
    adapter = profile.adapter()
    loaded = None
    custom_predictor = predictor is not None
    if predictor is None:
        job.phase = "loading_model"
        missing = adapter.validate_model_dir(managed_dir)
        if missing:
            job.status = "failed"
            job.phase = "failed"
            job.error = f"Context suggestions unavailable: {profile.display_name} is not installed. Manual tagging is still available."
            return
        try:
            loaded = adapter.load(managed_dir, device="cpu")
        except Exception as exc:
            job.status = "failed"
            job.phase = "failed"
            job.error = f"Context suggestions unavailable: {exc}. Manual tagging is still available."
            return

        def predictor(path: Path) -> List[TagPrediction]:
            with Image.open(path) as image:
                return adapter.predict(loaded, [image.convert("RGB")])[0]

    cache = _load_cache(job.project_root)
    entries = cache.setdefault("entries", {})
    revision = "test" if custom_predictor else _model_revision(managed_dir)
    config = {
        "model_profile": profile.key,
        "model_revision": revision,
        "threshold_strategy": "optimized",
        "sensitivity": job.sensitivity,
        "classification_version": classification.version,
        "suggestion_policy_version": SUGGESTION_POLICY_VERSION,
    }
    cache["config"] = config
    current_rels = {path.relative_to(dataset).as_posix() for path in images}
    for stale in set(entries) - current_rels:
        entries.pop(stale, None)
    need = 0
    for path in images:
        rel = path.relative_to(dataset).as_posix()
        entry = entries.get(rel) or {}
        if entry.get("fingerprint") != _image_fingerprint(path) or entry.get("config") != config or entry.get("error"):
            need += 1
    job.logs.extend(
        [
            f"[model] Selected: {profile.display_name}",
            f"[model] Runtime: {'ONNX CPUExecutionProvider' if profile.runtime == 'onnx' else 'Timm/PyTorch'}",
            f"[model] Preprocessor: {adapter.preprocessing_label}",
            f"[classifier] Jio7 loaded: {classification.source_rows or len(classification.categories)} classified rows",
            f"[inspect] {len(images) - need} cached / {need} need inference",
        ]
    )
    job.phase = "inspecting"
    for path in images:
        if job.cancel_event.is_set():
            job.status = "cancelled"
            job.phase = "cancelled"
            job.logs.append("[inspect] Cancelled; completed cache remains resumable")
            _atomic_json(cache_path(job.project_root), cache)
            return
        rel = path.relative_to(dataset).as_posix()
        job.current = rel
        fingerprint = _image_fingerprint(path)
        existing = entries.get(rel) or {}
        if existing.get("fingerprint") == fingerprint and existing.get("config") == config and not existing.get("error"):
            job.cached += 1
        else:
            try:
                predictions = predictor(path)  # type: ignore[misc]
                routed = filter_and_route_predictions(predictions, classification, job.segments, sensitivity=job.sensitivity)
                entries[rel] = {
                    "image_relative_path": rel,
                    "fingerprint": fingerprint,
                    "config": config,
                    "segments": routed,
                    "updated_at": _now(),
                }
                job.inferred += 1
                job.eligible += sum(len(rows) for rows in routed.values())
                _atomic_json(cache_path(job.project_root), cache)
            except Exception as exc:
                entries[rel] = {"image_relative_path": rel, "fingerprint": fingerprint, "config": config, "error": str(exc), "segments": {}}
                job.logs.append(f"[inspect] [ERROR] {rel}: {exc}")
                _atomic_json(cache_path(job.project_root), cache)
        job.processed += 1
        job.logs.append(f"[inspect] {job.processed}/{job.total} {rel}")
    job.current = ""
    job.eligible = sum(
        len(rows)
        for entry in entries.values()
        if isinstance(entry, dict) and entry.get("config") == config
        for rows in (entry.get("segments") or {}).values()
    )
    _atomic_json(cache_path(job.project_root), cache)
    job.status = "completed"
    job.phase = "completed"
    job.logs.extend(["[inspect] Completed", f"[suggest] {job.eligible} eligible recommendations cached"])


def _run_job(job: InspectionJob) -> None:
    try:
        inspect_dataset_sync(job)
    except Exception as exc:
        job.status = "failed"
        job.phase = "failed"
        job.error = f"Context suggestions unavailable: {exc}. Manual tagging is still available."


def start_inspection(project_root: Path, profile_key: str = CAFORMER_PROFILE, sensitivity: str = "normal", segments: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    profile = get_model_profile(profile_key)
    root = Path(project_root).resolve()
    if not (root / "dataset").is_dir():
        return {"ok": False, "error": f"Dataset folder not found: {root / 'dataset'}"}
    sensitivity = str(sensitivity or "normal").lower()
    if sensitivity not in SENSITIVITY_MULTIPLIERS:
        sensitivity = "normal"
    job = InspectionJob(uuid.uuid4().hex, root, profile.key, sensitivity, list(segments or []))
    with _LOCK:
        previous_id = _LATEST_BY_PROJECT.get(str(root))
        previous = _JOBS.get(previous_id or "")
        if previous and previous.status in {"queued", "running"}:
            return {"ok": True, "job": previous.payload(), "already_running": True}
        _JOBS[job.job_id] = job
        _LATEST_BY_PROJECT[str(root)] = job.job_id
    threading.Thread(target=_run_job, args=(job,), name=f"ContextInspection-{job.job_id[:8]}", daemon=True).start()
    return {"ok": True, "job": job.payload()}


def inspection_status(job_id: str = "", project_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    with _LOCK:
        resolved_id = str(job_id or "")
        if not resolved_id and project_root is not None:
            resolved_id = _LATEST_BY_PROJECT.get(str(Path(project_root).resolve()), "")
        job = _JOBS.get(resolved_id)
    return job.payload() if job else None


def cancel_inspection(job_id: str) -> bool:
    with _LOCK:
        job = _JOBS.get(str(job_id or ""))
    if not job or job.status not in {"queued", "running"}:
        return False
    job.cancel_event.set()
    return True


def suggestions_for_image(
    project_root: Path,
    image_rel: str,
    *,
    expected_profile: str = "",
    expected_sensitivity: str = "",
) -> Dict[str, Any]:
    rel = str(image_rel or "").replace("\\", "/").lstrip("/")
    parts = [part for part in rel.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        return {"ok": False, "error": "Invalid image path"}
    # Guided Tagging sessions store paths relative to the project root
    # (dataset/example.png), while inspection entries are relative to dataset/
    # itself (example.png). Normalize both API forms to the cache key.
    if parts and parts[0].lower() == "dataset":
        parts = parts[1:]
    rel = "/".join(parts)
    if not rel:
        return {"ok": False, "error": "Invalid image path"}
    cache = _load_cache(Path(project_root))
    entry = (cache.get("entries") or {}).get(rel)
    entry_config = entry.get("config") if isinstance(entry, dict) else {}
    stale = bool(
        entry
        and (
            (expected_profile and entry_config.get("model_profile") != expected_profile)
            or (expected_sensitivity and entry_config.get("sensitivity") != expected_sensitivity)
        )
    )
    return {
        "ok": True,
        "ready": bool(entry) and not stale,
        "stale": stale,
        "entry": None if stale else (entry or None),
        "config": cache.get("config") or {},
    }
