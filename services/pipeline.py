import json
import os
import shutil
import threading
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from utils.io import readable_path, readable_path_or_none
from utils.parse import parse_bool, parse_int, parse_float, parse_optional_int, parse_exts
from utils.tool_result import unpack_tool_result
from utils.tags import tag_compare_key
from services import (
    batch_adjust,
    combine_datasets,
    group_renamer,
    merge_groups_tool,
    normalizer,
    offline_tagger,
    tag_editor,
    webp_converter,
    webtoon_splitter,
)
from services.pipeline_workflows import (
    WORKFLOW_CATALOG,
    WORKFLOW_SCHEMA_VERSION,
    compile_workflow,
    plan_items_for_steps,
)

PIPELINE_STATUSES = {
    "QUEUED",
    "RUNNING",
    "WAITING_MANUAL",
    "COMPLETED",
    "FAILED",
    "STOPPED",
}

DEFAULT_STEP_ORDER = [
    "offline_tagger",
    "tag_editor",
    "normalize",
    "zip_final",
]

STEP_LABELS = {
    "offline_tagger": "Offline tagger (WD v3)",
    "tag_editor": "Dataset Tag Editor",
    "manual_review": "Manual caption review",
    "normalize": "Dataset normalization",
    "dataset_audit": "Dataset audit",
    "export_final": "Export final dataset",
    "zip_final": "Zip result",
    "dedup_tags": "Tag cleanup (dedup)",
    "webp_to_png": "Image -> PNG",
    "batch_adjust": "Photo adjust (preset)",
    "combine_datasets": "Combine dataset",
    "flatten_renumber": "Flatten & renumber",
    "merge_groups": "Stitch groups",
    "webtoon_split": "Webtoon panel splitter",
}

TAG_STEPS = {"offline_tagger", "tag_editor", "manual_review", "normalize", "dedup_tags", "dataset_audit", "export_final"}
IMAGE_ONLY_STEPS = {"webp_to_png", "batch_adjust", "merge_groups", "webtoon_split", "flatten_renumber"}
STRUCTURAL_STEPS = {"webtoon_split", "merge_groups", "flatten_renumber", "combine_datasets"}
EDITABLE_STEPS = {
    "offline_tagger",
    "tag_editor",
    "manual_review",
    "normalize",
    "dedup_tags",
    "webp_to_png",
    "batch_adjust",
    "combine_datasets",
    "flatten_renumber",
    "merge_groups",
    "webtoon_split",
}
STEP_STAGE = {
    "webtoon_split": 10,
    "merge_groups": 10,
    "flatten_renumber": 10,
    "combine_datasets": 10,
    "webp_to_png": 20,
    "batch_adjust": 20,
    "offline_tagger": 30,
    "tag_editor": 40,
    "manual_review": 40,
    "normalize": 50,
    "dedup_tags": 50,
    "dataset_audit": 60,
    "zip_final": 70,
    "export_final": 70,
}
IMAGE_EXTENSIONS_DEFAULT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VALID_COPY_MODES = {"copy", "hardlink", "incremental"}
PERSIST_INTERVAL_RUNNING_SEC = 0.75


@dataclass
class StepResult:
    status: str  # SUCCESS | WAIT | FAIL | STOP
    message: str = ""
    wait_status: Optional[str] = None  # WAITING_MANUAL


@dataclass
class PipelineJob:
    id: str
    created_at: float
    status: str
    current_step: str
    step_index: int
    steps: List[str]
    config: Dict[str, Any]
    log: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    waiting_reason: str = ""
    log_seq: int = 0

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["created_at"] = datetime.fromtimestamp(self.created_at).isoformat()
        return out


def _now_ts() -> float:
    return time.time()


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _zipdir(
    src: Path,
    dest_zip: Path,
    include_txt: bool = True,
    include_images_only: bool = False,
    should_stop: Optional[Callable[[], bool]] = None,
):
    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src):
            if should_stop and should_stop():
                raise InterruptedError("Stop requested while creating zip.")
            for fname in files:
                if should_stop and should_stop():
                    raise InterruptedError("Stop requested while creating zip.")
                p = Path(root) / fname
                rel_parts = [part.lower() for part in p.relative_to(src).parts]
                if "_temp" in rel_parts or p.suffix.lower() == ".bak" or p.name.lower() == "state.json":
                    continue
                if p.name.startswith("."):
                    continue
                if include_images_only and p.suffix.lower() in {".txt", ".json", ".md"}:
                    continue
                if include_txt is False and p.suffix.lower() == ".txt":
                    continue
                arcname = str(p.relative_to(src))
                zf.write(p, arcname)


def _coerce_copy_mode(raw: Any) -> str:
    val = str(raw or "copy").strip().lower()
    if val in {"link", "hardlink"}:
        return "hardlink"
    if val in {"incremental", "sync", "incremental_copy"}:
        return "incremental"
    return "copy"


def _same_file_meta(src: Path, dst: Path) -> bool:
    try:
        s = src.stat()
        d = dst.stat()
    except Exception:
        return False
    return s.st_size == d.st_size and int(s.st_mtime) == int(d.st_mtime)


def _transfer_file(src: Path, dst: Path, mode: str, incremental: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if incremental and dst.exists() and _same_file_meta(src, dst):
        return "skipped"

    if mode == "hardlink":
        if dst.exists():
            try:
                dst.unlink()
            except Exception:
                shutil.copy2(src, dst)
                return "copied"
        try:
            os.link(src, dst)
            return "linked"
        except Exception:
            shutil.copy2(src, dst)
            return "copied"

    shutil.copy2(src, dst)
    return "copied"


def _copy_dataset_fast(
    src: Path,
    dest: Path,
    image_exts: List[str],
    recursive: bool,
    copy_mode: str = "copy",
    incremental: bool = False,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, int]:
    mode = _coerce_copy_mode(copy_mode)
    if mode == "incremental":
        mode = "copy"
        incremental = True
    _ensure_dir(dest)

    copied = 0
    linked = 0
    skipped = 0
    errors = 0
    images_seen = 0
    txt_seen = 0
    stopped = 0

    def _eligible(path: Path) -> bool:
        return path.is_file() and (path.suffix.lower() in image_exts or path.suffix.lower() == ".txt")

    iterator = src.rglob("*") if recursive else src.iterdir()
    for p in iterator:
        if should_stop and should_stop():
            stopped = 1
            break
        if not _eligible(p):
            continue
        if p.suffix.lower() == ".txt":
            txt_seen += 1
        else:
            images_seen += 1
        rel = p.relative_to(src) if recursive else Path(p.name)
        dst = dest / rel
        try:
            result = _transfer_file(p, dst, mode=mode, incremental=incremental)
            if result == "copied":
                copied += 1
            elif result == "linked":
                linked += 1
            else:
                skipped += 1
        except Exception:
            errors += 1

    return {
        "copied": copied,
        "linked": linked,
        "skipped": skipped,
        "errors": errors,
        "images_seen": images_seen,
        "txt_seen": txt_seen,
        "stopped": stopped,
    }


def _find_images(folder: Path, recursive: bool, exts: List[str]) -> List[Path]:
    exts_lower = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    if recursive:
        return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts_lower])
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts_lower])


def _coerce_exts(raw: Any) -> List[str]:
    return parse_exts(raw, default=normalizer.DEFAULT_IMAGE_EXTS)


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except Exception:
        return path.absolute()


def _same_path(a: Path, b: Path) -> bool:
    ar = str(_safe_resolve(a))
    br = str(_safe_resolve(b))
    if os.name == "nt":
        return ar.lower() == br.lower()
    return ar == br


def _path_contains(parent: Path, child: Path) -> bool:
    parent_r = _safe_resolve(parent)
    child_r = _safe_resolve(child)
    try:
        child_r.relative_to(parent_r)
        return not _same_path(parent_r, child_r)
    except Exception:
        return False


def _split_tags_for_audit(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text or "").replace("\n", ",").split(","):
        tag = raw.strip()
        if tag:
            out.append(tag)
    return out


def _scan_dataset_quality(
    root: Path,
    image_exts: List[str],
    recursive: bool,
    require_caption_pairs: bool,
    include_token_check: bool,
    token_warning_limit: int,
) -> Dict[str, Any]:
    images = _find_images(root, recursive=recursive, exts=image_exts) if root.exists() else []
    txts = sorted(root.rglob("*.txt") if recursive else root.glob("*.txt")) if root.exists() else []
    txt_by_rel = {p.relative_to(root).with_suffix("").as_posix().lower(): p for p in txts if "_temp" not in [x.lower() for x in p.relative_to(root).parts]}
    image_stems = {
        img.relative_to(root).with_suffix("").as_posix().lower(): img
        for img in images
        if "_temp" not in [x.lower() for x in img.relative_to(root).parts]
    }
    valid_pairs: List[str] = []
    missing_txt: List[str] = []
    orphan_txt: List[str] = []
    empty_txt: List[str] = []
    duplicate_tags: List[Dict[str, Any]] = []
    malformed_tags: List[str] = []
    token_warnings: List[Dict[str, Any]] = []
    failed_reads: List[Dict[str, str]] = []

    for rel, img in image_stems.items():
        txt = img.with_suffix(".txt")
        if txt.exists():
            valid_pairs.append(img.relative_to(root).as_posix())
        else:
            missing_txt.append(img.relative_to(root).as_posix())

    for rel, txt in txt_by_rel.items():
        if rel not in image_stems:
            orphan_txt.append(txt.relative_to(root).as_posix())
        try:
            content = txt.read_text(encoding="utf-8-sig")
        except Exception as exc:
            failed_reads.append({"path": txt.relative_to(root).as_posix(), "error": str(exc)})
            continue
        if not content.strip():
            empty_txt.append(txt.relative_to(root).as_posix())
        raw_parts = str(content).replace("\n", ",").split(",")
        tags = [part.strip() for part in raw_parts if part.strip()]
        if any(part != part.strip() for part in raw_parts if part):
            malformed_tags.append(txt.relative_to(root).as_posix())
        counts = Counter(tag_compare_key(tag) for tag in tags)
        dupes = [key.replace("_", " ") for key, count in counts.items() if count > 1]
        if dupes:
            duplicate_tags.append({"path": txt.relative_to(root).as_posix(), "tags": dupes[:20]})
        if include_token_check and token_warning_limit > 0 and len(tags) > token_warning_limit:
            token_warnings.append(
                {
                    "path": txt.relative_to(root).as_posix(),
                    "tokens": len(tags),
                    "limit": token_warning_limit,
                }
            )

    temp_files = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and "_temp" in [part.lower() for part in p.relative_to(root).parts]
    ] if root.exists() else []
    bak_files = [p.relative_to(root).as_posix() for p in root.rglob("*.bak")] if root.exists() else []
    stems_by_name: Dict[str, List[str]] = defaultdict(list)
    for img in images:
        stems_by_name[img.stem.lower()].append(img.relative_to(root).as_posix())
    duplicate_stems = {stem: vals for stem, vals in stems_by_name.items() if len(vals) > 1}

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    info: List[Dict[str, Any]] = [
        {"severity": "info", "code": "totals", "message": f"Images: {len(image_stems)} | captions: {len(txt_by_rel)} | valid pairs: {len(valid_pairs)}"}
    ]
    if require_caption_pairs and missing_txt:
        errors.append({"severity": "error", "code": "missing_captions", "message": f"{len(missing_txt)} image(s) missing .txt captions", "items": missing_txt[:200]})
    if temp_files:
        errors.append({"severity": "error", "code": "temp_files", "message": f"{len(temp_files)} file(s) remain in _temp", "items": temp_files[:200]})
    if failed_reads:
        errors.append({"severity": "error", "code": "failed_reads", "message": f"{len(failed_reads)} caption file(s) could not be read", "items": failed_reads[:50]})
    if orphan_txt:
        warnings.append({"severity": "warning", "code": "orphan_txt", "message": f"{len(orphan_txt)} orphan .txt file(s)", "items": orphan_txt[:200]})
    if empty_txt:
        level = "error" if require_caption_pairs else "warning"
        (errors if level == "error" else warnings).append({"severity": level, "code": "empty_captions", "message": f"{len(empty_txt)} empty caption file(s)", "items": empty_txt[:200]})
    if duplicate_tags:
        warnings.append({"severity": "warning", "code": "duplicate_tags", "message": f"{len(duplicate_tags)} caption(s) contain duplicate tags", "items": duplicate_tags[:50]})
    if malformed_tags:
        warnings.append({"severity": "warning", "code": "malformed_tags", "message": f"{len(malformed_tags)} caption(s) contain extra whitespace around tags", "items": malformed_tags[:200]})
    if bak_files:
        warnings.append({"severity": "warning", "code": "bak_files", "message": f"{len(bak_files)} .bak file(s) found", "items": bak_files[:200]})
    if duplicate_stems:
        warnings.append({"severity": "warning", "code": "duplicate_stems", "message": f"{len(duplicate_stems)} duplicate filename stem(s) across folders", "items": dict(list(duplicate_stems.items())[:50])})
    if token_warnings:
        warnings.append({"severity": "warning", "code": "token_limit", "message": f"{len(token_warnings)} caption(s) exceed token warning limit", "items": token_warnings[:50]})

    return {
        "ok": not errors,
        "summary": {
            "total_images": len(image_stems),
            "total_txt": len(txt_by_rel),
            "valid_pairs": len(valid_pairs),
            "missing_txt": len(missing_txt),
            "orphan_txt": len(orphan_txt),
            "empty_txt": len(empty_txt),
            "temp_files": len(temp_files),
            "bak_files": len(bak_files),
            "duplicate_stems": len(duplicate_stems),
            "token_warnings": len(token_warnings),
        },
        "errors": errors,
        "warnings": warnings,
        "info": info,
    }


def _cfg_fingerprint_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Exclude runtime-only keys that should not affect dedupe checks.
    runtime_keys = {"fingerprint", "manual_resume_index", "manual_pending_index"}
    return {k: v for k, v in cfg.items() if k not in runtime_keys}


def _parse_bool(val: Any) -> bool:
    return parse_bool(val, default=False)


def _parse_int(val: Any, default: int) -> int:
    return parse_int(val, default=default)


def _parse_float(val: Any, default: float) -> float:
    parsed = parse_float(val, default=default)
    return parsed if parsed is not None else default


def _parse_optional_int(val: Any) -> Optional[int]:
    return parse_optional_int(val)


def _optional_path(raw: Any) -> Optional[Path]:
    return readable_path_or_none(raw)


def _required_path(raw: Any, field_name: str) -> Tuple[Optional[Path], Optional[str]]:
    path = readable_path_or_none(raw)
    if not path:
        return None, f"{field_name} required"
    return path, None


def _dedup_list(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in items:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _dedup_txt_folder(folder: Path):
    txts = list(folder.rglob("*.txt"))
    for path in txts:
        tf = normalizer.parse_tag_file(path)
        tf.main = _dedup_list(tf.main)
        tf.optional = _dedup_list(tf.optional)
        content = normalizer.format_tag_file(tf)
        path.write_text(content, encoding="utf-8")


class PipelineManager:
    def __init__(self):
        self.jobs: Dict[str, PipelineJob] = {}
        self.last_persist_ts: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.state_root = Path("_work") / "pipeline_jobs"
        _ensure_dir(self.state_root)
        self.step_runners = {
            "offline_tagger": self._step_autotag_offline,
            "tag_editor": self._step_tag_editor,
            "manual_review": self._step_manual_review,
            "normalize": self._step_normalize,
            "dataset_audit": self._step_dataset_audit,
            "export_final": self._step_export_final,
            "zip_final": self._step_zip_final,
            "dedup_tags": self._step_final_cleanup,
            "webp_to_png": self._step_webp_to_png,
            "batch_adjust": self._step_batch_adjust,
            "combine_datasets": self._step_combine_datasets,
            "flatten_renumber": self._step_flatten_renumber,
            "merge_groups": self._step_merge_groups,
            "webtoon_split": self._step_webtoon_split,
        }
        self._restore_jobs_from_disk()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    # ---------- persistence ----------
    def _job_state_path(self, job: PipelineJob) -> Path:
        return self.state_root / job.id / "state.json"

    def _job_workspace(self, job: PipelineJob) -> Optional[Path]:
        working, err = _required_path(job.config.get("working_dir"), "working_dir")
        if err or not working:
            return None
        return working / "jobs" / job.id

    def _persist_job(self, job: PipelineJob, force: bool = False):
        now = _now_ts()
        last = self.last_persist_ts.get(job.id, 0.0)
        if not force and job.status == "RUNNING" and (now - last) < PERSIST_INTERVAL_RUNNING_SEC:
            return
        path = self._job_state_path(job)
        _ensure_dir(path.parent)
        try:
            path.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
            self.last_persist_ts[job.id] = now
        except Exception:
            pass

    # ---------- job helpers ----------
    def _add_log(self, job: PipelineJob, message: str, level: str = "info"):
        job.log_seq += 1
        job.log.append(
            {"id": job.log_seq, "ts": datetime.utcnow().isoformat(), "level": level, "message": message}
        )
        job.log = job.log[-500:]

    def _state_candidates(self) -> List[Path]:
        candidates: List[Path] = []
        try:
            candidates.extend([p for p in self.state_root.glob("*/state.json") if p.exists()])
        except Exception:
            pass

        # Legacy fallback for old single-file states.
        roots = [Path("_work"), Path(".")]
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            direct = root / ".pipeline_job.json"
            if direct.exists():
                candidates.append(direct)
            try:
                first_level = [p for p in root.iterdir() if p.is_dir()]
            except Exception:
                first_level = []
            for child in first_level:
                state = child / ".pipeline_job.json"
                if state.exists():
                    candidates.append(state)
                try:
                    second_level = [p for p in child.iterdir() if p.is_dir()]
                except Exception:
                    second_level = []
                for grand in second_level:
                    deep_state = grand / ".pipeline_job.json"
                    if deep_state.exists():
                        candidates.append(deep_state)
        # preserve newest state files first
        uniq = {}
        for path in candidates:
            uniq[str(path.resolve())] = path
        return sorted(uniq.values(), key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)

    def _restore_jobs_from_disk(self):
        for path in self._state_candidates():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            job_id = str(raw.get("id") or "").strip()
            if not job_id or job_id in self.jobs:
                continue
            created_at_raw = raw.get("created_at")
            created_at = _now_ts()
            if isinstance(created_at_raw, (int, float)):
                created_at = float(created_at_raw)
            elif isinstance(created_at_raw, str) and created_at_raw.strip():
                try:
                    created_at = datetime.fromisoformat(created_at_raw).timestamp()
                except Exception:
                    created_at = _now_ts()

            status = str(raw.get("status") or "WAITING_MANUAL").strip().upper()
            if status in {"RUNNING", "QUEUED"}:
                status = "WAITING_MANUAL"

            steps = [str(x) for x in (raw.get("steps") or []) if str(x).strip()]
            config = raw.get("config") if isinstance(raw.get("config"), dict) else {}
            artifacts = raw.get("artifacts") if isinstance(raw.get("artifacts"), dict) else {}
            waiting_reason = str(raw.get("waiting_reason") or "")
            if status == "WAITING_MANUAL" and not waiting_reason:
                waiting_reason = "Recovered after restart. Review and resume when ready."
            logs_raw = raw.get("log") if isinstance(raw.get("log"), list) else []
            logs: List[Dict[str, Any]] = []
            max_log_id = 0
            for item in logs_raw[-500:]:
                if not isinstance(item, dict):
                    continue
                log_id = item.get("id")
                if not isinstance(log_id, int):
                    max_log_id += 1
                    log_id = max_log_id
                else:
                    max_log_id = max(max_log_id, log_id)
                logs.append(
                    {
                        "id": int(log_id),
                        "ts": str(item.get("ts") or datetime.utcnow().isoformat()),
                        "level": str(item.get("level") or "info"),
                        "message": str(item.get("message") or ""),
                    }
                )

            job = PipelineJob(
                id=job_id,
                created_at=created_at,
                status=status if status in PIPELINE_STATUSES else "WAITING_MANUAL",
                current_step=str(raw.get("current_step") or ""),
                step_index=max(0, int(raw.get("step_index") or 0)),
                steps=steps,
                config=config,
                log=logs,
                artifacts=artifacts,
                waiting_reason=waiting_reason,
                log_seq=max_log_id,
            )
            self.jobs[job.id] = job
            self._add_log(job, f"Recovered pipeline state from: {path}")
            self._persist_job(job, force=True)

    def _normalize_steps(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw = cfg.get("steps") or []
        if not raw:
            raw = [{"id": step_id, "config": {}} for step_id in DEFAULT_STEP_ORDER]

        steps: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, str):
                step_id = item
                step_cfg: Dict[str, Any] = {}
            elif isinstance(item, dict):
                step_id = (item.get("id") or item.get("step") or "").strip()
                step_cfg = item.get("config") if isinstance(item.get("config"), dict) else {}
            else:
                continue
            if step_id not in self.step_runners:
                continue
            steps.append({"id": step_id, "config": step_cfg})

        if not steps:
            steps = [{"id": step_id, "config": {}} for step_id in DEFAULT_STEP_ORDER if step_id in self.step_runners]

        return steps

    def _validate_step_order(self, steps: List[Dict[str, Any]]) -> Optional[str]:
        ids = [item.get("id") for item in steps if item.get("id")]
        if not ids:
            return "At least one step is required."
        export_ids = {"zip_final", "export_final"}
        for export_id in export_ids:
            if export_id in ids and ids[-1] != export_id:
                return "Final export must be the last step so the output is always up to date."
        if "export_final" in ids:
            if "dataset_audit" not in ids:
                return "Dataset Audit must run before Final Export."
            if ids.index("dataset_audit") > ids.index("export_final"):
                return "Dataset Audit must run before Final Export."
        if "offline_tagger" in ids and "normalize" in ids and ids.index("offline_tagger") > ids.index("normalize"):
            return "Offline Tagger must run before Dataset Normalization."
        if "manual_review" in ids and "normalize" in ids and ids.index("manual_review") > ids.index("normalize"):
            return "Manual Review must happen before Dataset Normalization."
        if "offline_tagger" in ids and "manual_review" in ids and ids.index("offline_tagger") > ids.index("manual_review"):
            return "Manual Review should happen after Auto-tagging."
        first_tag = None
        last_image_step = None
        for idx, step_id in enumerate(ids):
            if step_id in TAG_STEPS and first_tag is None:
                first_tag = idx
            if step_id in IMAGE_ONLY_STEPS:
                last_image_step = idx
        if first_tag is not None and last_image_step is not None and last_image_step > first_tag:
            offenders = [
                step_id
                for idx, step_id in enumerate(ids)
                if step_id in IMAGE_ONLY_STEPS and idx > first_tag
            ]
            labels = ", ".join(STEP_LABELS.get(step_id, step_id) for step_id in offenders)
            return (
                "Image-only steps must be placed before tag steps "
                f"(Offline Tagger / Tag Editor / Normalization / Tag Cleanup). "
                f"Move: {labels}."
            )
        previous_stage = -1
        for step_id in ids:
            stage = STEP_STAGE.get(step_id, previous_stage)
            if stage < previous_stage:
                return f"{STEP_LABELS.get(step_id, step_id)} is out of order for a safe dataset workflow."
            previous_stage = stage
        return None

    def _prepare_guided_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        workflow_id = str(cfg.get("workflow_id") or "").strip()
        if not workflow_id:
            return dict(cfg)
        guided = compile_workflow(workflow_id, cfg.get("workflow_options") or {})
        merged = dict(cfg)
        merged.update(guided)
        merged["workflow_id"] = guided.get("workflow_id")
        merged["schema_version"] = WORKFLOW_SCHEMA_VERSION
        return merged

    def _validate_paths_for_start(self, dataset: Path, output: Path, working: Path) -> Optional[str]:
        if _same_path(dataset, output):
            return "Source folder and final output folder must be different."
        if _path_contains(dataset, output):
            return "Final output folder cannot be inside the source folder."
        if _path_contains(output, dataset):
            return "Source folder cannot be inside the final output folder."
        workspace_root = working / "jobs"
        if _same_path(output, working) or _path_contains(working, output) or _path_contains(workspace_root, output):
            return "Final output folder cannot be inside the pipeline workspace."
        return None

    def plan_workflow(self, cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
        cfg = self._prepare_guided_config(dict(cfg))
        dataset, err = _required_path(cfg.get("dataset_path"), "dataset_path")
        if err or not dataset:
            return False, {}, err or "dataset_path required"
        working, err = _required_path(cfg.get("working_dir"), "working_dir")
        if err or not working:
            return False, {}, err or "working_dir required"
        output, err = _required_path(cfg.get("output_dir"), "output_dir")
        if err or not output:
            return False, {}, err or "output_dir required"
        if not dataset.exists():
            return False, {}, f"Dataset not found: {dataset}"
        if not dataset.is_dir():
            return False, {}, f"Dataset is not a folder: {dataset}"
        path_err = self._validate_paths_for_start(dataset, output, working)
        if path_err:
            return False, {}, path_err
        cfg["steps"] = self._normalize_steps(cfg)
        order_err = self._validate_step_order(cfg["steps"])
        if order_err:
            return False, {}, order_err

        warnings: List[str] = []
        image_exts = _coerce_exts(cfg.get("image_exts"))
        recursive = bool(cfg.get("recursive"))
        images = _find_images(dataset, recursive=recursive, exts=image_exts)
        if not images:
            warnings.append("No images found in the selected source folder.")
        workflow_id = str(cfg.get("workflow_id") or "")
        if workflow_id == "captioned_dataset":
            scan = _scan_dataset_quality(dataset, image_exts, recursive, True, True, int(cfg.get("token_warning_limit") or 77))
            missing = scan.get("summary", {}).get("missing_txt", 0)
            if missing:
                warnings.append(f"{missing} image(s) are missing .txt captions.")
            empty = scan.get("summary", {}).get("empty_txt", 0)
            if empty:
                warnings.append(f"{empty} caption file(s) are empty.")
        if workflow_id == "image_preparation":
            warnings.append("Image-only workflow will not create captions.")
        if any(item.get("id") == "normalize" for item in cfg["steps"]):
            preset_root = _optional_path(cfg.get("preset_root")) or Path("presets")
            for item in cfg["steps"]:
                if item.get("id") != "normalize":
                    continue
                scfg = item.get("config") or {}
                ptype = str(scfg.get("preset_type") or cfg.get("preset_type") or "anime")
                pfile = str(scfg.get("preset_file") or cfg.get("preset_file") or "")
                if not pfile:
                    return False, {}, "Normalization preset file is required."
                try:
                    normalizer.load_preset(preset_root, ptype, pfile)
                except Exception as exc:
                    return False, {}, f"Normalization preset is not available: {exc}"

        plan = plan_items_for_steps(cfg["steps"], str(output))
        return True, {"ok": True, "workflow_id": cfg.get("workflow_id") or "custom", "plan": plan, "warnings": warnings, "steps": cfg["steps"]}, ""

    def start_job(self, cfg: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
        cfg = self._prepare_guided_config(dict(cfg))
        dataset, err = _required_path(cfg.get("dataset_path"), "dataset_path")
        if err or not dataset:
            return False, "", err
        working, err = _required_path(cfg.get("working_dir"), "working_dir")
        if err or not working:
            return False, "", err
        output, err = _required_path(cfg.get("output_dir"), "output_dir")
        if err or not output:
            return False, "", err
        if not dataset.exists():
            return False, "", f"Dataset not found: {dataset}"
        if not dataset.is_dir():
            return False, "", f"Dataset is not a folder: {dataset}"
        path_err = self._validate_paths_for_start(dataset, output, working)
        if path_err:
            return False, "", path_err

        cfg = dict(cfg)
        cfg["dataset_path"] = str(dataset)
        cfg["working_dir"] = str(working)
        cfg["output_dir"] = str(output)
        cfg["steps"] = self._normalize_steps(cfg)
        order_err = self._validate_step_order(cfg["steps"])
        if order_err:
            return False, "", order_err
        if _coerce_copy_mode(cfg.get("copy_mode")) == "hardlink" and any(
            (step.get("id") in EDITABLE_STEPS) for step in cfg["steps"]
        ):
            cfg["copy_mode"] = "copy"
            cfg["copy_mode_forced_warning"] = (
                "Hardlink mode was changed to copy because this workflow edits files. "
                "Original dataset protection is enforced."
            )

        cfg_fingerprint = json.dumps(_cfg_fingerprint_payload(cfg), sort_keys=True, default=str)
        job_id = str(uuid4())
        cfg["fingerprint"] = cfg_fingerprint
        steps = self._build_step_names(cfg)
        job = PipelineJob(
            id=job_id,
            created_at=_now_ts(),
            status="QUEUED",
            current_step=steps[0] if steps else "",
            step_index=0,
            steps=steps,
            config=cfg,
        )
        job.config["stop_requested"] = False
        self._add_log(job, "Job created.")
        if cfg.get("copy_mode_forced_warning"):
            self._add_log(job, str(cfg.get("copy_mode_forced_warning")), level="warning")
        with self.lock:
            for existing in self.jobs.values():
                existing_fingerprint = existing.config.get("fingerprint")
                if existing_fingerprint and existing_fingerprint == cfg_fingerprint:
                    if existing.status in {"QUEUED", "RUNNING", "WAITING_MANUAL"}:
                        return False, "", f"A similar pipeline job is already active: {existing.id}"
            self.jobs[job.id] = job
        self.wake.set()
        self._persist_job(job, force=True)
        return True, job_id, None

    def pause_job(self, job_id: str) -> Tuple[bool, str]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False, "job not found"
            if job.status in {"COMPLETED", "FAILED", "STOPPED"}:
                return False, f"cannot pause from {job.status}"
            if job.status == "WAITING_MANUAL":
                return True, "already paused"
            job.status = "WAITING_MANUAL"
            job.waiting_reason = "Paused by user"
            self._add_log(job, "Paused by user.")
            self._persist_job(job, force=True)
        return True, "paused"

    def stop_job(self, job_id: str) -> Tuple[bool, str]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False, "job not found"
            if job.status == "STOPPED":
                return True, "already stopped"
            if job.status in {"COMPLETED", "FAILED"}:
                return False, f"cannot stop from {job.status}"
            job.config["stop_requested"] = True
            job.status = "STOPPED"
            self._add_log(job, "Stopped by user.")
            self._persist_job(job, force=True)
        return True, "stopped"

    def resume_job(self, job_id: str) -> Tuple[bool, str]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False, "job not found"
            if job.status in {"COMPLETED", "FAILED"}:
                return False, f"cannot resume from {job.status}"
            if job.status != "WAITING_MANUAL":
                return False, f"cannot resume from {job.status}"
            pending = job.config.get("manual_pending_index")
            if pending == job.step_index:
                job.config["manual_resume_index"] = pending
            job.config["stop_requested"] = False
            job.status = "QUEUED"
            job.waiting_reason = ""
            self._add_log(job, "Resumed.")
            self._persist_job(job, force=True)
        self.wake.set()
        return True, "resumed"

    def get_status(
        self, job_id: Optional[str] = None, since_log_id: Optional[int] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        with self.lock:
            if not job_id and self.jobs:
                # return latest job
                job_id = sorted(self.jobs.values(), key=lambda j: j.created_at)[-1].id
            job = self.jobs.get(job_id or "")
            if not job:
                return False, None, "job not found"
            data = job.to_dict()
            data["last_log_id"] = job.log_seq
            if since_log_id is not None and since_log_id >= 0:
                data["log"] = [item for item in job.log if int(item.get("id") or 0) > since_log_id]
                data["log_delta"] = True
            else:
                data["log_delta"] = False
        return True, data, ""

    def get_review_context(self, job_id: str) -> Tuple[bool, Dict[str, Any], str]:
        with self.lock:
            job = self.jobs.get(str(job_id or ""))
            if not job:
                return False, {}, "job not found"
            artifacts = dict(job.artifacts or {})
            ready = (
                job.status == "WAITING_MANUAL"
                and bool(artifacts.get("review_project_root"))
                and bool(artifacts.get("review_dataset_root"))
            )
            return True, {
                "job_id": job.id,
                "status": job.status,
                "review_project_root": artifacts.get("review_project_root") or "",
                "review_dataset_path": artifacts.get("review_dataset_root") or "",
                "review_temp_root": artifacts.get("review_temp_root") or "",
                "manual_step_message": job.waiting_reason or "",
                "ready_to_resume": ready,
            }, ""

    # ---------- steps ----------
    def _build_step_names(self, cfg: Dict[str, Any]) -> List[str]:
        steps = self._normalize_steps(cfg)
        return [STEP_LABELS.get(item["id"], item["id"]) for item in steps]

    def _worker(self):
        while True:
            self.wake.wait(timeout=1.0)
            job = self._next_job()
            if not job:
                self.wake.clear()
                continue
            try:
                self._run_job(job)
            except Exception as exc:  # pragma: no cover
                with self.lock:
                    job.status = "FAILED"
                    self._add_log(job, f"Fatal error: {exc}", level="error")
                    self._persist_job(job, force=True)

    def _next_job(self) -> Optional[PipelineJob]:
        with self.lock:
            for job in sorted(self.jobs.values(), key=lambda j: j.created_at):
                if job.status == "QUEUED":
                    return job
        return None

    def _run_job(self, job: PipelineJob):
        if not job.artifacts.get("prepared"):
            with self.lock:
                if self._should_stop(job):
                    self._persist_job(job, force=True)
                    return
                job.status = "RUNNING"
                job.config["stop_requested"] = False
                job.current_step = "Prepare workspace"
                self._persist_job(job)

            result = self._step_validate(job)

            with self.lock:
                if self._should_stop(job):
                    job.status = "STOPPED"
                    self._persist_job(job, force=True)
                    return
                if job.status == "WAITING_MANUAL" and result.status == "SUCCESS":
                    job.artifacts["prepared"] = True
                    if job.steps:
                        safe_idx = max(0, min(job.step_index, len(job.steps) - 1))
                        job.current_step = job.steps[safe_idx]
                    self._persist_job(job, force=True)
                    return
                if result.status == "SUCCESS":
                    job.artifacts["prepared"] = True
                    job.current_step = ""
                    self._persist_job(job)
                elif result.status == "WAIT":
                    job.status = result.wait_status or "WAITING_MANUAL"
                    job.waiting_reason = result.message
                    self._add_log(job, f"Waiting: {result.message}")
                    self._persist_job(job, force=True)
                    return
                elif result.status == "FAIL":
                    job.status = "FAILED"
                    self._add_log(job, result.message, level="error")
                    self._persist_job(job, force=True)
                    return
                elif result.status == "STOP":
                    job.status = "STOPPED"
                    self._add_log(job, "Stopped.")
                    self._persist_job(job, force=True)
                    return

        steps = self._build_steps(job)
        while job.step_index < len(steps):
            step_name, fn = steps[job.step_index]
            with self.lock:
                if self._should_stop(job):
                    job.status = "STOPPED"
                    self._persist_job(job, force=True)
                    return
                job.status = "RUNNING"
                job.config["stop_requested"] = False
                job.current_step = step_name
                self._persist_job(job)

            result = fn(job)

            with self.lock:
                if self._should_stop(job):
                    job.status = "STOPPED"
                    self._persist_job(job, force=True)
                    return
                if job.status == "WAITING_MANUAL" and result.status == "SUCCESS":
                    job.step_index += 1
                    if job.step_index < len(steps):
                        job.current_step = steps[job.step_index][0]
                    else:
                        job.current_step = "Done"
                    self._persist_job(job, force=True)
                    return
                if result.status == "SUCCESS":
                    job.step_index += 1
                    if job.step_index < len(steps):
                        job.current_step = steps[job.step_index][0]
                elif result.status == "WAIT":
                    job.status = result.wait_status or "WAITING_MANUAL"
                    job.waiting_reason = result.message
                    self._add_log(job, f"Waiting: {result.message}")
                    self._persist_job(job, force=True)
                    return
                elif result.status == "FAIL":
                    job.status = "FAILED"
                    self._add_log(job, result.message, level="error")
                    self._persist_job(job, force=True)
                    return
                elif result.status == "STOP":
                    job.status = "STOPPED"
                    self._add_log(job, "Stopped.")
                    self._persist_job(job, force=True)
                    return

                self._persist_job(job)

        with self.lock:
            if self._should_stop(job):
                job.status = "STOPPED"
                self._persist_job(job, force=True)
                return
            if job.status == "WAITING_MANUAL":
                self._persist_job(job, force=True)
                return
            job.status = "COMPLETED"
            job.current_step = "Done"
            self._add_log(job, "Pipeline completed.")
            self._persist_job(job, force=True)

    # Step builders
    def _build_steps(self, job: PipelineJob) -> List[Tuple[str, Any]]:
        steps: List[Tuple[str, Any]] = []
        for item in self._normalize_steps(job.config):
            step_id = item["id"]
            label = STEP_LABELS.get(step_id, step_id)
            runner = self.step_runners.get(step_id)
            if not runner:
                continue
            step_cfg = item.get("config") or {}
            steps.append((label, partial(runner, step_cfg=step_cfg)))
        return steps

    def _resolve_input_dir(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> Optional[Path]:
        override = _optional_path(step_cfg.get("input_dir") if step_cfg else None)
        if override:
            return override
        target = job.artifacts.get("working_copy") or job.config.get("dataset_path")
        return readable_path(str(target)) if target else None

    def _resolve_output_dir(
        self, job: PipelineJob, step_cfg: Dict[str, Any], default_subdir: str
    ) -> Optional[Path]:
        override = _optional_path(step_cfg.get("output_dir") if step_cfg else None)
        if override:
            return override
        workspace = self._job_workspace(job)
        if not workspace:
            return None
        return workspace / default_subdir

    def _resolve_trigger_tag(self, job: PipelineJob, step_cfg: Optional[Dict[str, Any]] = None) -> str:
        if step_cfg:
            raw = (step_cfg.get("trigger_tag") or "").strip()
            if raw:
                return raw
        cfg = job.config
        raw = (cfg.get("trigger_tag") or "").strip()
        if raw:
            return raw
        for item in cfg.get("steps") or []:
            if item.get("id") == "offline_tagger":
                val = (item.get("config") or {}).get("trigger_tag")
                if val:
                    return str(val).strip()
        return ""

    def _log_tool_output(self, job: PipelineJob, log: str):
        if not log:
            return
        with self.lock:
            for line in log.splitlines():
                if line.strip():
                    self._add_log(job, line)

    def _should_stop(self, job: PipelineJob) -> bool:
        return job.status == "STOPPED" or bool(job.config.get("stop_requested"))

    def _run_tool(self, job: PipelineJob, handler, form: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> StepResult:
        if self._should_stop(job):
            return StepResult(status="STOP", message="Stopped.")
        try:
            raw = handler(form, ctx or {})
        except Exception as exc:
            return StepResult(status="FAIL", message=str(exc))
        _, log, meta = unpack_tool_result(raw)
        output_log = log or ""
        self._log_tool_output(job, output_log)
        if self._should_stop(job):
            return StepResult(status="STOP", message="Stopped.")
        if not bool(meta.get("ok", True)):
            message = str(meta.get("error") or "").strip()
            if not message:
                for line in output_log.splitlines():
                    clean = line.strip()
                    if clean:
                        message = clean
                        break
            return StepResult(status="FAIL", message=message or "Tool step failed.")
        artifacts = meta.get("artifacts")
        if isinstance(artifacts, dict) and artifacts:
            with self.lock:
                job.artifacts.update(artifacts)
        return StepResult(status="SUCCESS", message="done")

    # ----- individual steps -----
    def _step_validate(self, job: PipelineJob) -> StepResult:
        cfg = job.config
        dataset, err = _required_path(cfg.get("dataset_path"), "dataset_path")
        if err or not dataset:
            return StepResult(status="FAIL", message=err or "dataset_path required")
        output, err = _required_path(cfg.get("output_dir"), "output_dir")
        if err or not output:
            return StepResult(status="FAIL", message=err or "output_dir required")
        workspace = self._job_workspace(job)
        if not workspace:
            return StepResult(status="FAIL", message="working_dir required")
        image_exts = _coerce_exts(cfg.get("image_exts"))
        recursive = bool(cfg.get("recursive"))
        copy_mode = _coerce_copy_mode(cfg.get("copy_mode"))
        incremental_copy = copy_mode == "incremental" or _parse_bool(cfg.get("incremental_copy"))
        clean_working_raw = cfg.get("clean_working")
        clean_working = (not incremental_copy) if clean_working_raw is None else _parse_bool(clean_working_raw)

        if not dataset.exists():
            return StepResult(status="FAIL", message=f"Dataset not found: {dataset}")
        if not dataset.is_dir():
            return StepResult(status="FAIL", message=f"Dataset is not a folder: {dataset}")

        normalized_dir = workspace / "normalized"
        if clean_working and normalized_dir.exists():
            try:
                shutil.rmtree(normalized_dir)
            except Exception as exc:
                return StepResult(status="FAIL", message=f"Failed cleaning working dir: {exc}")

        for d in [normalized_dir, output]:
            _ensure_dir(d)

        # copy dataset to normalized work area
        copy_stats = _copy_dataset_fast(
            dataset,
            normalized_dir,
            image_exts=image_exts,
            recursive=recursive,
            copy_mode=copy_mode,
            incremental=incremental_copy,
            should_stop=lambda: self._should_stop(job),
        )
        if copy_stats.get("stopped"):
            return StepResult(status="STOP", message="Stopped during workspace preparation.")
        if copy_stats["errors"] > 0:
            return StepResult(
                status="FAIL",
                message=(
                    f"Failed preparing workspace: {copy_stats['errors']} file(s) could not be copied/linked. "
                    "Fix source file errors and retry."
                ),
            )
        if copy_stats.get("images_seen", 0) <= 0:
            return StepResult(status="FAIL", message="No images found in dataset")

        with self.lock:
            job.artifacts.update(
                {
                    "dataset_name": dataset.name,
                    "job_workspace": str(workspace),
                    "normalized_dir": str(normalized_dir),
                    "output_dir": str(output),
                    "working_copy": str(normalized_dir),
                    "copy_mode": "incremental" if incremental_copy else copy_mode,
                    "copy_stats": copy_stats,
                }
            )
            self._add_log(
                job,
                "Prepared working dirs. "
                f"Images: {copy_stats.get('images_seen', 0)} | mode={job.artifacts['copy_mode']} | "
                f"copied={copy_stats['copied']} linked={copy_stats['linked']} "
                f"skipped={copy_stats['skipped']} errors={copy_stats['errors']}",
            )
        return StepResult(status="SUCCESS", message="Prepared")

    def _step_autotag_offline(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        target = self._resolve_input_dir(job, step_cfg)
        if not target or not target.exists():
            return StepResult(status="FAIL", message="No working folder to tag.")
        cfg = job.config
        policy = dict(offline_tagger.TAGGER_POLICY)
        policy["image_exts"] = _coerce_exts(cfg.get("image_exts"))

        form_opts = dict(step_cfg or {})
        form_opts["input_dir"] = str(target)
        form_opts.setdefault("recursive", cfg.get("recursive"))
        deprecated = offline_tagger._find_deprecated_keys(form_opts)
        opts = offline_tagger._effective_opts(form_opts, policy)
        with self.lock:
            self._add_log(job, "Offline autotag started.")
        ok, lines = offline_tagger.run_tagger(opts, deprecated_keys=deprecated)
        with self.lock:
            for line in lines:
                self._add_log(job, line)
        if not ok:
            return StepResult(status="FAIL", message="Offline autotag failed.")
        return StepResult(status="SUCCESS", message="Offline autotag done")

    def _step_normalize(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        cfg = job.config
        target = self._resolve_input_dir(job, step_cfg)
        if not target or not target.exists():
            return StepResult(status="FAIL", message="Nothing to normalize.")
        preset_type = (step_cfg.get("preset_type") or cfg.get("preset_type") or "anime").strip()
        preset_file = (step_cfg.get("preset_file") or cfg.get("preset_file") or "").strip()
        if not preset_file:
            return StepResult(status="FAIL", message="preset_file is required for normalization.")
        trigger_tag = self._resolve_trigger_tag(job, step_cfg)
        opts = normalizer.NormalizeOptions(
            dataset_path=Path(target),
            recursive=bool(cfg.get("recursive")),
            include_missing_txt=_parse_bool(step_cfg.get("include_missing_txt", True)),
            preset_type=preset_type or "anime",
            preset_file=preset_file,
            extra_remove=normalizer.clean_input_list(step_cfg.get("extra_remove") or ""),
            extra_keep=normalizer.clean_input_list(step_cfg.get("extra_keep") or ""),
            move_unknown_background_to_optional=_parse_bool(
                step_cfg.get("move_unknown_background_to_optional")
            ),
            background_threshold=None,
            normalize_order=_parse_bool(step_cfg.get("normalize_order", True)),
            preview_limit=30,
            backup_enabled=_parse_bool(step_cfg.get("backup_enabled", True)),
            image_exts=_coerce_exts(cfg.get("image_exts")),
            identity_tags=normalizer.clean_input_list(step_cfg.get("identity_tags") or ""),
            pinned_tags=[trigger_tag] if trigger_tag else [],
        )
        result = normalizer.apply_normalization(opts, Path(cfg.get("preset_root")))
        if not result.get("ok"):
            return StepResult(status="FAIL", message=result.get("error") or "normalize failed")
        processed = int(result.get("total_files") or result.get("processed") or 0)
        changed = int(result.get("changed_files") or result.get("changed") or 0)
        unchanged = max(0, processed - changed)
        backups = int(result.get("backups_made") or result.get("backups") or 0)
        failed = int(result.get("failed") or 0)
        with self.lock:
            self._add_log(
                job,
                "[Normalize tags] "
                f"Processed {processed} captions | changed: {changed} | unchanged: {unchanged} | "
                f"backups: {backups} | failed: {failed}",
            )
        return StepResult(status="SUCCESS", message="Normalized")

    def _step_manual_pause(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        if job.config.get("manual_resume_index") == job.step_index:
            job.config["manual_resume_index"] = None
            job.config["manual_pending_index"] = None
            with self.lock:
                self._add_log(job, "Manual step complete. Resuming.")
            return StepResult(status="SUCCESS", message="Manual step complete")
        msg = (
            (step_cfg.get("message") or "").strip()
            or "Pause for manual edits. Update files in the working folder, then resume."
        )
        job.config["manual_pending_index"] = job.step_index
        with self.lock:
            self._add_log(job, msg)
        return StepResult(status="WAIT", message=msg, wait_status="WAITING_MANUAL")

    def _copy_tree_for_review(self, src: Path, dst: Path) -> Tuple[int, int, str]:
        files = 0
        errors = 0
        first_error = ""
        for path in src.rglob("*"):
            if self._review_skip_path(path, src):
                continue
            if path.is_dir():
                continue
            rel = path.relative_to(src)
            target = dst / rel
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                files += 1
            except Exception as exc:
                errors += 1
                if not first_error:
                    first_error = f"{rel.as_posix()}: {exc}"
        return files, errors, first_error

    def _review_skip_path(self, path: Path, root: Path) -> bool:
        try:
            rel = path.relative_to(root)
        except Exception:
            return True
        parts = [part.lower() for part in rel.parts]
        return "_temp" in parts or path.name.lower() == "state.json"

    def _move_review_dataset_to_temp(self, dataset_root: Path, temp_root: Path) -> Tuple[int, int, str]:
        moved = 0
        errors = 0
        first_error = ""
        for path in sorted(dataset_root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path == temp_root or temp_root in path.parents:
                continue
            if path.is_dir():
                continue
            rel = path.relative_to(dataset_root)
            dest = temp_root / rel
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    raise FileExistsError(f"temp destination already exists: {dest}")
                shutil.move(str(path), str(dest))
                moved += 1
            except Exception as exc:
                errors += 1
                if not first_error:
                    first_error = f"{rel.as_posix()}: {exc}"
        for directory in sorted([p for p in dataset_root.rglob("*") if p.is_dir() and p != temp_root and temp_root not in p.parents], key=lambda p: len(p.parts), reverse=True):
            try:
                directory.rmdir()
            except Exception:
                pass
        return moved, errors, first_error

    def _restore_review_temp(self, dataset_root: Path, temp_root: Path) -> Tuple[int, int, str]:
        moved = 0
        errors = 0
        first_error = ""
        if not temp_root.exists():
            return 0, 0, ""
        for path in sorted(temp_root.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(temp_root)
            dest = dataset_root / rel
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    raise FileExistsError(f"review restore destination already exists: {dest}")
                shutil.move(str(path), str(dest))
                moved += 1
            except Exception as exc:
                errors += 1
                if not first_error:
                    first_error = f"{rel.as_posix()}: {exc}"
        leftovers = [p.relative_to(temp_root).as_posix() for p in temp_root.rglob("*") if p.is_file()]
        if leftovers and not first_error:
            first_error = f"{len(leftovers)} file(s) remain in _temp: {leftovers[:5]}"
            errors += len(leftovers)
        for directory in sorted([p for p in temp_root.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
            try:
                directory.rmdir()
            except Exception:
                pass
        return moved, errors, first_error

    def _step_manual_review(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        workspace = self._job_workspace(job)
        if not workspace:
            return StepResult(status="FAIL", message="Job workspace is not available for manual review.")
        review_root = workspace / "review_project"
        review_database = review_root / "database"
        review_dataset = review_root / "dataset"
        review_temp = review_dataset / "_temp"

        if job.config.get("manual_resume_index") == job.step_index:
            moved, errors, first_error = self._restore_review_temp(review_dataset, review_temp)
            if errors:
                return StepResult(status="FAIL", message=f"Manual review restore failed: {first_error}")
            leftovers = [p for p in review_temp.rglob("*") if p.is_file()] if review_temp.exists() else []
            if leftovers:
                return StepResult(status="FAIL", message=f"Manual review restore blocked: {len(leftovers)} file(s) remain in _temp.")
            with self.lock:
                job.config["manual_resume_index"] = None
                job.config["manual_pending_index"] = None
                job.artifacts["working_copy"] = str(review_dataset)
                job.artifacts["review_dataset_root"] = str(review_dataset)
                self._add_log(job, f"[Manual review] Restored {moved} file(s) from _temp. Continuing workflow.")
            return StepResult(status="SUCCESS", message="Manual review complete")

        target = self._resolve_input_dir(job, step_cfg)
        if not target or not target.exists():
            return StepResult(status="FAIL", message="No working dataset is available for manual review.")
        try:
            if review_root.exists():
                shutil.rmtree(review_root)
            review_database.mkdir(parents=True, exist_ok=True)
            review_dataset.mkdir(parents=True, exist_ok=True)
            review_temp.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return StepResult(status="FAIL", message=f"Cannot prepare review project: {exc}")

        copied, copy_errors, copy_error = self._copy_tree_for_review(Path(target), review_dataset)
        if copy_errors:
            return StepResult(status="FAIL", message=f"Cannot copy working dataset for review: {copy_error}")
        moved, move_errors, move_error = self._move_review_dataset_to_temp(review_dataset, review_temp)
        if move_errors:
            return StepResult(status="FAIL", message=f"Cannot stage review files into _temp: {move_error}")
        trigger = self._resolve_trigger_tag(job, step_cfg)
        try:
            (review_root / "prompt.txt").write_text((trigger.strip() or "trigger_word") + "\n", encoding="utf-8")
        except Exception as exc:
            return StepResult(status="FAIL", message=f"Cannot write review prompt.txt: {exc}")

        msg = (
            (step_cfg.get("message") or "").strip()
            or "Your working dataset is ready in Dataset Tag Editor. Review captions, then return here and click Resume."
        )
        job.config["manual_pending_index"] = job.step_index
        with self.lock:
            job.artifacts.update(
                {
                    "review_project_root": str(review_root),
                    "review_dataset_root": str(review_dataset),
                    "review_temp_root": str(review_temp),
                    "working_copy": str(review_dataset),
                }
            )
            self._add_log(
                job,
                f"[Manual review] Review project prepared. Files copied: {copied} | staged in _temp: {moved} | project: {review_root}",
            )
        return StepResult(status="WAIT", message=msg, wait_status="WAITING_MANUAL")

    def _step_final_cleanup(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        target = self._resolve_input_dir(job, step_cfg)
        if not target or not Path(target).exists():
            return StepResult(status="FAIL", message="No folder to clean.")
        _dedup_txt_folder(Path(target))
        with self.lock:
            self._add_log(job, "Final cleanup done (dedup tags).")
        return StepResult(status="SUCCESS", message="Cleanup done")

    def _step_dataset_audit(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        target = self._resolve_input_dir(job, step_cfg)
        if not target or not Path(target).exists():
            return StepResult(status="FAIL", message="No folder to audit.")
        cfg = job.config
        require_pairs = _parse_bool(step_cfg.get("require_caption_pairs", cfg.get("require_caption_pairs", False)))
        fail_on_errors = _parse_bool(step_cfg.get("fail_on_audit_errors", cfg.get("fail_on_audit_errors", require_pairs)))
        include_token_check = _parse_bool(step_cfg.get("include_token_check", cfg.get("include_token_check", require_pairs)))
        token_limit = _parse_int(step_cfg.get("token_warning_limit") or cfg.get("token_warning_limit"), 77)
        image_exts = _coerce_exts(cfg.get("image_exts"))
        recursive = bool(cfg.get("recursive", True))
        report = _scan_dataset_quality(Path(target), image_exts, recursive, require_pairs, include_token_check, token_limit)
        report["dataset_root"] = str(target)
        report["created_at"] = datetime.utcnow().isoformat()
        workspace = self._job_workspace(job)
        if not workspace:
            return StepResult(status="FAIL", message="Job workspace is not available for audit report.")
        reports_dir = workspace / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "dataset_audit.json"
        summary_path = reports_dir / "dataset_audit.txt"
        try:
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            lines = [
                "Dataset Audit",
                f"Dataset: {target}",
                f"Images: {report['summary']['total_images']}",
                f"Captions: {report['summary']['total_txt']}",
                f"Valid pairs: {report['summary']['valid_pairs']}",
                f"Errors: {len(report.get('errors') or [])}",
                f"Warnings: {len(report.get('warnings') or [])}",
            ]
            for item in (report.get("errors") or []) + (report.get("warnings") or []):
                lines.append(f"[{item.get('severity')}] {item.get('message')}")
            summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as exc:
            return StepResult(status="FAIL", message=f"Cannot write audit report: {exc}")
        with self.lock:
            job.artifacts["dataset_audit_report"] = str(report_path)
            job.artifacts["dataset_audit_summary"] = str(summary_path)
            job.artifacts["dataset_audit_passed"] = bool(report.get("ok"))
            self._add_log(
                job,
                "[Dataset audit] "
                f"Images: {report['summary']['total_images']} | captions: {report['summary']['total_txt']} | "
                f"valid pairs: {report['summary']['valid_pairs']} | errors: {len(report.get('errors') or [])} | "
                f"warnings: {len(report.get('warnings') or [])}",
                level="error" if (report.get("errors") and fail_on_errors) else "info",
            )
        if report.get("errors") and fail_on_errors:
            first = (report.get("errors") or [{}])[0]
            return StepResult(status="FAIL", message=first.get("message") or "Dataset audit failed.")
        return StepResult(status="SUCCESS", message="Dataset audit complete")

    def _export_file_allowed(self, path: Path, root: Path, include_txt: bool) -> bool:
        try:
            rel = path.relative_to(root)
        except Exception:
            return False
        parts = [part.lower() for part in rel.parts]
        if "_temp" in parts:
            return False
        if path.suffix.lower() == ".bak":
            return False
        if path.name.lower() in {"state.json", ".pipeline_job.json"}:
            return False
        if path.name.startswith("."):
            return False
        if not include_txt and path.suffix.lower() == ".txt":
            return False
        return True

    def _step_export_final(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        target = self._resolve_input_dir(job, step_cfg)
        if not target or not Path(target).exists():
            return StepResult(status="FAIL", message="No working dataset to export.")
        cfg = job.config
        fail_on_audit = _parse_bool(step_cfg.get("fail_on_audit_errors", cfg.get("fail_on_audit_errors", True)))
        if fail_on_audit and not job.artifacts.get("dataset_audit_passed"):
            return StepResult(status="FAIL", message="Final export requires a passed Dataset Audit.")
        output = _optional_path(step_cfg.get("output_dir"))
        if not output:
            output, err = _required_path(cfg.get("output_dir"), "output_dir")
            if err or not output:
                return StepResult(status="FAIL", message=err or "output_dir required")
        dataset_name = str(job.artifacts.get("dataset_name") or Path(target).name or "dataset")
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        final_dir = output / f"{dataset_name}__final__{ts}"
        if final_dir.exists():
            return StepResult(status="FAIL", message=f"Final output folder already exists: {final_dir}")
        include_txt = _parse_bool(step_cfg.get("include_txt", True))
        image_exts = set(_coerce_exts(cfg.get("image_exts"))) | IMAGE_EXTENSIONS_DEFAULT
        stats = {"images": 0, "captions": 0, "files": 0, "excluded": 0, "skipped": 0}
        try:
            final_dir.mkdir(parents=True, exist_ok=False)
            for path in Path(target).rglob("*"):
                if self._should_stop(job):
                    raise InterruptedError("Stop requested during final export.")
                if path.is_dir():
                    continue
                if not self._export_file_allowed(path, Path(target), include_txt):
                    stats["excluded"] += 1
                    continue
                rel = path.relative_to(Path(target))
                dest = final_dir / rel
                if dest.exists():
                    stats["skipped"] += 1
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix.lower() == ".txt" and include_txt:
                    tag_file = normalizer.parse_tag_file(path)
                    trigger = self._resolve_trigger_tag(job, step_cfg)
                    if trigger:
                        trigger_key = tag_compare_key(trigger)
                        tag_file.main = [
                            trigger if tag_compare_key(tag) == trigger_key else tag
                            for tag in tag_file.main
                        ]
                        tag_file.optional = [
                            trigger if tag_compare_key(tag) == trigger_key else tag
                            for tag in tag_file.optional
                        ]
                        tag_file.protected_literals = [trigger]
                    dest.write_text(normalizer.format_tag_file(tag_file), encoding="utf-8")
                else:
                    shutil.copy2(path, dest)
                stats["files"] += 1
                if path.suffix.lower() in image_exts:
                    stats["images"] += 1
                elif path.suffix.lower() == ".txt":
                    stats["captions"] += 1
        except InterruptedError:
            return StepResult(status="STOP", message="Stopped during final export.")
        except Exception as exc:
            return StepResult(status="FAIL", message=f"Final export failed: {exc}")

        final_zip: Optional[Path] = None
        if _parse_bool(step_cfg.get("create_zip", True)):
            final_zip = final_dir.with_suffix(".zip")
            try:
                _zipdir(final_dir, final_zip, include_txt=include_txt, should_stop=lambda: self._should_stop(job))
            except InterruptedError:
                return StepResult(status="STOP", message="Stopped while creating final ZIP.")
            except Exception as exc:
                return StepResult(status="FAIL", message=f"Final ZIP failed: {exc}")
        with self.lock:
            job.artifacts["final_dataset_dir"] = str(final_dir)
            job.artifacts["export_stats"] = stats
            if final_zip:
                job.artifacts["final_zip_path"] = str(final_zip)
            self._add_log(
                job,
                "[Export final dataset] "
                f"Images exported: {stats['images']} | captions exported: {stats['captions']} | "
                f"files excluded: {stats['excluded']} | skipped: {stats['skipped']} | folder: {final_dir}"
                + (f" | zip: {final_zip}" if final_zip else ""),
            )
        return StepResult(status="SUCCESS", message="Final dataset exported")

    def _step_zip_final(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        target = self._resolve_input_dir(job, step_cfg)
        if not target or not Path(target).exists():
            return StepResult(status="FAIL", message="No folder to zip.")
        output = _optional_path(step_cfg.get("output_dir"))
        if not output:
            output, err = _required_path(job.config.get("output_dir"), "output_dir")
            if err or not output:
                return StepResult(status="FAIL", message=err or "output_dir required")
        dataset_name = job.artifacts.get("dataset_name", "dataset")
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest_zip = output / f"{dataset_name}__final__{ts}.zip"
        include_txt = _parse_bool(step_cfg.get("include_txt", True))
        try:
            _zipdir(
                Path(target),
                dest_zip,
                include_txt=include_txt,
                should_stop=lambda: self._should_stop(job),
            )
        except InterruptedError:
            return StepResult(status="STOP", message="Stopped during zip step.")
        except Exception as exc:
            return StepResult(status="FAIL", message=f"Final zip failed: {exc}")
        with self.lock:
            job.artifacts["final_zip_path"] = str(dest_zip)
            self._add_log(job, f"Final zip built: {dest_zip}")
        return StepResult(status="SUCCESS", message="Final zip ready")

    def _stage_tag_editor_temp(self, job: PipelineJob, target: Path, image_exts: List[str]) -> StepResult:
        temp_root = target / "_temp"
        try:
            temp_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return StepResult(status="FAIL", message=f"Cannot create _temp folder for tag editor: {exc}")

        try:
            has_existing_files = any(p.is_file() for p in temp_root.rglob("*"))
        except Exception as exc:
            return StepResult(status="FAIL", message=f"Cannot inspect _temp folder: {exc}")
        if has_existing_files:
            return StepResult(
                status="FAIL",
                message=f"Tag editor staging requires empty _temp folder: {temp_root}",
            )

        images = [p for p in _find_images(target, recursive=True, exts=image_exts) if temp_root not in p.parents]
        if not images:
            return StepResult(status="FAIL", message="No images found to stage for tag editor.")

        moved: List[Tuple[Path, Path]] = []
        try:
            for src_img in images:
                if self._should_stop(job):
                    raise InterruptedError("Stop requested during tag editor staging.")
                rel = src_img.relative_to(target)
                dst_img = temp_root / rel
                src_txt = src_img.with_suffix(".txt")
                dst_txt = dst_img.with_suffix(".txt")
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_img), str(dst_img))
                moved.append((dst_img, src_img))
                if src_txt.exists():
                    shutil.move(str(src_txt), str(dst_txt))
                    moved.append((dst_txt, src_txt))
        except InterruptedError as exc:
            rollback_errors: List[str] = []
            for moved_path, original_path in reversed(moved):
                try:
                    if not moved_path.exists():
                        continue
                    original_path.parent.mkdir(parents=True, exist_ok=True)
                    if original_path.exists():
                        rollback_errors.append(f"{moved_path.name}: source already exists")
                        continue
                    shutil.move(str(moved_path), str(original_path))
                except Exception as rollback_exc:
                    rollback_errors.append(f"{moved_path.name}: {rollback_exc}")
            message = str(exc)
            if rollback_errors:
                message = f"{message} | rollback issues: {'; '.join(rollback_errors)}"
            return StepResult(status="STOP", message=message)
        except Exception as exc:
            rollback_errors: List[str] = []
            for moved_path, original_path in reversed(moved):
                try:
                    if not moved_path.exists():
                        continue
                    original_path.parent.mkdir(parents=True, exist_ok=True)
                    if original_path.exists():
                        rollback_errors.append(f"{moved_path.name}: source already exists")
                        continue
                    shutil.move(str(moved_path), str(original_path))
                except Exception as rollback_exc:
                    rollback_errors.append(f"{moved_path.name}: {rollback_exc}")
            message = f"Failed staging files for tag editor: {exc}"
            if rollback_errors:
                message = f"{message} | rollback issues: {'; '.join(rollback_errors)}"
            return StepResult(status="FAIL", message=message)

        with self.lock:
            self._add_log(job, f"Tag editor staging ready: {len(images)} image(s) moved to {temp_root}")
        return StepResult(status="SUCCESS", message="staged")

    def _restore_tag_editor_temp(self, job: PipelineJob, target: Path, image_exts: List[str]) -> StepResult:
        restore_form = {
            "folder": str(target),
            "mode": "undo",
            "exts": ",".join(image_exts),
            "backup": False,
        }
        try:
            raw = tag_editor.handle(restore_form, {})
        except Exception as exc:
            return StepResult(status="FAIL", message=f"Tag editor restore failed: {exc}")

        _, log, meta = unpack_tool_result(raw)
        self._log_tool_output(job, log or "")
        if not bool(meta.get("ok", True)):
            error = str(meta.get("error") or "").strip()
            if "No image files found in _temp." not in error:
                return StepResult(status="FAIL", message=error or "Tag editor restore failed.")

        temp_root = target / "_temp"
        leftovers = [p for p in temp_root.rglob("*") if p.is_file()] if temp_root.exists() else []
        moved_leftovers = 0
        renamed_leftovers = 0
        for src in leftovers:
            rel = src.relative_to(temp_root)
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            final_dst = dst
            bump = 1
            while final_dst.exists():
                final_dst = dst.with_name(f"{dst.stem}_{bump}{dst.suffix}")
                bump += 1
            if final_dst != dst:
                renamed_leftovers += 1
            try:
                shutil.move(str(src), str(final_dst))
                moved_leftovers += 1
            except Exception as exc:
                return StepResult(status="FAIL", message=f"Failed moving leftover temp file {src}: {exc}")

        if temp_root.exists():
            dirs = sorted([p for p in temp_root.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True)
            for d in dirs:
                try:
                    d.rmdir()
                except Exception:
                    pass

        with self.lock:
            self._add_log(
                job,
                "Tag editor staging restored."
                f" Leftovers moved: {moved_leftovers} (renamed: {renamed_leftovers}).",
            )
        return StepResult(status="SUCCESS", message="restored")

    def _step_tag_editor(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        mode = (step_cfg.get("mode") or "manual").strip().lower()
        if mode in {"move", "undo"}:
            with self.lock:
                self._add_log(
                    job,
                    f"Legacy tag editor mode '{mode}' is not supported in pipeline anymore. Switching to manual pause.",
                    level="warning",
                )
            mode = "manual"
        if mode in {"manual", "pause"}:
            return self._step_manual_pause(job, step_cfg)
        if mode not in {"insert", "delete", "replace", "dedup"}:
            return StepResult(status="FAIL", message=f"Unsupported tag editor mode for pipeline: {mode}")
        target = self._resolve_input_dir(job, step_cfg)
        if not target or not target.exists():
            return StepResult(status="FAIL", message="No folder for tag editor.")
        image_exts = _coerce_exts(step_cfg.get("exts") or job.config.get("image_exts"))
        stage = self._stage_tag_editor_temp(job, Path(target), image_exts)
        if stage.status != "SUCCESS":
            return stage
        form = {
            "folder": str(Path(target)),
            "mode": mode,
            "tags": step_cfg.get("tags") or "",
            "exts": ",".join(image_exts),
            "backup": _parse_bool(step_cfg.get("backup")),
            "create_missing_txt": _parse_bool(step_cfg.get("create_missing_txt")),
        }
        edit_result = self._run_tool(job, tag_editor.handle, form)
        restore_result = self._restore_tag_editor_temp(job, Path(target), image_exts)
        if restore_result.status != "SUCCESS":
            if edit_result.status == "SUCCESS":
                return restore_result
            return StepResult(
                status=edit_result.status,
                message=f"{edit_result.message} | restore issue: {restore_result.message}",
            )
        return edit_result

    def _step_webp_to_png(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        target = self._resolve_input_dir(job, step_cfg)
        if not target or not target.exists():
            return StepResult(status="FAIL", message="Source folder not found for Image -> PNG.")
        output = self._resolve_output_dir(job, step_cfg, "webp_to_png")
        if not output:
            return StepResult(status="FAIL", message="Output folder required for Image -> PNG.")
        form = {"src_png": str(target), "dst_png": str(output)}
        result = self._run_tool(job, webp_converter.handle, form)
        if result.status == "SUCCESS":
            with self.lock:
                job.artifacts["working_copy"] = str(output)
                job.artifacts["png_output"] = str(output)
                job.artifacts["webp_output"] = str(output)
        return result

    def _step_batch_adjust(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        target = self._resolve_input_dir(job, step_cfg)
        if not target or not target.exists():
            return StepResult(status="FAIL", message="Source folder not found for batch adjust.")
        output = self._resolve_output_dir(job, step_cfg, "adjusted")
        if not output:
            return StepResult(status="FAIL", message="Output folder required for batch adjust.")
        preset_name = (step_cfg.get("preset_name") or step_cfg.get("preset") or "").strip()
        presets = job.config.get("preset_library") or {}
        if not preset_name:
            return StepResult(status="FAIL", message="Preset name is required for batch adjust.")
        if preset_name not in presets:
            return StepResult(status="FAIL", message=f"Preset not found: {preset_name}")
        form = {
            "src_batch": str(target),
            "dst_batch": str(output),
            "preset": preset_name,
            "suffix": step_cfg.get("suffix") or "_adj",
            "limit": _parse_int(step_cfg.get("limit"), 0),
        }
        # Optional overrides (so preset can be tweaked per pipeline step)
        for k in (
            "exposure_ev","brightness","contrast","highlights","shadows",
            "saturation","warmth","tint","sharpness","vignette","output_format",
        ):
            if k in step_cfg and step_cfg.get(k) is not None:
                form[k] = step_cfg.get(k)
        if "recursive" in step_cfg:
            form["recursive"] = _parse_bool(step_cfg.get("recursive"))

        result = self._run_tool(job, batch_adjust.handle, form, ctx={"presets": presets})
        if result.status == "SUCCESS":
            with self.lock:
                job.artifacts["working_copy"] = str(output)
                job.artifacts["batch_adjust_output"] = str(output)
        return result

    def _step_combine_datasets(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        source_folders = step_cfg.get("source_folders") or ""
        legacy_extra = step_cfg.get("extra_folders") or ""
        if isinstance(source_folders, list):
            source_folders = "\n".join(str(x) for x in source_folders if str(x).strip())
        if isinstance(legacy_extra, list):
            legacy_extra = "\n".join(str(x) for x in legacy_extra if str(x).strip())

        raw_sources: List[str] = []
        legacy_folder_a = _optional_path(step_cfg.get("folder_a"))
        legacy_folder_b = _optional_path(step_cfg.get("folder_b"))
        if legacy_folder_a:
            raw_sources.append(str(legacy_folder_a))
        elif legacy_folder_b or str(legacy_extra).strip():
            # Backward compatibility for old saved pipeline configs that used folder_a default.
            fallback_input = self._resolve_input_dir(job, step_cfg)
            if fallback_input:
                raw_sources.append(str(fallback_input))
        if legacy_folder_b:
            raw_sources.append(str(legacy_folder_b))
        for block in (source_folders, legacy_extra):
            for line in str(block).splitlines():
                value = line.strip()
                if value:
                    raw_sources.append(value)

        source_paths: List[Path] = []
        seen = set()
        for raw in raw_sources:
            path = readable_path(raw)
            key = str(path)
            if os.name == "nt":
                key = key.lower()
            if key in seen:
                continue
            seen.add(key)
            source_paths.append(path)

        output = self._resolve_output_dir(job, step_cfg, "combined")
        if not output:
            return StepResult(status="FAIL", message="Output folder required for combine dataset.")
        if len(source_paths) < 2:
            return StepResult(status="FAIL", message="Provide at least two source folders to combine.")
        form = {
            "source_folders": "\n".join(str(path) for path in source_paths),
            "out_dir": str(output),
            "suffix_combine": step_cfg.get("suffix") or "_B",
            "exts_combine": step_cfg.get("exts") or ".jpg,.jpeg,.png,.webp",
            "move_instead": _parse_bool(step_cfg.get("move_instead")),
        }
        result = self._run_tool(job, combine_datasets.handle, form)
        if result.status == "SUCCESS":
            with self.lock:
                job.artifacts["working_copy"] = str(output)
                job.artifacts["combine_output"] = str(output)
        return result

    def _step_flatten_renumber(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        target = self._resolve_input_dir(job, step_cfg)
        if not target or not target.exists():
            return StepResult(status="FAIL", message="Source folder not found for renamer.")
        output = self._resolve_output_dir(job, step_cfg, "renamed")
        if not output:
            return StepResult(status="FAIL", message="Output folder required for renamer.")
        image_exts = step_cfg.get("exts") or ",".join(_coerce_exts(job.config.get("image_exts")))
        form = {
            "rn_root": str(target),
            "rn_out": str(output),
            "rn_exts": image_exts,
            "rn_start": _parse_int(step_cfg.get("start"), 1),
            "rn_pad": _parse_int(step_cfg.get("pad"), 3),
            "rn_suffix_pad": _parse_int(step_cfg.get("suffix_pad"), 0),
            "rn_sep": step_cfg.get("sep") or "_",
            "rn_top_order": step_cfg.get("top_order") or "name",
            "rn_folder_order": step_cfg.get("folder_order") or "name",
            "rn_inside_order": step_cfg.get("inside_order") or "name",
            "rn_include_txt": _parse_bool(step_cfg.get("include_txt", True)),
            "rn_move_instead": _parse_bool(step_cfg.get("move_instead")),
            "rn_dry_run": _parse_bool(step_cfg.get("dry_run")),
        }
        result = self._run_tool(job, group_renamer.handle, form)
        if result.status == "SUCCESS" and not _parse_bool(step_cfg.get("dry_run")):
            with self.lock:
                job.artifacts["working_copy"] = str(output)
                job.artifacts["renamer_output"] = str(output)
        return result

    def _step_merge_groups(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        target = self._resolve_input_dir(job, step_cfg)
        if not target or not target.exists():
            return StepResult(status="FAIL", message="Source folder not found for stitching.")
        output = self._resolve_output_dir(job, step_cfg, "merged")
        if not output:
            return StepResult(status="FAIL", message="Output folder required for stitching.")
        form = {
            "merge_folder": str(target),
            "merge_out_dir": str(output),
            "merge_glob": step_cfg.get("glob") or "*_*.*",
            "merge_exts": step_cfg.get("exts") or ".png,.jpg,.jpeg,.webp",
            "merge_skip_single": _parse_bool(step_cfg.get("skip_single")),
            "merge_reverse": _parse_bool(step_cfg.get("reverse")),
            "merge_orientation": step_cfg.get("orientation") or "v",
            "merge_resize": step_cfg.get("resize") or "auto",
            "merge_align": step_cfg.get("align") or "center",
            "merge_gap": _parse_int(step_cfg.get("gap"), 0),
            "merge_bg": step_cfg.get("bg") or "#FFFFFF",
            "merge_overwrite": _parse_bool(step_cfg.get("overwrite")),
            "merge_dry_run": _parse_bool(step_cfg.get("dry_run")),
        }
        result = self._run_tool(job, merge_groups_tool.handle, form)
        if result.status == "SUCCESS" and not _parse_bool(step_cfg.get("dry_run")):
            with self.lock:
                job.artifacts["working_copy"] = str(output)
                job.artifacts["merge_output"] = str(output)
        return result

    def _step_webtoon_split(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        target = self._resolve_input_dir(job, step_cfg)
        if not target or not target.exists():
            return StepResult(status="FAIL", message="Source folder not found for webtoon split.")
        output = self._resolve_output_dir(job, step_cfg, "panels")
        if not output:
            return StepResult(status="FAIL", message="Output folder required for webtoon split.")
        form = {
            "wt_folder": str(target),
            "wt_out_dir": str(output),
            "wt_glob": step_cfg.get("glob") or "*.*",
            "wt_exts": step_cfg.get("exts") or ".png,.jpg,.jpeg,.webp",
            "wt_sort_by": step_cfg.get("sort_by") or "name",
            "wt_sort_dir": step_cfg.get("sort_dir") or "asc",
            "wt_resize": step_cfg.get("resize") or "match-width",
            "wt_white_threshold": _parse_int(step_cfg.get("white_threshold"), 245),
            "wt_stripe_colors": step_cfg.get("stripe_colors") or "#FFFFFF",
            "wt_row_ratio": _parse_float(step_cfg.get("row_ratio"), 98),
            "wt_min_stripe": _parse_int(step_cfg.get("min_stripe"), 12),
            "wt_max_gap": _parse_int(step_cfg.get("max_gap"), 2),
            "wt_min_panel": _parse_int(step_cfg.get("min_panel"), 128),
            "wt_save_strip": _parse_bool(step_cfg.get("save_strip", True)),
            "wt_overwrite": _parse_bool(step_cfg.get("overwrite")),
            "wt_dry_run": _parse_bool(step_cfg.get("dry_run")),
        }
        result = self._run_tool(job, webtoon_splitter.handle, form)
        if result.status == "SUCCESS" and not _parse_bool(step_cfg.get("dry_run")):
            with self.lock:
                job.artifacts["working_copy"] = str(output)
                job.artifacts["webtoon_output"] = str(output)
        return result


# Global manager
PIPELINE_MANAGER = PipelineManager()
