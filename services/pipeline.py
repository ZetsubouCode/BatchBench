import json
import os
import shutil
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from utils.io import readable_path, readable_path_or_none
from utils.parse import parse_bool, parse_int, parse_float, parse_optional_int, parse_exts
from utils.tool_result import unpack_tool_result
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
    "normalize": "Dataset normalization",
    "zip_final": "Zip result",
    "dedup_tags": "Tag cleanup (dedup)",
    "webp_to_png": "Image -> PNG",
    "batch_adjust": "Photo adjust (preset)",
    "combine_datasets": "Combine dataset",
    "flatten_renumber": "Flatten & renumber",
    "merge_groups": "Stitch groups",
    "webtoon_split": "Webtoon panel splitter",
}

TAG_STEPS = {"offline_tagger", "tag_editor", "normalize", "dedup_tags"}
IMAGE_ONLY_STEPS = {"webp_to_png", "batch_adjust", "merge_groups", "webtoon_split"}
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
            "normalize": self._step_normalize,
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
        if "zip_final" in ids and ids[-1] != "zip_final":
            return "Zip result should be the last step so the output is always up to date."
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
        return None

    def start_job(self, cfg: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
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

        cfg = dict(cfg)
        cfg["dataset_path"] = str(dataset)
        cfg["working_dir"] = str(working)
        cfg["output_dir"] = str(output)
        cfg["steps"] = self._normalize_steps(cfg)
        order_err = self._validate_step_order(cfg["steps"])
        if order_err:
            return False, "", order_err

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
        with self.lock:
            self._add_log(job, f"Normalized tags: {result}")
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

    def _step_final_cleanup(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        target = self._resolve_input_dir(job, step_cfg)
        if not target or not Path(target).exists():
            return StepResult(status="FAIL", message="No folder to clean.")
        _dedup_txt_folder(Path(target))
        with self.lock:
            self._add_log(job, "Final cleanup done (dedup tags).")
        return StepResult(status="SUCCESS", message="Cleanup done")

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

    def _step_tag_editor(self, job: PipelineJob, step_cfg: Dict[str, Any]) -> StepResult:
        mode = (step_cfg.get("mode") or "manual").strip().lower()
        if mode in {"manual", "pause"}:
            return self._step_manual_pause(job, step_cfg)
        target = self._resolve_input_dir(job, step_cfg)
        if not target or not target.exists():
            return StepResult(status="FAIL", message="No folder for tag editor.")
        image_exts = step_cfg.get("exts") or ",".join(_coerce_exts(job.config.get("image_exts")))
        form = {
            "folder": str(target),
            "mode": mode,
            "edit_target": (step_cfg.get("edit_target") or "recursive").strip().lower(),
            "tags": step_cfg.get("tags") or "",
            "exts": image_exts,
            "backup": _parse_bool(step_cfg.get("backup")),
            "temp_dir": step_cfg.get("temp_dir") or "",
        }
        return self._run_tool(job, tag_editor.handle, form)

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
            "wt_resize": step_cfg.get("resize") or "match-width",
            "wt_white_threshold": _parse_int(step_cfg.get("white_threshold"), 245),
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
