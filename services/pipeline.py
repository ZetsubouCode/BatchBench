import json
import os
import shutil
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from utils.io import readable_path
from services import normalizer

PIPELINE_STATUSES = {
    "QUEUED",
    "RUNNING",
    "WAITING_MANUAL",
    "WAITING_DOWNLOAD",
    "COMPLETED",
    "FAILED",
    "STOPPED",
}


@dataclass
class StepResult:
    status: str  # SUCCESS | WAIT | FAIL | STOP
    message: str = ""
    wait_status: Optional[str] = None  # WAITING_MANUAL / WAITING_DOWNLOAD


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

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["created_at"] = datetime.fromtimestamp(self.created_at).isoformat()
        return out


def _now_ts() -> float:
    return time.time()


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _zipdir(src: Path, dest_zip: Path, include_txt: bool = True, include_images_only: bool = False):
    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src):
            for fname in files:
                p = Path(root) / fname
                if include_images_only and p.suffix.lower() in {".txt", ".json", ".md"}:
                    continue
                if include_txt is False and p.suffix.lower() == ".txt":
                    continue
                arcname = str(p.relative_to(src))
                zf.write(p, arcname)


def _copy_dataset(src: Path, dest: Path, image_exts: List[str], recursive: bool):
    _ensure_dir(dest)
    if recursive:
        for p in src.rglob("*"):
            if p.is_dir():
                continue
            rel = p.relative_to(src)
            if p.is_file():
                if p.suffix.lower() in image_exts or p.suffix.lower() == ".txt":
                    dest_path = dest / rel
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dest_path)
    else:
        for p in src.iterdir():
            if p.is_file() and (p.suffix.lower() in image_exts or p.suffix.lower() == ".txt"):
                dest_path = dest / p.name
                shutil.copy2(p, dest_path)


def _find_images(folder: Path, recursive: bool, exts: List[str]) -> List[Path]:
    exts_lower = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    if recursive:
        return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts_lower])
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts_lower])


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
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    # ---------- persistence ----------
    def _persist_job(self, job: PipelineJob):
        workdir = Path(job.config.get("working_dir") or ".")
        _ensure_dir(workdir)
        path = workdir / ".pipeline_job.json"
        try:
            path.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
        except Exception:
            pass

    # ---------- job helpers ----------
    def _add_log(self, job: PipelineJob, message: str, level: str = "info"):
        job.log.append({"ts": datetime.utcnow().isoformat(), "level": level, "message": message})
        job.log = job.log[-500:]

    def _register_job(self, job: PipelineJob):
        with self.lock:
            self.jobs[job.id] = job
        self.wake.set()

    def start_job(self, cfg: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
        dataset = readable_path(cfg.get("dataset_path", ""))
        working = readable_path(cfg.get("working_dir", ""))
        output = readable_path(cfg.get("output_dir", ""))
        if not dataset:
            return False, "", "dataset_path required"
        if not working:
            return False, "", "working_dir required"
        if not output:
            return False, "", "output_dir required"
        if not dataset.exists():
            return False, "", f"Dataset not found: {dataset}"

        job_id = str(uuid4())
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
        self._add_log(job, "Job created.")
        self._register_job(job)
        self._persist_job(job)
        return True, job_id, None

    def pause_job(self, job_id: str) -> Tuple[bool, str]:
        job = self.jobs.get(job_id)
        if not job:
            return False, "job not found"
        with self.lock:
            job.status = "WAITING_MANUAL"
            job.waiting_reason = "Paused by user"
            self._add_log(job, "Paused by user.")
            self._persist_job(job)
        return True, "paused"

    def stop_job(self, job_id: str) -> Tuple[bool, str]:
        job = self.jobs.get(job_id)
        if not job:
            return False, "job not found"
        with self.lock:
            job.status = "STOPPED"
            self._add_log(job, "Stopped by user.")
            self._persist_job(job)
        return True, "stopped"

    def resume_job(self, job_id: str) -> Tuple[bool, str]:
        job = self.jobs.get(job_id)
        if not job:
            return False, "job not found"
        with self.lock:
            if job.status in {"COMPLETED", "FAILED", "STOPPED"}:
                return False, f"cannot resume from {job.status}"
            job.status = "QUEUED"
            job.waiting_reason = ""
            self._add_log(job, "Resumed.")
            self._persist_job(job)
        self.wake.set()
        return True, "resumed"

    def get_status(self, job_id: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        if not job_id and self.jobs:
            # return latest job
            job_id = sorted(self.jobs.values(), key=lambda j: j.created_at)[-1].id
        job = self.jobs.get(job_id or "")
        if not job:
            return False, None, "job not found"
        return True, job.to_dict(), ""

    # ---------- steps ----------
    def _build_step_names(self, cfg: Dict[str, Any]) -> List[str]:
        names = ["Validate & Prepare"]
        run_autotag = bool(cfg.get("run_autotag", True))
        if run_autotag:
            names += [
                "Zip for CivitAI",
                "Manual upload",
                "Wait download",
                "Extract result",
            ]
        names.append("Normalize tags")
        names.append("Manual retag (pause)")
        names.append("Final cleanup")
        names.append("Build final zip")
        names.append("Done")
        return names

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
                    self._persist_job(job)

    def _next_job(self) -> Optional[PipelineJob]:
        with self.lock:
            for job in sorted(self.jobs.values(), key=lambda j: j.created_at):
                if job.status in {"QUEUED", "RUNNING"}:
                    return job
        return None

    def _run_job(self, job: PipelineJob):
        steps = self._build_steps(job)
        while job.step_index < len(steps):
            step_name, fn = steps[job.step_index]
            with self.lock:
                if job.status == "STOPPED":
                    self._persist_job(job)
                    return
                job.status = "RUNNING"
                job.current_step = step_name
                self._persist_job(job)

            result = fn(job)

            with self.lock:
                if result.status == "SUCCESS":
                    job.step_index += 1
                    if job.step_index < len(steps):
                        job.current_step = steps[job.step_index][0]
                elif result.status == "WAIT":
                    job.status = result.wait_status or "WAITING_MANUAL"
                    job.waiting_reason = result.message
                    self._add_log(job, f"Waiting: {result.message}")
                    self._persist_job(job)
                    return
                elif result.status == "FAIL":
                    job.status = "FAILED"
                    self._add_log(job, result.message, level="error")
                    self._persist_job(job)
                    return
                elif result.status == "STOP":
                    job.status = "STOPPED"
                    self._add_log(job, "Stopped.")
                    self._persist_job(job)
                    return

                self._persist_job(job)

        with self.lock:
            job.status = "COMPLETED"
            job.current_step = "Done"
            self._add_log(job, "Pipeline completed.")
            self._persist_job(job)

    # Step builders
    def _build_steps(self, job: PipelineJob) -> List[Tuple[str, Any]]:
        cfg = job.config
        run_autotag = bool(cfg.get("run_autotag", True))
        steps: List[Tuple[str, Any]] = [
            ("Validate & Prepare", lambda j: self._step_validate(j)),
        ]
        if run_autotag:
            steps.extend(
                [
                    ("Zip for CivitAI", lambda j: self._step_zip_for_autotag(j)),
                    ("Manual upload", lambda j: self._step_wait_manual_upload(j)),
                    ("Wait download", lambda j: self._step_wait_download(j)),
                    ("Extract result", lambda j: self._step_extract_result(j)),
                ]
            )
        steps.extend(
            [
                ("Normalize tags", lambda j: self._step_normalize(j)),
                ("Manual retag (pause)", lambda j: self._step_manual_pause(j)),
                ("Final cleanup", lambda j: self._step_final_cleanup(j)),
                ("Build final zip", lambda j: self._step_zip_final(j)),
                ("Done", lambda j: StepResult(status="SUCCESS", message="Done")),
            ]
        )
        return steps

    # ----- individual steps -----
    def _step_validate(self, job: PipelineJob) -> StepResult:
        cfg = job.config
        dataset = readable_path(cfg.get("dataset_path", ""))
        working = readable_path(cfg.get("working_dir", ""))
        output = readable_path(cfg.get("output_dir", ""))
        image_exts = cfg.get("image_exts") or [".jpg", ".jpeg", ".png", ".webp"]
        recursive = bool(cfg.get("recursive"))

        if not dataset.exists():
            return StepResult(status="FAIL", message=f"Dataset not found: {dataset}")

        images = _find_images(dataset, recursive=recursive, exts=image_exts)
        if not images:
            return StepResult(status="FAIL", message="No images found in dataset")

        raw_zip_dir = working / "raw_zip"
        civitai_dir = working / "civitai_result"
        normalized_dir = working / "normalized"
        final_dir = working / "final"
        extracted_dir = civitai_dir / "extracted"
        for d in [raw_zip_dir, civitai_dir, normalized_dir, final_dir, extracted_dir, output]:
            _ensure_dir(d)

        # copy dataset to normalized work area
        _copy_dataset(dataset, normalized_dir, image_exts=image_exts, recursive=recursive)

        with self.lock:
            job.artifacts.update(
                {
                    "dataset_name": dataset.name,
                    "raw_zip_dir": str(raw_zip_dir),
                    "civitai_dir": str(civitai_dir),
                    "normalized_dir": str(normalized_dir),
                    "final_dir": str(final_dir),
                    "extracted_dir": str(extracted_dir),
                    "output_dir": str(output),
                    "working_copy": str(normalized_dir),
                    "watch_started_at": _now_ts(),
                }
            )
            self._add_log(job, f"Prepared working dirs. Images: {len(images)}")
        return StepResult(status="SUCCESS", message="Prepared")

    def _step_zip_for_autotag(self, job: PipelineJob) -> StepResult:
        image_exts = job.config.get("image_exts") or [".jpg", ".jpeg", ".png", ".webp"]
        normalized_dir = Path(job.artifacts.get("working_copy"))
        raw_zip_dir = Path(job.artifacts.get("raw_zip_dir"))
        dataset_name = job.artifacts.get("dataset_name", "dataset")
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest_zip = raw_zip_dir / f"{dataset_name}__for_autotag__{ts}.zip"
        images = _find_images(normalized_dir, recursive=True, exts=image_exts)
        if not images:
            return StepResult(status="FAIL", message="No images to zip for autotag.")
        _zipdir(normalized_dir, dest_zip, include_txt=False, include_images_only=True)
        with self.lock:
            job.artifacts["source_zip_path"] = str(dest_zip)
            self._add_log(job, f"Zip created for CivitAI: {dest_zip}")
        return StepResult(status="SUCCESS", message="Zipped for autotag")

    def _step_wait_manual_upload(self, job: PipelineJob) -> StepResult:
        msg = "Upload the source zip to CivitAI, then resume."
        with self.lock:
            self._add_log(job, msg)
        return StepResult(status="WAIT", message=msg, wait_status="WAITING_MANUAL")

    def _step_wait_download(self, job: PipelineJob) -> StepResult:
        cfg = job.config
        auto_detect = bool(cfg.get("auto_detect_download", True))
        if not auto_detect:
            msg = "Place downloaded zip in watch folder then resume."
            with self.lock:
                self._add_log(job, msg)
            return StepResult(status="WAIT", message=msg, wait_status="WAITING_DOWNLOAD")

        watch_dir = readable_path(cfg.get("downloads_watch") or Path.home() / "Downloads")
        pattern = (cfg.get("download_pattern") or "").strip()
        timeout = int(cfg.get("download_timeout") or 300)
        poll_interval = int(cfg.get("download_poll_interval") or 2)
        started = float(job.artifacts.get("watch_started_at") or _now_ts())
        now = _now_ts()

        if now - started > timeout:
            return StepResult(status="FAIL", message="Timeout waiting for downloaded zip.")

        candidate = None
        if watch_dir.exists():
            for p in sorted(watch_dir.glob("*.zip"), key=lambda x: x.stat().st_mtime, reverse=True):
                if p.stat().st_mtime < started:
                    continue
                if pattern:
                    if pattern.lower() not in p.name.lower():
                        continue
                candidate = p
                break

        if not candidate:
            with self.lock:
                self._add_log(job, f"No download detected yet in {watch_dir}. Rechecking...")
            time.sleep(poll_interval)
            return StepResult(status="WAIT", message="Waiting download", wait_status="WAITING_DOWNLOAD")

        civitai_dir = Path(job.artifacts.get("civitai_dir"))
        dest = civitai_dir / candidate.name
        shutil.copy2(candidate, dest)
        with self.lock:
            job.artifacts["civitai_result_zip_path"] = str(dest)
            self._add_log(job, f"Downloaded zip detected: {candidate}")
        return StepResult(status="SUCCESS", message="Download detected")

    def _step_extract_result(self, job: PipelineJob) -> StepResult:
        zip_path = job.artifacts.get("civitai_result_zip_path")
        extracted_dir = Path(job.artifacts.get("extracted_dir"))
        if not zip_path or not Path(zip_path).exists():
            return StepResult(status="FAIL", message="No downloaded zip to extract.")
        _ensure_dir(extracted_dir)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extracted_dir)
        except Exception as exc:
            return StepResult(status="FAIL", message=f"Extract failed: {exc}")
        with self.lock:
            job.artifacts["extracted_folder"] = str(extracted_dir)
            self._add_log(job, f"Extracted download to {extracted_dir}")
        return StepResult(status="SUCCESS", message="Extracted")

    def _step_normalize(self, job: PipelineJob) -> StepResult:
        cfg = job.config
        target = job.artifacts.get("extracted_folder") or job.artifacts.get("working_copy")
        if not target or not Path(target).exists():
            return StepResult(status="FAIL", message="Nothing to normalize.")
        opts = normalizer.NormalizeOptions(
            dataset_path=Path(target),
            recursive=True,
            include_missing_txt=True,
            preset_type=cfg.get("preset_type") or "anime",
            preset_file=cfg.get("preset_file") or "",
            extra_remove=normalizer.clean_input_list(cfg.get("extra_remove") or ""),
            extra_keep=normalizer.clean_input_list(cfg.get("extra_keep") or ""),
            move_unknown_background_to_optional=bool(cfg.get("move_unknown_background_to_optional")),
            background_threshold=None,
            normalize_order=True,
            preview_limit=30,
            backup_enabled=True,
            image_exts=normalizer.DEFAULT_IMAGE_EXTS,
            identity_tags=normalizer.clean_input_list(cfg.get("identity_tags") or ""),
        )
        if not opts.preset_file:
            return StepResult(status="FAIL", message="preset_file is required for normalization.")
        result = normalizer.apply_normalization(opts, Path(cfg.get("preset_root")))
        if not result.get("ok"):
            return StepResult(status="FAIL", message=result.get("error") or "normalize failed")
        with self.lock:
            self._add_log(job, f"Normalized tags: {result}")
        return StepResult(status="SUCCESS", message="Normalized")

    def _step_manual_pause(self, job: PipelineJob) -> StepResult:
        msg = "Pause for manual retagging. Edit files in normalized folder, then resume."
        with self.lock:
            self._add_log(job, msg)
        return StepResult(status="WAIT", message=msg, wait_status="WAITING_MANUAL")

    def _step_final_cleanup(self, job: PipelineJob) -> StepResult:
        target = job.artifacts.get("extracted_folder") or job.artifacts.get("working_copy")
        if not target or not Path(target).exists():
            return StepResult(status="FAIL", message="No folder to clean.")
        _dedup_txt_folder(Path(target))
        with self.lock:
            self._add_log(job, "Final cleanup done (dedup tags).")
        return StepResult(status="SUCCESS", message="Cleanup done")

    def _step_zip_final(self, job: PipelineJob) -> StepResult:
        target = job.artifacts.get("extracted_folder") or job.artifacts.get("working_copy")
        output = readable_path(job.config.get("output_dir"))
        dataset_name = job.artifacts.get("dataset_name", "dataset")
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest_zip = output / f"{dataset_name}__final__{ts}.zip"
        try:
            _zipdir(Path(target), dest_zip, include_txt=True)
        except Exception as exc:
            return StepResult(status="FAIL", message=f"Final zip failed: {exc}")
        with self.lock:
            job.artifacts["final_zip_path"] = str(dest_zip)
            self._add_log(job, f"Final zip built: {dest_zip}")
        return StepResult(status="SUCCESS", message="Final zip ready")


# Global manager
PIPELINE_MANAGER = PipelineManager()
